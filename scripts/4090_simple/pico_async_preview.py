"""Bounded asynchronous MuJoCo EGL feedback; never physics, SDK or training RGB.

Importing the module loads no MuJoCo/GL backend. Model hashes, MjData creation,
FK, camera transforms, GL drawing and readback all belong to a spawn worker.
The caller copies only an actual archived integration state. Preview drops do
not drop or manufacture physical states, commands, observations or labels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import time

import numpy as np

CONTRACT = 'cvpr-asynchronous-mujoco-egl-preview-v1'
DEFAULT_CAMERAS = {'head_stereo_left': (640, 360)}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _number(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (int,float))
             and math.isfinite(value), 'Finite '+name+' required')
    return float(value)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class _Mailbox:
    """One immutable serialized value, no feeder thread or blocking control put."""
    def __init__(self, context, capacity):
        self.capacity = capacity
        self.buffer = context.RawArray('B', capacity)
        self.size = context.RawValue('I', 0)
        self.lock = context.Lock()

    def put(self, value, *, blocking=False):
        encoded = pickle.dumps(value, protocol=5)
        _require(len(encoded) <= self.capacity, 'Preview message exceeds fixed capacity')
        if not self.lock.acquire(blocking):
            return False
        try:
            memoryview(self.buffer).cast('B')[:len(encoded)] = encoded
            self.size.value = len(encoded)
        finally:
            self.lock.release()
        return True

    def get(self, *, consume=False, blocking=False):
        if not self.lock.acquire(blocking):
            return None
        try:
            value = bytes(memoryview(self.buffer).cast('B')[:self.size.value])
            if consume:
                self.size.value = 0
        finally:
            self.lock.release()
        return pickle.loads(value) if value else None

    def clear(self):
        with self.lock:
            self.size.value = 0


@dataclass(frozen=True)
class PreviewFrame:
    images: dict
    epoch: int
    source_tick50: int
    sim_time_s: float
    frame_id: int
    request_id: int
    state_sha256: str
    submitted_monotonic_s: float
    render_started_monotonic_s: float
    render_completed_monotonic_s: float
    received_monotonic_s: float | None
    render_wall_s: float
    evidence: dict

    @property
    def receive_age_s(self):
        return None if self.received_monotonic_s is None else self.received_monotonic_s-self.submitted_monotonic_s


class _MujocoEGLBackend:
    """Worker-owned renderer: setState + FK/cameras, with zero integration."""
    def __init__(self, config):
        # The worker chooses EGL before importing mujoco or creating a context.
        import mujoco
        _require(mujoco.gl_context.GLContext.__module__ == 'mujoco.egl', 'Worker must use actual EGL backend')
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_binary_path(config['model_path'])
        self.spec = config['state_spec']
        _require(self.spec == int(mujoco.mjtState.mjSTATE_INTEGRATION), 'Full mjSTATE_INTEGRATION required')
        self.width = mujoco.mj_stateSize(self.model, self.spec)
        _require(self.width == config['state_width'], 'Archived model integration width differs')
        _require(self.model.neq == self.model.nmocap == 0, 'Preview model has equality/mocap assistance')
        self.data = mujoco.MjData(self.model)
        self.renderers = {}
        self.before, self.after = np.empty(self.width), np.empty(self.width)
        try:
            for name, (width,height) in config['cameras'].items():
                camera = self.model.camera(name).id
                _require(int(self.model.cam_mode[camera]) == 0,
                    'Only fixed body-mounted cameras support FK-only preview: '+name)
                self.renderers[name] = mujoco.Renderer(self.model, width=width, height=height)
        except BaseException:
            self.close()
            raise

    def render(self, state, *, sim_time_s):
        mj = self.mujoco
        original = state.copy()
        mj.mj_setState(self.model, self.data, state, self.spec)
        mj.mj_getState(self.model, self.data, self.before, self.spec)
        _require(np.array_equal(self.before, original), 'State restoration changed archived values')
        _require(abs(float(self.data.time)-sim_time_s) <= 1e-9, 'State and preview clock differ')
        _require(not np.any(self.data.qfrc_applied) and not np.any(self.data.xfrc_applied),
                 'Nonzero external force in preview state')
        # No mj_forward/mj_step: original solved contacts/forces stay in the
        # physical process and are never reconstructed from rendering data.
        mj.mj_kinematics(self.model, self.data)
        mj.mj_camlight(self.model, self.data)
        images = {}
        for name, renderer in self.renderers.items():
            renderer.update_scene(self.data, camera=name)
            images[name] = renderer.render().copy()
        mj.mj_getState(self.model, self.data, self.after, self.spec)
        _require(np.array_equal(self.before, self.after) and np.array_equal(state, original),
                 'FK/rendering modified the archived integration state')
        return images, {'full_integration_state_unchanged':True,
            'restored_sim_time_s':float(self.data.time), 'external_forces_all_zero':True,
            'physics_steps':0, 'forward_dynamics_calls':0, 'camera_modes':'fixed_body_mount',
            'mujoco_version':mj.__version__, 'actual_gl_module':mj.gl_context.GLContext.__module__}

    def close(self):
        for renderer in getattr(self, 'renderers', {}).values():
            renderer.close()
        self.renderers = {}


def _cpu_policy(egl_device_id,cpu_affinity):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(egl_device_id), CUDA_VISIBLE_DEVICES='')
    policy = {'pid':os.getpid(), 'model_inference':False, 'physics_steps':0, 'SDK_started':False,
              'thread_limit':1, 'egl_device_id':egl_device_id}
    if hasattr(os, 'sched_getaffinity'):
        policy['cpus'] = list(cpu_affinity) if cpu_affinity is not None else sorted(os.sched_getaffinity(0))[-1:]
        os.sched_setaffinity(0, policy['cpus'])
    if hasattr(os, 'nice'):
        os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
        policy['nice'] = os.getpriority(os.PRIO_PROCESS,0)
    guard = None
    try:
        from threadpoolctl import threadpool_limits
        guard = threadpool_limits(limits=1)
    except ImportError:
        pass
    return policy, guard


def _preview_process(configurations, requests, results, statuses, stop, backend_factory, egl_device_id,cpu_affinity):
    backend, guard = None, None
    current_epoch, previous_request, frame_id = None, 0, 0
    state = {'contract':CONTRACT, 'status':'starting', 'ready':False, 'error':None,
             'epoch':None, 'rendered_frames':0, 'dropped_pending_requests':0}
    try:
        policy, guard = _cpu_policy(egl_device_id,cpu_affinity)
        state['worker'] = policy
        statuses.put(state, blocking=True)
        while not stop.value:
            configuration = configurations.get(blocking=True)
            if configuration is not None and configuration['epoch'] != current_epoch:
                if backend is not None:
                    backend.close()
                    backend = None
                current_epoch = configuration['epoch']
                state.update(status='configuring',ready=False,epoch=current_epoch,error=None,
                    rendered_frames=0,last_frame_id=None,last_source_tick50=None,
                    last_source_sim_time_s=None,last_render_wall_s=None,model_sha256=None)
                statuses.put(state,blocking=True)
                try:
                    _require(_sha(configuration['model_path']) == configuration['model_sha256'], 'Model MJB SHA mismatch')
                    backend = backend_factory(configuration)
                    _require(_sha(configuration['model_path']) == configuration['model_sha256'],'MJB changed during preview configuration')
                    state.update(status='ready',ready=True,model_sha256=configuration['model_sha256'],
                        state_width=configuration['state_width'],cameras=configuration['cameras'])
                except Exception as error:
                    state.update(status='error',ready=False,error=f'{type(error).__name__}: {error}'[:1000])
                statuses.put(state,blocking=True)
            request = requests.get(blocking=True)
            if (backend is None or request is None or request['request_id'] == previous_request
                    or request['epoch'] != current_epoch):
                time.sleep(.002)
                continue
            state['dropped_pending_requests'] += max(0,request['request_id']-previous_request-1)
            previous_request = request['request_id']
            start = time.monotonic()
            try:
                original = request['integration_state'].copy()
                images, details = backend.render(request['integration_state'],sim_time_s=request['sim_time_s'])
                completed = time.monotonic()
                _require(np.array_equal(request['integration_state'],original),'Backend mutated submitted state')
                _require(set(images) == set(configuration['cameras']), 'Preview camera set differs from configuration')
                pixels = {}
                for name, dimensions in configuration['cameras'].items():
                    image = np.asarray(images[name])
                    _require(image.dtype == np.uint8 and image.shape == (dimensions[1],dimensions[0],3),
                             'Renderer did not supply actual uint8 RGB dimensions')
                    pixels[name] = image.copy(order='C')
                latest_config = configurations.get(blocking=True)
                if stop.value or latest_config['epoch'] != current_epoch:
                    continue  # Never publish an earlier scene after reset.
                frame_id += 1
                evidence = {**details, 'contract':CONTRACT, 'model_sha256':configuration['model_sha256'],
                    'source_tick50':request['source_tick50'], 'source_sim_time_s':request['sim_time_s'],
                    'paused_render_request':request.get('paused_render_request',False),
                    'repeated_physical_source':request.get('repeated_physical_source',False),
                    'preview_only':True, 'training_observation':False, 'SDK_started':False,
                    'render_queue_wait_s':start-request['submitted_monotonic_s'],
                    'dropped_pending_requests':state['dropped_pending_requests'], 'worker':policy}
                frame = PreviewFrame(pixels,current_epoch,request['source_tick50'],request['sim_time_s'],
                    frame_id,request['request_id'],hashlib.sha256(original.tobytes()).hexdigest(),
                    request['submitted_monotonic_s'],start,completed,None,completed-start,evidence)
                results.put(frame,blocking=True)
                state.update(status='ready',ready=True,error=None,rendered_frames=state['rendered_frames']+1,
                    total_rendered_frames=frame_id,
                    last_frame_id=frame_id,last_source_tick50=request['source_tick50'],
                    last_source_sim_time_s=request['sim_time_s'],last_render_wall_s=completed-start)
                statuses.put(state,blocking=True)
            except Exception as error:
                state.update(status='error',ready=False,error=f'{type(error).__name__}: {error}'[:1000])
                statuses.put(state,blocking=True)
                backend.close()
                backend = None  # No black frame/fallback; explicit reconfigure required.
        state.update(status='closed',ready=False)
        statuses.put(state,blocking=True)
    except BaseException as error:
        state.update(status='error',ready=False,error=f'{type(error).__name__}: {error}'[:1000])
        statuses.put(state,blocking=True)
    finally:
        if backend is not None:
            backend.close()
        if guard is not None:
            guard.restore_original_limits()


class AsyncMujocoPreview:
    """Paused configure + nonblocking latest-only state submit/frame poll.

    A completed frame can lag a newer submitted state while remaining within
    ``max_frame_age_s``. Its real source tick/time/hash is always explicit. Only
    same-epoch new frame IDs are returned, at most once. Physical capture must
    run independently and retain every required state even when preview drops.
    """
    def __init__(self, *, backend_factory=None, max_cameras=2, max_width=640,
                 max_height=360, max_state_width=16384, max_frame_age_s=.1, egl_device_id=0,cpu_affinity=None):
        for name,value,upper in (('max_cameras',max_cameras,8),('max_width',max_width,4096),
                                 ('max_height',max_height,4096),('max_state_width',max_state_width,1000000)):
            _require(type(value) is int and 0<value<=upper,'Invalid '+name)
        age = _number(max_frame_age_s,'preview maximum frame age')
        _require(age>0,'Preview maximum frame age must be positive')
        _require(type(egl_device_id) is int and egl_device_id>=0,'Invalid EGL device id')
        if cpu_affinity is not None:
            cpu_affinity=tuple(cpu_affinity)
            _require(bool(cpu_affinity) and all(type(cpu) is int and cpu>=0 for cpu in cpu_affinity),'Invalid preview CPU affinity')
        self.cpu_affinity=cpu_affinity
        self._configure_failed=False
        self.max_cameras,self.max_width,self.max_height,self.max_state_width=max_cameras,max_width,max_height,max_state_width
        self.max_frame_age_s,self.egl_device_id=age,egl_device_id
        self.factory=backend_factory or _MujocoEGLBackend
        context=multiprocessing.get_context('spawn')
        self._context=context
        self._configurations=_Mailbox(context,16384)
        self._requests=_Mailbox(context,max_state_width*8+16384)
        self._results=_Mailbox(context,max_cameras*max_width*max_height*3+131072)
        self._statuses=_Mailbox(context,16384)
        self._stop=context.RawValue('b',0)
        self._process=None
        self._closed=False
        self._epoch=None
        self._configuration=None
        self._sequence=0
        self._last_tick=None
        self._last_time=None
        self._last_state=None
        self._delivered=0
        self._dropped_submits=0
        self._expired_results=0
        self._cached_status={'contract':CONTRACT,'status':'not_started','ready':False,'error':None,'epoch':None}

    def start(self):
        _require(not self._closed,'Closed preview cannot restart')
        if self._process is None:
            self._process=self._context.Process(target=_preview_process,
                args=(self._configurations,self._requests,self._results,self._statuses,self._stop,
                      self.factory,self.egl_device_id,self.cpu_affinity),daemon=True,name='pico-egl-preview')
            self._process.start()
            self._cached_status.update(status='starting')
        return self

    def configure(self, model_path, model_sha256, state_spec, state_width, cameras=None, epoch=0, *, timeout_s=20.):
        """Blocking setup, permitted only while the caller's physics is paused."""
        _require(self._process is not None and not self._closed,'Start preview before configuring')
        _require(type(epoch) is int and epoch>=0 and (self._epoch is None or epoch>self._epoch),
                 'Preview reset epochs must strictly increase')
        _require(isinstance(model_sha256,str) and len(model_sha256)==64
                 and all(c in '0123456789abcdef' for c in model_sha256),'Explicit model SHA256 required')
        _require(type(state_spec) is int and state_spec>0,'Explicit integration state specification required')
        _require(type(state_width) is int and 0<state_width<=self.max_state_width,'State width exceeds configured capacity')
        cameras=dict(DEFAULT_CAMERAS if cameras is None else cameras)
        _require(0<len(cameras)<=self.max_cameras,'Invalid preview camera count')
        normalized={}
        for name,dimensions in cameras.items():
            _require(isinstance(name,str) and bool(name),'Camera names must be nonempty strings')
            if isinstance(dimensions,dict):
                dimensions=(dimensions['width'],dimensions['height'])
            _require(len(dimensions)==2 and all(type(v) is int for v in dimensions),'Camera dimensions must be integer W,H')
            width,height=dimensions
            _require(0<width<=self.max_width and 0<height<=self.max_height,'Camera dimensions exceed fixed preview capacity')
            normalized[name]=(width,height)
        timeout=_number(timeout_s,'configure timeout')
        _require(timeout>0,'Configure timeout must be positive')
        configuration={'model_path':str(Path(model_path).resolve(strict=True)),'model_sha256':model_sha256,
            'state_spec':state_spec,'state_width':state_width,'cameras':normalized,'epoch':epoch}
        self._epoch=epoch
        self._configure_failed=False
        self._configuration=configuration
        self._last_tick=self._last_time=None
        self._last_state=None
        self._delivered=0
        self._cached_status={'contract':CONTRACT,'status':'configuring','ready':False,
            'epoch':epoch,'error':None,'rendered_frames':0,'last_frame_id':None}
        self._results.clear()
        self._configurations.put(configuration,blocking=True)
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            status=self.status()
            if status.get('epoch')==epoch:
                if status.get('ready'):
                    return status
                if status.get('status')=='error':
                    self._configure_failed=True
                    raise RuntimeError(status.get('error') or 'Preview configuration failed')
            if not self._process.is_alive():
                self._configure_failed=True
                raise RuntimeError('Preview worker exited during configuration')
            time.sleep(.002)
        self._configure_failed=True
        self._cached_status.update(status='error',ready=False,error='Preview configuration timeout')
        raise TimeoutError('Preview configuration timeout')

    def submit(self, integration_state, sim_time_s, epoch, source_tick50):
        """Active physics source: tick and simulation time must strictly advance."""
        return self._submit(integration_state,sim_time_s,epoch,source_tick50,paused=False)

    def submit_paused(self, integration_state, sim_time_s, epoch, source_tick50):
        """Explicit paused-only redraw, including the identical first source.

        The caller must have paused physics. A repeat is accepted only when
        tick, time and all integration values equal the last accepted source.
        Each redraw retains its real submit time and the same source identity;
        a slow first shader draw is never relabelled as fresh. Retry after it
        finishes to obtain a newly rendered frame within the unchanged age gate.
        This method must never supply physical observations or training RGB.
        """
        return self._submit(integration_state,sim_time_s,epoch,source_tick50,paused=True)

    def _submit(self, integration_state, sim_time_s, epoch, source_tick50, *, paused):
        _require(self._process is not None and not self._closed,'Start preview before submitting')
        _require(type(epoch) is int and epoch==self._epoch,'Cannot submit a stale/new unconfigured preview epoch')
        if not self.is_ready:
            self._dropped_submits+=1
            return False
        _require(type(source_tick50) is int and source_tick50>=0,'Nonnegative source tick required')
        stamp=_number(sim_time_s,'preview simulation clock')
        _require(stamp>=0,'Simulation clock cannot be negative')
        value=np.asarray(integration_state,dtype=np.float64)
        _require(value.shape==(self._configuration['state_width'],) and np.isfinite(value).all(),
                 'Complete finite integration state required')
        value=value.copy(order='C')
        repeated=False
        if self._last_tick is not None and (source_tick50<=self._last_tick or stamp<=self._last_time):
            repeated=bool(paused and source_tick50==self._last_tick and stamp==self._last_time
                          and np.array_equal(value,self._last_state))
            if not repeated:
                self._dropped_submits+=1
                return False
        request={'integration_state':value,'sim_time_s':stamp,'source_tick50':source_tick50,
            'epoch':epoch,'request_id':self._sequence+1,'submitted_monotonic_s':time.monotonic(),
            'paused_render_request':paused,'repeated_physical_source':repeated}
        if not self._requests.put(request):
            self._dropped_submits+=1
            return False
        self._sequence+=1
        self._last_tick,self._last_time=source_tick50,stamp
        self._last_state=value
        return True

    def poll(self):
        if self._closed or self._epoch is None or not self.is_ready:
            return None
        frame=self._results.get(consume=True)
        if frame is None:
            return None
        received=time.monotonic()
        if (frame.epoch!=self._epoch or frame.frame_id<=self._delivered
                or received-frame.submitted_monotonic_s>self.max_frame_age_s):
            self._expired_results+=1
            return None
        _require(self._configuration is not None and set(frame.images)==set(self._configuration['cameras']),
                 'Preview result cameras differ')
        for pixels in frame.images.values():
            pixels.flags.writeable=False
        self._delivered=frame.frame_id
        return replace(frame,received_monotonic_s=received,
            evidence={**frame.evidence,'receive_age_s':received-frame.submitted_monotonic_s,
                      'received_monotonic_s':received})

    @property
    def is_ready(self):
        status=self.status()
        return bool(status.get('ready') and status.get('epoch')==self._epoch)

    def status(self):
        if self._closed:
            return {**self._cached_status,'status':'closed','ready':False}
        incoming=self._statuses.get()
        if incoming is not None and incoming.get('epoch')==self._epoch and not self._configure_failed:
            self._cached_status=incoming
        status={**self._cached_status,'epoch':self._epoch,'dropped_submits':self._dropped_submits,
                'expired_results':self._expired_results,'accepted_new_dataset':False,'training_observation':False}
        if self._process is not None and not self._process.is_alive():
            status.update(status='error',ready=False,error=status.get('error') or 'Preview worker exited')
        return status

    def close(self, timeout_s=.5):
        if self._closed:
            return
        self._stop.value=1
        if self._process is not None:
            self._process.join(max(0.,float(timeout_s)))
            if self._process.is_alive():
                self._process.terminate(); self._process.join(.5)
            if self._process.is_alive():
                self._process.kill(); self._process.join(.5)
            _require(not self._process.is_alive(),'Preview worker failed to exit')
            self._process.close()
            self._process=None
        self._closed=True
        self._cached_status.update(status='closed',ready=False)

    def __enter__(self):
        return self.start()

    def __exit__(self,*unused):
        self.close()
