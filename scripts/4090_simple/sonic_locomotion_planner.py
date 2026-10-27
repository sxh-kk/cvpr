"""Official SONIC V2 planner references, with an isolated asynchronous CPU worker.

All generated futures are MODEL FORECAST REFERENCES, never recorded actions or
new dataset labels. Only a later actual execution can supply training labels.
The strict low-watermark replan request is an explicit adaptation required by
our complete46-frame encoder window; it is not the C++ default1s walk timer.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from handover_sonic_teleop_reference import (quaternion, finite, multiply_wxyz, conjugate, heading_wxyz, rotation)
from prepare_carrybox_v11 import BODY_NAMES

CONTRACT = 'cvpr-official-sonic11-locomotion-planner-v1'
MODEL_SHA256 = '39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea'
DEFAULT_HEIGHT = .788740
V2_MASK = (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
DEPLOY = SCRIPT_DIR.parents[1] / 'VLMExpert/third_party/GR00T-WholeBodyControl/gear_sonic_deploy'
SOURCE_FILES = tuple(DEPLOY / 'src/g1/g1_deploy_onnx_ref' / name for name in (
    'include/localmotion_kplanner.hpp', 'include/localmotion_kplanner_tensorrt.hpp',
    'include/policy_parameters.hpp', 'include/math_utils.hpp', 'src/g1_deploy_onnx_ref.cpp')) + (
    Path(__file__).resolve(), SCRIPT_DIR / 'prepare_carrybox_v11.py',
    SCRIPT_DIR / 'handover_sonic_teleop_reference.py')
INPUTS = {
    'context_mujoco_qpos': ('tensor(float)', (1, 4, 36)), 'target_vel': ('tensor(float)', (1,)),
    'mode': ('tensor(int64)', (1,)), 'movement_direction': ('tensor(float)', (1, 3)),
    'facing_direction': ('tensor(float)', (1, 3)), 'random_seed': ('tensor(int64)', (1,)),
    'has_specific_target': ('tensor(int64)', (1, 1)), 'specific_target_positions': ('tensor(float)', (1, 4, 3)),
    'specific_target_headings': ('tensor(float)', (1, 4)), 'allowed_pred_num_tokens': ('tensor(int64)', (1, 11)),
    'height': ('tensor(float)', (1,))}
OUTPUTS = {'mujoco_qpos': ('tensor(float)', (1, 64, 36)), 'num_pred_frames': ('tensor(int32)', (1,))}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def slerp(a, b, weight):
    a = quaternion(a, (4,), 'SLERP first'); b = quaternion(b, (4,), 'SLERP second')
    require(math.isfinite(weight) and 0 <= weight <= 1, 'SLERP weight out of range')
    cosine = float(np.dot(a, b))
    if cosine < 0: b = -b; cosine = -cosine
    cosine = np.clip(cosine, -1., 1.)
    if cosine > .9995:
        value = a + weight * (b - a)
        return value / np.linalg.norm(value)
    theta = math.acos(cosine)
    value = (math.sin((1-weight)*theta)*a + math.sin(weight*theta)*b) / math.sin(theta)
    return value / np.linalg.norm(value)


@dataclass(frozen=True)
class WorldAlignment:
    initial_actual_pose7: np.ndarray
    yaw_wxyz: np.ndarray
    origin_xy: np.ndarray

    @classmethod
    def from_actual(cls, actual_pose7):
        pose = finite(actual_pose7, (7,), 'actual initialization pose')
        q = quaternion(pose[3:], (4,), 'actual initialization wxyz')
        return cls(pose.copy(), heading_wxyz(q), pose[:2].copy())

    def direction_to_canonical(self, value): return finite(value, (3,), 'world direction') @ rotation(self.yaw_wxyz)

    def to_world(self, qpos):
        value = np.asarray(qpos, np.float64).copy()
        require(value.ndim == 2 and value.shape[1] == 36 and np.isfinite(value).all(), 'Require qpos[N,36]')
        value[:, :3] = value[:, :3] @ rotation(self.yaw_wxyz).T
        value[:, :2] += self.origin_xy
        value[:, 3:7] = np.stack([multiply_wxyz(self.yaw_wxyz, q) for q in value[:, 3:7]])
        return value

    def to_canonical(self, qpos):
        value = np.asarray(qpos, np.float64).copy()
        require(value.ndim == 2 and value.shape[1] == 36 and np.isfinite(value).all(), 'Require qpos[N,36]')
        value[:, :2] -= self.origin_xy
        value[:, :3] = value[:, :3] @ rotation(self.yaw_wxyz)
        value[:, 3:7] = np.stack([multiply_wxyz(conjugate(self.yaw_wxyz), q) for q in value[:, 3:7]])
        return value

    def evidence(self):
        return {'initial_actual_root_pose7': self.initial_actual_pose7.tolist(), 'initial_world_yaw_wxyz': self.yaw_wxyz.tolist(),
            'world_origin_xy_m': self.origin_xy.tolist(), 'world_Z_translation_m': 0.,
            'bootstrap_root_height_m': DEFAULT_HEIGHT, 'actual_initial_root_roll_pitch_used_for_bootstrap': False,
            'coordinate_mapping': 'world = initial_yaw * canonical; add initial XY only; preserve official world Z'}


def bootstrap_context(actual_body_q29):
    q = finite(actual_body_q29, (29,), 'actual measured BODY29')
    row = np.r_[0., 0., DEFAULT_HEIGHT, 1., 0., 0., 0., q]
    return np.tile(row, (4, 1)), {'kind': 'official_normalized_static_bootstrap_context',
        'four_measured_history_frames_claimed': False, 'joint_source': 'one actual measured BODY29 repeated4 times',
        'root_source': '[0,0,0.788740,1,0,0,0] repeated4 times; actual XY/yaw restored separately',
        'body_joint_names': list(BODY_NAMES)}


@dataclass(frozen=True)
class Forecast:
    qpos50: np.ndarray
    dq29: np.ndarray
    valid_forecast_mask: np.ndarray
    metadata: dict


def resample_30_to_50(valid_qpos30):
    value = np.asarray(valid_qpos30, np.float64)
    require(value.ndim == 2 and value.shape[1] == 36 and 2 <= len(value) <= 64 and np.isfinite(value).all(), 'Invalid valid planner output')
    quaternion(value[:, 3:7], (len(value), 4), 'planner forecast quaternions')
    count = len(value) * 50 // 30
    output = np.empty((count, 36)); valid = np.empty(count, dtype=bool)
    for f in range(count):
        at = f * 30 / 50; first = int(math.floor(at)); second = min(first+1, len(value)-1); w = at-first
        output[f] = (1-w)*value[first] + w*value[second]
        output[f, 3:7] = slerp(value[first, 3:7], value[second, 3:7], w)
        valid[f] = at <= len(value)-1 + 1e-12
    dq = np.diff(output[:, 7:], axis=0)*50
    dq = np.vstack([dq, dq[-1]])
    return Forecast(output, dq, valid, {'source_rate_hz': 30, 'reference_rate_hz': 50,
        'valid_model_output_frames30': len(value), 'official_resampled_frames50': count,
        'fully_interpolated_valid_forecast_frames50': int(valid.sum()),
        'official_endpoint_hold_rows50': np.flatnonzero(~valid).tolist(),
        'endpoint_holds_eligible_as_complete_reference_future': False,
        'dq_semantics': 'official forward finite differences at50Hz, final derivative repeated; forecast reference only',
        'future_action_or_observation_labels_claimed': False})


def context_from_forecast(forecast, current_frame, lookahead=2):
    require(type(current_frame) is int and current_frame >= 0 and lookahead == 2, 'Official nonnegative cursor/+2 lookahead required')
    at = current_frame + lookahead + np.arange(4)*50/30
    result = []
    for index in at:
        first = int(math.floor(index)); second = min(first+1, len(forecast.qpos50)-1); w = index-first
        require(first < len(forecast.qpos50) and forecast.valid_forecast_mask[first]
                and (w <= 1e-12 or forecast.valid_forecast_mask[second]), 'Context exceeds valid forecast; do not clamp or fabricate it')
        a,b = forecast.qpos50[first],forecast.qpos50[second]
        row = (1-w)*a+w*b;row[3:7] = slerp(a[3:7],b[3:7],w);result.append(row)
    return np.asarray(result), {'context_source': 'previous planner forecast, not actual robot history',
        'forecast_sample_indices50_fractional': at.tolist(), 'generated_start_frame50': current_frame+lookahead,
        'official_reference_context_semantics': True, 'clamping_used': False}


def future_window(forecast, current_frame, rows=46):
    require(type(current_frame) is int and current_frame >= 0 and rows == 46, 'Require nonnegative cursor and complete46-frame reference')
    stop = current_frame+rows
    require(stop <= len(forecast.qpos50) and forecast.valid_forecast_mask[current_frame:stop].all(),
            'Insufficient valid46-frame forecast; request a new plan instead of padding/clamping')
    return forecast.qpos50[current_frame:stop].copy(), forecast.dq29[current_frame:stop].copy()


def crossfade8(old, new, current_frame, generated_from_frame):
    require(type(current_frame) is int and type(generated_from_frame) is int
            and 0 <= current_frame < len(old.qpos50), 'Invalid reference blend cursor')
    count = generated_from_frame-current_frame+len(new.qpos50)
    require(count > 0, 'Planner result arrived after its full reference expired')
    start = max(0, generated_from_frame-current_frame);q=np.empty((count,36));dq=np.empty((count,29));valid=np.zeros(count,bool)
    for f in range(count):
        old_i=min(f+current_frame,len(old.qpos50)-1);new_i=np.clip(f+current_frame-generated_from_frame,0,len(new.qpos50)-1)
        w=float(np.clip((f-start)/8,0,1));a,b=old.qpos50[old_i],new.qpos50[new_i]
        q[f]=(1-w)*a+w*b;q[f,3:7]=slerp(a[3:7],b[3:7],w)
        dq[f]=(1-w)*old.dq29[old_i]+w*new.dq29[new_i]
        valid[f]=(w==1 or (f+current_frame<len(old.qpos50) and old.valid_forecast_mask[old_i])) and (w==0 or new.valid_forecast_mask[new_i])
    return Forecast(q,dq,valid,{'blend_frames':8,'rebased_cursor50':0,'new_start_on_rebased_timeline50':start,
        'reference_poses_and_velocities_crossfaded_separately_as_official':True,'future_labels_claimed':False})


def replan_reason(current, previous, since_last_replan_s, forecast, cursor, *, next_check_ticks50=5):
    """Official replan reasons plus explicit complete-window low-watermark."""
    require(since_last_replan_s >= 0 and next_check_ticks50 == 5, 'Planner checks are10Hz, not forced10Hz inference')
    if previous is None: return 'initialize'
    for key in ('mode','facing_direction','height'):
        if not np.array_equal(current[key],previous[key]): return 'official_'+key+'_changed'
    static={0,4,5,6,7,9};mode=int(current['mode'])
    if mode not in static:
        for key in ('speed','movement_direction'):
            if not np.array_equal(current[key],previous[key]): return 'official_'+key+'_changed'
        interval=.1 if mode==3 else .2 if mode==8 else 1.
        if mode in {11,12,13,15,16}:interval=1.
        if since_last_replan_s >= interval and current['speed'] != 0:return 'official_periodic_interval'
    try:future_window(forecast,cursor+next_check_ticks50)
    except ValueError:return 'strict_future_low_watermark_explicit_adapter_extension'
    return None


def joint_range_report(qpos, lower, upper):
    q=np.asarray(qpos)[:,7:];lo=finite(lower,(29,),'physical lower BODY29');hi=finite(upper,(29,),'physical upper BODY29')
    margin=np.minimum(q-lo,hi-q);bad=margin < -1e-6
    return {'all_BODY29_within_physical_ranges':bool(not bad.any()),'encoded_lower12_within_physical_ranges':bool(not bad[:,:12].any()),
        'minimum_margin_rad':float(margin.min()),'violating_sample_elements':int(bad.sum()),
        'violating_joint_names':[BODY_NAMES[i] for i in range(29) if bad[:,i].any()],
        'values_clipped_or_repaired':False}


class OfficialPlannerCPU:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.path=Path(model_path).resolve(strict=True);require(sha(self.path)==MODEL_SHA256,'Official pinned planner bytes changed')
        opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session=ort.InferenceSession(str(self.path),sess_options=opts,providers=['CPUExecutionProvider'])
        require(self.session.get_providers()==['CPUExecutionProvider'],'CPU provider required')
        for actual,expected in [(self.session.get_inputs(),INPUTS),(self.session.get_outputs(),OUTPUTS)]:
            require({a.name:(a.type,tuple(a.shape)) for a in actual}==expected,'Actual pinned ONNX tensor contract changed')
        self.source_sha256={str(p):sha(p) for p in SOURCE_FILES};self.inference_calls=0

    def infer(self, canonical_context, alignment, *, mode, movement_world, facing_world, speed, seed, height=-1.):
        context=finite(canonical_context,(4,36),'canonical forecast context')
        quaternion(context[:,3:7],(4,4),'context roots')
        require(type(mode) is int and 0 <= mode < 27 and type(seed) is int and 0 <= seed < 2**63,'V2 mode/seed out of range')
        require(math.isfinite(speed) and math.isfinite(height),'Finite command speed/height required')
        movement=alignment.direction_to_canonical(movement_world);facing=alignment.direction_to_canonical(facing_world)
        require(np.linalg.norm(facing[:2])>1e-8,'Facing direction must define a horizontal heading')
        inputs={'context_mujoco_qpos':context[None].astype(np.float32),'target_vel':np.array([speed],np.float32),
            'mode':np.array([mode],np.int64),'movement_direction':movement[None].astype(np.float32),
            'facing_direction':facing[None].astype(np.float32),'random_seed':np.array([seed],np.int64),
            'has_specific_target':np.zeros((1,1),np.int64),'specific_target_positions':np.zeros((1,4,3),np.float32),
            'specific_target_headings':np.zeros((1,4),np.float32),'allowed_pred_num_tokens':np.array([V2_MASK],np.int64),
            'height':np.array([height],np.float32)}
        start=time.perf_counter();output,count=self.session.run(['mujoco_qpos','num_pred_frames'],inputs);elapsed=time.perf_counter()-start
        self.inference_calls+=1
        require(output.shape==(1,64,36) and output.dtype==np.float32 and count.shape==(1,) and count.dtype==np.int32,'Planner output tensor contract differs')
        n=int(count[0]);require(n in (36,40,44),'Planner returned horizon outside the fixed official V2 mask')
        valid=output[0,:n].copy();require(np.isfinite(valid).all(),'Nonfinite valid planner output')
        canonical=resample_30_to_50(valid);world=Forecast(alignment.to_world(canonical.qpos50),canonical.dq29.copy(),canonical.valid_forecast_mask.copy(),dict(canonical.metadata))
        evidence={'contract':CONTRACT,'official_model_sha256':MODEL_SHA256,'inference_call':self.inference_calls,'CPU_seconds':elapsed,
            'input_sha256':{key:array_sha(value) for key,value in inputs.items()},'valid_model_output_sha256':array_sha(valid),
            'full_padded_model_output_sha256':array_sha(output),'num_pred_frames':n,'ignored_padded_model_output_rows':64-n,
            'alignment':alignment.evidence(),'resampling':canonical.metadata,'source_sha256':dict(self.source_sha256),
            'command_world':{'mode':mode,'movement_direction':list(movement_world),'facing_direction':list(facing_world),'speed':speed,'height':height,'seed':seed},
            'command_canonical':{'movement_direction':movement.tolist(),'facing_direction':facing.tolist()},
            'all_generated_futures_are_forecast_references_not_labels':True,'physics_steps':0,'GPU_used':False,
            'source_resources_modified':False}
        return world,canonical,evidence,inputs,output.copy()


@dataclass(frozen=True)
class PlannerResult:
    request_id: int
    world: Forecast
    canonical: Forecast
    evidence: dict
    submitted_monotonic: float
    completed_monotonic: float
    expires_monotonic: float


class _Mailbox:
    """Fixed-size latest-only storage, with no Queue feeder or blocking put."""
    def __init__(self, context, capacity):
        self.capacity = capacity
        self.buffer = context.RawArray('B', capacity)
        self.length = context.RawValue('I', 0)
        self.lock = context.Lock()

    def put(self, value, *, blocking=False):
        import pickle
        encoded = pickle.dumps(value, protocol=5)
        require(len(encoded) <= self.capacity, 'Planner IPC message exceeds bounded capacity')
        if not self.lock.acquire(blocking):
            return False
        try:
            memoryview(self.buffer).cast('B')[:len(encoded)] = encoded
            self.length.value = len(encoded)
        finally:
            self.lock.release()
        return True

    def get(self, *, consume=False, blocking=False, clear_event=None):
        import pickle
        if not self.lock.acquire(blocking):
            return None
        try:
            value = bytes(memoryview(self.buffer).cast('B')[:self.length.value])
            if consume:
                self.length.value = 0
            if clear_event is not None:
                clear_event.clear()
        finally:
            self.lock.release()
        return pickle.loads(value) if value else None


def _cpu_worker_policy(cpu_affinity=None):
    import os
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    policy = {'cpu_only': True, 'ORT_intra_op_threads': 1, 'ORT_inter_op_threads': 1}
    if hasattr(os, 'sched_getaffinity'):
        inherited = sorted(os.sched_getaffinity(0))
        cpus = list(cpu_affinity) if cpu_affinity is not None else inherited[-1:]
        require(bool(cpus) and set(cpus).issubset(inherited), 'Planner CPUs must be in inherited affinity')
        os.sched_setaffinity(0, cpus)
        policy.update(cpus=sorted(os.sched_getaffinity(0)), inherited_cpus=inherited,
            explicit_cpu_affinity=cpu_affinity is not None)
    if hasattr(os, 'nice'):
        os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
        policy['nice'] = os.getpriority(os.PRIO_PROCESS, 0)
    # Spawn imports numpy before this entry point. Limit any already-loaded
    # BLAS pools too, when the installed CPU runtime provides threadpoolctl.
    try:
        from threadpoolctl import threadpool_limits
        policy['_thread_limit_guard'] = threadpool_limits(limits=1)
    except ImportError:
        pass
    return policy


def _planner_process(model_path, factory, requests, results, statuses, wakeup, stop,cpu_affinity):
    import os
    policy = _cpu_worker_policy(cpu_affinity)
    guard = policy.pop('_thread_limit_guard', None)  # Keep the native limit alive.
    state = {'status': 'loading', 'ready': False, 'pid': os.getpid(), 'policy': policy,
        'inference_calls': 0, 'expired_results': 0, 'superseded_results': 0,
        'superseded_pending_requests': 0, 'error': None}
    statuses.put(state, blocking=True)
    try:
        planner = factory(Path(model_path))  # Hash/load/ORT creation happen here.
        state.update(status='idle', ready=True)
        statuses.put(state, blocking=True)
        previous_id = 0
        while not stop.is_set():
            wakeup.wait(.05)
            if stop.is_set():
                break
            request = requests.get(blocking=True, clear_event=wakeup)
            if request is None or request['request_id'] == previous_id:
                continue
            request_id = request['request_id']
            state['superseded_pending_requests'] += max(0, request_id-previous_id-1)
            previous_id = request_id
            observed = time.monotonic()
            state.update(request_id=request_id,
                last_request_submitted_monotonic=request['submitted_monotonic'],
                last_request_expires_monotonic=request['expires_monotonic'],
                last_queue_wait_s=observed-request['submitted_monotonic'],
                last_inference_started_monotonic=None,last_inference_completed_monotonic=None,
                last_inference_wall_s=None,last_inference_process_cpu_s=None,
                last_request_total_wall_s=observed-request['submitted_monotonic'],
                expired_before_inference=observed>=request['expires_monotonic'])
            if state['expired_before_inference']:
                state.update(status='expired', expired_results=state['expired_results']+1)
                statuses.put(state, blocking=True)
                continue
            inference_started = time.monotonic()
            process_cpu_started = time.process_time()
            state.update(status='working', request_id=request_id, error=None,
                last_request_submitted_monotonic=request['submitted_monotonic'],
                last_request_expires_monotonic=request['expires_monotonic'],
                last_inference_started_monotonic=inference_started,
                last_queue_wait_s=inference_started-request['submitted_monotonic'])
            statuses.put(state, blocking=True)
            try:
                world, canonical, evidence, _, _ = planner.infer(request['context'], request['alignment'],
                    **request['command'])
                completed = time.monotonic()
                state['inference_calls'] += 1
                state.update(last_inference_completed_monotonic=completed,
                    last_inference_wall_s=completed-inference_started,
                    last_inference_process_cpu_s=time.process_time()-process_cpu_started,
                    last_request_total_wall_s=completed-request['submitted_monotonic'])
                latest = requests.get(blocking=True)
                if stop.is_set():
                    break
                if completed >= request['expires_monotonic']:
                    state.update(status='expired', expired_results=state['expired_results']+1)
                elif latest is None or latest['request_id'] != request_id:
                    state.update(status='superseded', superseded_results=state['superseded_results']+1)
                else:
                    result = PlannerResult(request_id, world, canonical,
                        {**evidence, 'async_cpu_worker': policy, 'reference_only': True,
                         'async_worker_timing':{key:state[key] for key in ('last_queue_wait_s',
                            'last_inference_wall_s','last_inference_process_cpu_s','last_request_total_wall_s')}},
                        request['submitted_monotonic'], completed, request['expires_monotonic'])
                    results.put(result, blocking=True)
                    state.update(status='idle')
                statuses.put(state, blocking=True)
            except Exception as error:
                completed = time.monotonic()
                state.update(status='error', error=f'{type(error).__name__}: {error}'[:1000],
                    last_inference_completed_monotonic=completed,
                    last_inference_wall_s=completed-inference_started,
                    last_inference_process_cpu_s=time.process_time()-process_cpu_started,
                    last_request_total_wall_s=completed-request['submitted_monotonic'])
                statuses.put(state, blocking=True)
        state.update(status='closed', ready=False)
        statuses.put(state, blocking=True)
    except Exception as error:
        state.update(status='error', ready=False, error=f'{type(error).__name__}: {error}'[:1000])
        statuses.put(state, blocking=True)
    finally:
        # This worker owns the only planner/session; no inference thread or
        # simulator outlives the process. Parent close can terminate a hung ORT.
        if guard is not None:
            guard.restore_original_limits()


class AsyncPlanner:
    """Latest-only CPU planner; control submits/polls without waiting for ORT.

    ``submit`` returns a request ID, or None if the tiny request mailbox is
    momentarily busy (caller may retry at its next planning tick). ``poll``
    consumes at most one matching, unexpired result; it never reuses old output.
    Loading, inference and all model/source hashes are worker-only. The caller
    must explicitly request IDLE (mode=0); expiration supplies no fake command.

    ``planner_factory`` is a spawn-pickleable CPU fake hook for isolated tests.
    Production leaves it unset, binding OfficialPlannerCPU's pinned model SHA.
    """
    def __init__(self, model_path, *, timeout_s=.25, planner_factory=None,cpu_affinity=None):
        import multiprocessing
        import os
        if cpu_affinity is not None:
            require(isinstance(cpu_affinity,(tuple,list)) and bool(cpu_affinity)
                and all(type(cpu) is int and cpu>=0 for cpu in cpu_affinity)
                and len(set(cpu_affinity)) == len(cpu_affinity), 'Invalid planner CPU affinity')
            require(hasattr(os,'sched_getaffinity') and set(cpu_affinity).issubset(os.sched_getaffinity(0)),
                'Planner CPUs must be in inherited affinity')
            cpu_affinity = tuple(cpu_affinity)
        self.cpu_affinity = cpu_affinity
        require(math.isfinite(timeout_s) and timeout_s > 0, 'Positive finite planning timeout required')
        self.model_path = Path(model_path).absolute()
        self.timeout_s = float(timeout_s)
        self.factory = planner_factory or OfficialPlannerCPU
        self._context = multiprocessing.get_context('spawn')
        self._requests = _Mailbox(self._context, 16384)
        self._results = _Mailbox(self._context, 262144)
        self._statuses = _Mailbox(self._context, 16384)
        self._wakeup, self._stop = self._context.Event(), self._context.Event()
        self._process = None
        self._closed = False
        self._sequence = 0
        self._delivered = 0
        self._latest = None
        self._cached_status = {'status': 'not_started', 'ready': False, 'error': None}
        self._busy_submits = 0
        self._dropped_polled_results = 0

    def start(self):
        require(not self._closed, 'Closed planner cannot be restarted')
        if self._process is None:
            self._process = self._context.Process(target=_planner_process,
                args=(self.model_path, self.factory, self._requests, self._results,
                      self._statuses, self._wakeup, self._stop,self.cpu_affinity), daemon=True,
                name='sonic-locomotion-cpu')
            self._process.start()
            self._cached_status = {'status': 'starting', 'ready': False, 'error': None}
        return self

    def submit(self, canonical_context, alignment, *, mode, movement_world, facing_world,
               speed, seed, height=-1., timeout_s=None):
        require(self._process is not None and not self._closed, 'Start the planner before submitting')
        require(type(mode) is int and 0 <= mode < 27, 'Explicit official locomotion mode required')
        require(type(seed) is int and 0 <= seed < 2**63, 'Official nonnegative int64 seed required')
        require(isinstance(alignment, WorldAlignment), 'Explicit calibrated WorldAlignment required')
        context = finite(canonical_context, (4,36), 'canonical forecast context').copy()
        quaternion(context[:,3:7], (4,4), 'context roots')
        movement = finite(movement_world, (3,), 'movement world').copy()
        facing = finite(facing_world, (3,), 'facing world').copy()
        require(np.linalg.norm(facing[:2]) > 1e-8, 'Facing direction requires a horizontal heading')
        require(math.isfinite(speed) and math.isfinite(height), 'Finite speed/height required')
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        require(math.isfinite(timeout) and timeout > 0, 'Positive finite timeout required')
        submitted = time.monotonic()
        request_id = self._sequence + 1
        request = {'request_id': request_id, 'submitted_monotonic': submitted,
            'expires_monotonic': submitted+timeout, 'context': context,
            'alignment': WorldAlignment(alignment.initial_actual_pose7.copy(),
                alignment.yaw_wxyz.copy(), alignment.origin_xy.copy()),
            'command': {'mode':mode, 'movement_world':movement, 'facing_world':facing,
                        'speed':float(speed), 'height':float(height), 'seed':seed}}
        if not self._requests.put(request):
            self._busy_submits += 1
            return None
        self._sequence = request_id
        self._latest = request
        self._wakeup.set()
        return request_id

    def poll(self):
        if self._closed:
            return None
        result = self._results.get(consume=True)
        if result is None:
            return None
        if (result.request_id != self._sequence or result.request_id <= self._delivered
                or time.monotonic() >= result.expires_monotonic):
            self._dropped_polled_results += 1
            return None
        self._delivered = result.request_id
        return result

    def status(self):
        if self._closed:
            return {**self._cached_status, 'status':'closed', 'ready':False}
        status = self._statuses.get()
        if status is not None:
            self._cached_status = status
        status = {**self._cached_status, 'latest_request_id':self._sequence,
            'delivered_request_id':self._delivered, 'busy_submits':self._busy_submits,
            'dropped_polled_results':self._dropped_polled_results}
        if self._process is not None and not self._process.is_alive():
            status.update(ready=False, status='error', error=status.get('error') or
                f'Planner CPU process exited ({self._process.exitcode})')
        if self._latest is not None and self._delivered != self._sequence:
            status['latest_request_expired'] = time.monotonic() >= self._latest['expires_monotonic']
        else:
            status['latest_request_expired'] = False
        return status

    def close(self, timeout_s=.5):
        if self._closed:
            return
        self._stop.set()
        self._wakeup.set()
        if self._process is not None:
            self._process.join(max(0., float(timeout_s)))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(.5)
            require(not self._process.is_alive(), 'Planner worker did not exit')
            self._process.close()
            self._process = None
        self._closed = True
        self._cached_status.update(status='closed', ready=False)

    def __enter__(self):
        return self.start()

    def __exit__(self, *unused):
        self.close()
