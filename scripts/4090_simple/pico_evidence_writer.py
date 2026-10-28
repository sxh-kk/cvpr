"""Bounded, fail-closed evidence handling in a prestarted CPU spawn process.

The caller freezes each request with pickle *before* queueing its bytes.  The
worker owns the handler and its files; it never receives a live simulator or
SDK.  ``submit`` is nonblocking, while startup, ``call`` and ``close`` have
explicit deadlines.  Queue saturation is an evidence failure, never a drop
policy.  A successful close acknowledges every accepted request and the
handler's close, and observes a clean child exit.  It does not certify that a
handler fsyncs its files: that remains the handler's explicit responsibility.

Factories must be importable top-level callables or ``module:factory`` strings.
Handlers implement ``handle(op, payload)`` and ``close()``.  Imports have no
process, environment, affinity, GPU, simulator, or SDK side effects.
"""
from __future__ import annotations

import importlib
import math
import multiprocessing
import os
import pickle
import queue
import sys
import threading
import time
import traceback


CONTRACT = 'cvpr-pico-independent-cpu-evidence-writer-v1'
THREAD_ENVIRONMENT = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                      'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


class EvidenceWriterError(RuntimeError):
    """Sticky loss of the evidence-writing contract."""


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return float(value)


def _select_cpus(cpus):
    if not hasattr(os, 'sched_getaffinity'):
        raise ValueError('Evidence worker requires explicit Linux CPU affinity')
    inherited = sorted(os.sched_getaffinity(0))
    if cpus is None:
        cpus = inherited[-4:-3]
        if not cpus:
            raise ValueError('Fewer than four allowed CPUs; pass explicit evidence-worker cpus')
    if (not isinstance(cpus, (list, tuple)) or not cpus
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or len(set(cpus)) != len(cpus) or not set(cpus).issubset(inherited)):
        raise ValueError('Evidence worker cpus must be unique integer members of inherited affinity')
    return tuple(cpus), inherited


def _library_policy():
    # Spawn can import the caller before entering this function.  Limit any
    # already-loaded pools too; newly imported libraries see the environment.
    import threadpoolctl
    guard = threadpoolctl.threadpool_limits(limits=1)
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    cv2 = sys.modules.get('cv2')
    if cv2 is not None:
        cv2.setNumThreads(1)
    numexpr = sys.modules.get('numexpr')
    if numexpr is not None:
        numexpr.set_num_threads(1)
    pools = threadpoolctl.threadpool_info()
    if any(pool.get('num_threads') != 1 for pool in pools):
        raise RuntimeError('A native library pool did not accept one thread')
    return guard, pools


def _resolve_factory(factory):
    if isinstance(factory, str):
        module, separator, name = factory.partition(':')
        if not separator or not module or not name:
            raise ValueError('Factory path must be module:factory')
        factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError('Evidence handler factory must be callable')
    return factory


def _worker(factory_bytes, init_bytes, cpus, inherited, requests, replies):
    handler = None
    handler_closed = False
    handler_close_started = False
    guards = []

    def emit(message):
        # Serialize before acknowledging, without a child feeder that could
        # hide serialization errors or block interpreter exit during flush.
        replies.send_bytes(pickle.dumps(message, protocol=5))

    request_id = None
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = '1'
        os.sched_setaffinity(0, cpus)
        nice = os.getpriority(os.PRIO_PROCESS, 0)
        if nice < 19:
            os.nice(19 - nice)
        guard, _ = _library_policy()
        guards.append(guard)
        factory = _resolve_factory(pickle.loads(factory_bytes))
        handler = factory(**pickle.loads(init_bytes))
        if not callable(getattr(handler, 'handle', None)) or not callable(getattr(handler, 'close', None)):
            raise TypeError('Handler must implement handle(op, payload) and close()')
        guard, pools = _library_policy()
        guards.append(guard)
        actual_cpus = sorted(os.sched_getaffinity(0))
        if actual_cpus != sorted(cpus):
            raise RuntimeError('Worker affinity differs from the requested CPUs')
        emit({'kind': 'ready', 'worker': {'contract': CONTRACT, 'pid': os.getpid(),
              'cpus': actual_cpus, 'inherited_cpus': inherited,
              'nice': os.getpriority(os.PRIO_PROCESS, 0), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
              'thread_environment': {name: os.environ[name] for name in THREAD_ENVIRONMENT},
              'native_pools': pools, 'parent_runtime_modified': False,
              'library_threads': 1, 'GPU_used': False}})
        while True:
            request = pickle.loads(requests.get())
            request_id = request.get('id')
            if request['kind'] == 'close':
                handler_close_started = True
                result = handler.close()
                handler_closed = True
                emit({'kind': 'closed', 'result': result})
                return
            result = handler.handle(request['op'], request['payload'])
            emit({'kind': 'result', 'id': request_id,
                  'result': result if request['return_result'] else None})
    except BaseException as error:
        try:
            emit({'kind': 'error', 'id': request_id, 'error': type(error).__name__ + ': ' + str(error),
                  'traceback': traceback.format_exc(limit=8)})
        except BaseException:
            pass
    finally:
        if handler is not None and not handler_close_started:
            try:
                handler.close()
            except BaseException:
                pass
        # Keep threadpool limits alive until all handler work has finished.
        for guard in reversed(guards):
            guard.restore_original_limits()
        replies.close()


def _receive_replies(connection, inbox):
    """Only blocking pipe I/O lives here, never in the real-time caller.

    Connection.poll()+recv_bytes() can still block after a partial large pipe
    message. A dedicated reader keeps call/close deadlines enforceable if a
    worker dies or stalls mid-reply. The parent closes its sender immediately
    after spawn, so the owned child's exit always makes this reader see EOF.
    """
    try:
        while True:
            inbox.put(connection.recv_bytes())
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


class EvidenceWriter:
    """Single-caller FIFO writer.  Never hand this object to another process.

    Async results are discarded unless ``return_result=True`` is requested.
    For those requests ``result(id)`` consumes the returned value once.  The
    same sticky error is exposed by ``failed``, ``error`` and ``check``; caller
    timeouts do not cancel an operation or make its evidence complete.
    """

    def __init__(self, handler_factory, *, init_kwargs=None, cpus=None,
                 max_pending=64, max_request_bytes=2 * 1024 * 1024, startup_timeout_s=15.):
        timeout = _positive_seconds(startup_timeout_s, 'startup_timeout_s')
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        if type(max_request_bytes) is not int or max_request_bytes < 1:
            raise ValueError('max_request_bytes must be a positive integer')
        cpus, inherited = _select_cpus(cpus)
        if init_kwargs is not None and not isinstance(init_kwargs, dict):
            raise TypeError('init_kwargs must be a dictionary')
        factory_bytes = pickle.dumps(handler_factory, protocol=5)
        init_bytes = pickle.dumps({} if init_kwargs is None else init_kwargs, protocol=5)
        self._context = multiprocessing.get_context('spawn')
        self._requests = self._context.Queue(maxsize=max_pending)
        self._requests._ignore_epipe = True
        reply_reader, reply_sender = self._context.Pipe(duplex=False)
        self._replies = queue.Queue()
        self._reply_thread = None
        self._process = self._context.Process(target=_worker,
            args=(factory_bytes, init_bytes, cpus, inherited, self._requests, reply_sender),
            name='pico-evidence-writer', daemon=True)
        self._worker = None
        self._error = None
        self._fault = None
        self._accepted = self._acknowledged = 0
        self._max_request_bytes = max_request_bytes
        self._results = {}
        self._result_expected = set()
        self._handler_closed = False
        self._handler_close_result = None
        self._closing = self._closed = False
        self._receipt = None
        try:
            self._process.start()
            # This process is only a producer. Retaining its unused read end
            # could strand the feeder on a full pipe after child death. Close
            # that owned duplicate after spawn has copied it to the consumer.
            self._requests._reader.close()
            reply_sender.close()
            self._reply_thread = threading.Thread(target=_receive_replies,
                args=(reply_reader, self._replies), name='pico-evidence-replies', daemon=True)
            self._reply_thread.start()
            deadline = time.monotonic() + timeout
            while self._worker is None and not self.failed and time.monotonic() < deadline:
                time.sleep(.002)
            if self._worker is None and not self._error:
                self._fail('startup_timeout')
            self.check()
        except BaseException:
            reply_sender.close()
            if self._reply_thread is None:
                reply_reader.close()
            self.close(timeout_s=.5)
            raise

    def _fail(self, reason, **details):
        if self._error is None:
            self._error = str(reason)
            self._fault = {'reason': self._error, **details}

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                try:
                    message = pickle.loads(self._replies.get_nowait())
                except queue.Empty:
                    break
                kind = message['kind']
                if kind == 'ready' and self._worker is None:
                    self._worker = message['worker']
                elif kind == 'result':
                    request_id = message['id']
                    if type(request_id) is not int or request_id != self._acknowledged + 1 or request_id > self._accepted:
                        self._fail('invalid_or_non_fifo_acknowledgement')
                        continue
                    self._acknowledged = request_id
                    if request_id in self._result_expected:
                        self._results[request_id] = message['result']
                elif kind == 'error':
                    self._fail('handler_error: ' + message['error'], worker_error=message)
                elif kind == 'closed':
                    self._handler_closed = True
                    self._handler_close_result = message['result']
                else:
                    self._fail('invalid_worker_message')
            # closed is only valid after the caller's FIFO close request.
            if self._handler_closed and not self._closing:
                self._fail('unexpected_worker_close')
            if self._process.pid is not None and self._process.exitcode is not None:
                if not (self._closing and self._handler_closed and self._process.exitcode == 0):
                    # The reader may still be draining a final reply already
                    # in the pipe when the child exited. Do not replace that
                    # reply with a premature process-death failure.
                    if ((self._reply_thread is None or not self._reply_thread.is_alive())
                            and self._replies.empty()):
                        self._fail('worker_died: exitcode=' + str(self._process.exitcode))
        except (OSError, EOFError, ValueError, TypeError, KeyError, pickle.PickleError) as error:
            self._fail('reply_channel_error: ' + repr(error))

    @property
    def failed(self):
        self._poll()
        return self._error is not None

    @property
    def error(self):
        self._poll()
        return self._error

    def _status(self):
        return {'contract': CONTRACT, 'status': 'failed' if self._error else 'closed' if self._closed else 'ready',
                'accepted': self._accepted, 'acknowledged': self._acknowledged,
                'pending': self._accepted - self._acknowledged, 'error': self._error,
                'fault': self._fault, 'worker': self._worker}

    def check(self):
        self._poll()
        if self._error:
            raise EvidenceWriterError(self._error)
        return self._status()

    def submit(self, op, payload=None, *, return_result=False):
        self.check()
        if self._closing or self._closed:
            raise EvidenceWriterError('Evidence writer is closing or closed')
        if not isinstance(op, str) or not op or type(return_result) is not bool:
            self._fail('invalid_request_operation_or_result_flag')
            self.check()
        request_id = self._accepted + 1
        try:
            encoded = pickle.dumps({'kind': 'request', 'id': request_id, 'op': op,
                                    'payload': payload, 'return_result': return_result}, protocol=5)
        except Exception as error:
            self._fail('request_serialization_error: ' + repr(error))
            self.check()
        if len(encoded) > self._max_request_bytes:
            self._fail('request_payload_too_large', encoded_bytes=len(encoded),
                       maximum_bytes=self._max_request_bytes)
            self.check()
        try:
            self._requests.put_nowait(encoded)
        except queue.Full:
            self._fail('request_queue_full')
            self.check()
        except Exception as error:
            self._fail('request_queue_error: ' + repr(error))
            self.check()
        self._accepted = request_id
        if return_result:
            self._result_expected.add(request_id)
        return request_id

    def result(self, request_id, *, timeout_s=5.):
        timeout = _positive_seconds(timeout_s, 'timeout_s')
        if type(request_id) is not int or request_id not in self._result_expected:
            raise ValueError('Request did not reserve a result, or its result was already consumed')
        deadline = time.monotonic() + timeout
        while True:
            self.check()
            if request_id in self._results:
                self._result_expected.remove(request_id)
                return self._results.pop(request_id)
            if time.monotonic() >= deadline:
                self._fail('request_timeout: id=' + str(request_id))
                self.check()
            time.sleep(.001)

    def call(self, op, payload=None, *, timeout_s=5.):
        timeout = _positive_seconds(timeout_s, 'timeout_s')
        deadline = time.monotonic() + timeout
        request_id = self.submit(op, payload, return_result=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._fail('request_timeout: id=' + str(request_id))
            self.check()
        return self.result(request_id, timeout_s=remaining)

    def close(self, *, timeout_s=5.):
        timeout = _positive_seconds(timeout_s, 'timeout_s')
        if self._closed:
            return self._receipt
        self._closing = True
        deadline = time.monotonic() + timeout
        close_sent = False
        try:
            while self._process.pid is not None and time.monotonic() < deadline:
                self._poll()
                if self._process.exitcode is not None:
                    if ((self._reply_thread is None or not self._reply_thread.is_alive())
                            and self._replies.empty()):
                        break
                    time.sleep(.001)
                    continue
                if not close_sent:
                    try:
                        self._requests.put_nowait(pickle.dumps({'kind': 'close'}, protocol=5))
                        close_sent = True
                    except queue.Full:
                        pass
                    except (OSError, ValueError) as error:
                        self._fail('close_queue_error: ' + repr(error))
                        break
                time.sleep(.001)
            self._poll()
            if self._process.pid is not None and self._process.is_alive():
                self._fail('close_timeout_unfinished_evidence')
                self._process.terminate()
                self._process.join(.25)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(.25)
                if self._process.is_alive():
                    self._fail('worker_did_not_exit_after_kill')
            if self._process.pid is not None:
                self._process.join(0)
            self._poll()
            if not self._handler_closed or self._acknowledged != self._accepted or self._process.exitcode != 0:
                self._fail('close_incomplete_evidence')
        finally:
            # A stopped reader must never make interpreter shutdown join a
            # blocked parent feeder. Completion was checked before cleanup.
            self._requests.cancel_join_thread()
            self._requests.close()
            feeder = self._requests._thread
            if feeder is not None:
                feeder.join(.1)
                if feeder.is_alive():
                    self._fail('request_feeder_did_not_exit')
            if feeder is None or not feeder.is_alive():
                self._requests._writer.close()
            if self._process.pid is None:
                self._requests._reader.close()
            if self._reply_thread is not None:
                self._reply_thread.join(.1)
                if self._reply_thread.is_alive():
                    self._fail('reply_reader_did_not_exit')
            self._closed = True
        self._receipt = {**self._status(), 'status': 'failed' if self._error else 'completed',
            'completed': self._error is None, 'handler_close_result': self._handler_close_result,
            'handler_closed': self._handler_closed, 'exitcode': self._process.exitcode}
        if self._process.pid is not None and not self._process.is_alive():
            self._process.close()
        return self._receipt

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, exc_type, exc, tb):
        receipt = self.close()
        if exc_type is None and not receipt['completed']:
            raise EvidenceWriterError(receipt['error'])
