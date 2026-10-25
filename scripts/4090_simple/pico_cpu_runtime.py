"""Explicit one-thread CPU-library policy for this collector process only.

Importing this module changes nothing. Call configure_cpu_runtime once before
model/reference startup. It also limits already loaded BLAS/OpenMP libraries,
which setting environment variables after importing NumPy alone cannot do.
This is a library parallelism policy, not a claim of one total OS thread: SDK,
HTTP and writer threads still exist. No SDK, model, CUDA, affinity, or external
process is started or controlled here.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import threading


CONTRACT = "cvpr-pico-cpu-runtime-one-thread-v1"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_lock = threading.Lock()
_active_guard = None
_configured_once = False


class CpuRuntimeConfigurationError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = copy.deepcopy(receipt)


def _os_threads():
    try:
        return len(tuple(Path("/proc/self/task").iterdir()))
    except OSError:
        return None


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _count(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " returned an invalid thread count: " + repr(value))
    return value


def _runtime_state(torch, cv2, threadpoolctl, numexpr=None):
    state = {"torch_intraop_threads": _count(torch.get_num_threads(), "torch intra-op"),
        "torch_interop_threads": _count(torch.get_num_interop_threads(), "torch inter-op"),
        "opencv_threads": _count(cv2.getNumThreads(), "OpenCV"),
        "native_pools": _json_copy(threadpoolctl.threadpool_info()),
        "os_process_threads": _os_threads()}
    if numexpr is not None:
        state["numexpr_threads"] = _count(numexpr.get_num_threads(), "numexpr")
    return state


def _verify_one_thread(state):
    for name in ("torch_intraop_threads", "torch_interop_threads", "opencv_threads", "numexpr_threads"):
        if name in state and state[name] != 1:
            raise ValueError(name + " is " + str(state[name]) + ", expected 1")
    if not isinstance(state["native_pools"], list):
        raise ValueError("threadpoolctl did not return a list of native pools")
    if not any(pool.get("user_api") == "blas" for pool in state["native_pools"] if isinstance(pool, dict)):
        raise ValueError("No loaded BLAS pool could be verified by threadpoolctl")
    for pool in state["native_pools"]:
        if not isinstance(pool, dict) or type(pool.get("num_threads")) is not int or pool["num_threads"] != 1:
            raise ValueError("A loaded native pool did not accept one thread: " + repr(pool))


class CpuRuntimeGuard:
    """Keep native limits alive; close releases only threadpoolctl's scope.

    Torch inter-op is a process-wide once-only setting and cannot be safely
    restored after work begins. Torch intra-op, OpenCV, optional NumExpr and the
    environment stay at one for this process. close never pretends to roll back
    that global policy and does not touch any other process.
    """
    def __init__(self, limiter, receipt, modules):
        self._limiter = limiter
        self._receipt = receipt
        self._modules = modules
        self.closed = False

    @property
    def receipt(self):
        return _json_copy(self._receipt)

    def close(self):
        with _lock:
            if self.closed:
                return self.receipt
            try:
                self._limiter.restore_original_limits()
                self.closed = True
                self._receipt.update(status="closed", native_limit_scope_released=True,
                    global_process_policy_restored=False)
                self._receipt["after_close"] = _runtime_state(*self._modules)
                return self.receipt
            except Exception as error:
                self._receipt.update(status="close_failed", close_error=repr(error),
                    global_process_policy_restored=False)
                raise CpuRuntimeConfigurationError("CPU thread limit cleanup failed: " + str(error),
                    self._receipt) from error

    def __enter__(self):
        if self.closed:
            raise RuntimeError("CPU runtime guard is already closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def configure_cpu_runtime():
    """Apply and verify one-thread CPU policy, or raise with a failure receipt.

    A repeated call while the guard is active returns the same guard without
    re-setting Torch inter-op. A closed policy cannot be silently reactivated.
    Newly imported libraries see the environment policy; already loaded native
    pools receive an actual runtime limit. Call before any parallel Torch work.
    """
    global _active_guard, _configured_once
    with _lock:
        if _active_guard is not None and not _active_guard.closed:
            if _active_guard.receipt.get("status") != "applied":
                raise CpuRuntimeConfigurationError("CPU runtime guard is not in a verified active state",
                    _active_guard.receipt)
            return _active_guard
        if _configured_once:
            raise RuntimeError("CPU runtime policy was already configured and closed in this process")
        receipt = {"contract": CONTRACT, "status": "configuring", "process_id": os.getpid(),
            "python_executable": sys.executable,
            "scope": "current-process library parallelism; not a limit on total OS threads",
            "environment_before": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
            "loaded_before_call": {name: name in sys.modules for name in ("numpy", "torch", "cv2", "numexpr")},
            "os_process_threads_before_imports": _os_threads(),
            "before_measurement_scope": "after environment pinning and required imports, before runtime setters",
            "cuda_api_called": False, "external_processes_modified": False,
            "global_process_policy_restorable": False}
        limiter = None
        modules = None
        action = "set_environment"
        try:
            for name in THREAD_ENVIRONMENT:
                os.environ[name] = "1"
            receipt["environment_after"] = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
            action = "import_cpu_libraries"
            numpy = importlib.import_module("numpy")
            torch = importlib.import_module("torch")
            cv2 = importlib.import_module("cv2")
            threadpoolctl = importlib.import_module("threadpoolctl")
            numexpr = sys.modules.get("numexpr")
            modules = (torch, cv2, threadpoolctl, numexpr)
            receipt["versions"] = {"python": platform.python_version(),
                "numpy": str(numpy.__version__), "torch": str(torch.__version__),
                "opencv": str(cv2.__version__), "threadpoolctl": str(threadpoolctl.__version__)}
            if numexpr is not None:
                receipt["versions"]["numexpr"] = str(numexpr.__version__)
            action = "measure_before_runtime_limits"
            receipt["before"] = _runtime_state(*modules)
            action = "torch_intraop"
            torch.set_num_threads(1)
            action = "torch_interop"
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            action = "opencv"
            cv2.setNumThreads(1)
            if numexpr is not None:
                action = "numexpr"
                numexpr.set_num_threads(1)
            action = "loaded_native_pools"
            # Applies immediately and covers BLAS plus OpenMP already in memory.
            limiter = threadpoolctl.threadpool_limits(limits=1)
            action = "measure_and_verify_after"
            receipt["after"] = _runtime_state(*modules)
            _verify_one_thread(receipt["after"])
            if any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT):
                raise ValueError("A CPU thread environment limit changed during configuration")
            receipt.update(status="applied", verified_one_thread=True,
                native_limit_scope_released=False)
            receipt = _json_copy(receipt)
            _active_guard = CpuRuntimeGuard(limiter, receipt, modules)
            _configured_once = True
            return _active_guard
        except Exception as error:
            receipt.update(status="failed", verified_one_thread=False,
                failed_action=action, error=repr(error), global_process_policy_restored=False)
            receipt["environment_after_failure"] = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
            if modules is not None:
                try:
                    receipt["observed_after_failure"] = _runtime_state(*modules)
                except Exception as measurement_error:
                    receipt["failure_measurement_error"] = repr(measurement_error)
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    receipt["native_limit_scope_released"] = True
                except Exception as cleanup_error:
                    receipt["native_limit_cleanup_error"] = repr(cleanup_error)
            raise CpuRuntimeConfigurationError("CPU one-thread policy failed at " + action + ": " +
                str(error) + "; process-wide settings may be partially applied", receipt) from error
