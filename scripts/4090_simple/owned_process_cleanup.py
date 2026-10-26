"""Strict cleanup for a recorded attempt process group, including /proc exit races.

Install this independent helper on the runtime module without changing numerical,
controller or protocol source files. Live permission failures remain fatal.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import time

CONTRACT = 'cvpr-owned-group-cleanup-exit-race-v1'
PROC_ROOT = Path('/proc')
TOKEN_KEY = 'QUALITY100_ATTEMPT_TOKEN='
DEAD_STATES = frozenset(('Z', 'X'))


@dataclass(frozen=True)
class ProcessStat:
    pid: int
    state: str
    group: int
    starttime: int


def _read_stat(entry):
    text=(entry/'stat').read_text()
    fields=text.rsplit(')',1)[1].split()
    return ProcessStat(int(text.split('(',1)[0].strip()),fields[0],int(fields[2]),int(fields[19]))


def _read_environment(entry):
    return (entry/'environ').read_bytes().split(b'\0')


def _remember(stat, seen):
    previous=seen.setdefault(stat.pid,stat.starttime)
    if previous != stat.starttime:
        raise RuntimeError(f'Recorded process PID {stat.pid} was reused during owned cleanup')


def _members(group, token, seen):
    """Verify every live group member before granting group-wide signal authority."""
    result=[]
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdecimal():
            continue
        pid=int(entry.name)
        try:
            try:
                process_group=os.getpgid(pid)
            except PermissionError:
                # Some unrelated namespace processes deny getpgid. A readable
                # stat may prove they are outside the recorded group; no live
                # member of the recorded group gets a permission exemption.
                if _read_stat(entry).group != group:
                    continue
                raise
            if process_group != group:
                continue
            stat=_read_stat(entry)
            if stat.pid != pid:
                raise RuntimeError('Process stat PID differs from its /proc directory')
            if stat.group != group:
                continue
            _remember(stat,seen)
            if stat.state in DEAD_STATES:
                continue
            try:
                environment=_read_environment(entry)
            except PermissionError:
                # This is the only relaxed error path: a previously confirmed
                # member exited between non-Z stat and environ. A live EACCES,
                # unreadable follow-up stat, or reused PID still fails closed.
                after=_read_stat(entry)
                if after.pid != pid:
                    raise RuntimeError('Process identity changed after environ permission failure')
                _remember(after,seen)
                if after.group == group and after.state in DEAD_STATES:
                    continue
                raise
            if (TOKEN_KEY+token).encode() not in environment:
                raise RuntimeError(f'Recorded process group {group} has an unowned live member {pid}')
            after=_read_stat(entry)
            if after.pid != pid:
                raise RuntimeError('Process identity changed after ownership check')
            _remember(after,seen)
            if after.group != group or after.state in DEAD_STATES:
                continue
            result.append(pid)
        except (FileNotFoundError,ProcessLookupError):
            # Disappearance proves there is no member left to signal. In
            # particular this handles disappearance on the EACCES stat re-read.
            continue
    return sorted(result)


def stop_owned_group(group, token, timeout=8):
    """Stop a recorded group only after exact attempt-token and PID identity checks."""
    if type(group) is not int or group <= 1 or group == os.getpgrp():
        raise ValueError('Require a recorded independent process group, never the evaluator group')
    if not isinstance(token,str) or not token or '\0' in token:
        raise ValueError('Require an exact nonempty attempt token')
    if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError('Cleanup timeout must be finite and nonnegative')
    seen={}
    members=lambda:_members(group,token,seen)
    initial=members()
    report={'process_group':group,'live_members_before':initial,'live_members_after':[]}
    if not initial:
        return report

    def send(signum):
        # Re-scan the complete group immediately before each signal. New
        # foreign members and PID reuse invalidate earlier signal authority.
        if not members():
            return
        try:
            os.killpg(group,signum)
        except ProcessLookupError:
            pass

    def wait_until(deadline):
        while True:
            remaining=members()
            if not remaining:
                return []
            delay=deadline-time.monotonic()
            if delay <= 0:
                return remaining
            time.sleep(min(.1,delay))

    send(signal.SIGTERM)
    remaining=wait_until(time.monotonic()+timeout)
    if remaining:
        send(signal.SIGKILL)
        remaining=wait_until(time.monotonic()+3)
    # A final complete ownership scan is required even after wait saw no live
    # members; a newly visible foreign member must not be reported as clean.
    remaining=members()
    if remaining:
        raise RuntimeError(f'Owned process group {group} did not stop: {remaining}')
    return report


def install(runtime_module):
    """Replace only runtime.stop_owned_group; stop_owned resolves that global too."""
    previous=getattr(runtime_module,'stop_owned_group',None)
    if not callable(previous):
        raise ValueError('Runtime module has no callable stop_owned_group helper')
    runtime_module.stop_owned_group=stop_owned_group
    return {'contract':CONTRACT,'runtime_module':getattr(runtime_module,'__name__',type(runtime_module).__name__),
        'helper':'stop_owned_group','already_installed':previous is stop_owned_group,
        'numerical_runtime_modified':False,'controller_modified':False,'protocol_modified':False}
