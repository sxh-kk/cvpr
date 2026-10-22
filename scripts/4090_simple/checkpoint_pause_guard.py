#!/usr/bin/env python3
"""Conservative, leased pause at an already completed task checkpoint.

No model/trainer imports and no termination signals. ``bind-ranks`` is read-only;
``watch`` sends SIGSTOP/SIGCONT only with explicit --execute. A separate watchdog
resumes the identity-bound ranks when the controller exits or its lease expires.
Stack evidence proves only optimizer-update progress, not unsaved DataLoader
worker augmentation/prefetch RNG or bitwise future-training reproducibility.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import types
import zipfile

LAUNCH_COMMIT = "196ec733615be31ea60e299b268b3bf48a77ca1b"
TRAIN_REL = "VLMExpert/scripts/train.py"
POSTTRAIN_REL = "VLMExpert/src/psi/trainers/posttrain.py"
TRAINER_REL = "VLMExpert/src/psi/trainers/trainer.py"
TASK_REL = "scripts/4090_simple/a100_task_finetune.py"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def proc(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "state": raw[0], "ppid": int(raw[1]), "start_ticks": raw[19]}
    except (OSError, ValueError, IndexError):
        return None


def same_process(identity):
    actual = proc(identity["pid"])
    return actual is not None and actual["start_ticks"] == str(identity["start_ticks"]) and actual["state"] != "Z"


def all_threads_stopped(pid):
    tasks = list(Path(f"/proc/{pid}/task").glob("*/stat"))
    if not tasks:
        return False
    try:
        return all(path.read_text().rsplit(")", 1)[1].split()[0] in ("T", "t") for path in tasks)
    except (OSError, IndexError):
        return False


def selected_environment(pid):
    values = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, _, value = item.partition(b"=")
        if key in (b"RANK", b"LOCAL_RANK", b"WORLD_SIZE"):
            values[key.decode()] = value.decode()
    return values


def prepared_bindings(state):
    preparation = Path(state["preparation"]).resolve(strict=True)
    prepared = preparation.parent
    plan = read_json(preparation)
    config_path = prepared / "run_config.json"
    config_sha = digest(config_path)
    config = read_json(config_path)
    train, model = config["train"], config["model"]
    if (plan["timestamp"] != state["timestamp"] or plan["config_sha256"] != config_sha
            or plan["source_step"] != 38000 or plan["gpu_count"] != 8
            or plan["effective_global_batch"] != 32 or plan["checkpointing_steps"] != 4000
            or plan["max_optimizer_steps"] != 20000 or train["data_parallel"] != "ddp"
            or train["gradient_accumulation_steps"] != 1 or train["train_batch_size"] != 4
            or train.get("optimizer_fused", False) or train.get("optimizer_foreach") is not False
            or train["resume_from_checkpoint"] is not None or model["optim"] != "adamw_torch"):
        raise ValueError("Guard supports only the launch-bound38k fresh ordinary-AdamW DDP task")
    return {"preparation_sha256": digest(preparation), "config_sha256": config_sha}


def verify_rank(identity, *, manager, script, prepared, allow_stopped=False):
    actual = proc(identity["pid"])
    if (actual is None or not same_process(identity) or actual["ppid"] != manager["pid"]
            or not same_process(manager)):
        raise ValueError(f"Rank process identity/parent changed: rank {identity['rank']}")
    if actual["state"] in ("T", "t") and not allow_stopped:
        raise ValueError("Refuse ownership of a process that was already stopped")
    command_bytes = Path(f"/proc/{identity['pid']}/cmdline").read_bytes()
    if (identity.get("command_sha256") is not None
            and identity["command_sha256"] != hashlib.sha256(command_bytes).hexdigest()):
        raise ValueError("Rank command changed since identity binding")
    argv = [v.decode() for v in command_bytes.split(b"\0") if v]
    expected_script = str(Path(script).resolve())
    if expected_script not in argv or "train" not in argv or "--prepared" not in argv:
        raise ValueError("Rank command is not the expected task worker")
    position = argv.index("--prepared")
    if position + 1 >= len(argv) or Path(argv[position + 1]).resolve() != Path(prepared).resolve():
        raise ValueError("Rank belongs to a different prepared task")
    env = selected_environment(identity["pid"])
    if env != {"RANK": str(identity["rank"]), "LOCAL_RANK": str(identity["rank"]), "WORLD_SIZE": "8"}:
        raise ValueError("Rank environment differs from the eight-rank binding")
    return {**actual, "rank": identity["rank"], "selected_environment": env,
            "command_sha256": hashlib.sha256(command_bytes).hexdigest()}


def bind_ranks(state_path, repo):
    state = read_json(state_path)
    if state.get("state") != "running":
        raise ValueError("Only a running task may be bound")
    manager = proc(state["pid"])
    if manager is None or manager["state"] == "Z":
        raise ValueError("Task manager is not alive")
    prepared = Path(state["preparation"]).resolve().parent
    identities = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        candidate = proc(int(entry.name))
        if not candidate or candidate["ppid"] != manager["pid"]:
            continue
        try:
            rank = int(selected_environment(candidate["pid"]).get("RANK", "-1"))
            if rank not in range(8):
                continue
            identities.append(verify_rank({**candidate, "rank": rank}, manager=manager,
                script=Path(repo) / TASK_REL, prepared=prepared))
        except (ValueError, OSError):
            continue
    if len(identities) != 8 or sorted(i["rank"] for i in identities) != list(range(8)):
        raise ValueError("Need exactly one verified direct worker for every rank0..7")
    return {"created_utc": utc_now(), "boot_id": boot_id(), "run_timestamp": state["timestamp"],
            "run_directory": state["run_directory"], "prepared_directory": str(prepared),
            "prepared_bindings": prepared_bindings(state),
            "manager": manager, "ranks": sorted(identities, key=lambda row: row["rank"])}


def one(nodes, label):
    values = list(nodes)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(values)}")
    return values[0]


def is_call(node, receiver, attribute):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute and ast.unparse(node.func.value) == receiver)


def code_policy(train_source, posttrain_source, *, train_path, posttrain_path, task_path, trainer_path):
    train = one((n for n in ast.parse(train_source).body if isinstance(n, ast.FunctionDef) and n.name == "train"), "train function")
    step = one((n for n in ast.walk(train) if isinstance(n, ast.Assign)
                and is_call(n.value, "trainer", "step")), "canonical trainer.step call")
    completed = one((n for n in ast.walk(train) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "completed_step" for t in n.targets)), "completed_step assignment")
    increment = one((n for n in ast.walk(train) if isinstance(n, ast.AugAssign)
                     and isinstance(n.target, ast.Name) and n.target.id == "global_step"), "global step increment")
    loader_loop = one((n for n in ast.walk(train) if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Tuple)
                      and [getattr(t, "id", None) for t in n.target.elts] == ["local_step", "batch"]), "batch loop")
    klass = one((n for n in ast.parse(posttrain_source).body if isinstance(n, ast.ClassDef) and n.name == "PosttrainTrainer"), "PosttrainTrainer")
    inner = one((n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_training_step"), "_training_step")
    optimizer = one((n for n in ast.walk(inner) if is_call(n, "self.optimizer", "step")), "optimizer.step")
    if not step.lineno < completed.lineno < increment.lineno or optimizer.lineno <= inner.lineno:
        raise ValueError("Unexpected production training control flow")
    # A context-manager header can be the current line during __exit__, after
    # its body has already committed the optimizer update. In particular, the
    # enclosing `with gac` must never pass a simple lineno<optimizer check.
    ambiguous_headers = set()
    for node in ast.walk(inner):
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.For, ast.AsyncFor, ast.While)):
            if any(child is optimizer for child in ast.walk(node)):
                first_body = min(child.lineno for child in node.body)
                ambiguous_headers.update(range(node.lineno, first_body))
                if isinstance(node, ast.Try):
                    # Exception/finally paths make earlier-line reasoning too
                    # subtle; this pristine production method has no such try.
                    raise ValueError("Optimizer inside try/finally is unsupported")
    def nested_codes(code):
        yield code
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                yield from nested_codes(value)
    inner_code = one((code for code in nested_codes(compile(posttrain_source, str(posttrain_path), "exec"))
                      if code.co_name == "_training_step" and code.co_firstlineno == inner.lineno),
                     "compiled _training_step")
    pre_optimizer_lines = sorted({line for _, _, line in inner_code.co_lines()
        if line is not None and inner.lineno < line < optimizer.lineno and line not in ambiguous_headers})
    return {"train_path": str(Path(train_path).resolve()), "posttrain_path": str(Path(posttrain_path).resolve()),
            "task_path": str(Path(task_path).resolve()), "trainer_path": str(Path(trainer_path).resolve()),
            "train_step_call_line": step.lineno, "batch_fetch_line": loader_loop.lineno,
            "post_optimizer_start_line": completed.end_lineno + 1,
            "post_optimizer_end_line": increment.lineno,
            "pre_optimizer_lines": pre_optimizer_lines, "ambiguous_context_lines": sorted(ambiguous_headers),
            "optimizer_call_line": optimizer.lineno}


def bound_policy(repo, source_receipt):
    repo = Path(repo).resolve(strict=True)
    source = read_json(source_receipt)
    if source.get("git_commit") != LAUNCH_COMMIT or source.get("git_status") != "":
        raise ValueError("Require the pristine196ec73 launch receipt")
    bound = source.get("files_sha256", {})
    if POSTTRAIN_REL not in bound or TASK_REL not in bound:
        raise ValueError("Launch source bindings are missing")
    for relative, expected in bound.items():
        path = (repo / relative).resolve(strict=True)
        if not path.is_relative_to(repo) or digest(path) != expected:
            raise ValueError(f"Current source differs from running launch: {relative}")
    # The original receipt omitted these two loop files. Bind the clean launch
    # commit's exact content as well, instead of trusting current line numbers.
    texts = {}
    for relative in (TRAIN_REL, POSTTRAIN_REL, TRAINER_REL):
        committed = subprocess.check_output(["git", "show", f"{LAUNCH_COMMIT}:{relative}"], cwd=repo)
        path = repo / relative
        if committed != path.read_bytes():
            raise ValueError(f"Loop source differs from the known launch commit: {relative}")
        texts[relative] = committed.decode()
    policy = code_policy(texts[TRAIN_REL], texts[POSTTRAIN_REL], train_path=repo / TRAIN_REL,
                         posttrain_path=repo / POSTTRAIN_REL, task_path=repo / TASK_REL, trainer_path=repo / TRAINER_REL)
    policy["source_receipt_sha256"] = digest(source_receipt)
    policy["loop_source_sha256"] = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in texts.items()}
    policy["launch_commit"] = LAUNCH_COMMIT
    return policy


def frame_integer(frame, key):
    values = [row.get("repr") for row in frame.get("locals", []) if row.get("name") == key]
    if len(values) != 1 or not isinstance(values[0], str) or re.fullmatch(r"[0-9]+", values[0]) is None:
        raise ValueError(f"Missing/ambiguous integer local {key}")
    return int(values[0])


def validate_stack(snapshot, *, pid, checkpoint_step, policy):
    if not isinstance(snapshot, list) or not snapshot:
        raise ValueError("No Python threads captured")
    threads = []
    for thread in snapshot:
        if thread.get("pid") != pid or not isinstance(thread.get("frames"), list):
            raise ValueError("py-spy process/thread identity is ambiguous")
        frames = thread["frames"]
        for frame in frames:
            filename = str(frame.get("filename", "")).replace("\\", "/")
            if "/torch/optim/" in filename:
                raise ValueError("Optimizer call stack is not an allowed pause point")
            if (frame.get("name") == "step"
                    and filename not in (policy["task_path"], policy["trainer_path"])):
                raise ValueError("Unrecognized step call is not an allowed pause point")
        train = [frame for frame in frames if frame.get("name") == "train"
                 and frame.get("filename") == policy["train_path"]]
        if train:
            threads.append((one(train, "train frame"), frames))
    train, frames = one(threads, "thread executing the production train loop")
    global_step = frame_integer(train, "global_step")
    line = train.get("line")
    if type(line) is not int:
        raise ValueError("Missing train frame line")
    inner = [frame for frame in frames if frame.get("name") == "_training_step"
             and frame.get("filename") == policy["posttrain_path"]]
    wrappers = [frame for frame in frames if frame.get("name") == "step"
                and frame.get("filename") in (policy["task_path"], policy["trainer_path"])]
    if global_step == checkpoint_step - 1:
        if (frame_integer(train, "completed_step") != checkpoint_step or inner or wrappers
                or not policy["post_optimizer_start_line"] <= line <= policy["post_optimizer_end_line"]):
            raise ValueError("Previous counter is not proven post-optimizer at the checkpoint")
        phase = "checkpoint_update_completed_before_counter_increment"
    elif global_step == checkpoint_step:
        if line == policy["batch_fetch_line"] and not inner and not wrappers:
            phase = "fetching_next_batch_before_next_optimizer"
        elif line == policy["train_step_call_line"]:
            active = one(inner, "active production _training_step frame")
            active_line = active.get("line")
            if type(active_line) is not int or active_line not in policy["pre_optimizer_lines"]:
                raise ValueError("Next optimizer may have started or completed")
            for frame in wrappers:
                if frame_integer(frame, "global_step") != checkpoint_step:
                    raise ValueError("Wrapper step counter disagrees")
            phase = "next_training_step_proven_before_optimizer_call"
        else:
            raise ValueError("Next counter is outside the conservative pre-optimizer whitelist")
    else:
        raise ValueError(f"Checkpoint boundary missed: global_step={global_step}")
    return {"pid": pid, "global_step": global_step, "train_line": line, "phase": phase,
            "optimizer_call_line": policy["optimizer_call_line"], "checkpoint_step": checkpoint_step}


def marker_present(log, checkpoint):
    with Path(log).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 262144))
        tail = handle.read().decode(errors="replace")
    return f"Saved state to {checkpoint}" in tail


def checkpoint_complete(directory, log):
    directory = Path(directory).resolve(strict=True)
    if not marker_present(log, directory):
        raise ValueError("No explicit production Saved state completion marker")
    required = ["model.safetensors", "optimizer.bin", "scheduler.bin",
                *[f"random_states_{rank}.pkl" for rank in range(8)]]
    identities = {}
    for name in required:
        path = directory / name
        stat = path.stat()
        if stat.st_size <= 0:
            raise ValueError(f"Empty checkpoint payload: {name}")
        identities[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    with (directory / "model.safetensors").open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > 16777216:
            raise ValueError("Unexpected model header size")
        header = json.loads(handle.read(header_size))
    tensor_rows = [row for key, row in header.items() if key != "__metadata__"]
    if not tensor_rows or 8 + header_size + max(row["data_offsets"][1] for row in tensor_rows) != identities["model.safetensors"]["bytes"]:
        raise ValueError("Safetensors extent does not cover a complete model")
    for name in ("optimizer.bin", "scheduler.bin"):
        with zipfile.ZipFile(directory / name) as archive:
            if not archive.namelist():
                raise ValueError(f"Empty checkpoint archive: {name}")
    return {"checkpoint_directory": str(directory), "saved_state_marker": True,
            "payloads": identities, "model_stored_tensors": len(tensor_rows),
            "scope": "completion marker, all required files, model extent and ZIP directory; no repeated full tensor/CRC scan"}


def continue_identities(identities, errors=None):
    results = []
    for identity in identities:
        if same_process(identity):
            try:
                os.kill(identity["pid"], signal.SIGCONT)
                results.append(identity["pid"])
            except OSError as error:
                if errors is not None:
                    errors.append({"pid": identity["pid"], "error": str(error)})
    return results


def watchdog(lease_path):
    lease_path = Path(lease_path)
    initial = read_json(lease_path)
    atomic_json(lease_path.parent / "watchdog_ready.json", {"pid": os.getpid(), "utc": utc_now()})
    result = {"state": "auto_continued", "utc": utc_now()}
    try:
        while True:
            lease = read_json(lease_path)
            if (lease["controller"] != initial["controller"] or lease["ranks"] != initial["ranks"]
                    or lease["expires_monotonic"] != initial["expires_monotonic"]):
                raise ValueError("Pause lease identities/expiry changed")
            if not lease["active"]:
                result["state"] = "disarmed"
                break
            if not same_process(initial["controller"]) or time.monotonic() >= initial["expires_monotonic"]:
                break
            time.sleep(.02)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if result["state"] != "disarmed":
            errors = []
            result["continued_pids"] = continue_identities(initial["ranks"], errors)
            result["continue_errors"] = errors
    atomic_json(lease_path.parent / "watchdog_result.json", result)


def verify_live_binding(binding, state, repo, *, allow_stopped=False):
    if (binding["boot_id"] != boot_id() or binding["run_directory"] != state["run_directory"]
            or binding["run_timestamp"] != state["timestamp"]
            or state.get("state") != "running" or len(binding["ranks"]) != 8
            or state["pid"] != binding["manager"]["pid"]
            or Path(state["preparation"]).resolve().parent != Path(binding["prepared_directory"]).resolve()
            or sorted(i["rank"] for i in binding["ranks"]) != list(range(8))
            or len({i["pid"] for i in binding["ranks"]}) != 8):
        raise ValueError("Task/rank binding is not the expected live eight-rank run")
    if binding["prepared_bindings"] != prepared_bindings(state):
        raise ValueError("Prepared configuration changed after rank binding")
    return [verify_rank(i, manager=binding["manager"], script=Path(repo) / TASK_REL,
                        prepared=binding["prepared_directory"], allow_stopped=allow_stopped)
            for i in binding["ranks"]]


def inspect_paused(args, binding, policy):
    evidence = []
    for sample in range(2):
        for rank in binding["ranks"]:
            if not same_process(rank) or not all_threads_stopped(rank["pid"]):
                raise ValueError("A rank is not completely frozen")
            command = [str(args.py_spy), "dump", "--pid", str(rank["pid"]), "--locals",
                       "--full-filenames", "--json", "--nonblocking"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.dump_timeout)
            prefix = args.output / f"rank{rank['rank']}_snapshot{sample}"
            prefix.with_suffix(".stderr").write_text(result.stderr)
            prefix.with_suffix(".json").write_text(result.stdout)
            if result.returncode:
                raise ValueError(f"py-spy failed on rank{rank['rank']}")
            accepted = validate_stack(json.loads(result.stdout), pid=rank["pid"],
                                      checkpoint_step=args.checkpoint_step, policy=policy)
            if not same_process(rank) or not all_threads_stopped(rank["pid"]):
                raise ValueError("Rank was resumed while collecting stack evidence")
            evidence.append({"rank": rank["rank"], "sample": sample, **accepted,
                             "snapshot_sha256": digest(prefix.with_suffix(".json"))})
    return evidence


def leased_pause(args, binding, state, policy):
    # Reject a previously stopped or changed task before creating a watchdog
    # that could ever send SIGCONT to it.
    verify_live_binding(binding, state, args.repo)
    manager = proc(os.getpid())
    lease_path = args.output / "lease.json"
    lease = {"active": True, "controller": manager, "ranks": binding["ranks"],
             "expires_monotonic": time.monotonic() + args.hold_seconds}
    atomic_json(lease_path, lease)
    with (args.output / "watchdog.log").open("a") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_watchdog", str(lease_path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    started_watchdog = False
    stop_started = False
    receipt = {"state": "inspecting", "boot_id": boot_id(), "checkpoint_step": args.checkpoint_step,
               "run_timestamp": state["timestamp"], "run_directory": state["run_directory"],
               "controller": manager, "ranks": binding["ranks"], "lease_path": str(lease_path),
               "expires_monotonic": lease["expires_monotonic"], "source_policy": policy,
               "termination_signals_sent": False,
               "scope": "No completed optimizer update after this checkpoint; worker/prefetch RNG exact replay remains unproven"}
    try:
        # Confirm watchdog imports/startup before stopping any target.
        deadline = time.monotonic() + min(5, args.hold_seconds / 4)
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Watchdog exited before pause")
            ready = args.output / "watchdog_ready.json"
            if ready.is_file() and read_json(ready).get("pid") == child.pid:
                started_watchdog = True
                break
            time.sleep(.01)
        if not started_watchdog:
            raise RuntimeError("Independent watchdog did not become ready")
        receipt["watchdog"] = proc(child.pid)
        if receipt["watchdog"] is None:
            raise RuntimeError("Independent watchdog identity is unavailable")
        verify_live_binding(binding, state, args.repo)
        stop_started = True
        for identity in binding["ranks"]:
            if not same_process(identity):
                raise ValueError("Rank changed before SIGSTOP")
            os.kill(identity["pid"], signal.SIGSTOP)
        stop_deadline = time.monotonic() + args.stop_timeout
        while not all(same_process(i) and all_threads_stopped(i["pid"]) for i in binding["ranks"]):
            if time.monotonic() >= stop_deadline:
                raise TimeoutError("Not all eight ranks reached a complete group stop")
            time.sleep(.002)
        verify_live_binding(binding, read_json(args.run_state), args.repo, allow_stopped=True)
        checkpoint = Path(state["run_directory"]) / "checkpoints" / f"ckpt_{args.checkpoint_step}"
        receipt["checkpoint"] = checkpoint_complete(checkpoint, state["log"])
        receipt["stack_evidence"] = inspect_paused(args, binding, policy)
        if time.monotonic() >= lease["expires_monotonic"] or child.poll() is not None:
            raise TimeoutError("Pause lease expired during inspection")
        receipt.update(state="held_verified", generated_utc=utc_now())
        atomic_json(args.output / "receipt.json", receipt)
        print(json.dumps({"state": "held_verified", "receipt": str(args.output / "receipt.json"),
                          "expires_monotonic": lease["expires_monotonic"]}), flush=True)
        while time.monotonic() < lease["expires_monotonic"]:
            if not all(same_process(i) for i in binding["ranks"]):
                receipt["state"] = "targets_exited_while_held"
                break
            if not all(all_threads_stopped(i["pid"]) for i in binding["ranks"]):
                raise ValueError("A verified target was resumed during the hold")
            if child.poll() is not None:
                raise RuntimeError("Independent watchdog ended during the hold")
            time.sleep(.02)
        else:
            receipt["state"] = "released_at_lease_expiry"
        return receipt
    except BaseException as error:
        receipt.update(state="rejected_resumed", error=f"{type(error).__name__}: {error}")
        return receipt
    finally:
        # A second ordinary interrupt must not cut short the eight-rank release.
        # SIGKILL is handled independently by the watchdog's controller check.
        if __name__ == "__main__":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        receipt["continue_errors"] = []
        receipt["continued_pids"] = (continue_identities(binding["ranks"], receipt["continue_errors"])
                                     if stop_started else [])
        lease["active"] = bool(receipt["continue_errors"])
        if lease["active"]:
            receipt["state"] = "release_error_watchdog_active"
        atomic_json(lease_path, lease)
        receipt["finished_utc"] = utc_now()
        atomic_json(args.output / "receipt.json", receipt)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass  # Independent watchdog retains its finite lease; never kill target processes here.


def validate_held(receipt_path):
    receipt = read_json(receipt_path)
    lease = read_json(receipt["lease_path"])
    if (receipt.get("boot_id") != boot_id() or receipt.get("state") != "held_verified" or not lease.get("active")
            or len(receipt["ranks"]) != 8 or sorted(i["rank"] for i in receipt["ranks"]) != list(range(8))
            or lease["ranks"] != receipt["ranks"] or lease["controller"] != receipt["controller"]
            or lease["expires_monotonic"] != receipt["expires_monotonic"]
            or not same_process(receipt["watchdog"])
            or time.monotonic() >= lease["expires_monotonic"] or not same_process(receipt["controller"])
            or not all(same_process(i) and all_threads_stopped(i["pid"]) for i in receipt["ranks"])):
        raise ValueError("Held evidence expired, resumed, or no longer matches live processes")
    return receipt


def interrupted(signum, frame):
    raise InterruptedError(f"Guard interrupted by signal{signum}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_watchdog":
        lease_path = Path(sys.argv[2])
        watchdog(lease_path)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    bind = commands.add_parser("bind-ranks")
    bind.add_argument("--run-state", type=Path, required=True)
    bind.add_argument("--repo", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    held = commands.add_parser("validate-held")
    held.add_argument("--receipt", type=Path, required=True)
    watch = commands.add_parser("watch")
    for name in ("run-state", "rank-binding", "source-at-launch", "repo", "output", "py-spy"):
        watch.add_argument("--" + name, type=Path, required=True)
    watch.add_argument("--checkpoint-step", type=int, required=True)
    watch.add_argument("--poll-ms", type=float, default=20)
    watch.add_argument("--hold-seconds", type=float, default=120)
    watch.add_argument("--stop-timeout", type=float, default=5)
    watch.add_argument("--dump-timeout", type=float, default=5)
    watch.add_argument("--wait-timeout", type=float, default=21600)
    watch.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "bind-ranks":
        if args.output.exists():
            parser.error("Use a fresh rank binding output")
        atomic_json(args.output, bind_ranks(args.run_state, args.repo))
        return 0
    if args.mode == "validate-held":
        print(json.dumps(validate_held(args.receipt), ensure_ascii=False))
        return 0
    if (args.checkpoint_step not in (4000, 8000, 12000, 16000, 20000)
            or args.poll_ms < 5 or not 10 <= args.hold_seconds <= 180
            or args.stop_timeout <= 0 or args.dump_timeout <= 0 or args.wait_timeout <= 0):
        parser.error("Require positive boundary/timeouts, poll>=5ms, and lease10..180 seconds")
    state, binding = read_json(args.run_state), read_json(args.rank_binding)
    policy = bound_policy(args.repo, args.source_at_launch)
    verify_live_binding(binding, state, args.repo)
    if not args.execute:
        print(json.dumps({"state": "dry_run_no_signals", "checkpoint_step": args.checkpoint_step,
                          "source_policy": policy, "rank_count": 8}, ensure_ascii=False))
        return 0
    if args.output.exists():
        parser.error("Use a fresh guard output directory")
    args.output.mkdir(parents=True)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    checkpoint = Path(state["run_directory"]) / "checkpoints" / f"ckpt_{args.checkpoint_step}"
    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        if read_json(args.run_state).get("state") != "running":
            raise ValueError("Task is no longer running")
        if checkpoint.is_dir() and marker_present(state["log"], checkpoint):
            # Recheck pristine source and identities immediately before control.
            policy = bound_policy(args.repo, args.source_at_launch)
            verify_live_binding(binding, read_json(args.run_state), args.repo)
            checkpoint_complete(checkpoint, state["log"])
            result = leased_pause(args, binding, state, policy)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0 if result["state"] == "targets_exited_while_held" else 2
        time.sleep(args.poll_ms / 1000)
    atomic_json(args.output / "receipt.json", {"state": "boundary_wait_timeout_no_signals", "utc": utc_now()})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
