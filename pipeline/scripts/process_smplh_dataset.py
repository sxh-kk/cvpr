#!/usr/bin/env python3
"""Plan, calibrate and run resumable EgoSuite SMPL-H motion-only derivatives."""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np

from egomotion_pipeline.smplh_jobs import (
    assert_plan_current, create_plan, file_identity, read_source_motion,
    sequence_inputs, source_record_contract,
)
from egomotion_pipeline.storage import Archive, atomic_json, digest


DEFAULT_ROOT = Path("/gaozt-test1/datasets/MixEgoMotion")
DEFAULT_MODEL = DEFAULT_ROOT/"weights/smplh/male/model.npz"
DEFAULT_SPLITS = Path(__file__).resolve().parents[2]/"MotionExpert/exp/dense_k12/splits.json"
_worker_model = None
_worker_plan = None
_worker_config = None
_worker_shared = None
_worker_archive = None
_worker_model_identity = None


def now():
    return datetime.now(timezone.utc).isoformat()


def config_from_dict(value):
    from egomotion_pipeline.smplh_sequence import SequenceConfig
    value = dict(value)
    for name in ("offset_joints", "offset_anchors", "world_up"):
        if name in value:
            value[name] = tuple(value[name])
    return SequenceConfig(**value).validate()


def write_status(job, **updates):
    path = job/"status.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(updated_at=now(), **updates)
    atomic_json(path, value)
    atomic_json(job.parent/"current_smplh_sequence.json", value)


def worker_init(plan, shared):
    global _worker_model, _worker_plan, _worker_config, _worker_shared, _worker_archive, _worker_model_identity
    import torch
    from egomotion_pipeline.smplh_retarget import load_smplh_body22_model
    from egomotion_pipeline.smplh_sequence import SourceCenterPrior
    assert_plan_current(plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _worker_plan = plan
    _worker_config = config_from_dict(plan["config"])
    _worker_model = load_smplh_body22_model(plan["model_path"], shape_components=16)
    if _worker_model.metadata["model_sha256"] != plan["model_sha256"]:
        raise ValueError("Loaded model differs from the frozen job")
    _worker_model_identity = file_identity(plan["model_path"])
    _worker_shared = SourceCenterPrior.from_dict(shared) if shared is not None else None
    _worker_archive = Archive(plan["source_archive"])
    _worker_archive.records()
    assert_plan_current(plan)


def source_record(index):
    """Cache the inventory once per worker, but recheck each original record."""
    archive = _worker_archive
    assert_worker_code_current()
    archive.assert_ready()
    expected = archive.records()[int(index)]
    current = json.loads(archive.record_path(int(index)).read_text())
    if current != expected:
        raise ValueError("Source episode record changed after worker initialization")
    return archive, current


def assert_worker_code_current():
    # Do not let a running worker use already-imported code while recording the
    # hash of a subsequently edited file. The model byte hash was frozen at
    # planning/loading; its filesystem identity is also checked on each task.
    for path, expected in _worker_plan["implementation_sha256"].items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Producer changed during processing: "+path)
    if file_identity(_worker_plan["model_path"]) != _worker_model_identity:
        raise ValueError("Model asset changed during processing")


def limb_signature(arrays):
    p = arrays["observation.state.body_world"].reshape(-1, 22, 3)
    mask = arrays["observation.valid.body_position"]
    from egomotion_pipeline.schema import PARENTS
    result = []
    for child in (4, 5, 7, 8, 10, 11, 18, 19, 20, 21):
        parent = PARENTS[child]
        good = mask[:, child] & mask[:, parent]
        result.append(float(np.median(np.linalg.norm(p[good, child]-p[good, parent], axis=-1)))
                      if good.sum() >= 20 else None)
    return result


def signature_matches(first, second):
    paired = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    if len(paired) < 4:
        return False
    return all(abs(a-b) <= max(.01, .03*max(a, b)) for a, b in paired)


def calibrate_record(row, destination, phase):
    from egomotion_pipeline.smplh_sequence import calibrate_group
    archive, record = source_record(row["episode_index"])
    if record["metadata"]["operator_id"] != row["operator_id"]:
        raise ValueError("Calibration operator differs from source record")
    group_id = row.get("calibration_group_id", row["operator_id"])
    arrays, identity = read_source_motion(archive.data_path(record["episode_index"]))
    source_contract = source_record_contract(archive, record, arrays, identity)
    shared_sha = digest(_worker_shared.as_dict()) if _worker_shared is not None else None
    signature = digest(dict(job=_worker_plan["job_signature"], phase=phase, group_id=group_id,
                            source=source_contract, shared_sha256=shared_sha))
    destination = Path(destination)
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["signature"] != signature:
            raise ValueError("Calibration resume contract changed")
        return previous
    started = time.monotonic()
    args = sequence_inputs(arrays)
    calibration = calibrate_group(_worker_model, **args,
        shared_offsets=_worker_shared, group_id=group_id, source_id="egosuite",
        config=_worker_config)
    assert_worker_code_current()
    if identity != file_identity(archive.data_path(record["episode_index"])):
        raise ValueError("Source changed during calibration")
    value = dict(signature=signature, phase=phase, source=source_contract,
        calibration=calibration.as_dict(), group_id=group_id, operator_id=row["operator_id"], split=row["split"],
        limb_signature_m=limb_signature(arrays), shared_prior_sha256=shared_sha,
        elapsed_s=time.monotonic()-started, completed_at=now())
    atomic_json(destination, value)
    return value


def fit_record(row, calibration_path, output_root, job):
    from egomotion_pipeline.smplh_sequence import GroupCalibration, fit_sequence
    from egomotion_pipeline.smplh_sidecar import build_sidecar_contract, write_episode_sidecar
    index = int(row["episode_index"])
    archive, record = source_record(index)
    arrays, identity = read_source_motion(archive.data_path(index))
    envelope = json.loads(Path(calibration_path).read_text())
    if envelope["operator_id"] != row["operator_id"] or record["metadata"]["operator_id"] != row["operator_id"]:
        raise ValueError("Fitting operator differs from frozen calibration")
    if not signature_matches(envelope["limb_signature_m"], limb_signature(arrays)):
        # A different source skeleton never silently reuses this person's shape.
        destination = Path(job)/"calibration/recording-specific"/f"{index:08d}.json"
        own_row = dict(episode_index=index, operator_id=row["operator_id"], split=row["split"],
                       calibration_group_id=row["operator_id"]+":record:"+record["source_key"])
        envelope = calibrate_record(own_row, destination, "recording_specific")
        calibration_path = destination
    calibration = GroupCalibration.from_dict(envelope["calibration"])
    contract = build_sidecar_contract(archive.root, archive.data_path(index),
        _worker_plan["model_path"], _worker_plan["config"], calibration.as_dict(), arrays,
        target_joint_basis="smplh_body22_axes_v1")
    episode_dir = Path(output_root)/"episodes"/f"{index:08d}"
    if episode_dir.exists():
        from egomotion_pipeline.smplh_sidecar import validate_episode_sidecar
        report = validate_episode_sidecar(output_root, record, arrays, contract)
        return dict(episode_index=index, frames=record["length"], resumed=True,
                    validation=report, source_key=record["source_key"])
    started = time.monotonic()
    result = fit_sequence(_worker_model, **sequence_inputs(arrays), calibration=calibration,
                          config=_worker_config, source_id="egosuite", group_id=calibration.group_id)
    assert_worker_code_current()
    receipt = write_episode_sidecar(output_root, record, arrays, result.as_mapping(), contract)
    if identity != file_identity(archive.data_path(index)):
        raise ValueError("Source changed during sequence fitting")
    return dict(episode_index=index, source_key=record["source_key"], frames=record["length"],
                resumed=False, elapsed_s=time.monotonic()-started, receipt=receipt,
                calibration_path=str(calibration_path), completed_at=now())


def parallel(tasks, worker, plan, shared, workers, on_complete):
    failures = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(plan, shared)) as executor:
        futures = {executor.submit(worker, *args): args for args in tasks}
        for future in as_completed(futures):
            try:
                value = future.result()
            except Exception as exc:
                args = futures[future]
                row = args[0] if args and isinstance(args[0], dict) else {}
                failure = dict(event="worker_failed",episode_index=row.get("episode_index"),
                               operator_id=row.get("operator_id"),error_type=type(exc).__name__,error=str(exc))
                failures.append(failure)
                print(json.dumps(failure),flush=True)
            else:
                try:
                    on_complete(value)
                except Exception as exc:
                    failure = dict(event="callback_failed", error_type=type(exc).__name__, error=str(exc))
                    failures.append(failure)
                    print(json.dumps(failure), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} worker tasks failed; inspect worker_failed log records; successful episodes remain resumable")


def load_plan(job):
    plan = json.loads((job/"plan.json").read_text())
    assert_plan_current(plan)
    return plan


def run_calibration(args):
    from egomotion_pipeline.smplh_sequence import GroupCalibration, train_shared_calibration
    plan = load_plan(args.job)
    rows = plan["selection"]["calibration_records"]
    outputs = []
    tasks = [(row, args.job/"calibration/source-prior"/(row["operator_id"]+".json"), "source_prior") for row in rows]
    write_status(args.job, state="calibrating_shared_centers", completed=0, total=len(rows))
    def complete(value):
        outputs.append(value)
        print(json.dumps(dict(event="shared_calibration",group=value["group_id"],completed=len(outputs),total=len(rows),elapsed_s=value.get("elapsed_s"))),flush=True)
        write_status(args.job, completed=len(outputs))
    parallel(tasks, calibrate_record, plan, None, args.workers, complete)
    outputs.sort(key=lambda x:x["group_id"])
    prior = train_shared_calibration([GroupCalibration.from_dict(x["calibration"]) for x in outputs],
                                    group_ids=[x["group_id"] for x in outputs], source_id="egosuite")
    assert_plan_current(plan)
    prior_path = args.job/"shared-center-prior.json"
    document = dict(job_signature=plan["job_signature"], prior=prior.as_dict(),
        train_group_ids=[x["group_id"] for x in outputs], input_calibration_signatures=[x["signature"] for x in outputs],
        policy="equal_train_group_shape_conditioned_estimated_offsets; not_native_axis_certification")
    if prior_path.exists() and json.loads(prior_path.read_text()) != document:
        raise ValueError("Shared center prior changed on resume")
    atomic_json(prior_path, document)
    write_status(args.job, state="shared_calibration_complete", completed=len(outputs), total=len(rows))


def run_sequences(args):
    plan = load_plan(args.job)
    prior = json.loads((args.job/"shared-center-prior.json").read_text())
    if prior["job_signature"] != plan["job_signature"]:
        raise ValueError("Prior belongs to a different job")
    if args.scope == "full":
        gate_path = args.job/"reports/expanded-acceptance.json"
        if not gate_path.exists() or json.loads(gate_path.read_text()).get("full_candidate_processing_allowed") is not True:
            raise ValueError("Expanded independent acceptance is required before full candidate processing")
        archive = Archive(plan["source_archive"])
        split_by_group = {key:value["split"] for key,value in plan["groups"].items()}
        rows = [dict(episode_index=r["episode_index"],source_key=r["source_key"],
                     operator_id=r["metadata"]["operator_id"],split=split_by_group[r["metadata"]["operator_id"]]) for r in archive.records()]
    else:
        rows = plan["selection"]["validation_records"]
    needed = sorted({row["operator_id"] for row in rows})
    calibration_tasks = []
    for group in needed:
        item = plan["groups"][group]
        row = dict(episode_index=item["calibration_episode_index"],operator_id=group,split=item["split"])
        calibration_tasks.append((row,args.job/"calibration/groups"/(group+".json"),"group"))
    done = []
    def calibration_complete(value):
        done.append(value["group_id"])
        print(json.dumps(dict(event="group_calibration",group=value["group_id"],completed=len(done),total=len(needed))),flush=True)
        write_status(args.job, state="calibrating_groups", completed=len(done), total=len(needed))
    parallel(calibration_tasks,calibrate_record,plan,prior["prior"],args.workers,calibration_complete)
    output = args.output or args.job/("expanded-sidecars" if args.scope == "expanded" else "full-sidecars")
    if output.resolve() == Path(plan["source_archive"]).resolve():
        raise ValueError("Derived output cannot be the source archive")
    completed, results = [], []
    def fitted(value):
        completed.append(value["episode_index"])
        results.append(value)
        atomic_json(args.job/"results"/f"{value['episode_index']:08d}.json",value)
        print(json.dumps(dict(event="sequence",episode=value["episode_index"],frames=value["frames"],completed=len(completed),total=len(rows),elapsed_s=value.get("elapsed_s"),resumed=value["resumed"])),flush=True)
        write_status(args.job,state="fitting_"+args.scope,completed=len(completed),total=len(rows),
                     completed_frames=sum(x["frames"] for x in results),output_root=str(output))
    tasks=[(row,args.job/"calibration/groups"/(row["operator_id"]+".json"),output,args.job) for row in rows]
    write_status(args.job,state="fitting_"+args.scope,completed=0,total=len(rows),output_root=str(output))
    parallel(tasks,fit_record,plan,prior["prior"],args.workers,fitted)
    assert_plan_current(plan)
    atomic_json(args.job/(args.scope+"-execution.json"),dict(state="complete",scope=args.scope,
        job_signature=plan["job_signature"],output_root=str(output),episodes=len(results),frames=sum(x["frames"] for x in results),
        receipts=sorted(results,key=lambda x:x["episode_index"]),production_archive_modified=False,
        independently_validated=False,completed_at=now()))
    write_status(args.job,state=args.scope+"_fitting_complete",completed=len(results),total=len(rows),
                 output_root=str(output),independent_acceptance_pending=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("plan","calibrate","run","status"))
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--archive",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--splits",type=Path,default=DEFAULT_SPLITS)
    p.add_argument("--selection",type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--scope",choices=("expanded","full"),default="expanded")
    p.add_argument("--output",type=Path)
    return p


def execute(args):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.action=="plan":
        from egomotion_pipeline.smplh_sequence import SequenceConfig
        selection=json.loads((args.selection or args.job/"selection.json").read_text())
        cfg=config_from_dict(json.loads(args.config.read_text())) if args.config else SequenceConfig().validate()
        plan=create_plan(args.archive,args.splits,args.model,cfg,selection,args.job)
        write_status(args.job,state="planned",job_signature=plan["job_signature"],source_episodes=plan["source_episodes"],source_frames=plan["source_frames"])
        print(json.dumps(dict(state="planned",groups=len(plan["groups"]),episodes=plan["source_episodes"],signature=plan["job_signature"])))
    elif args.action=="calibrate":run_calibration(args)
    elif args.action=="run":run_sequences(args)
    else:print((args.job/"status.json").read_text())


def main():
    args=parser().parse_args()
    try:
        execute(args)
    except Exception as exc:
        if args.action != "status" and args.job.is_dir():
            write_status(args.job,state="failed",error_type=type(exc).__name__,error=str(exc),
                         failed_action=args.action)
        raise


if __name__=="__main__":
    main()
