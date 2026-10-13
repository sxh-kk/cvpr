"""Independent guard and episode-identity checks for the full ME reader proof."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/audit_motionexpert_smplh.py"
spec = importlib.util.spec_from_file_location("me_full_probe_under_test", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def guard_files(tmp_path):
    mix, hm = tmp_path / "Mix", tmp_path / "HM"
    root = mix / "parameters"
    script = tmp_path / "workspace/pipeline/scripts/audit.py"
    script.parent.mkdir(parents=True)
    script.write_text("# only the audit script is allowed")
    paths = {"owned": root / "motion.parquet", "old": mix / "data/motion.parquet", "hm_file": hm / "motion.parquet",
             "pipeline": script.parents[1] / "src/helper.py"}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet": pq.write_table(pa.table({"v": [1, 2]}), path)
        else: path.write_text("# forbidden pipeline helper")
    guard = probe.ReaderGuard(root, [mix, hm], script)
    return dict(root=root, mix=mix, hm=hm, script=script, guard=guard, **paths)


def test_guard_allows_owned_but_rejects_old_raw_pipeline_and_escape_symlink(guard_files):
    g = guard_files
    g["guard"].check(g["owned"])
    g["guard"].check(g["script"])
    for name in ("old", "hm_file"):
        with pytest.raises(PermissionError, match="old/source dependency"):
            g["guard"].check(g[name])
    with pytest.raises(PermissionError, match="pipeline source dependency"):
        g["guard"].check(g["pipeline"])
    escape = g["root"] / "escaping.parquet"; escape.symlink_to(g["old"])
    with pytest.raises(PermissionError, match="old/source dependency"):
        g["guard"].check(escape)
    assert len(g["guard"].old_source_open_attempts) == 3
    assert g["guard"].pipeline_imports


def test_installed_fresh_process_guard_blocks_python_arrow_and_pipeline_import(guard_files):
    g = guard_files
    # Audit hooks cannot be uninstalled; run the real installed guard separately
    # so no later pytest or cleanup fixture inherits monkeypatched OS functions.
    code = r'''
import importlib.util,json,os,pathlib,sys
import pyarrow.parquet as pq
paths=json.loads(sys.argv[1])
spec=importlib.util.spec_from_file_location("guard_module",paths["actual_script"])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
guard=m.ReaderGuard(paths["root"],[paths["mix"],paths["hm"]],paths["script"]).install()
assert pq.read_table(paths["owned"]).num_rows == 2
assert pathlib.Path(paths["owned"]).open("rb").read(4) == b"PAR1"
assert os.stat(pathlib.Path(paths["script"]).parent).st_mode
assert os.stat(pathlib.Path(paths["script"]).parents[1]).st_mode
failures=[]
for name,fn in [
 ("python",lambda:pathlib.Path(paths["old"]).read_bytes()),
 ("osopen",lambda:os.open(paths["old"],os.O_RDONLY)),
 ("arrowtable",lambda:pq.read_table(paths["hm_file"])),
 ("arrowfile",lambda:pq.ParquetFile(paths["old"])),
 ("pipeline",lambda:pathlib.Path(paths["pipeline"]).read_bytes()),
 ("pipelinestat",lambda:os.stat(paths["pipeline"])),
 ("import",lambda:__import__("egomotion_pipeline"))]:
 try:fn()
 except (PermissionError,ImportError):failures.append(name)
assert set(failures)=={"python","osopen","arrowtable","arrowfile","pipeline","pipelinestat","import"},failures
print(json.dumps({"passed":True,"blocked":failures}))
'''
    paths = {key: str(value) for key, value in g.items() if isinstance(value, Path)}
    paths["actual_script"] = str(SCRIPT)
    result = subprocess.run([sys.executable, "-c", code, json.dumps(paths)], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"]


def test_only_two_pipeline_directory_stats_are_allowed(guard_files):
    g = guard_files
    for directory in (g["script"].parent, g["script"].parents[1]):
        g["guard"].check(directory, 'stat')
    assert not g["guard"].pipeline_imports
    for directory in (g["script"].parent, g["script"].parents[1]):
        for operation in ('open', 'scandir'):
            with pytest.raises(PermissionError, match="pipeline source dependency"):
                g["guard"].check(directory, operation)
    with pytest.raises(PermissionError, match="pipeline source dependency"):
        g["guard"].check(g["pipeline"], 'stat')
    with pytest.raises(ImportError, match="pipeline import blocked"):
        g["guard"].hook('import', ('egomotion_pipeline', None))


def test_pipeline_import_search_paths_are_removed_before_guard(guard_files, monkeypatch):
    g = guard_files
    outside = str(g["mix"].parent)
    entries = [str(g["script"].parent), outside, str(g["script"].parents[1]), str(g["pipeline"].parent)]
    monkeypatch.setattr(sys, 'path', entries.copy())
    removed = probe.remove_pipeline_import_paths(g["script"])
    assert sys.path == [outside]
    assert removed == [entries[0], entries[2], entries[3]]


@pytest.mark.parametrize("field,value", [("episode_index", 4), ("seq_name", "wrong-source-key"), ("source", "egosuite"),
                                        ("idx", 90), ("start_idx", -1), ("end_idx", 100)])
def test_sample_episode_identity_and_bounds_fail_before_accessing_tensors(field, value):
    record = {"episode_index": 3, "source_key": "egoexo4d_hm:take", "source": "egoexo4d_hm", "length": 100}
    misc = {"episode_index": 3, "seq_name": record["source_key"], "source": record["source"], "idx": 9,
            "start_idx": 0, "end_idx": 60}
    misc[field] = value
    # Deliberately omit motion/image tensors: this proves the identity check
    # executes before potentially equal zero-valued arrays can hide misrouting.
    with pytest.raises(ValueError, match="requested episode/source/index bounds"):
        probe.sample_assertions({"misc": misc, "y": {}}, record, {}, [], requested_index=9)


def test_contact_mask_requires_source_ground_and_valid_adjacent_time_edges():
    feet = np.zeros((11, 2, 3)); feet[..., 2] = .05
    valid = np.ones((11, 2), bool); valid[3, 0] = False
    times = np.arange(11) / 5; times[7:] += 1
    contact, mask = probe.contacts_from_source(feet, valid, times, {"valid": True, "height_m": 0})
    assert not mask[3:5, 0].any() and not mask[7].any()
    assert np.array_equal(contact[mask], np.ones(mask.sum(), np.float32))
    assert not contact[~mask].any()
    contact, mask = probe.contacts_from_source(feet, valid, times, {"valid": False, "height_m": 0})
    assert not contact.any() and not mask.any()
