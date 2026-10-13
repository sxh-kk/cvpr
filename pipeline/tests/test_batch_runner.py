"""The long batch must isolate rejected inputs and keep one valid archive."""

import json
from pathlib import Path
import subprocess
import sys

from egomotion_pipeline.storage import Archive
from egomotion_pipeline.validation import validate
from mcap_fixtures import make_mcap


def test_batch_continues_after_bad_input_and_resumes_committed_episodes(tmp_path):
    source = tmp_path/"source"
    paths = [make_mcap(source/"first.mcap"),
             make_mcap(source/"bad.mcap", omit_camera=True),
             make_mcap(source/"upper.mcap", body_joints=14, upper_only=True),
             make_mcap(source/"last.mcap")]
    config = {"fps": 30, "sources": [{"type": "egosuite_mcap", "name": "egostandard",
                "root": str(source), "coordinate_verified": True, "require_camera": True}]}
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps([{"path": str(p), "uid": p.stem} for p in paths]))
    job = tmp_path/"job"
    job.mkdir()
    output = tmp_path/"archive"
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/"scripts/process_egostandard.py"),
               "--config", str(config_path), "--manifest", str(manifest),
               "--output", str(output), "--job-dir", str(job), "--workers", "2"]
    for converted, resumed in [(2, 0), (0, 2)]:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        assert result.returncode == 1, result.stdout+result.stderr
        status = json.loads((job/"status.json").read_text())
        assert status["stage"] == "completed_with_errors" and status["validation_passed"]
        assert status["processed_files"] == 4
        assert status["converted_files"] == converted and status["resumed_files"] == resumed
        assert status["failed_files"] == status["skipped_files"] == 1
        assert status["committed_episodes"] == 2
        assert validate(Archive(output))["frames"] == 10
    errors = [json.loads(line) for line in (job/"failures.jsonl").read_text().splitlines()]
    assert all(error["uid"] == "bad" for error in errors)
    assert json.loads((job/"validation.json").read_text())["failed_files"] == 1
