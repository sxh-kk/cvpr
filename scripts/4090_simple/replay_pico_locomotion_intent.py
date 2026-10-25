"""Offline same-packet PICO pelvis -> planner command diagnostic.

Reads an immutable archived commands.jsonl or raw packet JSON. No SDK, model,
simulator, current collector endpoint, or training dataset is opened. Replayed
arrival times are synthesized from device time; this cannot revalidate the
original network freshness or demonstrate physical human walking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pico_live_input import parse_tracking
from pico_locomotion_intent import PicoRootPlannerIntent


def packet(row):
    if not isinstance(row, dict):
        raise ValueError('Packet record must be an object')
    for key in ('source_raw_packet', 'raw_packet', 'data'):
        if key in row:
            return row[key]
    if row.get('functionName') == 'Tracking' or 'timeStampNs' in row:
        return row
    return None  # Explicit ledger headers are not tracking samples.


def replay(source, output, *, robot_yaw_rad=0.):
    import math
    if not math.isfinite(robot_yaw_rad):
        raise ValueError('Finite diagnostic robot heading required')
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = source.read_bytes()
    text = source_bytes.decode('utf-8')
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    root = [0., 0., .78, math.cos(robot_yaw_rad/2), 0., 0., math.sin(robot_yaw_rad/2)]
    estimator = PicoRootPlannerIntent()
    initial_ns = previous_ns = None
    count = moving = invalid = 0
    calibration = None
    with (output/'intents.jsonl').open('x') as stream:
        for row in rows:
            value = packet(row)
            if value is None:
                continue
            frame = parse_tracking(value, 0., count+1)
            timestamp = frame.device_timestamp_ns
            if timestamp is None or timestamp <= 0 or (previous_ns is not None and timestamp <= previous_ns):
                raise ValueError('Archived device samples must strictly increase; no duplicate padding')
            if initial_ns is None:
                initial_ns = timestamp
            elapsed = (timestamp-initial_ns)/1e9
            frame = parse_tracking(value, elapsed, count+1)
            if count == 0:
                calibration = estimator.calibrate(frame, root, now_s=elapsed)
                previous_ns = timestamp
                count += 1
                continue
            intent = estimator.update(frame, root, now_s=elapsed)
            result = {'elapsed_device_s': elapsed, 'source_sequence': row.get('source_sequence', frame.sequence),
                      'diagnostic_source_sequence': frame.sequence,
                      'device_timestamp_ns': timestamp, 'planner': intent.planner_fields(),
                      'intent': intent.as_dict(), 'diagnostic_only': True, 'training_eligible': False}
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            count += 1
            moving += int(intent.mode == 1)
            invalid += int(not intent.valid)
            previous_ns = timestamp
    if count < 2:
        raise ValueError('At least two actual archived packet samples required')
    receipt = {'contract': 'cvpr-pico-root-planner-offline-replay-v1',
        'source': str(source), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'input_packets': count, 'intent_rows': count-1, 'slow_walk_rows': moving, 'invalid_rows': invalid,
        'calibration': calibration, 'diagnostic_robot_root_pose7': root,
        'SDK_started': False, 'models_loaded': False, 'physics_steps': 0,
        'host_arrival_times': 'synthesized_from_device_time_only_for_offline_mapping',
        'original_network_freshness_revalidated': False, 'physical_human_following_validated': False,
        'diagnostic_only': True, 'accepted_new_dataset': False, 'training_eligible': False,
        'intents_sha256': hashlib.sha256((output/'intents.jsonl').read_bytes()).hexdigest()}
    (output/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--robot-yaw-rad', type=float, default=0.)
    args = parser.parse_args()
    print(json.dumps(replay(args.source, args.output, robot_yaw_rad=args.robot_yaw_rad), ensure_ascii=False))
