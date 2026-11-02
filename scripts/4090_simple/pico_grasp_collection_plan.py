"""Freeze all 100 official Grasp TRAIN initializations without robot conversion.

This plan preserves native AMO config bytes. The separately identified live
SONIC adapter owns any coordinate/robot conversion; this file never reads
demonstration state/action arrays or creates a simulator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import uuid

from pico_mobile_collection_plan import (canonical, digest, json_bytes, owned,
    parse_json, pose, regular, require, semantic_sha)

CONTRACT = 'cvpr-pico-grasp-official-train100-collection-plan-v1'
TASK_ID = 'cvpr/G1SonicTabletopGraspPico-v0'
SOURCE_TASK_ID = 'simple/G1WholebodyTabletopGraspMP-v0'
SOURCE_TASK_UID = 'g1_wholebody_tabletop_grasp_mp'
SOURCE_SHA256 = '183474d4439d4d5de2b261f3c6ce0b28328f000af0e1a4f5ae0b9305dfb2cfc7'
EPISODE_INDICES = list(range(100))
REPO = Path(__file__).resolve().parents[2]


def source_rows(payload):
    require(digest(payload) == SOURCE_SHA256, 'Require the fixed official Grasp TRAIN metadata SHA')
    records = [parse_json(line) for line in payload.decode().splitlines() if line.strip()]
    require(len(records) == 100 and all(type(row.get('episode_index')) is int for row in records)
        and sorted(row['episode_index'] for row in records) == EPISODE_INDICES,
        'Require exactly100 unique official Grasp TRAIN episodes0..99')
    result, identities = [], set()
    for number, record in enumerate(records):
        text = record.get('environment_config')
        require(isinstance(text, str) and text, 'Missing original environment_config bytes')
        config = parse_json(text)
        require(config.get('uid') == SOURCE_TASK_UID and config.get('metadata', {}).get('split') == 'train'
            and config.get('robot_cfg', {}).get('uid') == 'g1_wholebody', 'Wrong native Grasp TRAIN task/robot')
        dr, actors = config['dr_state_dict'], config['layout']['actors']
        roles = ('target', *('distractor_'+uid for uid in dr['distractors']))
        require(set(actors) == {'table', *roles} and len(dr['distractors']) == 3,
            'Grasp requires one table, target and three distractors; no container')
        spatial = dr['spatial']
        pose(spatial['g1_wholebody'], 'native robot reset')
        keys = {'g1_wholebody'}
        for role in roles:
            actor = actors[role]
            asset = actor['asset']
            uid = asset['uid']
            selected = dr['target'] if role == 'target' else dr['distractors'][uid]
            require(uid == actor['uid'] == selected['uid'] and asset['res_id'] == selected['res_id'],
                'Saved DR asset identity does not match '+role)
            require(uid not in keys, 'Ambiguous Grasp spatial identity')
            keys.add(uid)
            pose(actor['pose'], role)
            pose(spatial[uid], 'DR/'+uid)
            require(canonical(actor['pose']) == canonical(spatial[uid]),
                'Saved Grasp layout is not its initial DR pose: '+role)
        require(set(spatial) == keys, 'Unexpected Grasp spatial identities')
        require(dr['scene']['name'] == '107733960_175999701'
            and dr['scene']['uid'] == 'hssd:scene2', 'Unexpected official Grasp scene')
        identity = semantic_sha(dr)
        require(identity not in identities, 'Repeated official Grasp initialization')
        identities.add(identity)
        index, original = record['episode_index'], text.encode()
        relative = f'original/episode_{index:03d}.json'
        result.append({'row': {'official_episode_index': index, 'source_row_number': number,
            'original_config_path': relative, 'original_config_sha256': digest(original),
            'reset_config_path': relative, 'reset_config_sha256': digest(original),
            'dr_state_sha256': identity, 'spatial_state_sha256': semantic_sha(spatial)}, 'original': original})
    return sorted(result, key=lambda item: item['row']['official_episode_index'])


def build_plan(source_jsonl, output_dir):
    source = regular(source_jsonl)
    payload = source.read_bytes()
    rows = source_rows(payload)
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink()
        and not output.resolve().is_relative_to(REPO), 'Use a fresh plan directory outside Git')
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not any(part.is_symlink() for part in (output.parent, *output.parent.parents)), 'Plan parent traverses symlink')
    staging = output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.partial')
    staging.mkdir(mode=0o700)
    (staging/'original').mkdir()
    (staging/'source_episodes.jsonl').write_bytes(payload)
    for row in rows:
        (staging/row['row']['original_config_path']).write_bytes(row['original'])
    manifest = {'contract': CONTRACT, 'task_id': TASK_ID, 'source_official_task_id': SOURCE_TASK_ID,
        'source_sha256': SOURCE_SHA256, 'source_path': str(source), 'episode_count': 100,
        'official_episode_indices': EPISODE_INDICES, 'unique_dr_states': 100,
        'native_config_bytes_preserved': True, 'control_adapter_applied': False,
        'teacher_actions_or_states_read': False, 'episodes': [item['row'] for item in rows]}
    (staging/'manifest.json').write_bytes(json_bytes(manifest))
    require(source.read_bytes() == payload, 'Official metadata changed during preparation')
    require(not output.exists() and not output.is_symlink(), 'Plan output appeared during preparation')
    staging.rename(output)
    return output/'manifest.json'


def load_plan(path):
    path = regular(path)
    payload = path.read_bytes()
    actual = parse_json(payload.decode())
    frozen = owned(path.parent, 'source_episodes.jsonl').read_bytes()
    expected_rows = source_rows(frozen)
    expected = {'contract': CONTRACT, 'task_id': TASK_ID, 'source_official_task_id': SOURCE_TASK_ID,
        'source_sha256': SOURCE_SHA256, 'source_path': actual.get('source_path'), 'episode_count': 100,
        'official_episode_indices': EPISODE_INDICES, 'unique_dr_states': 100,
        'native_config_bytes_preserved': True, 'control_adapter_applied': False,
        'teacher_actions_or_states_read': False, 'episodes': [item['row'] for item in expected_rows]}
    require(isinstance(expected['source_path'], str) and Path(expected['source_path']).is_absolute()
        and canonical(actual) == canonical(expected), 'Frozen Grasp plan differs from original TRAIN metadata')
    result = []
    for item in expected_rows:
        row = deepcopy(item['row'])
        config = owned(path.parent, row['original_config_path'])
        require(config.read_bytes() == item['original'], 'Frozen native Grasp config changed')
        row.update(original_config_path=str(config), reset_config_path=str(config), task_id=TASK_ID,
            plan_sha256=digest(payload), manifest_path=str(path))
        result.append(row)
    require(path.read_bytes() == payload, 'Plan changed during verification')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    build = modes.add_parser('build')
    build.add_argument('--source-jsonl', type=Path, required=True)
    build.add_argument('--output-dir', type=Path, required=True)
    check = modes.add_parser('check')
    check.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    path = build_plan(args.source_jsonl, args.output_dir) if args.mode == 'build' else args.plan
    rows = load_plan(path)
    print(json.dumps({'manifest': str(path), 'plan_sha256': rows[0]['plan_sha256'],
        'official_episodes': len(rows), 'source_sha256': SOURCE_SHA256, 'simulator_started': False}))


if __name__ == '__main__':
    main()
