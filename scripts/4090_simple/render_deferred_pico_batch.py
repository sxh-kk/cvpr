"""Sequential frozen Grasp RTX acquisition with one environment / Isaac app.

The original single-episode renderer executes unchanged, including every reset,
scene binding, zero-step state restore and fresh RGB / full-state check. Only its
renderer factory is leased; per-episode close is deferred until this batch ends.
This is not independent admission and never resumes partial or rejected output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import render_deferred_pico as renderer


CONTRACT = 'cvpr-pico-grasp-deferred-rtx-batch-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class CompatibleGraspResources:
    """Reject cache aliasing; permit only poses/materials updated by the backend."""
    def __init__(self):
        self.fixed = None
        self.scenes = {}
        self.assets = {}

    def add(self, environment):
        # Camera resolutions/mounts are created once, unlike poses/intrinsics.
        # Requiring the complete sensor config is deliberately stronger here.
        fixed = {key: environment[key] for key in ('uid', 'metadata', 'robot_cfg', 'sensor_cfgs')}
        fixed['light_topology'] = [{key: light[key] for key in ('uid', 'type')}
            for light in environment['layout']['lights']]
        if self.fixed is None:
            self.fixed = canonical(fixed)
        require(self.fixed == canonical(fixed), 'Batch robot/camera/light topology differs')
        scene = environment['dr_state_dict']['scene']
        scene_identity = {key: scene[key] for key in ('uid', 'conf', 'name', 'data_dir')}
        previous = self.scenes.setdefault(scene['uid'], canonical(scene_identity))
        require(previous == canonical(scene_identity), 'Same scene UID aliases different geometry/config')
        for actor in environment['layout']['actors'].values():
            if 'asset' not in actor:
                continue
            asset = actor['asset']
            previous = self.assets.setdefault(asset['label'], canonical(asset))
            require(previous == canonical(asset), 'Same object label aliases different asset: ' + asset['label'])


class EnvironmentLease:
    def __init__(self, owner):
        self.owner = owner

    @property
    def unwrapped(self):
        return self.owner.environment.unwrapped

    def reset(self, *args, **kwargs):
        return self.owner.environment.reset(*args, **kwargs)

    def close(self):
        # The original renderer always closes its lease in finally. Even on an
        # exception the owning batch closes the real environment exactly once.
        pass


class RendererPool:
    def __init__(self, factory):
        self.factory = factory
        self.environment = None
        self.frame_binding = None
        self.task_id = self.sonic = None
        self.leases = 0
        self.compatibility = CompatibleGraspResources()

    def acquire(self, task_id, sonic_config, environment):
        from pico_deferred_capture import GRASP_TASK_ID
        require(task_id == GRASP_TASK_ID, 'Batch reuse only supports the private Grasp adapter')
        self.compatibility.add(environment)
        sonic = canonical(sonic_config)
        if self.environment is None:
            self.environment, self.frame_binding = self.factory(task_id, sonic_config, environment)
            self.task_id, self.sonic = task_id, sonic
        else:
            require((self.task_id, self.sonic) == (task_id, sonic), 'Batch task/SONIC configuration changed')
        self.leases += 1
        return EnvironmentLease(self), self.frame_binding

    def close(self):
        if self.environment is not None:
            environment, self.environment = self.environment, None
            environment.close()


def frozen_inputs(path):
    manifest = load_json(path)
    require(manifest.get('contract') == 'cvpr-grasp100-rtx-admission-frozen-inputs-v1', 'Unknown frozen input contract')
    rows = manifest['episodes']
    require(len(rows) == manifest['count'] == 100, 'Frozen batch must contain exactly 100 episodes')
    require(len({row['uuid'] for row in rows}) == 100, 'Repeated frozen episode UUID')
    require({row['official_episode_index'] for row in rows} == set(range(100)), 'Frozen official scene coverage differs')
    require(len({str(Path(row['directory']).resolve()) for row in rows}) == 100, 'Repeated frozen episode directory')
    compatibility = CompatibleGraspResources()
    for row in rows:
        episode = Path(row['directory']).resolve(strict=True)
        for name, frozen in row['source_files'].items():
            source = (episode / name).resolve(strict=True)
            require(source.is_relative_to(episode), 'Frozen source escapes episode directory')
            stat = source.stat()
            require(stat.st_size == frozen['size_bytes'] and stat.st_mtime_ns == frozen['mtime_ns'],
                    'Frozen file stat changed: ' + str(source))
            if 'sha256' in frozen:
                require(renderer.file_sha256(source) == frozen['sha256'], 'Frozen source hash changed: ' + str(source))
        metadata = load_json(episode / 'deferred_capture/capture.json')['metadata']
        require(metadata['human_demo_uuid'] == row['uuid'], 'Frozen episode UUID changed')
        renderer.recorded_task(metadata)
        compatibility.add(metadata['environment_config'])
    return manifest


def completed_output(episode):
    """Reuse only exact finished bytes; this does not replace independent audit."""
    episode = Path(episode)
    paths = [episode/'capture', episode/'rtx_completion.json', episode/'rtx_frame_bindings.jsonl',
             episode/'deferred_render_environment.json']
    if not any(path.exists() for path in paths):
        return None
    require(all(path.exists() for path in paths), 'Partial RTX output exists; preserve it and retry in a fresh episode copy')
    receipt = load_json(paths[1])
    require(receipt.get('status') == 'deferred_rtx_completed_pending_independent_audit',
            'Rejected/incomplete RTX output exists; never overwrite it')
    require(receipt.get('archived_physics_steps') == 0 and receipt.get('sonic_inferences') == 0
            and receipt.get('full_integration_state_unchanged') is True
            and receipt.get('state_event_numeric_checks_passed') is True
            and receipt.get('environment_binding', {}).get('matched') is True, 'Incomplete renderer proof')
    for kind, filename in [('manifest', 'capture.json'), ('events', 'events.jsonl'),
                           ('model', 'model.mjb'), ('states', 'states.npy')]:
        require(receipt.get('source_' + kind + '_sha256') == renderer.file_sha256(episode/'deferred_capture'/filename),
                'Completed RTX source hash changed: ' + kind)
    require(receipt.get('render_code_sha256') == renderer.file_sha256(renderer.__file__), 'Completed renderer source changed')
    require(receipt.get('capture_manifest_sha256') == renderer.file_sha256(episode/'capture/capture.json'), 'RTX manifest changed')
    require(receipt.get('capture') == load_json(episode/'capture/capture.json'), 'RTX capture receipt differs')
    require(receipt.get('frame_bindings_path') == 'rtx_frame_bindings.jsonl'
            and receipt.get('frame_bindings_sha256') == renderer.file_sha256(paths[2]), 'RTX frame bridge changed')
    events = [json.loads(line) for line in (episode/'capture/events.jsonl').read_text().splitlines()]
    bridge = [json.loads(line) for line in paths[2].read_text().splitlines()]
    require(len(events) == len(bridge) == receipt['completed_frames'] == receipt['frame_bindings_count'] > 0,
            'Completed RTX frame counts differ')
    for event, binding in zip(events, bridge):
        require(binding['rtx_event_sha256'] == renderer.canonical_event_sha256(event), 'Completed RTX event changed')
        png = (episode/'capture'/event['rgb']['path']).resolve(strict=True)
        require(png.is_relative_to((episode/'capture').resolve()), 'RTX image escapes capture directory')
        require(renderer.file_sha256(png) == event['rgb']['file_sha256'] == binding['rgb_file_sha256'], 'RTX PNG changed')
    return receipt


def run(args):
    manifest_path = args.manifest.resolve(strict=True)
    manifest = frozen_inputs(manifest_path)  # Validate all 100 even for a pilot.
    require(args.limit is None or 1 <= args.limit <= 100, 'Limit must be between 1 and 100')
    rows = manifest['episodes'][:args.limit]
    report = {'contract': CONTRACT, 'status': 'running', 'accepted_new_dataset': False,
        'manifest_path': str(manifest_path), 'manifest_sha256': renderer.file_sha256(manifest_path),
        'batch_code_sha256': renderer.file_sha256(__file__),
        'original_renderer_code_sha256': renderer.file_sha256(renderer.__file__),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_count': len(rows),
        'reuse_scope': 'one private Grasp environment; full original reset and binding per episode', 'episodes': []}
    # A report is a new run identity, never a reusable/overwritten failed run.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
    pool = RendererPool(renderer.make_recorded_renderer)
    try:
        with patch.object(renderer, 'make_recorded_renderer', pool.acquire):
            for row in rows:
                episode = Path(row['directory']).resolve(strict=True)
                receipt = completed_output(episode)
                reused = receipt is not None
                if receipt is None:
                    receipt = renderer.render(SimpleNamespace(episode=episode, data_root=args.data_root))
                    receipt['batch_execution'] = {'contract': CONTRACT,
                        'batch_code_sha256': report['batch_code_sha256'],
                        'original_renderer_code_sha256': report['original_renderer_code_sha256'],
                        'frozen_manifest_sha256': report['manifest_sha256'],
                        'environment_lease_number': pool.leases,
                        'real_environment_count': 1, 'original_reset_and_environment_binding_per_episode': True,
                        'factory_override': 'single environment lease; defer close until batch finally',
                        'batch_report_path': str(args.report.resolve())}
                    renderer.write_json(episode/'rtx_completion.json', receipt)
                report['episodes'].append({'uuid': row['uuid'], 'number': row['number'],
                    'directory': str(episode), 'reused_completed_output': reused,
                    'completed_frames': receipt['completed_frames'],
                    'environment_binding': receipt['environment_binding'],
                    'rtx_completion_sha256': renderer.file_sha256(episode/'rtx_completion.json')})
                renderer.write_json(args.report, report)
                print(json.dumps(report['episodes'][-1], ensure_ascii=False), flush=True)
        report['status'] = 'render_completed_pending_process_exit'
        report['render_finished_utc'] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report.update(status='rejected', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # SimulationApp.close may terminate the process without returning to
        # Python. Persist complete render evidence before calling it; only the
        # outer process supervisor may attest its exit code / final report SHA.
        report['new_environment_leases'] = pool.leases
        report['cleanup_requested'] = True
        report['cleanup_return_verified'] = False
        renderer.write_json(args.report, report)
        try:
            pool.close()
        except BaseException as error:
            report.update(status='rejected', close_error=repr(error), close_traceback=traceback.format_exc())
            raise
        finally:
            report['cleanup_return_verified'] = 'close_error' not in report
            if report['status'] == 'render_completed_pending_process_exit':
                report['status'] = 'completed_pending_independent_audit'
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            renderer.write_json(args.report, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Render only first N rows after validating all 100 frozen inputs')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
