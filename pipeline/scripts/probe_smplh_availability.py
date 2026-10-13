#!/usr/bin/env python3
"""Fresh-process owned-file availability proof, with no FK or data-page reads."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = "standalone-smplh-structural-availability-v1"
SCOPE = "structural_availability_only"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def secure_path(path):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('Availability path traversal is forbidden')
    path = path.absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Availability dependencies cannot contain symlinks')
    return path


def owned(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Availability dependencies must be owned relative paths')
    path = secure_path(root / relative)
    if not path.is_relative_to(root):
        raise ValueError('Availability dependency escapes the owned archive')
    return path


def identity(path):
    value = secure_path(path).stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Availability expects a regular file')
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns, device=value.st_dev, inode=value.st_ino)


def sha(path):
    value = hashlib.sha256()
    with secure_path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


class Guard:
    """Guard Python/Arrow owned-file entry points; this is not an OS sandbox."""
    def __init__(self, root, blocked):
        self.root = secure_path(root)
        self.blocked = [secure_path(path) for path in blocked]
        self.old_source_open_attempts, self.pipeline_imports = [], []
        self.pipeline_root = Path(__file__).absolute().parents[1]
        self.motionexpert_root = Path(__file__).absolute().parents[2] / 'MotionExpert'
        self.checking = False

    def check(self, value, kind='open'):
        if value is None or isinstance(value, int) or self.checking:
            return
        if isinstance(value, (list, tuple)):
            for item in value: self.check(item, kind)
            return
        try: path = Path(os.path.abspath(os.fsdecode(value)))
        except TypeError: return
        self.checking = True
        try: resolved = Path(os.path.realpath(path))
        finally: self.checking = False
        if resolved.is_relative_to(self.root):
            return
        if kind == 'stat' and self.root.is_relative_to(resolved):
            return  # Directory ancestors only; no open or enumeration exemption.
        if any(resolved.is_relative_to(prefix) for prefix in self.blocked):
            self.old_source_open_attempts.append(dict(operation=kind, path=str(resolved)))
            raise PermissionError('Availability old-source dependency blocked')
        if resolved.is_relative_to(self.pipeline_root) or resolved.is_relative_to(self.motionexpert_root):
            self.pipeline_imports.append(dict(operation=kind, path=str(resolved)))
            raise PermissionError('Availability pipeline source dependency blocked')

    def hook(self, event, args):
        if event == 'open': self.check(args[0])
        elif event == 'os.scandir': self.check(args[0], 'scandir')
        elif event == 'import' and (args[0] == 'pipeline' or args[0].startswith(('pipeline.', 'egomotion_pipeline', 'MotionExpert'))):
            self.pipeline_imports.append(args[0])
            raise ImportError('Availability cannot import a pipeline reader or MotionExpert')

    def install(self):
        forbidden = [name for name in sys.modules if name == 'pipeline' or name.startswith(('pipeline.', 'egomotion_pipeline', 'MotionExpert'))]
        for name, module in sys.modules.copy().items():
            filename = getattr(module, '__file__', None)
            if filename:
                code_path = Path(os.path.realpath(filename))
                if code_path.is_relative_to(self.motionexpert_root) or code_path.is_relative_to(self.pipeline_root / 'src'):
                    forbidden.append(name)
        if forbidden: raise ValueError('Numeric reader/model modules were already loaded')
        sys.addaudithook(self.hook)
        for name in ('stat', 'lstat', 'scandir', 'open'):
            original = getattr(os, name)
            kind = 'stat' if name in ('stat', 'lstat') else name
            def wrapper(path, *args, _original=original, _kind=kind, **kwargs):
                self.check(path, _kind); return _original(path, *args, **kwargs)
            setattr(os, name, wrapper)
        original = pq.ParquetFile
        def footer(path, *args, **kwargs):
            self.check(path, 'Arrow footer'); return original(path, *args, **kwargs)
        pq.ParquetFile = footer
        return self


def verify_protected(files, root):
    for name, value in files.items():
        path = secure_path(name)
        if not path.is_relative_to(root) or identity(path) != value['identity']:
            raise ValueError('Owned availability identity differs from the accepted snapshot')
        if not value['binary'] and sha(path) != value['sha256']:
            raise ValueError('Owned availability metadata/model checksum changed')


def check_schema(schema, specs):
    for name, spec in specs.items():
        if name not in schema.names:
            raise ValueError('Required availability column is missing: ' + name)
        width = spec['shape'][0]
        expected = pa.type_for_alias(spec['dtype'])
        if width != 1: expected = pa.list_(expected, width)
        if schema.field(name).type != expected:
            raise ValueError('Availability Arrow type/width changed: ' + name)


def run(envelope):
    root = secure_path(envelope['root'])
    # Remove the audit script's package directories before guarded module lookup.
    pipeline = Path(__file__).absolute().parents[1]
    motionexpert = pipeline.parent / 'MotionExpert'
    sys.path[:] = [entry for entry in sys.path if not any(Path(os.path.realpath(entry or os.getcwd())).is_relative_to(folder) for folder in (pipeline, motionexpert))]
    guard = Guard(root, envelope['blocked_roots']).install()
    report = dict(format=FORMAT, scope=SCOPE, passed=False, fresh_process=True,
        all_population_files_checked=False, numerical_checks_rerun=False, fk_or_model_executed=False,
        training_windows_revalidated=False, motionexpert_revalidated=False, data_pages_read=False,
        root=str(root), manifest_sha256=envelope['manifest_sha256'], population=envelope['population'],
        protected_snapshot_sha256=digest(envelope['protected_files']), blocked_roots=envelope['blocked_roots'],
        original_numeric_qualification_sha256=envelope['original_numeric_qualification_sha256'], rows=[], failures=[])
    report['model_sha256'] = envelope['model_sha256']
    try:
        verify_protected(envelope['protected_files'], root)
        records = [json.loads(path.read_text()) for path in sorted((root / '.pipeline/episodes').glob('*.json'))]
        fields = ('episode_index', 'source', 'source_key', 'length', 'split', 'data_path', 'vision_path')
        if [{key: r[key] for key in fields} for r in records] != envelope['records']:
            raise ValueError('Availability metadata does not cover the exact accepted population')
        info = json.loads((root / 'meta/info.json').read_text())
        if info['model']['model_sha256'] != envelope['model_sha256'] or sha(owned(root, info['model']['path'])) != envelope['model_sha256']:
            raise ValueError('Availability model asset binding differs')
        specs, optional = info['motion_features'], info.get('optional_features', {})
        core = {'smplh.root_orient': 3, 'smplh.pose_body': 63, 'smplh.trans': 3, 'smplh.betas': 16,
                'estimated.valid.body_position': 22, 'estimated.valid.body_rotation': 22}
        if any(specs.get(key, {}).get('shape') != [width] for key, width in core.items()):
            raise ValueError('Availability metadata lacks parameter16/body-mask schema')
        vision_specs = {'frame_index': {'dtype': 'int64', 'shape': [1]}, 'episode_index': {'dtype': 'int64', 'shape': [1]},
            'timestamp': {'dtype': 'float64', 'shape': [1]}, 'observation.valid.dino': {'dtype': 'bool', 'shape': [1]},
            'observation.features.dino': {'dtype': 'float32', 'shape': [5120]},
            'observation.source_timestamp': {'dtype': 'float64', 'shape': [1]}, 'source.video_frame_index': {'dtype': 'int64', 'shape': [1]},
            'observation.video_timestamp': {'dtype': 'float64', 'shape': [1]}, 'observation.video_time_error': {'dtype': 'float64', 'shape': [1]}}
        for record in records:
            motion, vision = owned(root, record['data_path']), owned(root, record['vision_path'])
            receipt_path = motion.parent / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            if receipt['record'] != record:
                raise ValueError('Availability receipt differs from accepted metadata')
            counts = {}
            for role, path in (('motion', motion), ('vision', vision)):
                protected = envelope['protected_files'][str(path)]
                if (identity(path) != receipt['output_identities'][role] or record[role + '_sha256'] != receipt[role]['sha256']
                        or protected['sha256'] != receipt[role]['sha256']):
                    raise ValueError('Availability output differs from its original receipt')
                parquet = pq.ParquetFile(path)
                try:
                    counts[role] = parquet.metadata.num_rows
                    check_schema(parquet.schema_arrow, specs if role == 'motion' else vision_specs)
                    if role == 'motion':
                        check_schema(parquet.schema_arrow, {k: v for k, v in optional.items() if k in parquet.schema_arrow.names})
                        if ('smplh.pose_hand_pca' in parquet.schema_arrow.names) != ('smplh.valid.hand_pose' in parquet.schema_arrow.names):
                            raise ValueError('Optional hand PCA and its validity mask must be paired')
                    if counts[role] != record['length']:
                        raise ValueError('Availability footer frame population changed')
                finally: parquet.close()
            report['rows'].append({**{key: record[key] for key in fields[:5]}, 'motion_footer_rows': counts['motion'],
                'vision_footer_rows': counts['vision'], 'passed': True})
        verify_protected(envelope['protected_files'], root)
        report.update(all_population_files_checked=True, episodes=len(records), frames=sum(r['length'] for r in records), passed=True)
    except Exception as error:
        report['failures'].append(repr(error))
    report.update(old_source_open_attempts=guard.old_source_open_attempts, pipeline_imports=guard.pipeline_imports)
    report['passed'] &= not report['old_source_open_attempts'] and not report['pipeline_imports']
    return report


if __name__ == '__main__':
    envelope = json.load(sys.stdin)
    result = run(envelope)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)
