"""Copy an accepted 14+100 package to a fresh A100 directory, preserving bytes.

This stage restores original nanosecond file identities after verifying SHA256.
It does not relocate DINO metadata or mark the remote package production-ready.
Previously uploaded old100 video/features may be copied into the fresh target
only after their bytes match the new local transfer inventory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12']
REMOTE_PYTHON = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126/bin/python'
OLD_RELEASE = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_quality100_20261023'
PACK_CONTRACT = 'cvpr-handover-original100-plus-new14-lerobot-pack-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Require a regular copied payload: {path}')
    before = path.stat()
    result = dict(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=sha(path))
    after = path.stat()
    require((after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns), f'Input changed: {path}')
    return result


def journal(paths, message):
    log = Path('/home/zxc/CVPR2027/log.md')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S +08:00')
    with log.open('a') as stream:
        stream.write(f'- {stamp} | ' + '、'.join(f'`{p}`' for p in paths) + f'：{message}。\n')


def inventory(root, remote):
    ready = read(root / 'packed/PACK_READY.json')
    require(ready.get('contract') == PACK_CONTRACT and ready.get('train_episodes') == 114
            and ready.get('total_frames') == 49089 and ready.get('train_windows') == 42363,
            'Require the complete existing100 plus actual new14 package')
    require((root / 'MERGED114_DATASET_READY.json').is_file(), 'Local production readiness is required before transfer')
    require(read(root / 'dino/manifest.json').get('state') == 'complete', 'Real DINO cache is incomplete')
    paths = []
    for name in ('packed', 'dino'):
        for path in sorted((root / name).rglob('*')):
            require(not path.is_symlink(), f'Transfer must not follow a payload alias: {path}')
            if path.is_file():
                paths.append(path)
    root_files = [p for p in sorted(root.iterdir()) if p.is_file()
                  and p.name not in ('A100_TRANSFER_MANIFEST.json', 'A100_BYTE_VERIFICATION.json')
                  and p.suffix not in ('.log', '.py')]
    paths.extend(root_files)
    files = {str(path.relative_to(root)): identity(path) for path in paths}
    for relative, expected in ready['payload_sha256'].items():
        require(files['packed/' + relative]['sha256'] == expected, f'Packed bytes differ: {relative}')
    return dict(contract='cvpr-handover-merged114-a100-byte-transfer-v1',
                local_root=str(root), remote_root=str(remote), files=files,
                created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                train_episodes=114, new14_first=True, original100_all_retained=True,
                data_directories=['packed', 'dino'], root_files=[p.name for p in root_files],
                total_bytes=sum(f['size_bytes'] for f in files.values()),
                uploader_sha256=sha(__file__), old_release=OLD_RELEASE,
                original_inputs_modified=False, remote_production_ready=False)


REMOTE_COMMON = r'''
import hashlib,json,os,pathlib,shutil,sys
value=json.load(sys.stdin);root=pathlib.Path(value['remote_root'])
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def owned(relative,exists=False):
 name=pathlib.Path(relative);require(not name.is_absolute() and '..' not in name.parts,'Escaping path')
 p=root/name;require(not p.is_symlink(),'No destination symlinks')
 require(p.resolve().is_relative_to(root.resolve()),'Resolved path escapes target')
 if exists:require(p.is_file(),'Missing copied file: '+relative)
 return p
'''

REMOTE_PREPARE = REMOTE_COMMON + r'''
require(root.is_absolute() and not root.is_symlink(),'Require independent absolute target')
marker=root/'A100_TRANSFER_MANIFEST.json'
if root.exists():
 require(marker.is_file() and json.loads(marker.read_text())==value,'Existing target is not this pending transfer')
else:
 root.mkdir(parents=True)
 marker.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
for name in value['data_directories']:(root/name).mkdir(exist_ok=True)
old=pathlib.Path(value['old_release']);reused=[]
for relative,expected in value['files'].items():
 name=pathlib.Path(relative);source=None
 if relative.startswith('packed/train/videos/') and name.suffix=='.mp4':
  index=int(name.stem.removeprefix('episode_'))
  if index>=14:source=old/'packed-quality100-v1'/name.relative_to('packed').parent/f'episode_{index-14:06d}.mp4'
 elif relative.startswith('dino/episodes/') and name.name in ('features.npy','frames.npz'):
  index=int(name.parent.name)
  if index>=14:source=old/'dino-quality100-v1/episodes'/f'{index-14:08d}'/name.name
 if source is None:continue
 destination=owned(relative)
 if destination.exists():continue
 require(source.is_file() and not source.is_symlink(),'Missing old100 copy candidate')
 require(source.stat().st_size==expected['size_bytes'] and sha(source)==expected['sha256'],'Old100 bytes differ from bound local source: '+relative)
 destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination);reused.append(relative)
print(json.dumps({'remote_target':str(root),'verified_reused_old100_files':len(reused),'reused_bytes':sum(value['files'][r]['size_bytes'] for r in reused)}),flush=True)
'''

REMOTE_VERIFY = REMOTE_COMMON + r'''
require(json.loads((root/'A100_TRANSFER_MANIFEST.json').read_text())==value,'Transfer inventory changed')
actual={str(p.relative_to(root)) for name in value['data_directories'] for p in (root/name).rglob('*') if p.is_file()}
expected={r for r in value['files'] if pathlib.Path(r).parts[0] in value['data_directories']}
require(actual==expected,'Missing/extra package/cache files')
restored=[]
for relative,wanted in value['files'].items():
 p=owned(relative,True);st=p.stat()
 require(st.st_size==wanted['size_bytes'] and sha(p)==wanted['sha256'],'Copied bytes differ: '+relative)
 if st.st_mtime_ns!=wanted['mtime_ns']:
  os.utime(p,ns=(st.st_atime_ns,wanted['mtime_ns']));restored.append(relative)
 require(p.stat().st_mtime_ns==wanted['mtime_ns'],'Nanosecond identity restoration failed')
report={'contract':'cvpr-handover-merged114-a100-byte-verification-v1','status':'all_copied_bytes_and_identities_verified',
 'verified_files':len(value['files']),'verified_bytes':value['total_bytes'],'restored_mtime_file_count':len(restored),
 'inventory_sha256':sha(root/'A100_TRANSFER_MANIFEST.json'),'all_original_copied_file_identities':value['files'],
 'remote_root':str(root),'DINO_relocation_pending':True,'production_ready':False,'GPU_used':False,'training_started':False}
target=root/'A100_BYTE_VERIFICATION.json';require(not target.exists(),'Preserve earlier verification')
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='all_original_copied_file_identities'}),flush=True)
'''


def remote_python(host, code, value):
    command = shlex.join([REMOTE_PYTHON, '-B', '-c', code])
    subprocess.run([*SSH, host, command], input=json.dumps(value, allow_nan=False), text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--remote-root', type=Path, required=True)
    parser.add_argument('--host', default='115.191.40.142')
    args = parser.parse_args()
    root = args.local_root.resolve(strict=True)
    require(args.remote_root.is_absolute(), 'Remote root must be absolute')
    path = root / 'A100_TRANSFER_MANIFEST.json'
    if path.exists():
        value = read(path)
        require(value['local_root'] == str(root) and value['remote_root'] == str(args.remote_root), 'Pending transfer target differs')
        require(all(identity(root/r) == v for r,v in value['files'].items()), 'Pending transfer input changed')
    else:
        value = inventory(root, args.remote_root)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        journal([str(path)], f'冻结114条数据集独立A100传输清单，共{len(value["files"])}个文件、{value["total_bytes"]}字节')
    print(json.dumps({'stage':'local_inventory_bound','files':len(value['files']),'bytes':value['total_bytes']}),flush=True)
    remote_python(args.host, REMOTE_PREPARE, value)
    journal([str(args.remote_root)], '建立114条独立远端目录，按实际SHA校验后只在新目录复制已有原100视频与特征，原目录不改')
    transport = shlex.join(SSH)
    common = ['rsync','-a','--checksum','--protect-args','--info=stats2','-e',transport]
    for name in value['data_directories']:
        print(json.dumps({'stage':'copying','directory':name}),flush=True)
        subprocess.run([*common,str(root/name)+'/',f'{args.host}:{args.remote_root/name}/'],check=True)
    for name in value['root_files']:
        subprocess.run([*common,str(root/name),f'{args.host}:{args.remote_root}/'],check=True,stdout=subprocess.DEVNULL)
    remote_python(args.host, REMOTE_VERIFY, value)
    target = root / 'A100_BYTE_VERIFICATION.json'
    subprocess.run([*common,f'{args.host}:{args.remote_root/target.name}',str(target)],check=True,stdout=subprocess.DEVNULL)
    require(read(target)['all_original_copied_file_identities'] == value['files'], 'Returned byte report differs')
    require(all(identity(root/r) == v for r,v in value['files'].items()), 'Read-only local input changed during transfer')
    journal([str(target),str(args.remote_root)], '114条完整数据集与DINO跨主机字节及纳秒文件身份校验通过，尚待远端路径重绑定和生产读取验收')
    print(json.dumps({'stage':'byte_transfer_complete','report':str(target),'report_sha256':sha(target)}),flush=True)


if __name__ == '__main__':
    main()
