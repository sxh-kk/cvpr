"""Observe one isolated pick RTX chunk exit; never infer successful cleanup from exit alone."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
from process_pico_dataset import require_idle, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('python','manifest','data-root','report','log'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--profile',required=True,choices=('xmove_pick','bend_pick','xmove_bend_pick'))
    p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,required=True)
    a=p.parse_args();require_idle();a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.report.exists() or a.log.exists():raise ValueError('Preserve previous chunk report/log; use new names')
    script=Path(__file__).resolve();digest=sha(script);snapshot=a.report.parent/'driver_sources'/digest/script.name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists() and sha(snapshot)!=digest:raise ValueError('Driver snapshot changed')
    if not snapshot.exists():snapshot.write_bytes(script.read_bytes())
    command=[str(a.python),str(script.with_name('render_deferred_pick_batch.py')),
        '--manifest',str(a.manifest),'--profile',a.profile,'--offset',str(a.offset),
        '--limit',str(a.limit),'--data-root',str(a.data_root),'--report',str(a.report)]
    environment=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR='/dev/shm')
    with a.log.open('x') as output:
        child=subprocess.Popen(command,cwd=script.parents[2],env=environment,stdin=subprocess.DEVNULL,
            stdout=output,stderr=subprocess.STDOUT);code=child.wait()
    if not a.report.is_file():raise ValueError('Renderer exited without a batch report; original log retained')
    receipt=a.report.with_name(a.report.stem+'_EXIT.json')
    if receipt.exists():raise ValueError('Preserve prior observed process exit')
    write(receipt,{'contract':'cvpr-pick-rtx-observed-process-exit-v1','exit_code':code,'child_pid':child.pid,
        'finished_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'report_path':str(a.report),
        'report_sha256':sha(a.report),'argv':command,'parentdriver_path':str(snapshot),
        'parentdriver_sha256':digest})
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S +08:00')
    with Path('/home/zxc/CVPR2027/log.md').open('a') as log:
        log.write(f'- {stamp} | `{os.path.relpath(a.report,"/home/zxc/CVPR2027")}`、`{os.path.relpath(receipt,"/home/zxc/CVPR2027")}`、`{os.path.relpath(a.log,"/home/zxc/CVPR2027")}`及该批片段派生补图目录：完成{a.profile}第{a.offset}起{a.limit}条分块RTX进程，记录退出码{code}及实际驱动源码快照；原始物理状态未积分、原件未改写。\n')
    if code:raise SystemExit(code)


if __name__=='__main__':main()
