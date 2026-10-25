"""Finish saved PICO episodes after the real-time collector has exited."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_handover_teacher import write_json


def finish(session):
    session=Path(session).resolve(strict=True)
    script=Path(__file__).resolve().parent
    environment=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    report={'session':str(session),'episodes':[],'eligible_for_packing':0,
        'accepted_new_dataset':False,'ckpt_feature_pack_ready':False,'technical_errors':0}
    for episode in sorted((session/'episodes').glob('*')):
        if not (episode/'episode.json').is_file():
            continue
        item={'episode':str(episode),'status':'skipped_unsaved_or_rejected'}
        report['episodes'].append(item)
        try:
            outcome=json.loads((episode/'episode.json').read_text())
            if outcome.get('status')=='pending_rtx':
                receipt=episode/'rtx_completion.json'
                if not receipt.exists():
                    with (episode/'rtx_render.log').open('x') as log:
                        child=subprocess.run([sys.executable,str(script/'render_deferred_pico.py'),str(episode)],
                            env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
                        item['renderer_returncode']=child.returncode
                    if child.returncode:
                        raise RuntimeError('Deferred renderer process failed with return code '+str(child.returncode))
                completion=json.loads(receipt.read_text()) if receipt.is_file() else {}
                if completion.get('status')!='deferred_rtx_completed_pending_independent_audit':
                    raise RuntimeError(completion.get('error','Renderer did not complete'))
            elif outcome.get('state',{}).get('status')!='pending_audit':
                continue
            # This audit may legitimately reject unsuccessful demonstrations.
            # Its rejection remains evidence, never an automatic success label.
            from audit_pico_teleop_episode import audit_episode
            audit=audit_episode(episode)
            item.update(status=audit['status'],eligible_for_packing=audit['eligible_for_packing'],
                errors=audit.get('errors',[]))
            report['eligible_for_packing']+=int(audit['eligible_for_packing'])
        except Exception as error:
            item.update(status='technical_error',error=f'{type(error).__name__}: {error}',eligible_for_packing=False)
            report['technical_errors']+=1
        finally:
            # Persist each result so an unrelated corrupt episode or a later
            # renderer failure cannot hide already processed demonstrations.
            write_json(session/'session_finalization.json',report)
    write_json(session/'session_finalization.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    report=finish(parser.parse_args().session)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(1 if report['technical_errors'] else 0)


if __name__=='__main__':
    main()
