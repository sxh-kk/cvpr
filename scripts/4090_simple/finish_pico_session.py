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
            excluded = False
            for name in ('operator_discard.json','collection_record.json'):
                marker = episode/name
                if marker.is_symlink():
                    raise ValueError('Discard/collection record must not be a symbolic link')
                if not marker.exists():
                    continue
                state=json.loads(marker.read_text())
                if not isinstance(state,dict):
                    raise ValueError('Invalid discard/collection record')
                if name == 'operator_discard.json' or state.get('disposition') == 'discarded':
                    item.update(status='skipped_discarded',eligible_for_packing=False,
                        reasons=state.get('reasons',[state.get('reason','operator discard')]))
                    excluded=True
                    break
                if state.get('disposition') != 'kept':
                    raise ValueError('Invalid collection disposition')
            if excluded:
                continue
            from pico_episode_review import discard_requested
            if discard_requested(episode):
                item.update(status='skipped_discarded',eligible_for_packing=False,
                    reasons=['操作员已申请丢弃，后台正在完成分流'])
                continue
            outcome=json.loads((episode/'episode.json').read_text())
            if outcome.get('geometry_pending') is True:
                submission=episode/'review_queue_submission.json'
                if not submission.is_file() or submission.is_symlink():
                    raise ValueError('Background review submission marker is missing or invalid')
                from pico_episode_review import discard_requested
                if discard_requested(episode):
                    item.update(status='skipped_discarded',eligible_for_packing=False,
                        reasons=['操作员已申请丢弃，后台正在完成分流'])
                    continue
                marker=episode/'collection_record.json'
                collection=json.loads(marker.read_text()) if marker.is_file() else {}
                if collection.get('status')!='committed' or collection.get('disposition')!='kept':
                    item.update(status='pending_background_review',eligible_for_packing=False,
                        reasons=['后台物理复核尚未完成，暂不启动RTX或数据准入'])
                    continue
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
    session=parser.parse_args().session
    from pico_episode_review import wait_session
    wait_session(session)  # The CLI is invoked only after live collection exits.
    report=finish(session)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(1 if report['technical_errors'] else 0)


if __name__=='__main__':
    main()
