"""Durable official-initialization progress, advanced by exact native review UUID.

This is collection progress only. Kept raw physics still needs RTX and independent
dataset admission. No simulator, headset, model, or GPU is imported here.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import uuid

CONTRACT = 'cvpr-mobile-official99-collection-progress-v1'
OPERATOR_ABORT_REASONS = ('operator_discard', 'operator_pause', 'operator_reset')


def _validate_abort_reason(value):
    if value is not None and (type(value) is not str or value not in OPERATOR_ABORT_REASONS):
        raise ValueError('只有明确的操作员丢弃、暂停或重置可以标记为主动中断')


class CollectionProgress:
    def __init__(self, rows, path):
        self.rows = rows
        self.path = Path(path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('本轮初始化计划已由另一个采集器使用') from None
        identity = rows[0]['plan_sha256']
        self.state = {'contract': CONTRACT, 'plan_sha256': identity,
            'accepted': {}, 'pending': None, 'attempts': [], 'error': None, 'error_phase':None,'revision': 0}
        if self.path.exists():
            if self.path.is_symlink():
                raise ValueError('进度不能为符号链接')
            value = json.loads(self.path.read_text())
            if (value.get('contract') != CONTRACT or value.get('plan_sha256') != identity
                    or not isinstance(value.get('accepted'), dict)
                    or not isinstance(value.get('attempts'), list)):
                raise ValueError('进度不属于当前冻结初始化计划')
            known = {str(row['official_episode_index']) for row in rows}
            if not set(value['accepted']) <= known:
                raise ValueError('进度含计划外初始化')
            identities = [item['uuid'] for item in value['attempts']]
            for item in identities:
                uuid.UUID(item)
            if len(set(identities)) != len(identities):
                raise ValueError('重复的采集尝试UUID')
            attempts = {item['uuid']:item for item in value['attempts']}
            for item in attempts.values():
                _validate_abort_reason(item.get('operator_abort_reason'))
            if any(type(item.get('official_episode_index')) is not int
                    or str(item['official_episode_index']) not in known for item in attempts.values()):
                raise ValueError('采集尝试含计划外初始化')
            accepted = value['accepted']
            if (len(set(accepted.values())) != len(accepted)
                    or any(identity not in attempts or str(attempts[identity]['official_episode_index']) != key
                        or attempts[identity].get('operator_abort_reason') is not None
                        for key, identity in accepted.items())):
                raise ValueError('已完成初始化与采集UUID没有一一绑定')
            pending = value.get('pending')
            if pending and (pending != attempts.get(pending.get('uuid'))
                    or str(pending['official_episode_index']) in accepted):
                raise ValueError('待复核尝试与当前初始化不一致')
            self.state = value
        else:
            self._write()

    def close(self):
        self.lock.close()

    def __del__(self):
        if hasattr(self, 'lock'):
            self.lock.close()

    def _write(self):
        self.state['revision'] += 1
        data = json.dumps(self.state, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
        fd, temporary = tempfile.mkstemp(prefix='.progress-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @property
    def current(self):
        return next((row for row in self.rows
            if str(row['official_episode_index']) not in self.state['accepted']), None)

    @property
    def blocked(self):
        return self.state['pending'] is not None or self.state['error'] is not None or self.current is None

    def receipt(self):
        pending = self.state['pending']
        row = (next(item for item in self.rows if item['official_episode_index'] == pending['official_episode_index'])
            if pending else self.current)
        return {'contract': CONTRACT, 'plan_sha256': self.state['plan_sha256'],
            'progress_path': str(self.path), 'total': len(self.rows),
            'accepted_count': len(self.state['accepted']),
            'current_official_episode_index': row['official_episode_index'] if row else None,
            'current_position': self.rows.index(row)+1 if row else None,
            'status': ('error' if self.state['error'] else 'awaiting_review' if self.state['pending']
                else 'ready' if row else 'complete'),
            'pending_uuid': (self.state['pending'] or {}).get('uuid'),
            'error': self.state['error'], 'dataset_admission_complete': False}

    def submit(self, identity, directory, *, operator_abort_reason=None):
        if self.blocked:
            raise ValueError('初始化计划正在等待复核或已经结束')
        _validate_abort_reason(operator_abort_reason)
        uuid.UUID(identity)
        if any(item['uuid'] == identity for item in self.state['attempts']):
            raise ValueError('采集尝试UUID重复')
        attempt = {'uuid': identity, 'directory': str(Path(directory).absolute()),
            'official_episode_index': self.current['official_episode_index'],
            'operator_abort_reason': operator_abort_reason}
        self.state['attempts'].append(attempt)
        self.state['pending'] = copy.deepcopy(attempt)
        self._write()

    def fail(self, message, *, transient_review=False):
        if transient_review and self.state['error'] and self.state.get('error_phase') != 'review_transient':
            return
        phase = 'review_transient' if transient_review else 'technical'
        if self.state['error'] != str(message) or self.state.get('error_phase') != phase:
            self.state['error'] = str(message)
            self.state['error_phase'] = phase
            self._write()

    def observe(self, snapshot):
        """Return True only when a completed review requires a fresh scene reset."""
        candidates = [snapshot.get('latest_episode'), snapshot.get('latest_completed')]
        candidates.extend(reversed(snapshot.get('latest_results') or []))
        changed = False
        pending = self.state['pending']
        # A later manual discard revokes only its own accepted initialization.
        for key, identity in list(self.state['accepted'].items()):
            match = next((v for v in candidates if isinstance(v, dict) and v.get('uuid') == identity), {})
            if (match.get('pending') is False and match.get('disposition') == 'discarded'):
                del self.state['accepted'][key]
                changed = True
        if pending:
            match = next((v for v in candidates if isinstance(v, dict)
                and v.get('uuid') == pending['uuid']), None)
            if match and (match.get('status') == 'error' or match.get('error')):
                self.fail(match.get('error') or '后台复核技术异常；保留当前初始化及原件',transient_review=True)
            if match and match.get('pending') is False:
                if self.state['error'] and self.state.get('error_phase') != 'review_transient':
                    if changed:
                        self._write()
                    return changed
                selection = match.get('selection_status')
                intentional_discard = (pending.get('operator_abort_reason') in OPERATOR_ABORT_REASONS
                    and match.get('disposition') == 'discarded'
                    and match.get('status') in ('discarded', 'technical_blocked')
                    and not match.get('error') and match.get('collection_enabled') is True
                    and match.get('selection_policy') == 'native_only'
                    and match.get('acceptance_gate_version') == 'mobile-native-task-success-v2'
                    and selection in ('technical_evidence_incomplete', 'operator_discarded', 'official_task_not_successful'))
                if intentional_discard:
                    # Partial raw capture remains technical/incomplete in History.
                    # An explicit operator abort only permits retrying this same
                    # initialization after its exact UUID was archived; it never
                    # counts as task failure, native success or accepted data.
                    self.state['pending'] = None
                    self.state['error'] = None
                    self.state['error_phase'] = None
                    changed = True
                elif (match.get('status') == 'error' or match.get('error')
                        or selection == 'technical_evidence_incomplete'
                        or match.get('status') == 'technical_blocked'):
                    self.fail(match.get('error') or '原始证据技术复核未通过；保留当前初始化及原件')
                elif (match.get('selection_policy') == 'native_only'
                        and pending.get('operator_abort_reason') is None
                        and match.get('status') == 'ready' and match.get('available') is True
                        and match.get('collection_enabled') is True
                        and match.get('acceptance_gate_version') == 'mobile-native-task-success-v2'
                        and selection == 'selected' and match.get('disposition') == 'kept'
                        and match.get('official_task_success_observed') is True
                        and not match.get('discarded')):
                    self.state['accepted'][str(pending['official_episode_index'])] = pending['uuid']
                    self.state['pending'] = None
                    self.state['error'] = None
                    self.state['error_phase'] = None
                    changed = True
                elif (match.get('disposition') == 'discarded'
                        and selection in ('official_task_not_successful', 'operator_discarded')):
                    self.state['pending'] = None
                    self.state['error'] = None
                    self.state['error_phase'] = None
                    changed = True
                else:
                    self.fail('复核结果没有完整官方成功/丢弃归因，不能切换初始化')
        if changed:
            self._write()
        return changed
