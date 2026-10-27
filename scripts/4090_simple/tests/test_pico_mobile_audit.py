"""Synthetic CPU evidence tests; fixtures are never actual human admissions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_pico_teleop_episode as audit
import pico_mobile_audit as mobile
import pico_mobile_vr_reference as vr
from handover_sonic_teleop_reference import build_teleop_reference
from test_audit_pico_teleop_episode import make_episode, write, write_rows


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def mobile_fixture(temporary):
    root = Path(temporary)/'collection/kept/001'
    root.mkdir(parents=True)
    identities, decoder_sha = make_episode(root)
    metadata, outcome = read(root/'metadata.json'), read(root/'episode.json')
    metadata.pop('initial_target_z_m')
    helper = vr.MobileVRReference()
    metadata.update(task_id=mobile.MOBILE_TASK_ID,
        actual_task_source_sha256=mobile.TASK_SOURCE_SHA256, actual_native_success_criteria=.9,
        mobile_control_protocol=mobile.CONTROL_CONTRACT, mobile_control_policy='continuous_vr_mode1',
        mobile_encoder_mode=1, mobile_planner_model_sha256=mobile.PLANNER_MODEL_SHA256,
        mobile_vr_source=helper.identity, quality_geometry_mode='deferred_geometry_with_original_solved_support',
        hand_control_profile={'profile_id':'sonic_v1_1_middle_close_local_max_trigger_grip_v1',
                              'torque_collision_physics_modified':False})
    source = Path(temporary)/'historical/pico_simple_teleop.py'
    source.parent.mkdir(); source.write_text('# Synthetic original collector bytes\n')
    metadata['source_code_sha256_at_session_start'] = {source.name:audit.sha(source)}
    historical = Path(temporary)/'historical.json'
    write(historical, {'status':'complete','sources':[{'filename':source.name,
        'sha256':audit.sha(source),'snapshot_path':str(source),'git_blob':'fixture-only'}]})
    events, ledger = rows(root/'capture/events.jsonl'), rows(root/'commands.jsonl')
    first = ledger[1]
    frame = audit.parse_tracking(first['source_raw_packet'], first['source_host_received_s'], first['source_sequence'])
    root7 = np.array(events[0]['root_pose7'])
    helper.calibrate(frame.body24x7, root7,
        np.array([[.2,.2,.9],[.2,-.2,.9],[0.,0.,.85]]), np.array([[1.,0,0,0]]*3))
    for index, command in enumerate(ledger[1:]):
        event = events[index]
        tick = command['physical_tick50']
        frame = audit.parse_tracking(command['source_raw_packet'], command['source_host_received_s'], command['source_sequence'])
        points, quats, live = helper.prepare(frame.body24x7, event['root_pose7'])
        _, evidence = build_teleop_reference(np.zeros((46,12)), np.zeros((46,12)), 0,
            event['root_pose7'], points, quats, event['root_pose7'][3:])
        evidence.update(contract=mobile.CONTROL_CONTRACT, reference_policy='continuous_vr_mode1',
            producer='planner_lower_body_plus_explicit_VR_mode1', model_forecasts_are_training_labels=False,
            model_inference_performed=True, encoder_sha256=audit.ENCODER_SHA256,
            token_sha256=hashlib.sha256(np.asarray(command['body_token64'],np.float32).tobytes()).hexdigest(),
            actual_state_written=False, upper_body_frozen=False, planner_upper_body_used_as_reference=False,
            synchronous_diagnostic_only=False, epoch=1, tick50=100+tick, timeline_origin_tick50=100+tick,
            cursor50=0, pending_request_id=None,
            source={'source_sequence':frame.sequence,'device_timestamp_ns':frame.device_timestamp_ns,
                'source_packet_reused_as_new_sample':False, 'source_packet_held_for_control_interval':False,
                'source_age_s':.001, 'axes_left_x_y_right_x_y':[0.]*4},
            live_vr=dict(live,source_sequence=frame.sequence,source_device_timestamp_ns=frame.device_timestamp_ns),
            live_upper_reference={'source':'caller_supplied_current_VR_three_point_world_targets',
                'source_sequence':frame.sequence,'device_timestamp_ns':frame.device_timestamp_ns,
                'point_order':list(vr.POINT_ORDER),'world_points_already_offset':True,
                'world_points_m':points.tolist(),'world_quaternions_wxyz':quats.tolist()},
            last_inference={'accepted_tick50':100+tick,'generated_tick50':100+tick,
                'submitted_tick50':100+tick,'evidence':{
                'contract':'cvpr-official-sonic11-locomotion-planner-v1',
                'official_model_sha256':mobile.PLANNER_MODEL_SHA256,
                'all_generated_futures_are_forecast_references_not_labels':True,
                'physics_steps':0,'GPU_used':False,'reference_only':True,
                'async_cpu_worker':{'cpu_only':True}, 'valid_model_output_sha256':'a'*64,
                'full_padded_model_output_sha256':'b'*64,
                'resampling':{'source_rate_hz':30,'reference_rate_hz':50,
                    'endpoint_holds_eligible_as_complete_reference_future':False,
                    'future_action_or_observation_labels_claimed':False,
                    'fully_interpolated_valid_forecast_frames50':72}}})
        command['reference_evidence'] = evidence
        hands = {}
        for side, sign in (('left',1.),('right',-1.)):
            for suffix, angle in zip(('thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1'),
                                    (0.,.7,.7,-1.,-1.5,-1.,-1.5)):
                hands[side+'_hand_'+suffix+'_joint'] = sign*angle*.2
        command['named_hand_targets'] = hands
        if event['action'] is not None:
            event['action']['hand14'] = [hands[name] for name in event['action']['hand_joint_names']]
    ledger[0]['metadata'] = metadata
    write_rows(root/'commands.jsonl', ledger); write_rows(root/'capture/events.jsonl', events)
    write(root/'metadata.json', metadata)
    for relative in ('capture/capture.json','capture/async_capture.json'):
        value = read(root/relative); value.update(metadata)
        value['events_sha256'] = audit.sha(root/'capture/events.jsonl'); write(root/relative,value)
    qualities = [{'completed_steps':i+1,'sim_time_s':3.+(i+1)/50,
        'solver_stage':'post_mj_step_before_observation_refresh','official_task_success':i==5} for i in range(150)]
    write_rows(root/'raw_quality_samples.jsonl', qualities)
    outcome['state']['start_tick50'] = 100
    outcome['capture'] = read(root/'capture/async_capture.json')
    outcome['ledger']['sha256'] = audit.sha(root/'commands.jsonl')
    write(root/'episode.json',outcome)
    session = Path(temporary)/'collection/live_session'
    source_alias = session/'episodes'/metadata['human_demo_uuid']
    source_alias.parent.mkdir(parents=True); source_alias.symlink_to(root, target_is_directory=True)
    record = {'status':'committed','disposition':'kept','source_kind':'live_pico_human',
        'human_demo_uuid':metadata['human_demo_uuid'],
        'source_episode_directory':str(source_alias),'episode_json_sha256':audit.sha(root/'episode.json'),
        'physical_report':{'native_flag_source_file':'raw_quality_samples.jsonl',
            'native_flag_source_sha256':audit.sha(root/'raw_quality_samples.jsonl')}}
    write(root/'collection_record.json', record)
    by_tick = {command['physical_tick50']:command for command in ledger[1:]}
    trace, latest = [], None
    for tick in range(150):
        command = by_tick.get(tick)
        if command is not None: latest = command
        before = {'sim_time_s':3.+tick/50,'root_pose7':root7.tolist(),
            'state_finite':True,'body_joint_names':list(audit.BODY_NAMES),'body_q29':[0.]*29,
            'body_dq29':[0.]*29,'hand_joint_names':list(audit.MODEL_HAND_NAMES),'hand_q14':[0.]*14}
        after = dict(before,sim_time_s=3.+(tick+1)/50)
        trace.append({'phase':'pico_follow','tick50':100+tick,
            'source_sequence':latest['source_sequence'],'source_device_timestamp_ns':latest['device_timestamp_ns'],
            'new_command30':command is not None,'reference':command['reference_evidence'] if command else None,
            'named_hand_targets':latest['named_hand_targets'],'actual_before_step':before,'actual_after_step':after,
            'terminated':qualities[tick]['official_task_success'],'truncated':False,
            'loop_wall_timing':{'lag_s':.002}})
    write_rows(session/'control_trace.jsonl',trace)
    return root, identities, decoder_sha, historical, session/'control_trace.jsonl'


class MobileAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root,self.identities,self.decoder_sha,self.historical,self.trace = mobile_fixture(temporary.name)
        for item in (patch.object(audit,'DECODER_SHA256',self.decoder_sha),
                     patch.object(audit,'verify_runtime_sources',return_value=self.identities)):
            item.start(); self.addCleanup(item.stop)

    def run_audit(self):
        return audit.audit_episode(self.root,write_report=False,historical_source_manifest=self.historical)

    def rewrite(self,name,mutate):
        path=self.root/name; value=read(path); mutate(value); write(path,value)

    def rebind_ledger(self):
        self.rewrite('episode.json',lambda v:v['ledger'].update(sha256=audit.sha(self.root/'commands.jsonl')))
        self.rewrite('collection_record.json',lambda v:v.update(episode_json_sha256=audit.sha(self.root/'episode.json')))

    def raw_quality(self,mutate,*,rebind=True,trace_flags=False):
        path=self.root/'raw_quality_samples.jsonl'; value=rows(path); mutate(value);write_rows(path,value)
        if rebind:
            self.rewrite('collection_record.json',lambda r:r['physical_report'].update(native_flag_source_sha256=audit.sha(path)))
        if trace_flags:
            trace=rows(self.trace)
            for i,row in enumerate(trace):row['terminated']=value[i].get('official_task_success')
            write_rows(self.trace,trace)

    def test_full_cpu_fixture_uses_original_native_in_warmup_without_extra_success_gates(self):
        before={path:path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        result=self.run_audit()
        self.assertEqual(result['errors'],[])
        self.assertTrue(result['eligible_for_packing'])
        self.assertEqual(result['contract'],mobile.CONTRACT)
        self.assertEqual(result['selection_policy'],'native_only')
        self.assertEqual(result['physical_gate']['excluded_unsupervised_warmup_ticks50'],0)
        self.assertEqual(set(Path(path).name for path in result['auditor_source_sha256']),mobile.AUDITOR_FILES)
        self.assertTrue(mobile.BASE_CHECKS<=set(result['checks']))
        self.assertNotIn('original_solved_support_and_deferred_geometry',result['checks'])
        self.assertFalse(result['mobile_reference_audit']['reference_numeric_inference_independently_reproduced'])
        self.assertEqual(before,{path:path.read_bytes() for path in before})

    def test_complete_native_false_is_task_failure_not_technical_failure(self):
        self.raw_quality(lambda v:[r.update(official_task_success=False) for r in v],trace_flags=True)
        result=self.run_audit()
        self.assertFalse(result['eligible_for_packing']);self.assertFalse(result['technical_blocked'])
        self.assertEqual(result['failure_kind'],'official_task_not_successful')
        self.assertIs(result['official_task_success'],False)

    def test_packet_used_by_prior50hz_tick_can_still_be_new_to30hz_encoder(self):
        ledger=rows(self.root/'commands.jsonl')
        ledger[3]['reference_evidence']['source']['source_packet_held_for_control_interval']=True
        write_rows(self.root/'commands.jsonl',ledger);self.rebind_ledger()
        trace=rows(self.trace);tick=ledger[3]['physical_tick50']
        trace[tick]['reference']=ledger[3]['reference_evidence'];write_rows(self.trace,trace)
        result=self.run_audit();self.assertEqual(result['errors'],[])
        self.assertTrue(result['eligible_for_packing'])

    def test_missing_native_bool_and_nonzero_force_are_technical_not_task_failure(self):
        for kind in ('missing','force'):
            with self.subTest(kind=kind):
                if kind=='missing':self.raw_quality(lambda v:v[4].pop('official_task_success'))
                else:
                    self.raw_quality(lambda v:v[4].update(official_task_success=False))
                    path=self.root/'applied_force_samples.jsonl';v=rows(path);v[2]['qfrc_applied_nonzero_count']=1;write_rows(path,v)
                result=self.run_audit()
                self.assertTrue(result['technical_blocked']);self.assertIsNone(result['official_task_success'])
                self.assertEqual(result['failure_kind'],'technical_evidence_incomplete')

    def test_raw_native_source_mutation_is_not_rescued_by_other_positive_flags(self):
        self.raw_quality(lambda v:v[6].update(official_task_success=True),rebind=False,trace_flags=True)
        result=self.run_audit();self.assertTrue(result['technical_blocked'])
        self.assertIn('raw-native source SHA differs',' '.join(result['errors']))

    def test_task_source_threshold_and_forecast_label_claims_fail_closed(self):
        for key,value in (('actual_task_source_sha256','0'*64),('actual_native_success_criteria',True)):
            original=(self.root/'metadata.json').read_bytes()
            self.rewrite('metadata.json',lambda v:v.update({key:value}))
            result=self.run_audit();self.assertTrue(result['technical_blocked'])
            (self.root/'metadata.json').write_bytes(original)
        ledger=rows(self.root/'commands.jsonl');ledger[1]['reference_evidence']['model_forecasts_are_training_labels']=True
        write_rows(self.root/'commands.jsonl',ledger)
        self.rebind_ledger()
        self.assertTrue(self.run_audit()['technical_blocked'])

    def test_same_tick_root_and_raw_body_vr_are_recomputed_not_trusted(self):
        ledger=rows(self.root/'commands.jsonl')
        ledger[1]['reference_evidence']['live_vr']['actual_root_pose7_used_for_world_lift'][0]+=.2
        write_rows(self.root/'commands.jsonl',ledger)
        self.rebind_ledger()
        result=self.run_audit();self.assertTrue(result['technical_blocked'])
        self.assertIn('same-tick VR root',' '.join(result['errors']))

    def test_all_original50hz_trace_not_just30hz_capture_is_required(self):
        value=rows(self.trace);value[1]['actual_before_step']['body_q29'][3]=.1;write_rows(self.trace,value)
        result=self.run_audit();self.assertTrue(result['technical_blocked'])
        self.assertIn('continuous actual50Hz',' '.join(result['errors']))

    def test_trace_native_mismatch_truncation_or_deadline_is_technical(self):
        original=self.trace.read_bytes()
        for key,value in (('terminated',True),('truncated',True),('lag',.10001)):
            trace=rows(self.trace)
            if key=='lag':trace[1]['loop_wall_timing']['lag_s']=value
            else:trace[1][key]=value
            write_rows(self.trace,trace)
            self.assertTrue(self.run_audit()['technical_blocked'])
            self.trace.write_bytes(original)

    def test_missing_trace_or_original_source_snapshot_never_claims_official_task_failure(self):
        self.trace.unlink()
        result=self.run_audit();self.assertTrue(result['technical_blocked']);self.assertIsNone(result['official_task_success'])
        source=Path(read(self.historical)['sources'][0]['snapshot_path']);source.write_text('changed')
        result=self.run_audit();self.assertTrue(result['technical_blocked'])
        self.assertIn('snapshot bytes changed',' '.join(result['errors']))


if __name__=='__main__':unittest.main()
