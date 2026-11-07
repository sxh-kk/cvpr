"""Synthetic CPU fixtures: raw training coverage, never real human admission."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_pico_teleop_episode as shared
import pico_grasp_audit as audit
from test_audit_pico_teleop_episode import write, write_rows
from test_pico_mobile_audit import mobile_fixture, read, rows


def grasp_fixture(temporary):
    root, identities, decoder_sha, historical, trace = mobile_fixture(temporary)
    metadata = read(root/'metadata.json')
    metadata.update(task_id=audit.GRASP_TASK_ID, mobile_control_protocol=audit.CONTROL_CONTRACT,
                    quality_geometry_mode='grasp_native_height_no_container_geometry')
    ledger, events, quality, execution = (rows(root/'commands.jsonl'), rows(root/'capture/events.jsonl'),
        rows(root/'raw_quality_samples.jsonl'), rows(trace))
    idle = {'mode':0, 'movement_direction':[0.,0.,0.], 'facing_direction':[1.,0.,0.],
            'speed':-1., 'height':-1.}
    for command in ledger[1:]:
        evidence = command['reference_evidence']
        evidence['contract'] = audit.CONTROL_CONTRACT
        evidence['source'].update(applied_axes_left_x_y_right_x_y=[0.,0.,0.,0.],
            ignored_axes=['left_axis_x','left_axis_y','right_axis_x','right_axis_y'],
            ignored_reason=['tabletop_grasp_standing_reference'],
            axis_policy='grasp_standing_leg_reference_all_joystick_axes_ignored')
        evidence.update(desired_command=deepcopy(idle), applied_plan_command=deepcopy(idle))
    ledger[0]['metadata'] = metadata
    for event in events:
        event['objects'].pop('container')
    for index, row in enumerate(quality):
        success = index >= 60
        target = [0.,0.,.57 if success else .55]
        row.update(official_task_success=success, native_task_reward=1. if success else 0.,
                   native_baseline_target_z_m=.55, actual_target_xyz_m=target,
                   target_pose7=target+[1.,0.,0.,0.], target_body_name='target')
    by_tick = {command['physical_tick50']:command for command in ledger[1:]}
    for index,row in enumerate(execution):
        row['terminated'] = quality[index]['official_task_success']
        if index in by_tick:
            row['reference'] = by_tick[index]['reference_evidence']
    write_rows(root/'commands.jsonl', ledger)
    write_rows(root/'capture/events.jsonl', events)
    write_rows(root/'raw_quality_samples.jsonl', quality)
    write_rows(trace, execution)
    write(root/'metadata.json', metadata)
    for relative in ('capture/capture.json','capture/async_capture.json'):
        manifest = read(root/relative)
        manifest.update(metadata, events_sha256=audit.sha(root/'capture/events.jsonl'))
        write(root/relative, manifest)
    outcome = read(root/'episode.json')
    outcome['capture'] = read(root/'capture/async_capture.json')
    outcome['ledger']['sha256'] = audit.sha(root/'commands.jsonl')
    write(root/'episode.json', outcome)
    record = read(root/'collection_record.json')
    record['episode_json_sha256'] = audit.sha(root/'episode.json')
    record['physical_report']['native_flag_source_sha256'] = audit.sha(root/'raw_quality_samples.jsonl')
    write(root/'collection_record.json', record)
    return root, identities, decoder_sha, historical, trace


class GraspAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root,self.identities,self.decoder_sha,self.historical,self.trace = grasp_fixture(temporary.name)
        # Synthetic fixture deliberately has no real pinned SONIC artifacts or
        # Grasp world adaptation. Only those resolvers are mocked; command/VR,
        # trace, dense RGB/state/action, native reward and force checks are real.
        for item in (patch.object(shared,'DECODER_SHA256',self.decoder_sha),
                     patch.object(shared,'verify_runtime_sources',return_value=self.identities),
                     patch.object(audit,'validate_metadata',return_value={'passed':True})):
            item.start(); self.addCleanup(item.stop)

    def run_audit(self):
        return audit.audit_episode(self.root,write_report=False,historical_source_manifest=self.historical)

    def quality(self, mutate):
        quality = rows(self.root/'raw_quality_samples.jsonl'); mutate(quality)
        write_rows(self.root/'raw_quality_samples.jsonl', quality)
        execution = rows(self.trace)
        for index,row in enumerate(execution):
            row['terminated'] = quality[index]['official_task_success']
        write_rows(self.trace, execution)
        record = read(self.root/'collection_record.json')
        record['physical_report']['native_flag_source_sha256'] = audit.sha(self.root/'raw_quality_samples.jsonl')
        write(self.root/'collection_record.json',record)

    def test_target_only_complete_fixture_preserves_all_source_bytes(self):
        before = {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = self.run_audit()
        self.assertEqual(result['errors'],[])
        self.assertTrue(result['eligible_for_packing'])
        self.assertEqual(result['contract'],audit.CONTRACT)
        self.assertEqual(result['selection_policy'],'native_only')
        self.assertEqual(set(Path(p).name for p in result['auditor_source_sha256']),audit.AUDITOR_FILES)
        self.assertTrue(audit.BASE_CHECKS <= set(result['checks']))
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_single_last_supervised_success_needs_no_hold_contact_or_final_height_gate(self):
        self.quality(lambda q:[r.update(official_task_success=i==100,native_task_reward=1. if i==100 else 0.)
                              for i,r in enumerate(q)])
        result = self.run_audit()
        self.assertTrue(result['eligible_for_packing'],result['errors'])

    def test_recovery_in_warmup_retains_visual_history_and_successful_supervised_suffix(self):
        def recovery(q):
            for i,row in enumerate(q[:50]):
                row['actual_target_xyz_m'] = [0.,0.,.8-i*.005]
        self.quality(recovery)
        result = self.run_audit()
        self.assertTrue(result['eligible_for_packing'],result['errors'])
        self.assertTrue(result['grasp_supervised_coverage']['warmup_actions_excluded_from_supervision'])

    def test_native_success_only_in_warmup_is_training_coverage_blocker(self):
        self.quality(lambda q:[r.update(official_task_success=i==5,native_task_reward=1. if i==5 else 0.)
                              for i,r in enumerate(q)])
        result = self.run_audit()
        self.assertFalse(result['eligible_for_packing'])
        self.assertTrue(result['physical_gate']['official_task_success'])
        self.assertIn('unlabeled warmup',' '.join(result['errors']))

    def test_complete_native_failure_is_not_misclassified_as_technical_failure(self):
        self.quality(lambda q:[r.update(official_task_success=False,native_task_reward=0.) for r in q])
        result = self.run_audit()
        self.assertFalse(result['eligible_for_packing'])
        self.assertFalse(result['technical_blocked'])
        self.assertEqual(result['failure_kind'],'official_task_not_successful')

    def test_original_reward_boolean_inconsistency_is_technical_failure(self):
        self.quality(lambda q:q[90].update(native_task_reward=0.))
        result = self.run_audit()
        self.assertTrue(result['technical_blocked'])
        self.assertIn('reward and check_success',' '.join(result['errors']))

    def test_missing_original50hz_trace_and_nonzero_external_force_rejected(self):
        self.trace.unlink()
        force = rows(self.root/'applied_force_samples.jsonl')
        force[10]['xfrc_applied_max_abs'] = .1
        write_rows(self.root/'applied_force_samples.jsonl',force)
        result = self.run_audit()
        self.assertTrue(result['technical_blocked'])
        self.assertFalse(result['checks']['all_original200hz_external_forces_zero']['passed'])
        self.assertFalse(result['checks']['independent_grasp_continuous_reference']['passed'])


class StandingReferenceTests(unittest.TestCase):
    def test_moving_reference_rejected_before_shared_vr_validation(self):
        evidence = {'contract':audit.CONTROL_CONTRACT,'source':{
            'applied_axes_left_x_y_right_x_y':[0.,1.,0.,0.]}}
        with self.assertRaisesRegex(ValueError,'ignore all joystick'):
            audit.check_reference(None,None,evidence,None,{})


class BatchProducerBindingTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'episode'; root.mkdir()
        raw = root/'original.json'; write(raw,{'actual':'frozen physical evidence'})
        stat = raw.stat()
        identity = 'fixture-uuid'
        frozen = [{'uuid':identity if index==0 else f'other-{index}',
                   'official_episode_index':index, 'number':index+1, 'directory':str(root),
                   'source_files':{'original.json':{'size_bytes':stat.st_size,
                       'mtime_ns':stat.st_mtime_ns,'sha256':audit.sha(raw)}}} for index in range(100)]
        manifest = Path(directory)/'frozen.json'
        write(manifest,{'contract':'cvpr-grasp100-rtx-admission-frozen-inputs-v1','count':100,'episodes':frozen})
        wrapper = Path(audit.__file__).with_name('render_deferred_pico_batch.py')
        renderer = Path(audit.__file__).with_name('render_deferred_pico.py')
        report_path = Path(directory)/'batch-report.json'
        execution = {'contract':'cvpr-pico-grasp-deferred-rtx-batch-v1',
            'batch_code_sha256':audit.sha(wrapper), 'original_renderer_code_sha256':audit.sha(renderer),
            'frozen_manifest_sha256':audit.sha(manifest), 'environment_lease_number':1,
            'real_environment_count':1,'original_reset_and_environment_binding_per_episode':True,
            'factory_override':'single environment lease; defer close until batch finally',
            'batch_report_path':str(report_path)}
        completion = {'batch_execution':execution,'completed_frames':2,'environment_binding':{'matched':True}}
        write(root/'rtx_completion.json', completion)
        report = {'contract':execution['contract'],'status':'completed_pending_independent_audit',
            'cleanup_requested':True,'cleanup_return_verified':True,
            'accepted_new_dataset':False,'batch_code_sha256':execution['batch_code_sha256'],
            'original_renderer_code_sha256':execution['original_renderer_code_sha256'],
            'manifest_path':str(manifest),'manifest_sha256':audit.sha(manifest),
            'selected_count':1,'new_environment_leases':1,
            'episodes':[{'uuid':identity,'number':1,'directory':str(root),
                'rtx_completion_sha256':audit.sha(root/'rtx_completion.json'),
                'completed_frames':2,'environment_binding':{'matched':True},'reused_completed_output':False}]}
        write(report_path,report)
        return root, completion, {'human_demo_uuid':identity}, report_path, manifest

    def test_completed_batch_binds_wrapper_report_frozen_cohort_and_exact_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root,completion,metadata,report,manifest=self.fixture(directory)
            bindings={}
            result=audit._check_batch_execution(root,completion,metadata,bindings)
            self.assertTrue(result['completed_report_binds_this_receipt'])
            self.assertEqual(bindings[str(report)],audit.sha(report))
            self.assertEqual(bindings[str(manifest)],audit.sha(manifest))
            self.assertIn(str(Path(audit.__file__).with_name('render_deferred_pico_batch.py')),bindings)

    def test_changed_wrapper_or_report_receipt_and_unfinished_report_rejected(self):
        for change in ('wrapper','receipt','unfinished','frozen'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root,completion,metadata,report_path,manifest=self.fixture(directory)
                if change=='wrapper':completion['batch_execution']['batch_code_sha256']='0'*64
                elif change=='frozen':
                    value=read(manifest);value['episodes'][0]['uuid']='different';write(manifest,value)
                else:
                    value=read(report_path)
                    if change=='receipt':value['episodes'][0]['rtx_completion_sha256']='0'*64
                    else:value['status']='running'
                    write(report_path,value)
                with self.assertRaises(ValueError):audit._check_batch_execution(root,completion,metadata,{})

    def test_original_nonbatch_receipt_needs_no_batch_producer(self):
        self.assertIsNone(audit._check_batch_execution(Path('/unused'),{}, {},{}))

    def legacy_fixture(self, directory):
        root,completion,metadata,report_path,manifest=self.fixture(directory)
        snapshot=Path(directory)/'legacy_batch.py'
        snapshot.write_text('# Synthetic reviewed first-pilot source, never executable evidence\n')
        legacy_sha=audit.sha(snapshot)
        completion['batch_execution']['batch_code_sha256']=legacy_sha
        write(root/'rtx_completion.json',completion)
        report=read(report_path)
        report.update(status='running',selected_count=2,batch_code_sha256=legacy_sha,
                      created_utc='2026-11-07T10:00:00+00:00')
        report.pop('new_environment_leases')
        report['episodes'][0]['rtx_completion_sha256']=audit.sha(root/'rtx_completion.json')
        report['episodes'].append(dict(report['episodes'][0],uuid='other-1',number=2))
        write(report_path,report)
        exit_path=Path(directory)/'exit.json'
        write(exit_path,{'exit_code':0,'finished_beijing':'2026-11-07T18:01:00+08:00'})
        recovery_path=report_path.with_name(report_path.stem+'.recovery.json')
        recovery={'contract':'cvpr-pico-grasp-batch-close-exit-recovery-v1',
            'status':'rendered_outputs_completed_process_exit_zero','cleanup_return_verified':False,
            'original_report':{'path':str(report_path),'sha256':audit.sha(report_path)},
            'producer_source':{'path':str(snapshot),'sha256':legacy_sha},
            'process_exit':{'path':str(exit_path),'sha256':audit.sha(exit_path)},
            'frozen_manifest':{'path':str(manifest),'sha256':audit.sha(manifest)},
            'original_renderer_code_sha256':completion['batch_execution']['original_renderer_code_sha256'],
            'episodes':report['episodes'],'new_environment_leases':2}
        write(recovery_path,recovery)
        return root,completion,metadata,legacy_sha,recovery_path,exit_path

    def test_only_reviewed_legacy_producer_with_external_exit0_can_recover_unchanged_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root,completion,metadata,legacy_sha,recovery_path,exit_path=self.legacy_fixture(directory)
            with patch.object(audit,'LEGACY_CLOSE_EXIT_WRAPPER_SHA',legacy_sha):
                result=audit._check_batch_execution(root,completion,metadata,{})
            self.assertFalse(result['cleanup_return_verified'])
            self.assertEqual(result['original_report_status_preserved'],'running')
            self.assertEqual(result['recovery_receipt_path'],str(recovery_path))
            # Even re-binding a nonzero exit into the recovery cannot bypass it.
            write(exit_path,{'exit_code':1,'finished_beijing':'2026-11-07T18:01:00+08:00'})
            recovery=read(recovery_path);recovery['process_exit']['sha256']=audit.sha(exit_path)
            write(recovery_path,recovery)
            with patch.object(audit,'LEGACY_CLOSE_EXIT_WRAPPER_SHA',legacy_sha),self.assertRaisesRegex(ValueError,'exit0'):
                audit._check_batch_execution(root,completion,metadata,{})

    def test_new_preclose_report_requires_exact_external_exit_and_driver_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,completion,metadata,report_path,manifest=self.fixture(directory)
            report=read(report_path)
            report.update(status='render_completed_pending_process_exit',cleanup_requested=True,
                          cleanup_return_verified=False,render_finished_utc='2026-11-07T10:00:00+00:00')
            write(report_path,report)
            driver=Path(directory)/'parentdriver.py';driver.write_text('# Synthetic parent wait() driver\n')
            exit_path=report_path.with_name(report_path.stem+'_EXIT.json')
            exit_receipt={'contract':'cvpr-grasp-rtx-observed-process-exit-v1',
                'exit_code':0,'child_pid':123,'finished_beijing':'2026-11-07T18:01:00+08:00',
                'report_path':str(report_path),'report_sha256':audit.sha(report_path),
                'argv':['python','render_deferred_pico_batch.py','--manifest',str(manifest)],
                'parentdriver_path':str(driver),'parentdriver_sha256':audit.sha(driver)}
            write(exit_path,exit_receipt)
            result=audit._check_batch_execution(root,completion,metadata,{})
            self.assertFalse(result['cleanup_return_verified'])
            self.assertEqual(result['process_exit_path'],str(exit_path))
            self.assertEqual(result['parentdriver_sha256'],audit.sha(driver))
            driver.write_text('# Changed independent driver bytes\n')
            with self.assertRaisesRegex(ValueError,'driver source bytes'):
                audit._check_batch_execution(root,completion,metadata,{})


if __name__ == '__main__':
    unittest.main()
