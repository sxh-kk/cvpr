"""CPU-only protocol tests; no physical replay, model or GPU is started."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(Path(__file__).resolve().parent)]
import collect_handover_supplement as supplement
import replay_handover_teacher as replay
import test_plan_handover_quality100 as fixtures


def make_plan(**kwargs):
    manifest, selection = fixtures.fixture()
    controls = {str(i): "measured_pose_pd" for i in range(24)}
    return supplement.build_plan(manifest, selection, manifest_sha256="b"*64,
        selection_sha256="a"*64, warmup_controls=controls, source_ids=list(range(6)), **kwargs)


def prepare_fixture(root):
    metadata = root / "source-metadata";metadata.mkdir()
    manifest, selection = fixtures.fixture()
    selection_file = metadata / "selection.json";supplement.write(selection_file, selection)
    manifest["selection_sha256"] = supplement.sha256(selection_file)
    manifest_file = metadata / "manifest.json";supplement.write(manifest_file, manifest)
    schedule = metadata / "warmup.json"
    supplement.write(schedule, {"contract": supplement.quality100.WARMUP_SCHEDULE_CONTRACT,
        "scope": supplement.quality100.WARMUP_SCOPE,
        "controls": {str(i): "measured_pose_pd" for i in range(24)}})
    inputs = {}
    for key in ("snapshot_root", "data_root", "sonic_resources"):
        inputs[key] = root / key;inputs[key].mkdir()
    args = SimpleNamespace(manifest=manifest_file, selection=selection_file, warmup_schedule=schedule,
        **inputs, python=Path(sys.executable), output=root / "new-queue", sources=list(range(6)),
        target=6, seed=20261024, xy_m=.02, yaw_rad=.075)
    supplement.prepare(args)
    return args.output


class SupplementProtocolTests(unittest.TestCase):
    def test_separate_contract_balanced_changed_proposals_and_no_recovery_claim(self):
        plan = make_plan()
        self.assertEqual(plan["contract"], supplement.PLAN_CONTRACT)
        self.assertEqual(plan["candidate_count"], 24)
        self.assertFalse(plan["policy_recovery_data"])
        self.assertFalse(plan["material_seed_applied"])
        self.assertFalse(plan["generation_seed_applied"])
        self.assertEqual(Counter(c["ancestor"]["source_episode_index"] for c in plan["candidates"]), {i:4 for i in range(6)})
        self.assertEqual({c["ancestor"]["source_episode_index"] for c in plan["candidates"][:6]}, set(range(6)))
        self.assertEqual(plan, make_plan())
        self.assertNotEqual(plan["candidates"], make_plan(seed=9)["candidates"])
        for candidate in plan["candidates"]:
            entries=[v for role in candidate["initial_object_deltas"].values() for v in role["xy_delta_m"]]
            self.assertEqual(sum(v != 0 for v in entries), 1)
            self.assertAlmostEqual(max(abs(v) for v in entries), .02)

    def test_quotas_and_hard_bounds_cannot_be_weakened(self):
        for kwargs in ({"target":5},{"target":13},{"target":True},{"xy_m":.031},{"yaw_rad":.101},{"xy_m":float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):make_plan(**kwargs)
        for key,value in (("maximum_accepted_per_ancestor",3),("minimum_distinct_accepted_ancestors",5),
                          ("physical_acceptance_gate","weak_contact_only"),("policy_recovery_data",True)):
            plan=make_plan();plan[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):supplement.validate_plan(plan)

    def test_candidate_identity_bounds_and_lineage_tampering_rejected(self):
        for mutate in (lambda c:c.update(candidate_spec_sha256='f'*64),
                       lambda c:c.update(split='val'),
                       lambda c:c['ancestor'].update(source_episode_index=39),
                       lambda c:c['initial_object_deltas']['target'].update(xy_delta_m=[.04,0])):
            plan=make_plan();mutate(plan['candidates'][0])
            with self.assertRaises(ValueError):supplement.validate_plan(plan)
        plan=make_plan();plan['candidates'].append(deepcopy(plan['candidates'][0]));plan['candidate_count']+=1
        with self.assertRaises(ValueError):supplement.validate_plan(plan)

    def test_current_legacy_contract_still_rejects_expanded_bounds(self):
        manifest,selection=fixtures.fixture();plan=fixtures.build(manifest,selection)
        candidate=plan['candidates'][0]
        candidate['initial_object_deltas']['target']['xy_delta_m']=[.02,0.]
        candidate['candidate_spec_sha256']=replay.canonical_sha256({k:candidate[k] for k in replay.CANDIDATE_SPEC_KEYS})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.json';path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'exceeds registered small bounds'):
                replay.read_candidate(path,candidate['candidate_id'])

    def test_new_replay_contract_accepts_2cm_only_with_verified_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            queue=prepare_fixture(Path(directory));plan=supplement.read(queue/'candidate_plan.json')
            candidate=plan['candidates'][0]
            bound=replay.read_candidate(queue/'candidate_plan.json',candidate['candidate_id'])
            self.assertEqual(bound['proposal']['initial_object_deltas'],candidate['initial_object_deltas'])
            state,actual=supplement.verified_queue(queue)
            self.assertEqual(state['accepted_count'],0);self.assertFalse(state['final_ready'])
            Path(plan['source_files']['selection']['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'metadata changed'):supplement.load_plan(queue/'candidate_plan.json')

    def test_source_coverage_reserved_before_second_acceptance(self):
        plan=make_plan(target=6)
        state={'attempts':[{'status':'accepted','source_episode_index':i} for i in range(6)]}
        result=supplement.queue_summary(state,plan)
        self.assertTrue(result['capture_target_satisfied'])
        self.assertFalse(result['final_ready']);self.assertFalse(result['DINO_cache_generated'])
        state={'attempts':[{'status':'accepted','source_episode_index':i//2} for i in range(6)]}
        self.assertFalse(supplement.queue_summary(state,plan)['capture_target_satisfied'])
        state={'attempts':[{'status':'accepted','source_episode_index':0} for _ in range(3)]}
        with self.assertRaises(ValueError):supplement.queue_summary(state,plan)

    def test_failed_cpu_screen_cannot_reach_capture_or_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            queue=prepare_fixture(Path(directory))
            with patch.object(supplement,'_execute',side_effect=RuntimeError('CPU fixture rejected')) as child:
                result=supplement.process(queue,phase='screen')
                self.assertEqual(child.call_count,1)
            self.assertEqual(result['attempts'][0]['status'],'screen_rejected')
            with patch.object(supplement,'_execute') as child:
                result=supplement.process(queue,phase='capture')
                child.assert_not_called()
            self.assertEqual(result['accepted_count'],0)


if __name__ == '__main__':
    unittest.main()
