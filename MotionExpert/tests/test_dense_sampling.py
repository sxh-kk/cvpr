"""Exercise task-aware dataset → sampler → physical world metrics end to end."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from MotionExpert.compute_joint_motion_stats import compute_stats
from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.evaluate import sampled_validation,compute_motion_metrics
from MotionExpert.tests.test_dense_dataset import synthetic_arrays,synthetic_metadata,write_archive


class DenseSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actual_fore_sample_uses_prefix_floor_and_reports_future_world_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"archive";root.mkdir()
            arrays=synthetic_arrays(9);meta=synthetic_metadata();meta["ground"]["valid"]=False
            # Future foot heights change full-window floor, but must not change the fore prefix floor.
            body=arrays["observation.state.body_world"].reshape(-1,22,3)
            body[4:,[10,11],2]+=.5
            manifest,_=write_archive(root,arrays,meta)
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,do_normalization=False,
                rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1",window=4,forecast_prefix=2,segment_stride=4)
            stats=compute_stats(dataset,limit_windows=1);normalizer=MotionNormalizer(stats)
            reconstruction=dataset[0];forecast=dataset.get_sample(0,task="fore")
            self.assertNotEqual(reconstruction["misc"]["floor_height"],forecast["misc"]["floor_height"])
            self.assertAlmostEqual(forecast["misc"]["floor_height"],-.7,places=6)
            calls=[]
            def sample(y):
                # This deterministic oracle tests integration and masks, not model quality.
                calls.append(y)
                torch.randn(5)
                self.assertTrue(bool(y["traj_mask"][0,2:].all()))
                self.assertTrue(bool(y["img_mask"][0,2:].all()))
                self.assertTrue(bool(y["feature_valid_mask"][0,2:].any()))
                torch.testing.assert_close(normalizer.denormalize(y["traj"],"traj")[0,:2],forecast["y"]["traj"][:2])
                return normalizer.normalize(forecast["misc"]["motion"]).unsqueeze(0)
            module=SimpleNamespace(model=torch.nn.Linear(1,1),normalizer=normalizer,
                                   cfg=SimpleNamespace(TRAIN=SimpleNamespace(SEED=13)),sample=sample)
            rng_before=torch.random.get_rng_state().clone()
            metrics=sampled_validation(module,dataset,max_samples=1,forecast_prefix=2,tasks=("fore",))
            self.assertEqual(len(calls),1)
            torch.testing.assert_close(torch.random.get_rng_state(),rng_before)
            self.assertEqual(metrics["fore/sample_count"],1)
            self.assertEqual(metrics["fore/body_world_mpjpe_m_count"],24)
            self.assertLess(metrics["fore/body_world_mpjpe_m"],1e-5)
            self.assertLess(metrics["fore/floor_ate_m"],1e-5)
            self.assertEqual(metrics["fore/reference_chain_coverage"],1.)

    def test_future_floor_uses_observed_anchor_and_predicted_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"archive";root.mkdir();manifest,_=write_archive(root,synthetic_arrays(9))
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,do_normalization=False,
                cond_img_feat=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1",window=4,forecast_prefix=2)
            sample=dataset.get_sample(0,task="fore");target=sample["misc"]["motion"]
            kwargs=dict(task="fore",forecast_prefix=2,fps=15,world_from_window=sample["misc"]["world_from_window"],
                        floor_transform=sample["misc"]["floor_transform"],floor_height=sample["misc"]["floor_height"])
            truth=compute_motion_metrics(target,target,sample["misc"]["feature_valid_mask"],**kwargs)
            predicted=target.clone();predicted[2:,114]+=.1
            changed=compute_motion_metrics(predicted,target,sample["misc"]["feature_valid_mask"],**kwargs)
            self.assertLess(truth["body_world_mpjpe_m"],1e-6)
            self.assertGreater(changed["body_world_mpjpe_m"],.1)
            self.assertGreater(changed["floor_final_drift_m"],.19)
            self.assertEqual(changed["body_local_mpjpe_m"],0.)
            # Changing future GT camera transforms must not alter a forecast result.
            kwargs["floor_transform"]=kwargs["floor_transform"].clone()
            kwargs["floor_transform"][2:,:3,3]+=999
            repeat=compute_motion_metrics(predicted,target,sample["misc"]["feature_valid_mask"],**kwargs)
            self.assertEqual(repeat["body_world_mpjpe_m"],changed["body_world_mpjpe_m"])


if __name__=="__main__":unittest.main()
