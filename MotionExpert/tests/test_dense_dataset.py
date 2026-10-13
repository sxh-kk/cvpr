"""Dense archive regressions with nontrivial world poses and projected Parquet I/O."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from MotionExpert.config import K12_JOINTS
from MotionExpert.dataset.dense_representation import build_dense_window,decode_dense_motion,quaternion_matrix
from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset,MOTION_COLUMNS
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.dataset.split_manifest import create_split_manifest,validate_manifest,archive_fingerprint
from MotionExpert.compute_joint_motion_stats import compute_stats,_accumulate,_finalize_accumulators
from MotionExpert.merge_joint_motion_stats import _merge_accumulator,_load


def synthetic_arrays(count=17):
    frames=np.arange(count)
    yaw=.4+frames*.03
    q=np.stack([np.cos(yaw/2),np.zeros(count),np.zeros(count),np.sin(yaw/2)],-1)
    body=np.zeros((count,22,3),dtype=np.float32)
    body[...,0]=3+frames[:,None]*.02+np.arange(22)[None,:]*.01
    body[...,1]=5+frames[:,None]*.005+np.arange(22)[None,:]*.015
    body[...,2]=.3+np.arange(22)[None,:]*.02
    body[:,[10,11],2]=-.68
    head=np.stack([3+frames*.02,5+frames*.005,np.ones(count)*1.1],-1).astype(np.float32)
    joints=np.arange(22)
    joint_yaw=.2+joints[None,:]*.04+frames[:,None]*.01
    body_q=np.stack([np.cos(joint_yaw/2),np.zeros_like(joint_yaw),np.zeros_like(joint_yaw),np.sin(joint_yaw/2)],-1)
    arrays={"timestamp":(frames/30).astype(np.float32),
            "observation.state.body_world":body.reshape(count,-1),
            "observation.valid.body_position":np.ones((count,22),bool),
            "observation.state.body_world_rotation":body_q.reshape(count,-1).astype(np.float32),
            "observation.valid.body_rotation":np.ones((count,22),bool),
            "observation.state.head_reference_world":head,
            "observation.valid.head_reference_position":np.ones(count,bool),
            "observation.state.head_reference_world_rotation":q.astype(np.float32),
            "observation.valid.head_reference_rotation":np.ones(count,bool),
            "observation.features.dino":np.repeat(frames[:,None].astype(np.float32),5120,axis=1),
            "observation.valid.dino":np.ones(count,bool)}
    for name,joint in (("hand_left",20),("hand_right",21)):
        hands=body[:,joint,None]+np.arange(21)[None,:,None]*np.array([.001,.002,.003])[None,None]
        arrays[f"observation.state.{name}_world"]=hands.reshape(count,63).astype(np.float32)
        arrays[f"observation.valid.{name}_position"]=np.ones((count,21),bool)
    return arrays


def synthetic_metadata(operator="a"):
    return {"operator_id":operator,"source_file_uid":operator,"source_episode_uuid":operator,
            "coordinate_verified":True,"head_reference":{"basis_verified":True},
            "rotation_basis_verified":False,"rotation_semantics":"source_global_wxyz","body_basis_correction":None,
            "ground":{"valid":True,"height_m":-.7,"mode":"estimate","verified":False,
                      "window_fallback":{"mode":"mean_valid_feet","offset_m":.02}}}


def write_archive(root,arrays,metadata=None):
    (root/"meta").mkdir(); (root/".pipeline/episodes").mkdir(parents=True)
    (root/".pipeline/config.json").write_text(json.dumps({"schema_version":"egomotion-body22-camera-floor-dino5-v4"}))
    (root/"meta/info.json").write_text(json.dumps({"fps":30,"features":{"observation.features.dino":{"shape":[5120]}}}))
    records=[]
    for i in range(3):
        columns={}
        for key,values in arrays.items():
            values=np.asarray(values)
            columns[key]=pa.array(values) if values.ndim==1 else pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)),values.shape[1])
        path=root/f"data/chunk-000/file-{i:03d}.parquet"; path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.table(columns),path,row_group_size=5)
        meta=synthetic_metadata(str(i)) if metadata is None else dict(metadata,operator_id=str(i),source_file_uid=str(i),source_episode_uuid=str(i))
        record={"episode_index":i,"source":"egosuite","source_key":f"egosuite:{i}","length":len(arrays["timestamp"]),"metadata":meta,"dino":{"state":"complete"}}
        (root/f".pipeline/episodes/{i:08d}.json").write_text(json.dumps(record)); records.append(record)
    manifest={"format":"motionexpert-archive-splits-v1","archive_fingerprint":archive_fingerprint(records),
              "splits":{"train":["egosuite:0"],"val":["egosuite:1"],"test":["egosuite:2"]}}
    manifest_path=root.parent/"splits.json"; manifest_path.write_text(json.dumps(manifest))
    return manifest_path,records


class DenseDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_nontrivial_world_roundtrip_and_fore_anchor(self):
        original=synthetic_arrays(11)
        selected={key:value[::2] for key,value in original.items()}
        encoded=build_dense_window(selected,synthetic_metadata(),rotation_basis_verified=True)
        meta=encoded.metadata["training_window"]
        decoded=decode_dense_motion(encoded.motion,window_to_world=meta["archive_world_from_window"],floor_height=meta["floor_height_m"])
        np.testing.assert_allclose(decoded["body_positions"],selected["observation.state.body_world"].reshape(-1,22,3)[:,K12_JOINTS],atol=2e-6)
        np.testing.assert_allclose(decoded["body_rotations"],quaternion_matrix(selected["observation.state.body_world_rotation"].reshape(-1,22,4))[:,K12_JOINTS],atol=2e-6)
        np.testing.assert_allclose(decoded["left_hand"],selected["observation.state.hand_left_world"].reshape(-1,21,3),atol=2e-6)
        np.testing.assert_allclose(decoded["right_hand"],selected["observation.state.hand_right_world"].reshape(-1,21,3),atol=2e-6)
        np.testing.assert_allclose(decoded["floor_positions"][:,:2],selected["observation.state.head_reference_world"][:,:2],atol=2e-6)
        anchor=np.eye(4); anchor[:3,:3]=meta["floor_rotation"][1]; anchor[:3,3]=meta["floor_position"][1]
        future=decode_dense_motion(encoded.motion,start_frame=2,initial_floor_transform=anchor,window_to_world=meta["archive_world_from_window"],floor_height=meta["floor_height_m"])
        torch.testing.assert_close(future["body_positions"],decoded["body_positions"][2:],atol=2e-6,rtol=1e-5)
        self.assertFalse(encoded.metadata["training_window"]["rotation_basis_verified"])

    def test_rotations_positions_and_hands_have_independent_validity(self):
        arrays=synthetic_arrays(5)
        arrays["observation.valid.body_rotation"][2,5]=False
        arrays["observation.valid.body_position"][3,4]=False
        arrays["observation.valid.hand_left_position"][1,7]=False
        encoded=build_dense_window(arrays,synthetic_metadata(),rotation_basis_verified=True)
        body=encoded.motion_valid[:,:108].reshape(5,12,9)
        self.assertFalse(body[2,K12_JOINTS.index(5),:6].any())
        self.assertTrue(body[2,K12_JOINTS.index(5),6:].all())
        self.assertTrue(body[3,K12_JOINTS.index(4),:6].all())
        self.assertFalse(body[3,K12_JOINTS.index(4),6:].any())
        self.assertFalse(encoded.motion_valid[1,117+7*3:117+8*3].any())
        np.testing.assert_array_equal(encoded.motion[~encoded.motion_valid],0)

    def test_fore_floor_uses_only_observed_prefix(self):
        arrays=synthetic_arrays(6)
        meta=synthetic_metadata(); meta["ground"]["valid"]=False
        baseline=build_dense_window(arrays,meta,task="fore",forecast_prefix=2,rotation_basis_verified=True)
        changed={k:v.copy() for k,v in arrays.items()}
        body=changed["observation.state.body_world"].reshape(-1,22,3)
        body[2:,[10,11],2]+=9
        future_changed=build_dense_window(changed,meta,task="fore",forecast_prefix=2,rotation_basis_verified=True)
        np.testing.assert_allclose(baseline.trajectory[:2],future_changed.trajectory[:2],atol=0)
        self.assertEqual(baseline.metadata["ground"]["height_m"],future_changed.metadata["ground"]["height_m"])
        recon=build_dense_window(changed,meta,task="recon",rotation_basis_verified=True)
        self.assertNotEqual(recon.metadata["ground"]["height_m"],baseline.metadata["ground"]["height_m"])

    def test_reader_selects_even_frames_and_only_requested_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive"; root.mkdir(); arrays=synthetic_arrays()
            # An unused dense22 joint is invalid throughout: K12 must survive.
            arrays["observation.valid.body_position"][:,1]=False
            manifest,_=write_archive(root,arrays)
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,segment_stride=3,
                do_normalization=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")
            dataset.reader.read_log.clear()
            sample=dataset[0]
            self.assertEqual(sample["misc"]["motion"].shape,(4,245))
            np.testing.assert_array_equal(sample["y"]["img_embs"][:,0,0],[0,2,4,6])
            self.assertEqual({entry[1] for entry in dataset.reader.read_log},{0,1})
            self.assertLessEqual(dataset.reader.cached_bytes,dataset.reader.max_bytes)
            self.assertEqual(json.loads(sample["misc"]["metadata_json"])["training_window"]["source_frame_indices"],[0,2,4,6])
            # At15Hz each yaw increment spans TWO30Hz source frames.
            matrix=quaternion_matrix(np.array([[np.cos(.06/2),0,0,np.sin(.06/2)]]))[0]
            np.testing.assert_allclose(sample["misc"]["motion"][1,108:114],matrix[:2].reshape(-1),atol=1e-6)
            with self.assertRaises(ValueError): dataset.get_from_seq_and_st(dataset.seq_names[0],1)
            forecast=dataset.get_sample(0,task="fore")
            self.assertTrue(bool(forecast["y"]["traj_mask"][2:].all()))
            self.assertTrue(bool(forecast["y"]["img_mask"][2:].all()))
            self.assertTrue(bool(forecast["y"]["feature_valid_mask"][2:].any()))

    def test_train_statistics_exclude_invalid_and_do_not_read_dino(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive"; root.mkdir(); arrays=synthetic_arrays()
            arrays["observation.valid.hand_right_position"][:]=False
            arrays["observation.state.hand_right_world"][:]=123456
            manifest,_=write_archive(root,arrays)
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,segment_stride=3,
                cond_img_feat=False,do_normalization=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")
            dataset.reader.read_log.clear()
            stats=compute_stats(dataset)
            self.assertEqual(stats["motion_mean"].shape,(245,))
            self.assertTrue(bool((stats["motion_count"][180:243]==0).all()))
            torch.testing.assert_close(stats["motion_mean"][108:117],stats["traj_mean"][9:18])
            torch.testing.assert_close(stats["motion_std"][108:117],stats["traj_std"][9:18])
            self.assertFalse(any("observation.features.dino" in columns for _,_,columns in dataset.reader.read_log))
            self.assertFalse(any("observation.valid.dino" in columns for _,_,columns in dataset.reader.read_log))
            norm=MotionNormalizer(stats)
            value=torch.rand(4,245)
            torch.testing.assert_close(norm.normalize(value)[:,243:],value[:,243:],atol=0,rtol=0)
            torch.testing.assert_close(norm.denormalize(norm.normalize(value)),value,atol=1e-5,rtol=1e-5)
            self.assertEqual(stats["split"],"train")
            validation=DenseMotionDataset(archive_root=root,split_manifest=manifest,split="val",window=4,forecast_prefix=2,segment_stride=3,
                cond_img_feat=False,do_normalization=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")
            with self.assertRaises(ValueError): compute_stats(validation)

    def test_native_axes_are_explicit_and_unified_requires_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive"; root.mkdir(); manifest,records=write_archive(root,synthetic_arrays())
            with self.assertRaisesRegex(ValueError,"unverified"):
                DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,do_normalization=False,cond_img_feat=False)
            record_path=root/".pipeline/episodes/00000000.json"
            record=records[0];record["source"]="egoexo_hm";record["source_key"]="egoexo_hm:0"
            record_path.write_text(json.dumps(record))
            manifest_data=json.loads(manifest.read_text());manifest_data["splits"]["train"]=["egoexo_hm:0"]
            manifest_data["archive_fingerprint"]=archive_fingerprint(records);manifest.write_text(json.dumps(manifest_data))
            with self.assertRaisesRegex(ValueError,"mixed sources"):
                DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,do_normalization=False,cond_img_feat=False,
                    rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")

    def test_invalid_initial_camera_filtered_and_image_threshold_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive"; root.mkdir(); arrays=synthetic_arrays()
            arrays["observation.valid.head_reference_rotation"][0]=False
            arrays["observation.valid.dino"][6:]=False
            manifest,_=write_archive(root,arrays)
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,segment_stride=1,
                do_normalization=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1",min_valid_image_fraction=.25)
            self.assertNotIn((0,0),dataset.idx_to_sidx_fidx)
            self.assertGreater(dataset.rejected_windows["invalid_initial_camera"],0)
            self.assertGreater(dataset.rejected_windows["insufficient_real_DINO"],0)

    def test_split_prevents_operator_and_recording_leakage(self):
        records=[]
        for i in range(10):
            records.append({"episode_index":i,"source_key":f"egosuite:{i}","source":"egosuite","length":20+i,
                            "metadata":{"operator_id":str(i//2),"source_file_uid":str(i)}})
        # Different operators can appear in multiple cuts of the same original recording.
        records[2]["metadata"]["source_file_uid"]=records[0]["metadata"]["source_file_uid"]
        manifest=create_split_manifest(records,seed=3)
        validate_manifest(manifest,records)
        owner={key:split for split,keys in manifest["splits"].items() for key in keys}
        self.assertEqual(owner["egosuite:0"],owner["egosuite:2"])
        malicious=json.loads(json.dumps(manifest))
        first=malicious["splits"]["train"][0]
        malicious["splits"]["val"].append(first)
        with self.assertRaises(ValueError): validate_manifest(malicious,records)

    def test_documented_unified_correction_is_postmultiplied(self):
        arrays=synthetic_arrays(5)
        correction=np.broadcast_to(np.array([[1,0,0],[0,0,-1],[0,1,0.]]),(22,3,3)).copy()
        encoded=build_dense_window(arrays,synthetic_metadata(),rotation_basis_verified=True,body_basis_correction=correction)
        details=encoded.metadata["training_window"]
        decoded=decode_dense_motion(encoded.motion,window_to_world=details["archive_world_from_window"],floor_height=details["floor_height_m"])
        expected=quaternion_matrix(arrays["observation.state.body_world_rotation"].reshape(-1,22,4)) @ correction
        np.testing.assert_allclose(decoded["body_rotations"],expected[:,K12_JOINTS],atol=2e-6)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive";root.mkdir();manifest,_=write_archive(root,arrays)
            basis=Path(temporary)/"basis.json"
            mapping={"format":"motionexpert-joint-basis-v1","target_basis":"synthetic_axes_v1",
                     "sources":{"egosuite":{"verified":True,"provenance":"synthetic test calibration only",
                       "source_rotation_semantics":"source_global_wxyz","corrections":correction.tolist()}}}
            basis.write_text(json.dumps(mapping))
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,cond_img_feat=False,
                do_normalization=False,rotation_basis_path=basis,rotation_basis_mode="unified",target_joint_basis="synthetic_axes_v1")
            sample=dataset[0];n=sample["misc"]["num_real_frames"]
            self.assertTrue(bool(sample["misc"]["feature_valid_mask"][:n,:108].reshape(n,12,9)[...,:6].all()))
            self.assertNotEqual(dataset.basis_mapping_fingerprint,"none")
            mapping["sources"]["egosuite"]["verified"]=False;basis.write_text(json.dumps(mapping))
            with self.assertRaisesRegex(ValueError,"verified=true"):
                DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,cond_img_feat=False,
                    do_normalization=False,rotation_basis_path=basis,rotation_basis_mode="unified",target_joint_basis="synthetic_axes_v1")

    def test_normalizer_rejects_nonfinite_clip_and_decoder_trims_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive";root.mkdir();manifest,_=write_archive(root,synthetic_arrays(5))
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,cond_img_feat=False,
                do_normalization=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")
            sample=dataset[0]
            self.assertEqual(dataset.ret_to_full_sequence(sample)["body_positions"].shape,(3,12,3))
            stats=compute_stats(dataset,limit_windows=1)
            with self.assertRaises(ValueError): MotionNormalizer(stats,clip=float("nan"))
            with self.assertRaises(ValueError): MotionNormalizer(stats,clip=float("inf"))
            sample["misc"]["feature_valid_mask"][1,108:117]=False
            with self.assertRaisesRegex(ValueError,"invalid"):
                dataset.ret_to_full_sequence(sample)

    def test_sharded_statistics_equal_single_pass_and_reject_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"archive";root.mkdir();manifest,_=write_archive(root,synthetic_arrays())
            dataset=DenseMotionDataset(archive_root=root,split_manifest=manifest,window=4,forecast_prefix=2,segment_stride=3,
                do_normalization=False,cond_img_feat=False,rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")
            full=compute_stats(dataset)
            first=_accumulate(dataset,range(0,len(dataset),2),progress=False)
            second=_accumulate(dataset,range(1,len(dataset),2),progress=False)
            merged=dict(first,**_merge_accumulator(_merge_accumulator(None,first),second))
            merged["num_windows"]=first["num_windows"]+second["num_windows"]
            finalized=_finalize_accumulators(merged)
            for key in ("motion_mean","motion_std","traj_mean","traj_std","motion_count","traj_count"):
                torch.testing.assert_close(finalized[key],full[key],atol=0,rtol=0)
            shard=Path(temporary)/"shard.pt"
            first.update(kind="joint_masked_partial",num_shards=2,shard_index=0)
            torch.save(first,shard)
            self.assertEqual(_load(shard)["motion_acc"]["sum"].shape,(245,))
            first["stats_schema_version"]=2;torch.save(first,shard)
            with self.assertRaisesRegex(ValueError,"legacy153"):_load(shard)

    def test_decode_rejects_gap_and_missing_forecast_anchor(self):
        encoded=build_dense_window(synthetic_arrays(4),synthetic_metadata(),rotation_basis_verified=True)
        with self.assertRaises(ValueError): decode_dense_motion(encoded.motion,start_frame=1)
        with self.assertRaisesRegex(ValueError,"invalid"):
            decode_dense_motion(encoded.motion,delta_valid=[True,False,True,True])


if __name__=="__main__": unittest.main()
