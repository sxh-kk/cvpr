"""Load the current SMPL-H245 forecasting data for training and validation."""
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
from .smplh_motion_dataset import SMPLHMotionDataset


def cache_training_collate(samples):
    """Transfer only training conditions, labels and source/logging identities."""
    return default_collate([
        {"y": sample["y"],
         "misc": {key: sample["misc"][key] for key in ("motion", "source", "seq_name")},
         "pred": {}}
        for sample in samples
    ])


class SMPLHMotionDataModule(pl.LightningDataModule):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self._cache_checked_epochs = {}

    def setup(self,stage=None):
        data=self.cfg.DATA
        def make(split):
            dataset_class = SMPLHMotionDataset
            return dataset_class(archive_root=getattr(data,"ARCHIVE_ROOT",None) or getattr(data,"DATA_DIR",None),
                split=split,split_manifest=getattr(data,"SPLIT_MANIFEST",None),stats_path=getattr(data,"STATS_PATH",None),
                do_normalization=True,cond_img_feat=data.COND_IMG_FEAT,cond_traj=data.COND_TRAJ,cond_betas=data.COND_BETAS,
                window=data.WINDOW,fps=data.FPS,forecast_prefix=data.FORECAST_PREFIX,
                segment_stride=data.SEGMENT_STRIDE,sparse_joint_indices=self.cfg.SPARSE_JOINTS.INDICES,
                repre_type=data.REPRE_TYPE,img_feat_type=data.IMG_FEAT_TYPE,
                row_group_cache_mb=getattr(data,"ROW_GROUP_CACHE_MB",128),parquet_file_cache=getattr(data,"PARQUET_FILE_CACHE",8),
                rotation_basis_path=getattr(data,"ROTATION_BASIS_PATH",None),rotation_basis_mode=getattr(data,"ROTATION_BASIS_MODE","smplh_model_defined"),
                target_joint_basis=getattr(data,"TARGET_JOINT_BASIS","smplh_body22_axes_v1"),
                require_verified_rotations=getattr(data,"REQUIRE_VERIFIED_ROTATIONS",True),normalization_clip=getattr(data,"NORMALIZATION_CLIP",None),
                contact_speed_m_s=getattr(data,"CONTACT_SPEED_M_S",.2),contact_height_m=getattr(data,"CONTACT_HEIGHT_M",.1),
                min_valid_image_fraction=getattr(data,"MIN_VALID_IMAGE_FRACTION",0.),
                train_task=getattr(data,"TRAIN_TASK",None) or "fore",require_full_window=getattr(data,"REQUIRE_FULL_WINDOW",False),
                motion_supervision=getattr(data,"MOTION_SUPERVISION",None),
                vision_profile_root=getattr(data,"VISION_PROFILE_ROOT",None),image_profile=getattr(data,"IMAGE_PROFILE",None),
                window_cache_root=getattr(data,"WINDOW_CACHE_ROOT",None))
        if stage in (None,"fit","train") and not hasattr(self,"train_dataset"): self.train_dataset=make("train")
        if stage in (None,"fit","validate","val") and not hasattr(self,"val_dataset"): self.val_dataset=make("val")
        if stage in ("test","predict") and not hasattr(self,"test_dataset"): self.test_dataset=make("test")

    def _loader(self,dataset,shuffle=False):
        data=self.cfg.DATA; workers=int(data.NUM_WORKERS)
        options={"pin_memory":getattr(data,"PIN_MEMORY",False),"persistent_workers":getattr(data,"PERSISTENT_WORKERS",False) and workers>0}
        if workers>0: options["prefetch_factor"]=getattr(data,"PREFETCH_FACTOR",2)
        sampler=None
        if shuffle and getattr(data,"SOURCE_BALANCE",False):
            counts={}
            for sequence,_ in dataset.idx_to_sidx_fidx:
                source=dataset.records[sequence]["source"]; counts[source]=counts.get(source,0)+1
            weights=getattr(data,"SOURCE_SAMPLE_WEIGHTS",{})
            weights=vars(weights) if hasattr(weights,"__dict__") else weights
            values=[]
            for sequence,_ in dataset.idx_to_sidx_fidx:
                source=dataset.records[sequence]["source"]; weight=float(weights.get(source,1.))
                if not np.isfinite(weight) or weight<=0: raise ValueError("Source weights must be finite positive")
                values.append(weight/counts[source])
            sampler=WeightedRandomSampler(values,num_samples=len(values),replacement=True); shuffle=False
        return DataLoader(dataset,batch_size=data.BATCH_SIZE,num_workers=workers,shuffle=shuffle,sampler=sampler,
                          collate_fn=cache_training_collate if getattr(dataset,"window_cache",None) is not None else default_collate,
                          drop_last=getattr(data,"DROP_LAST",False) if dataset.split=="train" else False,**options)

    def train_dataloader(self): return self._loader(self.train_dataset,shuffle=True)
    def val_dataloader(self,shuffle=False): return self._loader(self.val_dataset,shuffle=shuffle)
    def test_dataloader(self): return self._loader(self.test_dataset)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Keep source/cache identity checks out of every DataLoader worker sample."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return batch
        name = ("train_dataset" if trainer.training else "val_dataset" if trainer.validating
                else "test_dataset" if trainer.testing or trainer.predicting else None)
        dataset = getattr(self, name, None) if name is not None else None
        if dataset is not None and getattr(dataset, "window_cache", None) is not None:
            epoch = int(trainer.current_epoch)
            identity = (epoch, id(dataset.window_cache))
            if self._cache_checked_epochs.get(name) != identity:
                dataset.assert_cache_current()
                self._cache_checked_epochs[name] = identity
        return batch
