import glob
import os

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


AMD_MAPPING = {
    "not.in.icd.table": 0,
    "no.amd.diagnosis": 0,
    "early.dry": 1,
    "intermediate.dry": 2,
    "advanced.atrophic.dry.with.subfoveal.involvement": 3,
    "advanced.atrophic.dry.without.subfoveal.involvement": 3,
    "wet.amd.active.choroidal.neovascularization": 3,
    "wet.amd.inactive.choroidal.neovascularization": 3,
    "wet.amd.inactive.scar": 3,
}

DR_MAPPING = {
    "not.in.icd.table": 0,
    "no.dr.diagnosis": 0,
    "mild.npdr": 0,
    "moderate.npdr": 0,
    "severe.npdr": 1,
    "pdr": 1,
}

SPLIT_MAP = {"train": "Training", "val": "Validation", "test": "Test"}


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


class FairVisionDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        disease,
        mode="train",
        model_base="transformer",
        condition="normal",
        condition_name="Gaussian",
        noise_std=0.05,
        seed=11,
        fundus_size=384,
        oct_size=(96, 96, 96),
    ):
        self.dataset_root = dataset_root
        self.disease = disease
        self.mode = mode.lower()
        self.model_base = model_base
        self.condition = condition
        self.condition_name = condition_name
        self.noise_std = float(noise_std)
        self.seed = seed
        self.fundus_size = fundus_size if model_base == "transformer" else 512
        self.oct_size = oct_size if model_base == "transformer" else (128, 256, 128)

        split_name = SPLIT_MAP[self.mode]
        split_candidates = [
            os.path.join(dataset_root, disease, split_name),
            os.path.join(dataset_root, split_name),
            dataset_root,
        ]
        self.data_dir = next((path for path in split_candidates if os.path.isdir(path)), None)
        if self.data_dir is None:
            raise FileNotFoundError(f"Could not find FairVision split directory for {disease}/{split_name} under {dataset_root}")

        self.file_list = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.npz"), recursive=True))
        if not self.file_list:
            raise FileNotFoundError(f"No NPZ files found in {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def _resize_oct(self, oct_img, size):
        oct_img = np.asarray(oct_img, dtype=np.float32)
        scale = [size[0] / oct_img.shape[0], size[1] / oct_img.shape[1], size[2] / oct_img.shape[2]]
        return ndimage.zoom(oct_img, scale, order=1)

    def _prepare_fundus(self, fundus):
        fundus = np.asarray(fundus, dtype=np.float32)
        if fundus.ndim == 2:
            fundus = cv2.resize(fundus, (self.fundus_size, self.fundus_size), interpolation=cv2.INTER_CUBIC)
            fundus = np.repeat(fundus[..., None], 3, axis=2)
        elif fundus.ndim == 3:
            if fundus.shape[0] in (1, 3):
                fundus = np.transpose(fundus, (1, 2, 0))
            fundus = cv2.resize(fundus, (self.fundus_size, self.fundus_size), interpolation=cv2.INTER_CUBIC)
            if fundus.shape[2] == 1:
                fundus = np.repeat(fundus, 3, axis=2)
        else:
            raise ValueError(f"Unexpected fundus shape: {fundus.shape}")
        if fundus.max() > 1.0:
            fundus = fundus / 255.0
        return np.clip(fundus, 0.0, 1.0)

    def _label_from_npz(self, raw_data):
        if self.disease == "AMD":
            return AMD_MAPPING[_to_scalar(raw_data["amd_condition"])]
        if self.disease == "DR":
            return DR_MAPPING[_to_scalar(raw_data["dr_subtype"])]
        if self.disease == "Glaucoma":
            return int(_to_scalar(raw_data["glaucoma"]))
        raise ValueError(f"Unsupported FairVision disease: {self.disease}")

    def _augment(self, fundus, oct_img, rng):
        if self.mode == "train" and rng.rand() < 0.5:
            fundus = fundus[:, ::-1, :].copy()
            oct_img = oct_img[:, :, ::-1].copy()
        return fundus, oct_img

    def _make_views(self, fundus, oct_img, rng):
        fundus_low = fundus.copy()
        oct_low = oct_img.copy()
        fundus_high = fundus.copy()
        oct_high = oct_img.copy()

        if self.condition == "noise":
            if self.condition_name == "Gaussian":
                sigma = max(self.noise_std, 1e-6)
                fundus_high = np.clip(fundus_high + rng.normal(0, sigma, fundus_high.shape), 0.0, 1.0)
                oct_high = np.clip(oct_high + rng.normal(0, sigma, oct_high.shape), 0.0, 1.0)
            elif self.condition_name == "SaltPepper":
                prob = min(max(self.noise_std, 1e-4), 0.2)
                salt_mask = rng.rand(*fundus_high.shape) < prob / 2.0
                pepper_mask = rng.rand(*fundus_high.shape) < prob / 2.0
                fundus_high[salt_mask] = 1.0
                fundus_high[pepper_mask] = 0.0
                salt_mask = rng.rand(*oct_high.shape) < prob / 2.0
                pepper_mask = rng.rand(*oct_high.shape) < prob / 2.0
                oct_high[salt_mask] = 1.0
                oct_high[pepper_mask] = 0.0

        return (fundus_low, oct_low), (fundus_high, oct_high)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        raw_data = np.load(file_path, allow_pickle=True)

        fundus = self._prepare_fundus(raw_data["slo_fundus"])
        oct_img = self._resize_oct(raw_data["oct_bscans"], self.oct_size)
        if oct_img.max() > 1.0:
            oct_img = oct_img / 255.0
        oct_img = np.clip(oct_img.astype(np.float32), 0.0, 1.0)

        rng = np.random.RandomState(self.seed + idx)
        fundus, oct_img = self._augment(fundus, oct_img, rng)
        (fundus_low, oct_low), (fundus_high, oct_high) = self._make_views(fundus, oct_img, rng)

        data_low = {
            0: torch.from_numpy(np.transpose(fundus_low, (2, 0, 1))).float(),
            1: torch.from_numpy(oct_low).unsqueeze(0).float(),
        }
        data_high = {
            0: torch.from_numpy(np.transpose(fundus_high, (2, 0, 1))).float(),
            1: torch.from_numpy(oct_high).unsqueeze(0).float(),
        }

        label = self._label_from_npz(raw_data)
        return (data_low, data_high), torch.tensor(label, dtype=torch.long)
