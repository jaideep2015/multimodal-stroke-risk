"""
Phase 2: PyTorch Dataset for the MRI branch.

Slice range: keep [70:185) out of the 256 slices per volume. This is the
reference `mri` repo's own `data_process.ipynb` range (verified against its
raw source), not CLAUDE.md's rough "0-49/205-255 are noise" README
paraphrase -- the two roughly agree (both center on the same middle band),
but 70:185 is what the reference repo actually runs, and it also lines up
with an independent check here: average pixel intensity across a 20-patient
sample peaks around slice 127-128 and is well inside this band.

Each item is a single 2D slice, not a whole patient. With only 298 patients
in this cohort, treating every clean slice as its own training example
(298 x 115 = 34,270 items) gives the CNN branch far more to learn from than
298 patient-level volumes would -- the same approach the reference repo
takes. Phase 4's training script aggregates slice-level predictions or
embeddings back up to one-per-patient (e.g. mean over a patient's slices)
for anything that needs a single patient-level number, such as the fusion
embedding.

Each slice is replicated across 3 channels rather than adapting the CNN's
first conv layer to 1-channel input (which is what the reference repo
does). That keeps torchvision's `IMAGENET1K_V1` pretrained weights usable
unmodified in Phase 4 -- adapting the first conv to grayscale would have
meant reinitializing exactly the layer the pretrained weights are for.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLEAN_SLICE_RANGE = (70, 185)  # [start, end) -- see module docstring
RESIZE_TO = 224  # standard input size for torchvision's pretrained ResNet18/34
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_clean_slices(dicom_path: str) -> np.ndarray:
    """Load one patient's DICOM volume and return only the clean-range slices."""
    volume = pydicom.dcmread(dicom_path).pixel_array
    start, end = CLEAN_SLICE_RANGE
    return volume[start:end]


def slice_to_tensor(slice_2d: np.ndarray) -> torch.Tensor:
    """Normalize one grayscale slice to [0, 1] (scaled by its own max, matching
    the reference repo's normalization) and replicate to 3 channels."""
    slice_float = slice_2d.astype(np.float32)
    peak = slice_float.max()
    normalized = slice_float / peak if peak > 0 else slice_float
    tensor = torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
    tensor = TF.resize(tensor, [RESIZE_TO, RESIZE_TO], antialias=True)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


class MRIDataset(Dataset):
    """One item = one clean slice from one patient's brain MRI.

    Built from the Phase 1 patient cohort dataframe (patient_id, label,
    mri_file_path columns); rows without an MRI are dropped automatically,
    so this naturally covers the same 298-patient subset used across
    Phases 4-6 for the imaging, EHR, and fused comparison.

    For train/val splitting, filter cohort_df to one fold's patient_ids
    (see folds.py) *before* constructing this Dataset -- e.g.
    `MRIDataset(cohort_df[cohort_df.patient_id.isin(train_ids)])`. Never
    construct one MRIDataset and split its items afterward: since each
    patient contributes 115 consecutive slice-items, a post-hoc split
    (by index, or any split not keyed on patient_id) risks putting some
    of a patient's slices in train and others in validation.
    """

    def __init__(self, cohort_df: pd.DataFrame):
        has_mri = cohort_df["mri_file_path"].notna()
        cohort_with_mri = cohort_df[has_mri].reset_index(drop=True)
        logger.info("MRIDataset: %d / %d patients have an MRI", len(cohort_with_mri), len(cohort_df))

        self.patient_ids = cohort_with_mri["patient_id"].tolist()
        self.mri_paths = cohort_with_mri["mri_file_path"].tolist()
        self.labels = cohort_with_mri["label"].tolist()
        self.slices_per_patient = CLEAN_SLICE_RANGE[1] - CLEAN_SLICE_RANGE[0]

    def __len__(self) -> int:
        return len(self.patient_ids) * self.slices_per_patient

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        patient_idx, slice_offset = divmod(idx, self.slices_per_patient)
        clean_slices = load_clean_slices(self.mri_paths[patient_idx])
        image = slice_to_tensor(clean_slices[slice_offset])
        return image, self.labels[patient_idx], self.patient_ids[patient_idx]


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cohort = pd.read_csv(repo_root / "data" / "processed" / "patient_cohort.csv")
    dataset = MRIDataset(cohort)
    logger.info("Dataset size: %d slice-level items from %d patients", len(dataset), len(dataset.patient_ids))

    image, label, patient_id = dataset[0]
    logger.info("Sample item 0 -> image %s, label %s, patient_id %s", tuple(image.shape), label, patient_id)
