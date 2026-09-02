"""
Phase 4-6 prep: one canonical patient-level fold assignment, shared by the
imaging, EHR-only, and fusion training scripts so all three train and
evaluate on identical splits -- otherwise the three-way comparison table
isn't apples-to-apples.

Assigned at the patient level, not the slice level. MRIDataset (see
mri_dataset.py) yields 115 slices per patient; if folds were assigned per
slice instead, slices from the same patient could end up split across
train and validation -- the model would then be "evaluated" on near-
identical scans of a patient it already saw during training, inflating
validation metrics. Every caller must filter its cohort dataframe down to
a fold's patient_ids *before* constructing any per-slice Dataset -- never
split a Dataset's items by fold after construction.

Only the 298-patient MRI subset gets folds assigned (not the full
1278-patient cohort), since that's the shared base cohort for imaging,
EHR-only, and fused models per the Phase 4-6 cohort-scope decision.
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 5 folds -> ~60 patients validated per fold. No separate held-out test set:
# every fold serves as validation exactly once, and with N=298, carving out
# a further untouched test set would shrink already-small training folds
# for little benefit -- report mean +/- std across the 5 folds instead.
N_SPLITS = 5
RANDOM_SEED = 42


def assign_folds(cohort_df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """One row per MRI patient: patient_id, label, fold (0..n_splits-1).
    Stratified on the label so each fold keeps roughly the same class balance."""
    mri_cohort = cohort_df[cohort_df["mri_file_path"].notna()].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_of = pd.Series(index=mri_cohort.index, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(mri_cohort["patient_id"], mri_cohort["label"])):
        fold_of.iloc[val_idx] = fold

    return pd.DataFrame({
        "patient_id": mri_cohort["patient_id"],
        "label": mri_cohort["label"],
        "fold": fold_of.values,
    })


def load_patient_folds(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def patients_in_split(folds_df: pd.DataFrame, fold: int, split: str) -> list:
    """split='val' -> patients held out for this fold; split='train' -> everyone else."""
    if split == "val":
        return folds_df.loc[folds_df["fold"] == fold, "patient_id"].tolist()
    if split == "train":
        return folds_df.loc[folds_df["fold"] != fold, "patient_id"].tolist()
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cohort = pd.read_csv(repo_root / "data" / "processed" / "patient_cohort.csv")
    folds = assign_folds(cohort)

    out_path = repo_root / "data" / "processed" / "patient_folds.csv"
    folds.to_csv(out_path, index=False)
    logger.info("Saved %d patients across %d folds to %s", len(folds), N_SPLITS, out_path)
    logger.info("Per-fold size and label balance:\n%s", folds.groupby("fold")["label"].value_counts().unstack())
