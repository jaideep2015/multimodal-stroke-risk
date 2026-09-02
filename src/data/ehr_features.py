"""
Phase 3: EHR feature engineering for the 298-patient MRI cohort (same base
cohort as the imaging and fusion branches, per the Phase 4-6 scope decision).

This module builds an *unfitted* sklearn ColumnTransformer rather than a
static "processed" CSV. The reason is the same leakage concern folds.py
exists for, just applied to preprocessing instead of splitting: median
imputation and StandardScaler both compute statistics from the data they're
fit on. Fitting those once on all 298 patients and then handing out a
single processed table would leak each fold's validation patients into the
imputation/scaling statistics used to transform them -- the exact scorecard-
modeling caution about fitting WOE bins or normalization on the full
population instead of train-only. So: call build_ehr_pipeline() fresh, fit()
it on a fold's *train* patients only, and use that fitted pipeline to
transform() both train and val for that fold. Phases 5 and 6 do this once
per fold, using patient_folds.csv (see folds.py) for the split.

Missing-value handling mirrors standard scorecard practice: numeric fields
get a `<col>_missing` indicator column (via SimpleImputer(add_indicator=
True)) before median imputation, so "value wasn't recorded" stays visible
to the model as its own signal rather than being silently smoothed over.
Categorical missingness gets its own explicit "Unknown" category instead of
being folded into the mode, for the same reason.
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["age", "systolic_bp", "diastolic_bp", "total_cholesterol", "hdl_cholesterol", "bmi", "glucose"]
CATEGORICAL_FEATURES = ["sex", "smoking_status"]


def build_ehr_pipeline() -> ColumnTransformer:
    """Returns an unfitted ColumnTransformer. Caller is responsible for
    fitting it on train-only data for whichever fold is in use."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from folds import load_patient_folds, patients_in_split

    repo_root = Path(__file__).resolve().parents[2]
    cohort = pd.read_csv(repo_root / "data" / "processed" / "patient_cohort.csv")
    folds = load_patient_folds(repo_root / "data" / "processed" / "patient_folds.csv")

    mri_cohort = cohort[cohort["patient_id"].isin(folds["patient_id"])].reset_index(drop=True)
    logger.info("EHR feature cohort: %d patients (matches the 298-patient MRI subset)", len(mri_cohort))

    fold = 0
    train_ids = set(patients_in_split(folds, fold, "train"))
    val_ids = set(patients_in_split(folds, fold, "val"))
    train_df = mri_cohort[mri_cohort["patient_id"].isin(train_ids)]
    val_df = mri_cohort[mri_cohort["patient_id"].isin(val_ids)]

    pipeline = build_ehr_pipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)

    logger.info("Fold %d: fit on %d train patients, transformed %d val patients", fold, len(train_df), len(val_df))
    logger.info("Feature matrix shapes -> train %s, val %s", X_train.shape, X_val.shape)
    logger.info("Output feature names: %s", list(pipeline.get_feature_names_out()))
