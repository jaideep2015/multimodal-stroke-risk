"""
Phase 5: EHR-only baseline. Evaluated with the same 5-fold patient-level
CV and the same metrics (AUROC/sensitivity/specificity) as Phase 4's
imaging baseline, so Phase 6/9 can drop both into one directly comparable
table without reconciling different evaluation setups.

Same fold-safety rule as Phase 3 (ehr_features.py): build_ehr_pipeline()
returns a fresh, unfitted transformer, fit here on each fold's train
patients only and applied (never refit) to that fold's val patients.
Reusing a pipeline already fit on a different fold, or fitting one across
all 298 patients before splitting, would leak validation statistics into
training the same way an un-pooled slice split would.

Default model is logistic regression, not gradient boosting, even though
CLAUDE.md leaves either as an option: this project's EHR feature
engineering already leans on scorecard-style practice (missing
indicators, explicit "Unknown" categories), logistic regression is the
natural continuation of that lineage, its coefficients are exactly what
Phase 7's tabular feature-importance step wants, and it's the model that
actually benefits from build_ehr_pipeline()'s StandardScaler step (a
gradient-boosted tree wouldn't care about feature scale). class_weight=
"balanced" mirrors Phase 4's pos_weight-based class weighting so both
baselines handle the label imbalance the same way. --model gbm swaps in
HistGradientBoostingClassifier for comparison.

Output: data/processed/ehr_baseline_metrics.csv -- same columns as
Phase 4's mri_baseline_metrics.csv (fold, n_val, auroc, sensitivity,
specificity) by design.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
from ehr_features import build_ehr_pipeline  # noqa: E402
from folds import load_patient_folds, patients_in_split  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

_MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    "gbm": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_SEED),
}


def run_fold(fold: int, mri_cohort: pd.DataFrame, folds_df: pd.DataFrame, model_name: str) -> dict:
    train_ids = set(patients_in_split(folds_df, fold, "train"))
    val_ids = set(patients_in_split(folds_df, fold, "val"))
    train_df = mri_cohort[mri_cohort["patient_id"].isin(train_ids)]
    val_df = mri_cohort[mri_cohort["patient_id"].isin(val_ids)]

    pipeline = build_ehr_pipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    model = _MODELS[model_name]()
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]

    auroc = roc_auc_score(y_val, val_probs)
    val_preds = (val_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    logger.info(
        "Fold %d val metrics (n=%d): AUROC=%.3f sensitivity=%.3f specificity=%.3f",
        fold, len(val_df), auroc, sensitivity, specificity,
    )
    return {
        "fold": fold,
        "n_val": len(val_df),
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: EHR-only baseline, 5-fold CV")
    parser.add_argument("--model", choices=list(_MODELS), default="logreg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(REPO_ROOT / "data" / "processed" / "patient_cohort.csv")
    folds_df = load_patient_folds(REPO_ROOT / "data" / "processed" / "patient_folds.csv")
    mri_cohort = cohort[cohort["patient_id"].isin(folds_df["patient_id"])].reset_index(drop=True)
    logger.info("EHR baseline cohort: %d patients (%s model)", len(mri_cohort), args.model)

    all_metrics = [run_fold(fold, mri_cohort, folds_df, args.model) for fold in sorted(folds_df["fold"].unique())]

    metrics_df = pd.DataFrame(all_metrics)
    logger.info("\n%s", metrics_df.to_string(index=False))
    summary = metrics_df[["auroc", "sensitivity", "specificity"]].agg(["mean", "std"])
    logger.info("Summary across folds:\n%s", summary)

    metrics_df.to_csv(REPO_ROOT / "data" / "processed" / "ehr_baseline_metrics.csv", index=False)


if __name__ == "__main__":
    main()
