"""
Phase 6: fusion model. Concatenates each patient's MRI image embedding
with their EHR feature vector into one joint classifier -- the "joint
fusion" architecture named in CLAUDE.md. This produces the headline
result: imaging-only vs. EHR-only vs. fused, evaluated identically across
all three via the same 5-fold patient-level CV (patient_folds.csv).

Per fold, two inputs get merged by patient_id (never assumed to already
be in matching row order -- see load_fold_data):
  - EHR features: build_ehr_pipeline() (ehr_features.py), fit fresh on
    that fold's train patients only, transformed for train+val. Same
    leakage-safe pattern as Phases 3 and 5.
  - Image features: the RAW per-patient embedding columns (emb_0..
    emb_511) from that fold's mri_embeddings_fold{k}.csv -- deliberately
    NOT the `prob` column in that same file. `prob` is the imaging-only
    model's own standalone prediction, already collapsed to a single
    scalar; feeding that into fusion instead of the 512-dim embedding
    would throw away exactly the richer representation Phase 4 pooled
    embeddings (not probabilities) specifically to preserve for this step.

Frozen CNN: the encoder that produced those embeddings is not
fine-tuned here. Phase 4's 5-fold training already cost ~6 hours on
Kaggle's free GPU tier, so this script reuses its output rather than
re-running it -- CLAUDE.md's "frozen CNN encoder + trainable fusion head"
option for when compute is limited.

Dimensionality: concatenating the raw 512-dim embedding with 18 EHR
features would give 530 features against ~238 training patients per
fold -- more features than training examples, a real overfitting risk
for any linear model even with regularization. This script first reduces
the embedding to 32 components with a per-fold PCA (whitened, so its
output scale matches the EHR pipeline's already-standardized numeric
features), fit on that fold's train patients only -- same leakage rule
as everything else here. 18 + 32 = 50 features against ~238 patients is
a much healthier ratio than 530.

Final classifier: logistic regression (class_weight="balanced", matching
Phases 4-5), not a deeper MLP fusion head. With a frozen CNN and only
~238 training patients per fold, a simple, well-regularized linear model
over [EHR features, PCA-reduced embedding] is more defensible than a
higher-capacity network that would just overfit this little data.

Outputs:
  - data/processed/fusion_metrics.csv     -- same columns as Phase 4/5's
                                              baseline metrics files
  - data/processed/comparison_table.csv   -- the headline imaging-only
                                              vs. EHR-only vs. fused
                                              comparison, mean +/- std
                                              per model across folds
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
from ehr_features import build_ehr_pipeline  # noqa: E402
from folds import load_patient_folds  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_EMBEDDING_COMPONENTS = 32


def build_embedding_reducer() -> Pipeline:
    """Unfitted -- fit per fold on train patients only, same rule as
    build_ehr_pipeline()."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=N_EMBEDDING_COMPONENTS, whiten=True, random_state=RANDOM_SEED)),
    ])


def load_fold_data(fold: int, cohort: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, list]:
    """One row per patient (this fold's 298), with EHR raw fields, split,
    label, and this fold's embedding columns all aligned by an explicit
    patient_id merge -- row order from the CSV or from `cohort` is never
    assumed to already match between the two sources."""
    emb_df = pd.read_csv(out_dir / f"mri_embeddings_fold{fold}.csv")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    merged = cohort.merge(emb_df[["patient_id", "split"] + emb_cols], on="patient_id", how="inner")
    if len(merged) != len(emb_df):
        raise ValueError(
            f"Fold {fold}: merge lost patients (cohort={len(cohort)}, "
            f"embeddings={len(emb_df)}, merged={len(merged)}) -- cohort and "
            f"embeddings file are not the same 298-patient set."
        )
    return merged, emb_cols


def run_fold(fold: int, cohort: pd.DataFrame, out_dir: Path) -> dict:
    merged, emb_cols = load_fold_data(fold, cohort, out_dir)
    train_df = merged[merged["split"] == "train"]
    val_df = merged[merged["split"] == "val"]

    ehr_pipeline = build_ehr_pipeline()
    X_ehr_train = ehr_pipeline.fit_transform(train_df)
    X_ehr_val = ehr_pipeline.transform(val_df)

    reducer = build_embedding_reducer()
    X_emb_train = reducer.fit_transform(train_df[emb_cols].values)
    X_emb_val = reducer.transform(val_df[emb_cols].values)

    X_train = np.hstack([X_ehr_train, X_emb_train])
    X_val = np.hstack([X_ehr_val, X_emb_val])
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
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


def build_comparison_table(out_dir: Path) -> pd.DataFrame:
    """Combine Phase 4/5/6's per-fold metrics into the headline
    imaging-only vs. EHR-only vs. fused comparison: mean +/- std per model
    across the 5 folds."""
    tables = {
        "imaging-only": pd.read_csv(out_dir / "mri_baseline_metrics.csv"),
        "ehr-only": pd.read_csv(out_dir / "ehr_baseline_metrics.csv"),
        "fused": pd.read_csv(out_dir / "fusion_metrics.csv"),
    }
    rows = []
    for model_name, df in tables.items():
        summary = df[["auroc", "sensitivity", "specificity"]].agg(["mean", "std"])
        rows.append({
            "model": model_name,
            "auroc_mean": summary.loc["mean", "auroc"],
            "auroc_std": summary.loc["std", "auroc"],
            "sensitivity_mean": summary.loc["mean", "sensitivity"],
            "sensitivity_std": summary.loc["std", "sensitivity"],
            "specificity_mean": summary.loc["mean", "specificity"],
            "specificity_std": summary.loc["std", "specificity"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = REPO_ROOT / "data" / "processed"
    cohort = pd.read_csv(out_dir / "patient_cohort.csv")
    folds_df = load_patient_folds(out_dir / "patient_folds.csv")
    mri_cohort = cohort[cohort["patient_id"].isin(folds_df["patient_id"])].reset_index(drop=True)
    logger.info("Fusion cohort: %d patients", len(mri_cohort))

    # cast to plain int -- folds_df["fold"] loads as float64 from CSV, and
    # mri_embeddings_fold{fold}.csv filenames don't carry a ".0" suffix
    available_folds = sorted(int(f) for f in folds_df["fold"].unique())
    all_metrics = [run_fold(fold, mri_cohort, out_dir) for fold in available_folds]

    metrics_df = pd.DataFrame(all_metrics)
    logger.info("\n%s", metrics_df.to_string(index=False))
    summary = metrics_df[["auroc", "sensitivity", "specificity"]].agg(["mean", "std"])
    logger.info("Summary across folds:\n%s", summary)
    metrics_df.to_csv(out_dir / "fusion_metrics.csv", index=False)

    comparison = build_comparison_table(out_dir)
    logger.info("\n=== Headline comparison: imaging-only vs EHR-only vs fused ===\n%s", comparison.to_string(index=False))
    comparison.to_csv(out_dir / "comparison_table.csv", index=False)


if __name__ == "__main__":
    main()
