"""
Phase 6: fusion models. Two alternatives, both evaluated identically
across the same 5-fold patient-level CV (patient_folds.csv):

1. JOINT fusion (run_fold): concatenates each patient's MRI image
   embedding with their EHR feature vector into one classifier -- the
   "joint fusion" architecture named in CLAUDE.md.
2. LATE fusion (run_late_fusion_fold): trains no new classifier over
   concatenated features at all. Instead it takes the imaging-only and
   EHR-only MODELS' predicted probabilities and blends them with a
   single tuned weight, alpha * imaging_prob + (1 - alpha) * ehr_prob.
   Added specifically to test whether joint fusion's inability to beat
   the EHR-only baseline is a limitation of concatenation itself (a
   linear classifier over concatenated features has no way to partially
   discount a noisy branch -- every input gets folded into one weight
   vector) rather than a limitation of combining these two modalities at
   all. Late fusion CAN discount a noisy branch: if alpha tunes toward 0,
   that's the model explicitly learning "ignore imaging, trust EHR."

Both approaches are cheap and require no GPU/retraining of the CNN --
they only ever consume Phase 4's already-computed embeddings/probabilities
and cheaply-refittable EHR models on CPU.

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
  - data/processed/fusion_metrics.csv       -- joint fusion, per fold
  - data/processed/fusion_late_metrics.csv  -- late fusion, per fold
                                                (same columns as the
                                                baseline metrics files,
                                                plus the tuned `alpha`)
  - data/processed/comparison_table.csv     -- the headline four-way
                                                comparison (imaging-only,
                                                EHR-only, joint fusion,
                                                late fusion), mean +/- std
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
from sklearn.model_selection import cross_val_predict
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
    label, this fold's embedding columns, AND the imaging model's own
    `prob` column (needed by late fusion) -- all aligned by an explicit
    patient_id merge. Row order from the CSV or from `cohort` is never
    assumed to already match between the two sources."""
    emb_df = pd.read_csv(out_dir / f"mri_embeddings_fold{fold}.csv")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    merged = cohort.merge(emb_df[["patient_id", "split", "prob"] + emb_cols], on="patient_id", how="inner")
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


def _ehr_pipeline_with_classifier() -> Pipeline:
    """EHR preprocessing + classifier as one Pipeline, so cross_val_predict
    below refits imputation/scaling fresh within each inner fold instead of
    leaking outer-train-wide statistics into the inner split it's tuning on."""
    return Pipeline([
        ("features", build_ehr_pipeline()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def build_oof_imaging_probs(out_dir: Path, folds_df: pd.DataFrame) -> pd.Series:
    """A genuinely out-of-fold imaging probability for every one of the
    298 patients, sourced entirely from Phase 4's existing embeddings
    files -- no CNN retraining involved.

    Each patient is the *val* patient of exactly one outer fold (the 5
    folds partition the cohort). That fold's CNN never trained on them,
    so that fold's `prob` for them is already a legitimate holdout
    prediction -- it's exactly what run_fold's own baseline metrics use.
    Concatenating just the val-split rows across all 5 embeddings files
    gives one lookup, indexed by patient_id, with zero in-sample entries
    anywhere. This replaces the old (biased) approach of using a fold's
    own train-split `prob` -- i.e. that fold's model scoring the same
    patients it was fit on -- as a stand-in for a held-out estimate."""
    available_folds = sorted(int(f) for f in folds_df["fold"].unique())
    oof_frames = []
    for fold in available_folds:
        emb_df = pd.read_csv(out_dir / f"mri_embeddings_fold{fold}.csv")
        oof_frames.append(emb_df.loc[emb_df["split"] == "val", ["patient_id", "prob"]])
    oof_df = pd.concat(oof_frames, ignore_index=True)
    if oof_df["patient_id"].duplicated().any():
        raise ValueError("Each patient should appear in exactly one fold's val split")
    return oof_df.set_index("patient_id")["prob"]


def tune_late_fusion_weight(train_df: pd.DataFrame, oof_imaging_prob: pd.Series) -> float:
    """Pick alpha (the imaging-vs-EHR blend weight) using only this fold's
    training patients -- never the val patients being held out for
    evaluation. Both sides of the blend now use genuine out-of-fold
    predictions: EHR via cross_val_predict (refit per inner fold), and
    imaging via oof_imaging_prob (each patient's prediction from the one
    outer fold that actually held them out) -- previously this used the
    current outer fold's own in-sample train-set `prob`, which is
    optimistic since that's the same model scoring patients it was fit on."""
    y_train = train_df["label"].values
    imaging_prob_train = train_df["patient_id"].map(oof_imaging_prob).values
    oof_ehr_prob = cross_val_predict(
        _ehr_pipeline_with_classifier(), train_df, y_train, cv=5, method="predict_proba"
    )[:, 1]

    best_alpha, best_auroc = 0.0, -np.inf
    for alpha in np.linspace(0, 1, 21):
        blended = alpha * imaging_prob_train + (1 - alpha) * oof_ehr_prob
        auroc = roc_auc_score(y_train, blended)
        if auroc > best_auroc:
            best_alpha, best_auroc = alpha, auroc
    return best_alpha


def run_late_fusion_fold(fold: int, cohort: pd.DataFrame, out_dir: Path, oof_imaging_prob: pd.Series) -> dict:
    merged, _ = load_fold_data(fold, cohort, out_dir)
    train_df = merged[merged["split"] == "train"]
    val_df = merged[merged["split"] == "val"]

    alpha = tune_late_fusion_weight(train_df, oof_imaging_prob)

    ehr_pipeline = build_ehr_pipeline()
    X_ehr_train = ehr_pipeline.fit_transform(train_df)
    X_ehr_val = ehr_pipeline.transform(val_df)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_ehr_train, train_df["label"].values)
    ehr_prob_val = clf.predict_proba(X_ehr_val)[:, 1]

    # equivalent to val_df["prob"] (fold k's val patients ARE fold k's oof
    # patients by construction) -- sourced from the same lookup as the
    # training side for one single, provably-consistent source of truth
    imaging_prob_val = val_df["patient_id"].map(oof_imaging_prob).values
    y_val = val_df["label"].values
    blended_val = alpha * imaging_prob_val + (1 - alpha) * ehr_prob_val

    auroc = roc_auc_score(y_val, blended_val)
    val_preds = (blended_val >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    logger.info(
        "Fold %d late-fusion val metrics (n=%d, alpha=%.2f): AUROC=%.3f sensitivity=%.3f specificity=%.3f",
        fold, len(val_df), alpha, auroc, sensitivity, specificity,
    )
    return {
        "fold": fold,
        "n_val": len(val_df),
        "alpha": alpha,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def build_comparison_table(out_dir: Path) -> pd.DataFrame:
    """Combine Phase 4/5/6's per-fold metrics into the headline four-way
    comparison: mean +/- std per model across the 5 folds."""
    tables = {
        "imaging-only": pd.read_csv(out_dir / "mri_baseline_metrics.csv"),
        "ehr-only": pd.read_csv(out_dir / "ehr_baseline_metrics.csv"),
        "joint-fusion": pd.read_csv(out_dir / "fusion_metrics.csv"),
        "late-fusion": pd.read_csv(out_dir / "fusion_late_metrics.csv"),
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

    logger.info("=== Joint fusion (concatenated features) ===")
    joint_metrics = [run_fold(fold, mri_cohort, out_dir) for fold in available_folds]
    joint_df = pd.DataFrame(joint_metrics)
    logger.info("\n%s", joint_df.to_string(index=False))
    logger.info("Summary across folds:\n%s", joint_df[["auroc", "sensitivity", "specificity"]].agg(["mean", "std"]))
    joint_df.to_csv(out_dir / "fusion_metrics.csv", index=False)

    logger.info("=== Late fusion (weighted average of model probabilities) ===")
    oof_imaging_prob = build_oof_imaging_probs(out_dir, folds_df)
    late_metrics = [run_late_fusion_fold(fold, mri_cohort, out_dir, oof_imaging_prob) for fold in available_folds]
    late_df = pd.DataFrame(late_metrics)
    logger.info("\n%s", late_df.to_string(index=False))
    logger.info("Summary across folds:\n%s", late_df[["auroc", "sensitivity", "specificity"]].agg(["mean", "std"]))
    late_df.to_csv(out_dir / "fusion_late_metrics.csv", index=False)

    comparison = build_comparison_table(out_dir)
    logger.info(
        "\n=== Headline comparison: imaging-only vs EHR-only vs joint-fusion vs late-fusion ===\n%s",
        comparison.to_string(index=False),
    )
    comparison.to_csv(out_dir / "comparison_table.csv", index=False)


if __name__ == "__main__":
    main()
