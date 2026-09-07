"""
Phase 8: real fused-model inference for the deployed demo. This is the
fix for the pattern CLAUDE.md flagged in the reference `mri` repo --
`model.py`'s inference function was a hardcoded stub that ignored its
input and returned a fixed string. Here, an uploaded MRI slice and a
short clinical form actually drive the CNN and EHR models respectively,
and the two get blended into one real risk score.

Uses LATE fusion (not joint), matching Phase 6's corrected finding: late
fusion roughly matches EHR-only (AUROC 0.672 vs. 0.674) while joint
fusion trails behind (0.621) -- the stronger of the two approaches is
what gets deployed.

"Final" model choices -- Phases 4-6 only ever produced 5 fold-specific
models for cross-validation, never one production model, so this phase
has to make that call:
  - CNN: reuses fold 0's checkpoint (data/processed/checkpoints/
    mri_fold0.pt), the only one available locally (checkpoints are
    gitignored by default; fold 0 was the one deliberately committed
    for Grad-CAM in Phase 7). Training a CNN on all 298 patients for a
    "true" final model would need Kaggle GPU time again -- reusing
    fold 0 here is an explicit, documented tradeoff, not an oversight.
  - EHR model: refit fresh on ALL 298 patients (cheap, CPU-only, no
    tradeoff needed) -- strictly more data than any single fold's ~238
    training patients saw.
  - Alpha: retuned on ALL 298 patients' out-of-fold predictions, the
    same build_oof_imaging_probs()-based approach fusion_model.py uses
    per fold, just applied to the whole cohort -- one final production
    weight instead of 5 fold-specific ones.

All three are computed once at import time and cached at module level --
cheap enough (EHR fit: milliseconds; alpha tuning: a couple seconds; CNN
checkpoint load: a couple seconds) to just do on app startup rather than
persisting a separate serialized artifact.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import: fusion_model.py
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
from ehr_features import build_ehr_pipeline  # noqa: E402
from folds import load_patient_folds  # noqa: E402
from fusion_model import _ehr_pipeline_with_classifier, build_oof_imaging_probs  # noqa: E402
from mri_dataset import IMAGENET_MEAN, IMAGENET_STD, RESIZE_TO  # noqa: E402
from mri_model import MRIClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = REPO_ROOT / "data" / "processed"
RANDOM_SEED = 42

NUMERIC_FIELDS = ["age", "systolic_bp", "diastolic_bp", "total_cholesterol", "hdl_cholesterol", "bmi", "glucose"]
CATEGORICAL_FIELDS = ["sex", "smoking_status"]


class LogitOnly(nn.Module):
    """Same Grad-CAM wrapper as notebooks/03_explainability.ipynb --
    pytorch-grad-cam expects output it can index like class scores, but
    MRIClassifier.forward() returns (logit, embedding), a tuple."""

    def __init__(self, mri_classifier: MRIClassifier):
        super().__init__()
        self.mri_classifier = mri_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit, _ = self.mri_classifier(x)
        return logit.unsqueeze(-1)


def _load_cnn() -> MRIClassifier:
    model = MRIClassifier(arch="resnet18")
    state_dict = torch.load(DATA_DIR / "checkpoints" / "mri_fold0.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def _fit_final_ehr_model(cohort: pd.DataFrame):
    pipeline = build_ehr_pipeline()
    X = pipeline.fit_transform(cohort)
    y = cohort["label"].values
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X, y)
    return pipeline, clf


def _tune_final_alpha(cohort: pd.DataFrame, oof_imaging_prob: pd.Series) -> float:
    """Same approach as fusion_model.tune_late_fusion_weight, applied to
    the full 298-patient cohort instead of one fold's train patients --
    one final production weight instead of 5 fold-specific ones."""
    y = cohort["label"].values
    imaging_prob = cohort["patient_id"].map(oof_imaging_prob).values
    oof_ehr_prob = cross_val_predict(
        _ehr_pipeline_with_classifier(), cohort, y, cv=5, method="predict_proba"
    )[:, 1]

    best_alpha, best_auroc = 0.0, -np.inf
    for alpha in np.linspace(0, 1, 21):
        blended = alpha * imaging_prob + (1 - alpha) * oof_ehr_prob
        auroc = roc_auc_score(y, blended)
        if auroc > best_auroc:
            best_alpha, best_auroc = alpha, auroc
    return best_alpha


def _initialize():
    cohort = pd.read_csv(DATA_DIR / "patient_cohort.csv")
    folds_df = load_patient_folds(DATA_DIR / "patient_folds.csv")
    mri_cohort = cohort[cohort["patient_id"].isin(folds_df["patient_id"])].reset_index(drop=True)

    logger.info("Loading fold 0 CNN checkpoint...")
    cnn = _load_cnn()
    logger.info("Fitting final EHR model on all %d patients...", len(mri_cohort))
    ehr_pipeline, ehr_clf = _fit_final_ehr_model(mri_cohort)
    logger.info("Tuning final late-fusion alpha...")
    oof_imaging_prob = build_oof_imaging_probs(DATA_DIR, folds_df)
    alpha = _tune_final_alpha(mri_cohort, oof_imaging_prob)
    logger.info("Final late-fusion alpha (tuned on all 298 patients): %.2f", alpha)
    return cnn, ehr_pipeline, ehr_clf, alpha


CNN_MODEL, EHR_PIPELINE, EHR_CLASSIFIER, ALPHA = _initialize()


def _normalize_and_resize(image: np.ndarray) -> torch.Tensor:
    """Grayscale-or-RGB uploaded image -> (3, RESIZE_TO, RESIZE_TO)
    tensor, scaled to [0,1] by its own max and replicated to 3 channels
    -- the shared first step for both the model input (which then also
    gets ImageNet-normalized) and the Grad-CAM display background
    (which doesn't). Mirrors mri_dataset.slice_to_tensor(), rewritten
    here for a general uploaded image rather than a raw DICOM pixel
    array."""
    if image.ndim == 3:
        image = image.mean(axis=2)
    image = image.astype(np.float32)
    peak = image.max()
    normalized = image / peak if peak > 0 else image
    tensor = torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1)
    return TF.resize(tensor, [RESIZE_TO, RESIZE_TO], antialias=True)


def preprocess_slice_image(image: np.ndarray) -> torch.Tensor:
    tensor = _normalize_and_resize(image)
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _display_image(image: np.ndarray) -> np.ndarray:
    return _normalize_and_resize(image).permute(1, 2, 0).numpy()


def imaging_probability(image: np.ndarray) -> float:
    tensor = preprocess_slice_image(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit, _ = CNN_MODEL(tensor)
        return torch.sigmoid(logit).item()


def ehr_probability(clinical_fields: dict) -> float:
    row = pd.DataFrame([clinical_fields])
    X = EHR_PIPELINE.transform(row)
    return EHR_CLASSIFIER.predict_proba(X)[:, 1].item()


def grad_cam_overlay(image: np.ndarray) -> np.ndarray:
    tensor = preprocess_slice_image(image).unsqueeze(0).to(DEVICE)
    wrapped_model = LogitOnly(CNN_MODEL)
    cam = GradCAM(model=wrapped_model, target_layers=[CNN_MODEL.backbone.layer4[-1]])
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
    return show_cam_on_image(_display_image(image), grayscale_cam, use_rgb=True)


def predict_risk(image: np.ndarray, clinical_fields: dict) -> dict:
    """The real fused-model inference the reference repo's stub never
    did: MRI slice -> CNN -> imaging probability, clinical form -> EHR
    pipeline -> EHR probability, blended by the tuned ALPHA -> one risk
    score, plus the Grad-CAM overlay for the same slice."""
    img_prob = imaging_probability(image)
    ehr_prob = ehr_probability(clinical_fields)
    risk_score = ALPHA * img_prob + (1 - ALPHA) * ehr_prob
    return {
        "risk_score": risk_score,
        "imaging_probability": img_prob,
        "ehr_probability": ehr_prob,
        "alpha": ALPHA,
        "grad_cam_overlay": grad_cam_overlay(image),
    }
