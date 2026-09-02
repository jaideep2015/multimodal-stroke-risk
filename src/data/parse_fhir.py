"""
Phase 1: parse FHIR bundles from the Coherent dataset into one patient-level
dataframe: patient_id | tabular EHR fields | label | mri_file_path.

Uses plain dict/json traversal rather than the `fhir.resources` pydantic
models -- we only need a handful of fields per resource, and raw dicts are
more forgiving of the quirks synthetic FHIR data tends to have.

Not runnable yet: DATA_DIR is unset until the Coherent dataset is downloaded
(see README.md / CLAUDE.md for the download link). Two TODOs below
(_find_imaging_study_ref, build_patient_cohort's dicom_root) need to be
confirmed against the real folder layout once the data is available.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: set once the Coherent dataset has been downloaded, e.g.
# DATA_DIR = Path("D:/data/coherent-11-07-2022")
DATA_DIR: Optional[Path] = None

# LOINC codes for the CVD-relevant EHR fields we extract from Observation resources.
OBSERVATION_CODES = {
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "2093-3": "total_cholesterol",
    "2085-9": "hdl_cholesterol",
    "13457-7": "ldl_cholesterol",
    "39156-5": "bmi",
    "2339-0": "glucose",
    "72166-2": "smoking_status",
}

# Keywords matched against Condition.code.text / coding[].display to derive the
# CVD label. Deliberately excludes risk factors like hypertension (those are
# predictors, not the outcome) -- this targets actual cardiovascular events/disease.
CVD_KEYWORDS = [
    "coronary heart disease",
    "myocardial infarction",
    "heart failure",
    "cardiac arrest",
    "stroke",
    "atrial fibrillation",
    "peripheral vascular disease",
]


def _load_bundle(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resources_of_type(bundle: dict, resource_type: str) -> Iterator[dict]:
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            yield resource


def _latest_reference_date(bundle: dict) -> Optional[datetime]:
    """Age is computed relative to the patient's most recent recorded
    Encounter/Observation, not real wall-clock time -- Synthea patients live
    on synthetic historical timelines, not the actual calendar."""
    dates = []
    for resource_type in ("Encounter", "Observation"):
        for resource in _resources_of_type(bundle, resource_type):
            date_str = resource.get("period", {}).get("start") or resource.get("effectiveDateTime")
            if date_str:
                try:
                    dates.append(datetime.fromisoformat(date_str[:10]))
                except ValueError:
                    continue
    return max(dates) if dates else None


def _extract_demographics(patient: dict, reference_date: Optional[datetime]) -> Dict:
    birth_date_str = patient.get("birthDate")
    age = None
    if birth_date_str and reference_date:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        age = (reference_date - birth_date).days // 365
    return {"sex": patient.get("gender"), "age": age}


def _extract_observations(bundle: dict) -> Dict:
    fields = {}
    for obs in _resources_of_type(bundle, "Observation"):
        for coding in obs.get("code", {}).get("coding", []):
            field_name = OBSERVATION_CODES.get(coding.get("code"))
            if not field_name:
                continue
            if "valueQuantity" in obs:
                fields[field_name] = obs["valueQuantity"].get("value")
            elif "valueCodeableConcept" in obs:
                fields[field_name] = obs["valueCodeableConcept"].get("text")
    return fields


def _extract_cvd_label(bundle: dict) -> int:
    for condition in _resources_of_type(bundle, "Condition"):
        text = (condition.get("code", {}).get("text") or "").lower()
        displays = [c.get("display", "").lower() for c in condition.get("code", {}).get("coding", [])]
        haystack = " ".join([text] + displays)
        if any(keyword in haystack for keyword in CVD_KEYWORDS):
            return 1
    return 0


def _find_imaging_study_ref(bundle: dict) -> Optional[str]:
    studies = list(_resources_of_type(bundle, "ImagingStudy"))
    if not studies:
        return None
    # TODO: verify against real data -- likely identifier[].value or
    # series[].uid links to the DICOM folder for this study.
    identifiers = studies[0].get("identifier", [])
    return identifiers[0].get("value") if identifiers else None


def _find_mri_path(imaging_study_ref: Optional[str], dicom_root: Path) -> Optional[str]:
    if not imaging_study_ref or not dicom_root.exists():
        return None
    matches = list(dicom_root.rglob(f"*{imaging_study_ref}*"))
    return str(matches[0]) if matches else None


def parse_patient_bundle(path: Path, dicom_root: Path) -> Optional[Dict]:
    bundle = _load_bundle(path)
    patients = list(_resources_of_type(bundle, "Patient"))
    if not patients:
        logger.warning("No Patient resource found in %s, skipping", path)
        return None
    patient = patients[0]

    reference_date = _latest_reference_date(bundle)
    record = {"patient_id": patient.get("id")}
    record.update(_extract_demographics(patient, reference_date))
    record.update(_extract_observations(bundle))
    record["label"] = _extract_cvd_label(bundle)
    imaging_ref = _find_imaging_study_ref(bundle)
    record["mri_file_path"] = _find_mri_path(imaging_ref, dicom_root)
    return record


def build_patient_cohort(data_dir: Path) -> pd.DataFrame:
    fhir_dir = data_dir / "fhir"
    dicom_root = data_dir / "dicom"  # TODO: confirm actual folder name once dataset is downloaded
    bundle_paths = sorted(fhir_dir.glob("*.json"))
    logger.info("Found %d FHIR bundles in %s", len(bundle_paths), fhir_dir)

    records = [parse_patient_bundle(path, dicom_root) for path in bundle_paths]
    df = pd.DataFrame([r for r in records if r is not None])
    logger.info("Built cohort dataframe: %d patients, %d columns", *df.shape)
    return df


if __name__ == "__main__":
    if DATA_DIR is None:
        raise SystemExit(
            "DATA_DIR is not set. Download the Coherent dataset from "
            "https://synthea.mitre.org/downloads, then set DATA_DIR to its "
            "local path at the top of this file before running it."
        )
    cohort_df = build_patient_cohort(DATA_DIR)
    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "patient_cohort.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(out_path, index=False)
    logger.info("Saved patient cohort to %s", out_path)
