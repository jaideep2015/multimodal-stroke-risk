"""
Phase 1: parse FHIR bundles from the Coherent dataset into one patient-level
dataframe: patient_id | tabular EHR fields | label | mri_file_path.

Uses plain dict/json traversal rather than the `fhir.resources` pydantic
models -- we only need a handful of fields per resource, and raw dicts are
more forgiving of the quirks synthetic FHIR data tends to have.

Confirmed against the real Coherent download: FHIR bundles live in
`<DATA_DIR>/fhir/*.json`, brain MRI DICOMs in `<DATA_DIR>/dicom/*.dcm`. MRI
linkage is done by matching the patient's FHIR id as a substring of the
DICOM filename (verified unique across all 298 DICOMs vs. 1280 patients --
no need to go through the ImagingStudy resource at all).
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR: Optional[Path] = Path(__file__).resolve().parents[2] / "data" / "raw" / "coherent"

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
# CVD label. Scoped to stroke/cerebrovascular disease specifically, not the
# broader CVD basket (AFib, coronary heart disease, heart failure, etc.) --
# the only imaging modality here is brain MRI, which has no real signal for
# cardiac-only conditions, so a broader label would leave the imaging branch
# with little to contribute and undercut the fusion-vs-baseline comparison.
CVD_KEYWORDS = [
    "stroke",
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


def _matching_field(coding_holder: dict) -> Optional[str]:
    for coding in coding_holder.get("code", {}).get("coding", []):
        field_name = OBSERVATION_CODES.get(coding.get("code"))
        if field_name:
            return field_name
    return None


def _value_of(value_holder: dict):
    if "valueQuantity" in value_holder:
        return value_holder["valueQuantity"].get("value")
    if "valueCodeableConcept" in value_holder:
        return value_holder["valueCodeableConcept"].get("text")
    return None


def _extract_observations(bundle: dict) -> Dict:
    """Most vitals/labs are standalone Observations, but blood pressure is a
    panel (code 85354-9) whose systolic/diastolic values live inside a
    `component[]` array instead of the top-level valueQuantity -- so both
    the observation itself and its components need checking."""
    fields = {}
    for obs in _resources_of_type(bundle, "Observation"):
        field_name = _matching_field(obs)
        if field_name:
            fields[field_name] = _value_of(obs)
        for component in obs.get("component", []):
            field_name = _matching_field(component)
            if field_name:
                fields[field_name] = _value_of(component)
    return fields


def _extract_cvd_label(bundle: dict) -> int:
    for condition in _resources_of_type(bundle, "Condition"):
        text = (condition.get("code", {}).get("text") or "").lower()
        displays = [c.get("display", "").lower() for c in condition.get("code", {}).get("coding", [])]
        haystack = " ".join([text] + displays)
        if any(keyword in haystack for keyword in CVD_KEYWORDS):
            return 1
    return 0


def _find_mri_path(patient_id: str, dicom_root: Path) -> Optional[str]:
    if not dicom_root.exists():
        return None
    matches = list(dicom_root.glob(f"*{patient_id}*"))
    return str(matches[0]) if matches else None


def parse_patient_bundle(path: Path, dicom_root: Path) -> Optional[Dict]:
    bundle = _load_bundle(path)
    patients = list(_resources_of_type(bundle, "Patient"))
    if not patients:
        logger.warning("No Patient resource found in %s, skipping", path)
        return None
    patient = patients[0]
    patient_id = patient.get("id")

    reference_date = _latest_reference_date(bundle)
    record = {"patient_id": patient_id}
    record.update(_extract_demographics(patient, reference_date))
    record.update(_extract_observations(bundle))
    record["label"] = _extract_cvd_label(bundle)
    record["mri_file_path"] = _find_mri_path(patient_id, dicom_root)
    return record


def build_patient_cohort(data_dir: Path) -> pd.DataFrame:
    fhir_dir = data_dir / "fhir"
    dicom_root = data_dir / "dicom"
    bundle_paths = sorted(fhir_dir.glob("*.json"))
    logger.info("Found %d FHIR bundles in %s", len(bundle_paths), fhir_dir)

    records = [parse_patient_bundle(path, dicom_root) for path in bundle_paths]
    df = pd.DataFrame([r for r in records if r is not None])
    logger.info("Built cohort dataframe: %d patients, %d columns", *df.shape)
    return df


if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise SystemExit(
            f"DATA_DIR does not exist: {DATA_DIR}. Download the Coherent "
            "dataset from https://synthea.mitre.org/downloads and extract it "
            "there, or update DATA_DIR at the top of this file."
        )
    cohort_df = build_patient_cohort(DATA_DIR)
    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "patient_cohort.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(out_path, index=False)
    logger.info("Saved patient cohort to %s", out_path)
