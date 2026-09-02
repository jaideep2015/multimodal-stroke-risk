# Multimodal CVD Risk Prediction — Project Build Plan

## Objective
Build a portfolio-ready multimodal deep learning project: predict cardiovascular disease (CVD) risk by fusing **MRI brain imaging** with **structured EHR (electronic health record) data**. This is for a CV/GitHub portfolio to demonstrate healthcare/pharma domain knowledge, coming from a credit-risk/scorecard modeling background (no prior deep learning experience — explain concepts as you go, don't assume familiarity with CNNs or PyTorch internals).

## Reference repos (study for approach — do not blindly copy/paste; the code here has real bugs and gaps you'll need to fix)
- `https://github.com/multimodal-healthcare/mri` — DICOM processing + CNN training code. Known issues to fix: `resnet18()` is initialized from scratch, not pretrained; `model.py`'s inference function is a hardcoded stub, not real inference.
- `https://github.com/multimodal-healthcare/ehr` — essentially empty (README + license only). This is the piece we're building from scratch.
- `https://github.com/multimodal-healthcare/fusion` — has two notebooks (`01_coherent_clean.ipynb`, `02_fusion_models.ipynb`) showing the intended data-cleaning and fusion approach. Use as a rough template.
- **Skip**: `https://github.com/multimodal-healthcare/genomics` — requires UK Biobank access (formal application, not obtainable for a personal project).
- **Optional stretch, later only**: `ecg`, `Clinical-Notes-and-DNA` repos in the same org.

## Dataset
- Source: MITRE's **Coherent Dataset** — public, synthetic, no application required. Download page: `https://synthea.mitre.org/downloads`
- ~9 GB. Contains FHIR-linked synthetic patient records: DICOM MRI scans, structured EHR (conditions, observations, medications, vitals, demographics), physiological waveforms, clinical notes, and genomic data.
- All modalities for the same synthetic patient are linked via FHIR resources: `Observation`, `Procedure`, `ImagingStudy`, `DiagnosticReport`, `DocumentReference`, `Binary`. Same patient ID ties every modality together — that's what makes this a real multimodal dataset rather than several unrelated ones.
- **Action before Phase 1**: user will download the dataset locally and provide the local path — do not attempt to fetch it automatically.

## Target architecture (joint/intermediate fusion)
```
MRI scan  → CNN encoder     → image embedding  ┐
                                                  ├→ concatenate → classifier head → CVD risk score
EHR record → feature engineering → feature vector ┘
```
Each modality gets its own encoder first; their outputs are combined only at the end, before the final prediction layer. This is the standard "joint fusion" pattern — name it explicitly in the README and any interview prep.

## Success criteria
- End-to-end pipeline: raw data → preprocessed → trained fused model → evaluation → deployed demo
- A results table comparing: imaging-only baseline vs. EHR-only baseline vs. fused model (this comparison is the single most compelling result for a CV/README — it's the direct evidence that fusion helped)
- README covering: problem statement, dataset (explicitly synthetic — no PHI, no compliance concerns, say so plainly), architecture, results, limitations, how to reproduce
- Live demo (Gradio on Hugging Face Spaces, free tier)
- Clean repo: sensible commit history, `.gitignore` for data/checkpoints, no large binaries committed

## Phase-by-phase plan

### Phase 0 — Scaffolding
- Init git repo. Folder structure:
  ```
  data/           (gitignored)
  notebooks/
  src/data/
  src/models/
  src/fusion/
  src/eval/
  app/
  README.md
  requirements.txt
  .gitignore
  ```
- Core deps: `torch`, `torchvision`, `pytorch-lightning`, `pandas`, `numpy`, `scikit-learn`, `pydicom`, `fhir.resources` (or `fhirclient`), `matplotlib`, `gradio`, `pytorch-grad-cam`.

### Phase 1 — Data ingestion & EDA
- Parse FHIR bundles from the local Coherent dataset path to extract:
  - Patient demographics (age, sex, etc.)
  - CVD-relevant `Observation` fields (blood pressure, cholesterol, BMI, glucose, smoking status)
  - `Condition` resources to derive the CVD label
  - `ImagingStudy` references linking each patient to their MRI DICOM file
- Build one patient-level dataframe: `patient_id | tabular EHR fields | label | mri_file_path`
- EDA: class balance of the label (expect imbalance — plan for it), feature correlations, sample MRI slice visualization
- Deliverable: `notebooks/01_data_exploration.ipynb`, `data/processed/patient_cohort.csv`

### Phase 2 — MRI preprocessing (reference: `mri` repo's `data_process` notebooks)
- Convert DICOM volumes to usable slices; drop noisy slices (repo notes: roughly indices 0–49 and 205–255 are noise-only)
- Build a PyTorch `Dataset`/`DataLoader` for the image branch
- Deliverable: `src/data/mri_dataset.py`

### Phase 3 — Tabular EHR preprocessing
- Standard feature engineering: missing-value handling, categorical encoding, scaling
- This step leans directly on scorecard/credit-risk feature-engineering experience — treat it as familiar territory, not new territory
- Deliverable: `src/data/ehr_features.py`

### Phase 4 — Imaging model (CNN branch)
- Pretrained ResNet18/34 (`weights='IMAGENET1K_V1'`), fine-tuned, with augmentation (flips/rotation/intensity jitter) and class-weighted loss for imbalance
- Train standalone first as an imaging-only baseline; evaluate with AUROC/sensitivity/specificity, not just accuracy
- Extract the penultimate layer as the image embedding for later fusion
- Deliverable: `src/models/mri_model.py`, baseline checkpoint + metrics

### Phase 5 — Tabular model (EHR branch, baseline)
- Baseline tabular model (logistic regression or gradient boosting) trained standalone, same metrics, as the second comparison baseline
- Deliverable: `src/models/ehr_model.py`, baseline metrics

### Phase 6 — Fusion model (reference: `fusion` repo's `02_fusion_models.ipynb`)
- Concatenate the CNN image embedding with the tabular feature vector (or a small MLP-encoded version of it) → final classifier head
- Train end-to-end, or with a frozen CNN encoder + trainable fusion head if compute is limited
- Compare fused metrics against both standalone baselines from Phases 4–5 — this comparison table is the headline result
- Deliverable: `src/fusion/fusion_model.py`, comparison table (imaging-only vs. EHR-only vs. fused)

### Phase 7 — Explainability
- Grad-CAM on the CNN branch — visualize which brain regions drove each prediction
- Basic feature importance (coefficients, or SHAP) on the tabular branch
- Deliverable: `notebooks/03_explainability.ipynb`

### Phase 8 — Deployment
- Fix the inference path properly (the reference repo's version is a hardcoded stub — replace with real fused-model inference)
- Gradio app: input = MRI slice + short clinical form; output = risk score + Grad-CAM overlay
- Deploy free on Hugging Face Spaces
- Deliverable: `app/gradio_app.py`, live demo link

### Phase 9 — Documentation & polish
- README: problem statement, dataset (synthetic, no PHI), architecture diagram, results table, limitations (synthetic data, small sample, not clinically validated), reproduction steps
- Clean commit history, license, a short project summary suitable for a CV bullet / LinkedIn post

## Constraints
- Never commit the raw Coherent dataset or large binaries/checkpoints to git — `.gitignore` them, document the download step in the README instead
- Get Phases 1–6 working end-to-end (even roughly) before polishing any single phase — a complete simple pipeline beats a perfect but unfinished one
- Explain deep learning concepts (CNNs, embeddings, fusion, Grad-CAM) in plain terms as they come up — this project doubles as the user's first real deep learning learning experience
