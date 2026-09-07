# Multimodal CVD Risk Prediction

Predicting cardiovascular disease (CVD) risk by fusing MRI brain imaging with structured EHR data, using MITRE's synthetic Coherent Dataset.

**Status:** early scaffolding — full writeup (problem statement, architecture, results, reproduction steps) lands in the final documentation phase.

## Known limitations (so far)

- **Imaging baseline was trained for only 5 epochs**, due to compute/time constraints -- 5 epochs already took ~6 hours on Kaggle's free-tier GPU (T4x2). (Colab's free tier was tried first but kept hitting quota walls before training could complete; Kaggle is where the run actually finished.) This yields a weaker-than-ideal standalone imaging AUROC (mean 0.551 across 5 folds, one fold near chance) compared to the EHR baseline (mean 0.674). Critically, this looks like undertraining rather than a lack of signal in the MRI data: the per-patient image embeddings themselves show real discriminative structure (individual embedding dimensions correlate with the stroke label up to 0.37), the linear classification head just didn't have enough training to fully exploit it. Documenting this as an honest, defensible compute tradeoff for a personal portfolio project, not a hidden flaw.

See `CLAUDE.md` for the full build plan.
