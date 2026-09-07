# Multimodal CVD Risk Prediction

Predicting cardiovascular disease (CVD) risk by fusing MRI brain imaging with structured EHR data, using MITRE's synthetic Coherent Dataset.

**Status:** early scaffolding — full writeup (problem statement, architecture, results, reproduction steps) lands in the final documentation phase.

## Known limitations (so far)

- **Imaging baseline was trained for only 5 epochs**, due to compute/time constraints -- 5 epochs already took ~6 hours on Kaggle's free-tier GPU (T4x2). (Colab's free tier was tried first but kept hitting quota walls before training could complete; Kaggle is where the run actually finished.) This yields a weaker-than-ideal standalone imaging AUROC (mean 0.557 across 5 folds, most folds hovering near chance) compared to the EHR baseline (mean 0.674). Critically, this looks like undertraining rather than a lack of signal in the MRI data: the per-patient image embeddings themselves show real discriminative structure (individual embedding dimensions correlate with the stroke label up to 0.37), the linear classification head just didn't have enough training to fully exploit it. Documenting this as an honest, defensible compute tradeoff for a personal portfolio project, not a hidden flaw. (Fold 0 was retrained once after the fact specifically to get a checkpoint for Grad-CAM -- see Phase 7 -- which is why its numbers differ slightly from the original run; every number in this section reflects that retrained fold 0.)
- **Joint fusion (naive concatenation) can't reliably discount a noisy branch; late fusion, correctly tuned, can and does.** See "Results so far" below.

## Results so far (Phases 4-6 baselines; full writeup pending Phase 9)

| Model | AUROC | Sensitivity | Specificity |
|---|---|---|---|
| Imaging-only | 0.557 ± 0.049 | 0.530 ± 0.349 | 0.538 ± 0.321 |
| EHR-only | 0.674 ± 0.065 | 0.631 ± 0.053 | 0.608 ± 0.079 |
| Joint fusion (concatenated features) | 0.621 ± 0.078 | 0.685 ± 0.079 | 0.492 ± 0.195 |
| Late fusion (tuned probability blend) | **0.672 ± 0.057** | 0.655 ± 0.118 | 0.646 ± 0.142 |

(mean ± std AUROC/sensitivity/specificity across the 5 CV folds; see `data/processed/comparison_table.csv`)

- **Joint fusion** (concatenating the 32-PCA-reduced image embedding with EHR features into one logistic regression) landed *below* EHR-only (0.621 vs. 0.674). A single linear classifier over concatenated features has no mechanism to selectively discount a noisy input -- every feature folds into one weight vector, so a weak imaging branch can drag down an otherwise-strong EHR signal instead of adding to it. This is a known limitation of simple/early fusion architectures in general, not something specific to this dataset.
- **Late fusion** (a weighted average of the imaging-only and EHR-only models' own predicted probabilities, `alpha * imaging_prob + (1 - alpha) * ehr_prob`, with alpha tuned via 5-fold CV on each fold's training patients) now lands at AUROC 0.672 -- **within 0.002 of EHR-only**, i.e. statistically indistinguishable given a std of ~0.06, and it edges out EHR-only on both sensitivity (0.655 vs. 0.631) and specificity (0.646 vs. 0.608). This is a corrected result: an earlier version of this tuning procedure had a methodological asymmetry -- the EHR side of the blend used genuine out-of-fold predictions (via `cross_val_predict`), but the imaging side reused each fold's own *in-sample* training-set probabilities, which look more discriminative than the model's true generalization since it's the same model scoring patients it was fit on. That bias pushed every fold's tuned weight toward over-trusting imaging (alphas of 0.80-0.90), dragging late fusion down to joint fusion's level (AUROC 0.620). Fixed by building a genuine out-of-fold imaging probability for every patient from Phase 4's own embeddings files -- each patient is the *val* patient of exactly one of the 5 outer folds, so that fold's prediction for them is already a legitimate holdout estimate, no CNN retraining required. With that fix, the tuned weights dropped to a far more sensible [0.35, 0.45, 0.25, 0.00, 0.30] -- correctly reflecting that imaging is the weaker branch -- and fold 3 tunes to alpha=0.00 (pure EHR) outright.
- **Bottom line:** late fusion, once correctly tuned, roughly matches the best single-modality baseline (EHR-only) rather than trailing it, and does so via a real mechanism (learned per-fold weights that mostly discount the weak imaging branch, occasionally to zero) rather than luck. It does not clearly *beat* EHR-only -- the ~0.002 AUROC gap is noise, not signal -- but it doesn't cost anything either, while gaining on sensitivity and specificity. Joint fusion remains the weaker of the two fusion approaches, for the structural reason described above. Both fusion results would plausibly improve further with a properly-trained (more epochs, more compute) imaging model.

See `CLAUDE.md` for the full build plan.
