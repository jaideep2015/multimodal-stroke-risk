# Multimodal CVD Risk Prediction

Predicting cardiovascular disease (CVD) risk by fusing MRI brain imaging with structured EHR data, using MITRE's synthetic Coherent Dataset.

**Status:** early scaffolding — full writeup (problem statement, architecture, results, reproduction steps) lands in the final documentation phase.

## Known limitations (so far)

- **Imaging baseline was trained for only 5 epochs**, due to compute/time constraints -- 5 epochs already took ~6 hours on Kaggle's free-tier GPU (T4x2). (Colab's free tier was tried first but kept hitting quota walls before training could complete; Kaggle is where the run actually finished.) This yields a weaker-than-ideal standalone imaging AUROC (mean 0.557 across 5 folds, most folds hovering near chance) compared to the EHR baseline (mean 0.674). Critically, this looks like undertraining rather than a lack of signal in the MRI data: the per-patient image embeddings themselves show real discriminative structure (individual embedding dimensions correlate with the stroke label up to 0.37), the linear classification head just didn't have enough training to fully exploit it. Documenting this as an honest, defensible compute tradeoff for a personal portfolio project, not a hidden flaw. (Fold 0 was retrained once after the fact specifically to get a checkpoint for Grad-CAM -- see Phase 7 -- which is why its numbers differ slightly from the original run; every number in this section reflects that retrained fold 0.)
- **Simple fusion architectures can't reliably discount a noisy branch.** See "Results so far" below -- neither fusion approach tried here beats the EHR-only baseline, and the reasons are diagnosable, not mysterious.

## Results so far (Phases 4-6 baselines; full writeup pending Phase 9)

| Model | AUROC | Sensitivity | Specificity |
|---|---|---|---|
| Imaging-only | 0.557 ± 0.049 | 0.530 ± 0.349 | 0.538 ± 0.321 |
| EHR-only | **0.674 ± 0.065** | 0.631 ± 0.053 | 0.608 ± 0.079 |
| Joint fusion (concatenated features) | 0.621 ± 0.078 | 0.685 ± 0.079 | 0.492 ± 0.195 |
| Late fusion (tuned probability blend) | 0.620 ± 0.052 | 0.589 ± 0.349 | 0.615 ± 0.289 |

(mean ± std AUROC/sensitivity/specificity across the 5 CV folds; see `data/processed/comparison_table.csv`)

**Neither fusion approach beats the EHR-only baseline.** Given the imaging branch's known undertraining above, that's a legitimate, explainable result, not a bug -- and stating it plainly matters more here than a flattering headline would.

- **Joint fusion** (concatenating the 32-PCA-reduced image embedding with EHR features into one logistic regression) landed *below* EHR-only (0.621 vs. 0.674). A single linear classifier over concatenated features has no mechanism to selectively discount a noisy input -- every feature folds into one weight vector, so a weak imaging branch can drag down an otherwise-strong EHR signal instead of adding to it. This is a known limitation of simple/early fusion architectures in general, not something specific to this dataset.
- **Late fusion** (a weighted average of the imaging-only and EHR-only models' own predicted probabilities, `alpha * imaging_prob + (1 - alpha) * ehr_prob`, with alpha tuned via 5-fold CV on each fold's training patients only) landed at essentially the same AUROC as joint fusion (0.620 vs. 0.621) -- both below EHR-only. Per-fold tuned weights: [0.80, 0.90, 0.45, 0.80, 0.90]. The mechanism to discount a noisy branch does exist (alpha can land anywhere from 0 to 1), but in practice every fold's tuned weight leaned toward trusting imaging *more* than its true held-out performance would justify -- e.g. fold 0's alpha=0.80 despite that fold's imaging model being one of the weakest (AUROC 0.502, barely above chance), which dragged that fold's late-fusion AUROC down to 0.577. The likely reason is a methodological asymmetry in how alpha gets tuned: the EHR side of the tuning uses genuine out-of-fold predictions (via `cross_val_predict`, refit per inner fold), but the imaging side reuses the same model's *in-sample* training-set probabilities (retraining the CNN inside an inner CV loop would need GPU time this project doesn't have to spare). In-sample predictions from the same model that was fit on those exact patients can look more discriminative than the model's true generalization -- so the tuning procedure is biased toward over-trusting imaging. This is a real limitation of the late-fusion setup as implemented, not fixed here.
- **Bottom line:** with an undertrained imaging branch, EHR-only is currently the strongest model. Neither fusion approach shows a lift over the best single modality, and late fusion's specific mechanism for handling this (discount the noisy branch) is undermined by an optimism bias in how its blend weight gets tuned. Both results would plausibly look different with a properly-trained imaging model (more epochs, more compute) and, for late fusion specifically, a tuning scheme that doesn't lean on in-sample imaging predictions.

See `CLAUDE.md` for the full build plan.
