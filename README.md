# Multimodal CVD Risk Prediction

Predicting cardiovascular disease (CVD) risk by fusing MRI brain imaging with structured EHR data, using MITRE's synthetic Coherent Dataset.

**Status:** early scaffolding — full writeup (problem statement, architecture, results, reproduction steps) lands in the final documentation phase.

## Known limitations (so far)

- **Imaging baseline was trained for only 5 epochs**, due to compute/time constraints -- 5 epochs already took ~6 hours on Kaggle's free-tier GPU (T4x2). (Colab's free tier was tried first but kept hitting quota walls before training could complete; Kaggle is where the run actually finished.) This yields a weaker-than-ideal standalone imaging AUROC (mean 0.551 across 5 folds, one fold near chance) compared to the EHR baseline (mean 0.674). Critically, this looks like undertraining rather than a lack of signal in the MRI data: the per-patient image embeddings themselves show real discriminative structure (individual embedding dimensions correlate with the stroke label up to 0.37), the linear classification head just didn't have enough training to fully exploit it. Documenting this as an honest, defensible compute tradeoff for a personal portfolio project, not a hidden flaw.
- **Simple linear fusion architectures can't discount a noisy branch.** See "Results so far" below -- neither fusion approach tried here beats the EHR-only baseline, and the reason is diagnosable, not mysterious.

## Results so far (Phases 4-6 baselines; full writeup pending Phase 9)

| Model | AUROC | Sensitivity | Specificity |
|---|---|---|---|
| Imaging-only | 0.551 ± 0.059 | 0.506 ± 0.317 | 0.577 ± 0.264 |
| EHR-only | **0.674 ± 0.065** | 0.631 ± 0.053 | 0.608 ± 0.079 |
| Joint fusion (concatenated features) | 0.599 ± 0.066 | 0.673 ± 0.074 | 0.469 ± 0.189 |
| Late fusion (tuned probability blend) | 0.658 ± 0.076 | 0.531 ± 0.281 | 0.715 ± 0.167 |

(mean ± std AUROC/sensitivity/specificity across the 5 CV folds; see `data/processed/comparison_table.csv`)

**Neither fusion approach beats the EHR-only baseline.** Given the imaging branch's known undertraining above, that's a legitimate, explainable result, not a bug -- and stating it plainly matters more here than a flattering headline would.

- **Joint fusion** (concatenating the 32-PCA-reduced image embedding with EHR features into one logistic regression) landed *below* EHR-only (0.599 vs. 0.674). A single linear classifier over concatenated features has no mechanism to selectively discount a noisy input -- every feature folds into one weight vector, so a weak imaging branch can drag down an otherwise-strong EHR signal instead of adding to it. This is a known limitation of simple/early fusion architectures in general, not something specific to this dataset.
- **Late fusion** (a weighted average of the imaging-only and EHR-only models' own predicted probabilities, `alpha * imaging_prob + (1 - alpha) * ehr_prob`, with alpha tuned via 5-fold CV on each fold's training patients only) got closer to EHR-only (0.658 vs. 0.674) and clearly beat joint fusion (0.658 vs. 0.599), because it *can* discount a noisy branch: fold 0's tuned weight came out to alpha=0.0, meaning the model correctly learned to ignore imaging entirely and exactly reproduced that fold's EHR-only result. But it didn't fully close the gap -- fold 1's tuned weight (alpha=0.90) leaned heavily on imaging despite that fold's imaging model being one of the weakest (AUROC ~0.52), dragging that fold's late-fusion AUROC down to 0.555. With only ~238 training patients per fold, the CV-based weight selection is itself noisy enough to occasionally pick badly.
- **Bottom line:** with an undertrained imaging branch, EHR-only is currently the strongest model. Late fusion is a meaningfully better fusion strategy than naive concatenation here, but neither fusion approach yet shows a lift over the best single modality -- a result that would plausibly change with a properly-trained imaging model (more epochs, more compute).

See `CLAUDE.md` for the full build plan.
