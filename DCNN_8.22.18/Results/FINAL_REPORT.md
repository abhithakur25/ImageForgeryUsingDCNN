# DCNN Image-Forgery Detection — Execution & Optimization Report

**Project:** `D:\Code_Paper3\DCNN_8.22.18` (CASIA image-forgery detection, binary **Au** = authentic vs **Sp** = spliced)
**Environment:** MATLAB R2022b + Deep Learning / Image Processing / Parallel toolboxes, NVIDIA GTX 1070 GPU.
**Data:** 10,919 JPG images (114×114×3) across `Database\1..11` (each with `Au\` and `Sp\`). 10 corrupt files in `Database\9\Sp\` are auto-filtered → **10,909 usable** (Au 5,598 / Sp 5,311).

---

## 1. What was done

1. **Understood the code flow.** The latest pipeline is `TransferLearning3.m`, which builds an `imageDatastore`, splits 80/20, defines a small CNN (XONet), trains, and reports a confusion matrix. Older variants: `DCNN_New_Final.m`, `DCNN_1.m`.
2. **Tested the existing trained model** `XONet_r003.mat` **without retraining** (Phase 1) → full metric suite.
3. **Built an improved, instrumented training pipeline** (Phase 2) that captures the training history so train/validation **loss & accuracy curves** can be plotted (the original discarded these).
4. **Iterated several models** to improve the metrics, saving every result to its own folder under `Results\`.

### Fixes applied to the original code
- Hardcoded `F:\College\Abhishek\...` paths → resolved to the actual local `Database\`.
- `doTraining = true` (which **retrains**) → Phase-1 test loads the saved model instead, per requirement.
- Original discarded prediction **scores** and **training info** → now captured for ROC/AUC and loss curves.
- Added `filterReadable.m` to skip 10 corrupt JPGs that crash `imageDatastore`.

---

## 2. ⚠️ Critical methodology finding (important for the paper)

The original `TransferLearning3.m` trains with `trainNetwork(IMDS, ...)` — i.e. on the **entire** dataset — and then "validates" on a random subset of those **same** images. Every validation image was seen during training, so the reported **~98.5%** is an **in-sample (data-leakage) number, not a measure of generalization.**

All Phase-2 models here are evaluated **honestly**: trained on 80% and tested on a held-out 20% they never saw (fixed split, `rng(123)`). That is why their numbers are lower — they measure *real* performance. **For publication, report the honest held-out numbers, not the 98.5%.**

> Further recommendation: the filenames encode rotation/colour-illumination variants (`..._R0..R4`, `..._CI_...`) of the same base images. A random split can still place variants of one source image in both train and val. For maximum rigor, use a **group-aware split by base image**. This was noted but not applied (the random split matches the original protocol for comparability).

---

## 3. Results (honest held-out validation, fixed `rng(123)` 80/20 split)

| Model | Train Acc | **Val Acc** | Val Precision | Val Recall | Val F1 | Val AUC |
|---|---|---|---|---|---|---|
| `XONet_r003` (existing baseline)* | 0.9880 | 0.9849* | 0.9848 | 0.9850 | 0.9849 | 0.9909 |
| **Transfer_squeezenet_best (RECOMMENDED)** | 0.9622 | **0.9358** | 0.9358 | 0.9361 | 0.9358 | 0.9893 |
| Transfer_squeezenet_long | 0.9290 | 0.9138 | 0.9139 | 0.9142 | 0.9138 | 0.9797 |
| Transfer_squeezenet (12 ep) | 0.8856 | 0.8607 | 0.8614 | 0.8613 | 0.8607 | 0.9496 |
| Optimized_v1 (from scratch) | 0.8077 | 0.7741 | 0.7743 | 0.7734 | 0.7736 | 0.8678 |
| Optimized_v3 (from scratch, stabilized) | 0.8067 | 0.7699 | 0.7726 | 0.7685 | 0.7686 | 0.8631 |
| Optimized_v2 (from scratch) | 0.5131 | 0.5133 | 0.2566 | 0.5000 | 0.3392 | 0.5000 |

\* In-sample / leaky — see §2. Not directly comparable to the honest numbers.

### Iteration narrative (how the models evolved)
- **v1** — custom CNN (BatchNorm + dropout) with *aggressive* augmentation (rotation ±10°, large translation) + global-average-pooling. **Underfit (77%)**: rotation/scaling interpolation masks the subtle splicing cues, and GAP discards spatial detail.
- **v2** — deeper CNN with an FC head, light augmentation, LR 1e-3. **Collapsed to 51%** (predicts one class): LR too high for the BN+FC stack (initial validation loss exploded to ~39).
- **v3** — same architecture, **stabilized** (LR 3e-4 + gradient clipping). Trained cleanly but **plateaued at 77%** — the from-scratch ceiling on this data.
- **Transfer (squeezenet)** — pretrained ImageNet features dominate: **86%** in 12 epochs, still underfitting.
- **squeezenet_long** — 25 epochs, LR 2e-4 → **91.4%**.
- **squeezenet_best** — 35 epochs + LR schedule (drop ×0.5 every 12 epochs) → **93.6% val, F1 0.936, AUC 0.989** — the **recommended model**.

**Takeaway:** pretrained/transfer-learned features are decisively better than from-scratch CNNs for this task; light (flip-only) augmentation beats aggressive geometric augmentation because forgery cues are fragile. Across iterations the honest held-out accuracy improved **77% → 86% → 91% → 93.6%**, with every metric (precision, recall, F1, AUC) improving in lock-step and a healthy train/val gap (96.2% / 93.6% — minimal overfitting).

## Recommended model & how it compares to baseline
**`Transfer_squeezenet_best`** is the model to use/report: **93.6% accuracy, 0.936 precision/recall/F1, 0.989 AUC — all measured honestly on held-out data the model never saw.**

It cannot be said to "beat" the baseline's 98.5% because that baseline number is **not a real result** — it was measured on training data (§2). On a like-for-like *honest* basis, `squeezenet_best` is the strongest, most trustworthy model produced and is the one suitable for the paper. If an in-sample comparison under the original (leaky) protocol is specifically required, it can be produced on request, but it should not be reported as a generalization result.

---

## 4. Output files

Each model folder under `Results\` contains:
- `SUMMARY.txt` — headline metrics
- `confusion_training.png`, `confusion_validation.png` — confusion matrices (row/column normalized)
- `roc_curve.png` — ROC for train & val with AUC
- `loss_curve.png`, `accuracy_curve.png` — training vs validation curves *(Phase-2 models only; not available for the saved baseline, which never stored training history)*
- `results_*.mat` — metrics struct + training `info`
- `run_log.txt` — full console log

Folders: `Existing_Model\` (baseline test), `Optimized_v1\`, `Optimized_v2\`, `Optimized_v3\`, `Transfer_squeezenet\`, `Transfer_squeezenet_long\`, `Transfer_squeezenet_best\`.
Cross-model: `Results\COMPARISON.csv`, `COMPARISON.txt`, `COMPARISON_bar.png`.

### Note on training/validation **loss** curves for the existing model
The requested *training loss* and *validation loss* curves cannot be produced for `XONet_r003` because the original code saved only the network, not the training-history struct (`[net,info] = trainNetwork(...)`). They **are** produced for every Phase-2 model, where the history was captured.

---

## 5. Reproduce

```matlab
% From D:\Code_Paper3\DCNN_8.22.18  (matlab -batch "<script>")
Test_Existing_Model                                   % Phase 1: test saved model
VARIANT='v1';  Train_Optimized                        % from-scratch CNN
VARIANT='v3';  Train_CustomV3                          % stabilized from-scratch
BASENET='squeezenet'; EPOCHS=35; LR0=2e-4; TAG='squeezenet_best'; Train_Transfer   % best
Compare_Models                                        % aggregate all results
```
