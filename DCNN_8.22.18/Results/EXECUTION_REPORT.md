# Image-Forgery Detection (CASIA) — Complete Execution Report

**Task:** Binary image-splicing detection — classify each JPEG as **`Au`** (authentic) or
**`Sp`** (spliced/forged).
**Dataset:** CASIA, stored at `DCNN_8.22.18\Database\1..11\{Au,Sp}\`.
**Environment:** MATLAB R2022b, Deep Learning Toolbox, single NVIDIA GTX 1070 (8 GB), `ExecutionEnvironment = 'auto'`.
**Generated from the actual run logs and `SUMMARY.txt` files in `Results\`.**

---

## 1. Which algorithm we used

The core algorithm is a **Deep Convolutional Neural Network (DCNN) image classifier trained
by supervised learning, using Transfer Learning** as the winning strategy.

Concretely the pipeline combines:

1. **Transfer Learning** — take a CNN already pretrained on ImageNet (1.2M natural images,
   1000 classes), keep its convolutional feature extractor, and **replace only the
   classification head** with a fresh 2-class head (`Au` / `Sp`). The new head is given a
   `WeightLearnRateFactor` / `BiasLearnRateFactor` of **10** so it adapts quickly while the
   pretrained backbone is fine-tuned gently.
2. **Optimizer:** **Adam** (adaptive moment estimation).
3. **Loss:** softmax + **cross-entropy** (MATLAB `classificationLayer`).
4. **Learning-rate schedule:** piecewise — start `1e-4`, drop ×0.5 every 12 epochs.
5. **Early-stopping / model selection:** `ValidationPatience = 8`, and
   `OutputNetwork = 'best-validation-loss'` (the network kept is the epoch with the lowest
   validation loss, **not** the last epoch — this is what makes the reported numbers honest).
6. **Data augmentation (deliberately minimal):** horizontal + vertical flips and small
   ±4-pixel translations **only**. Rotation and scaling are intentionally excluded because
   their interpolation smears the very splicing/edge artifacts the network must detect.

A second family of **from-scratch custom CNNs** (`Optimized_v1/v2/v3`) was also trained as a
baseline to prove the value of transfer learning — they plateau near 77 % while transfer
learning reaches 99 %.

---

## 2. Which models we used

| Family | Model | Pretrained on | Role |
|---|---|---|---|
| Transfer learning | **ResNet-50** | ImageNet | **Best model — 99.40 % val** |
| Transfer learning | **ResNet-18** | ImageNet | 98.81 % val |
| Transfer learning | **AlexNet** | ImageNet | 97.48 % val |
| Transfer learning | **VGG-16** | ImageNet | 95.33 % val |
| Transfer learning | **SqueezeNet** | ImageNet | 93.58 % val (best variant) |
| Baseline / existing | **XONet_r003** (`SeriesNetwork`, 114×114×3) | — (original project net) | 98.49 % val *(leaky split — see §6)* |
| From scratch | **Optimized_v1 / v2 / v3** | — | 77 % / 51 % / 77 % baselines |

> Note on naming: the report covers **SqueezeNet** (you wrote "SequenxNet") and **AlexNet**
> (you wrote "LXNet").

---

## 3. Architecture of each backbone

All five backbones are standard ImageNet CNNs. We swap their final 1000-class head for a
2-class (`Au`/`Sp`) head.

### ResNet-18
- **Depth:** 18 layers · ~**11.7 M** parameters · **input 224×224×3**.
- **Idea — Residual learning:** layers are grouped into *residual blocks* that learn a
  residual `F(x)` and add the input back via a **skip/shortcut connection**: output =
  `F(x) + x`. This lets gradients flow through deep stacks without vanishing.
- **Structure:** 7×7 conv (stride 2) + max-pool → **4 stages of BasicBlocks** (each block =
  two 3×3 convs + BatchNorm + ReLU + identity shortcut), with channel widths 64→128→256→512
  and spatial downsampling between stages → global average pooling → FC(1000) → softmax.
- In our pipeline the final FC is replaced by **FC(2)**.

### ResNet-50  *(our best)*
- **Depth:** 50 layers · ~**25.6 M** parameters · **input 224×224×3**.
- Same residual idea as ResNet-18 but uses deeper **"bottleneck" blocks**: 1×1 conv (reduce
  channels) → 3×3 conv → 1×1 conv (restore channels) + shortcut. This makes each block cheaper
  yet deeper, giving a much richer feature extractor — which is why it wins here.
- **Structure:** stem (7×7 conv + pool) → 4 stages of bottleneck blocks
  (3,4,6,3 blocks) → global average pool → FC(1000)→ replaced by **FC(2)**.

### AlexNet  *(written by you as "LXNet")*
- **Depth:** 8 learnable layers (5 conv + 3 FC) · ~**61 M** parameters · **input 227×227×3**.
- The classic 2012 ImageNet-winning CNN. **No residual connections, no BatchNorm.**
- **Structure:** Conv(11×11, s4) → ReLU → LRN → MaxPool → Conv(5×5) → … → 3 more conv layers
  → MaxPool → **FC(4096) → FC(4096) → FC(1000)**, with **dropout** on the FC layers.
- We replace the last **FC(1000)** with **FC(2)**.

### VGG-16
- **Depth:** 16 weight layers (13 conv + 3 FC) · ~**138 M** parameters · **input 224×224×3**.
- **Idea — uniform small filters:** every conv is **3×3**; depth comes from stacking many of
  them. Very simple and regular, but **very heavy** (138 M params) — that 478 MB network is why
  it needs `MiniBatchSize = 24` to fit the 8 GB GPU, and why it trains/generalises a bit worse
  here than ResNet.
- **Structure:** 5 conv blocks (2-2-3-3-3 convs, each block ended by 2×2 max-pool) →
  **FC(4096) → FC(4096) → FC(1000)** → replaced by **FC(2)**.

### SqueezeNet  *(written by you as "SequenxNet")*
- **Depth:** 18 layers · only ~**1.24 M** parameters (~3 MB!) · **input 227×227×3**.
- **Idea — "Fire modules":** a *squeeze* 1×1 conv (few channels) followed by an *expand* mix of
  1×1 and 3×3 convs. Achieves AlexNet-level accuracy with ~50× fewer parameters.
- **Head is convolutional, not FC:** ends in a 1×1 conv → global average pool → softmax. So
  here we replace the final **1×1 conv** (not an FC layer) with a **1×1 conv producing 2
  channels**. Smallest/fastest model; lowest accuracy of the transfer set.

*(For reference, the original existing model **XONet_r003** is a small from-scratch
`SeriesNetwork` with input **114×114×3** — not an ImageNet backbone.)*

---

## 4. How the data was split — training vs validation

Identical for **every** model so comparisons are apples-to-apples:

```
imageDatastore(Database, IncludeSubfolders, .jpg, LabelSource=foldernames)
filterReadable(...)              % drop 10 corrupt JPGs in Database\9\Sp
rng(123);                        % fixed seed — same split every run
splitEachLabel(IMDS, 0.8, 'randomized')   % 80% train / 20% validation
```

Resulting counts (from the run logs):

| | Au | Sp | **Total** |
|---|---:|---:|---:|
| Raw on disk | — | — | 10 919 |
| Corrupt dropped | — | 10 | −10 |
| **Readable** | 5 598 | 5 311 | **10 909** |
| **Training (80 %)** | 4 478 | 4 249 | **8 727** |
| **Validation (20 %)** | 1 120 | 1 062 | **2 182** |

- **Training performed on 8 727 images** (per epoch), augmented on-the-fly.
- **Validation performed on 2 182 held-out images** — never seen during training, scored at
  every epoch (`ValidationFrequency` = one full pass) to drive early-stopping and best-model
  selection, and scored once more at the end for the final metrics.

---

## 5. How much training each model received

Same protocol for the transfer sweep (`run_queue.ps1`):

| Model | Max epochs | Initial LR | Mini-batch | Optimizer | Notes |
|---|---:|---:|---:|---|---|
| ResNet-50 | 30 | 1e-4 | 64 | Adam | best result |
| ResNet-18 | 30 | 1e-4 | 64 | Adam | |
| AlexNet | 30 | 1e-4 | 64 | Adam | |
| VGG-16 | 30 | 1e-4 | **24** | Adam | small batch — 138 M params on 8 GB GPU |
| SqueezeNet | 30 | 1e-4 | 64 | Adam | |

Shared settings: `Shuffle='every-epoch'`, LR drop ×0.5 / 12 epochs,
`ValidationPatience=8`, `OutputNetwork='best-validation-loss'`. With 8 727 training images and
batch 64 → ~136 iterations/epoch (~363/epoch for VGG-16 at batch 24). The full 3-job queue
(resnet50 + alexnet + vgg16) ran in **10 h 07 m** on the GTX 1070.

---

## 6. Final results (honest, held-out validation)

Sorted by validation accuracy — from `Results\COMPARISON.txt`:

| Rank | Model | Train Acc | **Val Acc** | Val F1 | Val AUC |
|---:|---|---:|---:|---:|---:|
| 1 | **Transfer_resnet50** | 0.9986 | **0.9940** | 0.9940 | 0.9999 |
| 2 | Transfer_resnet18 | 0.9985 | 0.9881 | 0.9881 | 0.9996 |
| 3 | Existing_Model (XONet, *leaky*) | 0.9880 | 0.9849 | 0.9849 | 0.9909 |
| 4 | Transfer_alexnet | 0.9881 | 0.9748 | 0.9748 | 0.9975 |
| 5 | Transfer_vgg16 | 0.9807 | 0.9533 | 0.9532 | 0.9928 |
| 6 | Transfer_squeezenet_best | 0.9622 | 0.9358 | 0.9358 | 0.9893 |
| 7 | Transfer_squeezenet_long | 0.9290 | 0.9138 | 0.9138 | 0.9797 |
| 8 | Transfer_squeezenet | 0.8856 | 0.8607 | 0.8607 | 0.9496 |
| 9 | Optimized_v1 (scratch) | 0.8077 | 0.7741 | 0.7736 | 0.8678 |
| 10 | Optimized_v3 (scratch) | 0.8067 | 0.7699 | 0.7686 | 0.8631 |
| 11 | Optimized_v2 (scratch, collapsed) | 0.5131 | 0.5133 | 0.3392 | 0.5000 |

**Best model — ResNet-50 — per-class validation performance (2 182 images):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Au | 0.9955 | 0.9929 | 0.9942 | 1 120 |
| Sp | 0.9925 | 0.9953 | 0.9939 | 1 062 |

**Leakage caveat:** the original `XONet_r003` never saved its train/val split, so under the
fixed `rng(123)` split it has very likely already *seen* part of today's validation set. Its
98.49 % is therefore **optimistic** and not directly comparable. All Phase-2 transfer models
are measured on a genuinely held-out 20 %.

---

## 7. How the results were detected — step by step

This is exactly what each training script (`Train_Transfer.m`) does end-to-end, and what
`run_queue.ps1` orchestrates across backbones:

**Step 1 — Load & clean data.** Build an `imageDatastore` over `Database\` (labels from
folder names), then `filterReadable()` drops the 10 corrupt JPGs → 10 909 usable images.

**Step 2 — Deterministic split.** `rng(123)` + `splitEachLabel(0.8)` → 8 727 train / 2 182
validation, identical for every model.

**Step 3 — Build the model.** Load the ImageNet-pretrained backbone via `setupPretrained()`,
read its native `InputSize`, then `replaceLayer` the final learnable layer (FC or 1×1 conv)
and the classification layer with a fresh **2-class** head (LR factor ×10).

**Step 4 — Pre-process & augment.** Wrap data in `augmentedImageDatastore` that resizes to the
backbone's input size, converts grayscale→RGB if needed, and applies flips + ±4-px shifts on
the **training** stream only.

**Step 5 — Train.** `trainNetwork` with Adam (LR 1e-4, 30 epochs, the chosen batch size).
At every epoch the validation set is scored; training stops early if val loss stops improving
for 8 checks, and the **best-validation-loss** network is the one kept. The full per-iteration
history is captured in the `info` struct.

**Step 6 — Save curves.** `plotTrainCurves(info)` writes `loss_curve.png` and
`accuracy_curve.png` (training vs validation) from the captured history.

**Step 7 — Score both sets.** `classify()` the trained net on training and validation streams
to get predicted labels **and** class scores (probabilities).

**Step 8 — Compute metrics.** `computeMetrics()` produces per-class + macro/weighted
**Precision, Recall, F1, Accuracy**; `plotROC()` computes the **ROC curve and AUC** (positive
class = `Sp`). These are printed by `printMetrics()` and written to `SUMMARY.txt`.

**Step 9 — Save confusion matrices.** `saveConfusion()` writes `confusion_training.png` and
`confusion_validation.png`.

**Step 10 — Persist everything.** The trained network (`net_<backbone>.mat`), the metrics
(`results_transfer.mat`), the human-readable `SUMMARY.txt`, all PNGs, and the full stdout
(`run_log.txt`) are saved per-model; the script prints the `DONE_TRAIN_OPTIMIZED` sentinel.

**Step 11 — Aggregate & rank.** After the queue finishes, `Compare_Models.m` scans every
`Results\**\results_*.mat`, sorts by validation accuracy, and writes `COMPARISON.csv`,
`COMPARISON.txt`, and `COMPARISON_bar.png` — the leaderboard in §6.

---

## 8. Per-model artifacts (where to look)

Each `Results\<Model>\` folder contains:
`SUMMARY.txt`, `confusion_training.png`, `confusion_validation.png`, `roc_curve.png`,
`loss_curve.png`, `accuracy_curve.png`, `results_*.mat`, `net_*.mat` (trained weights),
`run_log.txt`. Cross-model: `Results\COMPARISON.{csv,txt}`, `COMPARISON_bar.png`,
`FINAL_REPORT.md`, and this `EXECUTION_REPORT.md`.

> Storage note: `net_vgg16.mat` (479 MB) and `net_alexnet.mat` (205 MB) exceed GitHub's
> 100 MB limit and are `.gitignore`d (regenerable via `run_queue.ps1`). All other networks,
> metrics, and plots are committed.
