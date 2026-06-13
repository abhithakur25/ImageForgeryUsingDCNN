# Image Forgery Detection using a Deep Convolutional Neural Network (DCNN)

Detecting **spliced / forged images** in the CASIA dataset as a binary classification
problem: **`Au`** (authentic) vs **`Sp`** (spliced). This repository turns a one-off
research script into a fully **reproducible, instrumented MATLAB pipeline** that trains,
evaluates, and compares multiple CNN architectures and reports *honest, held-out*
performance with a complete metric suite (confusion matrices, precision / recall / F1,
ROC / AUC, and training-vs-validation loss & accuracy curves).

---

## Table of contents
1. [Introduction](#1-introduction)
2. [Environment & requirements](#2-environment--requirements)
3. [Repository structure](#3-repository-structure)
4. [The dataset](#4-the-dataset)
5. [Step-by-step: how to execute the code](#5-step-by-step-how-to-execute-the-code)
6. [Parameters explained](#6-parameters-explained)
7. [Models used and how they work](#7-models-used-and-how-they-work)
8. [Training time](#8-training-time)
9. [Comparison & results](#9-comparison--results)
10. [Methodology note (data leakage)](#10-methodology-note-data-leakage)
11. [Output files](#11-output-files)
12. [What is / isn't tracked in this repo](#12-what-is--isnt-tracked-in-this-repo)

---

## 1. Introduction

Image splicing — copying a region from one image into another — is one of the most common
forms of digital image forgery. The cues it leaves behind (boundary artifacts, inconsistent
noise, colour-illumination mismatches) are subtle, so the task is well suited to deep
convolutional neural networks that learn discriminative features directly from pixels.

This project takes the CASIA forgery dataset and frames detection as **binary
classification**. The original code (`TransferLearning3.m`) trained a small custom network
("XONet") and reported ~98.5% accuracy — but it validated on images it had already trained
on (see §10). The pipeline here corrects that: every model is trained on 80% of the data and
evaluated on a **held-out 20%** it never sees, using a fixed random seed so all models are
compared on exactly the same split. Several architectures were trained and benchmarked, from
small from-scratch CNNs to ImageNet-pretrained backbones fine-tuned by transfer learning.

## 2. Environment & requirements

| Component | Version / detail |
|---|---|
| MATLAB | **R2022b** |
| Toolboxes | Deep Learning Toolbox, Image Processing Toolbox, Parallel Computing Toolbox |
| Pretrained-network add-ons | `resnet18`, `resnet50`, `alexnet`, `vgg16` (`squeezenet` is built-in) |
| GPU | NVIDIA GTX 1070 (8 GB) — training uses `ExecutionEnvironment 'auto'` |
| OS | Windows 11 |

MATLAB is launched non-interactively (`-batch`) so the whole pipeline can be scripted from
PowerShell.

## 3. Repository structure

```
DCNN_8.22.18/                 <-- ACTIVE project (do all work here)
├── Database/1..11/{Au,Sp}/   active CASIA images (tracked, ~27 MB)
├── XONet_r003.mat            original pretrained model (baseline)
├── Test_Existing_Model.m     Phase 1: evaluate the saved model
├── Train_Transfer.m          Phase 2: transfer learning (parameterised)
├── Train_Optimized.m         Phase 2: from-scratch CNN (v1)
├── Train_CustomV2.m / V3.m    Phase 2: from-scratch CNN variants
├── Compare_Models.m          aggregate all results -> COMPARISON.*
├── run_queue.ps1             resumable backbone sweep + auto-compare
├── computeMetrics.m / printMetrics.m / saveConfusion.m / plotROC.m /
│   writeSummary.m / filterReadable.m / setupPretrained.m   (shared helpers)
└── Results/<ModelName>/      per-model outputs (SUMMARY, curves, logs, net_*.mat)
DCNN_1/ DCNN_2/ DCNN_8.21.18/ Review8.8.18/   <-- legacy dated snapshots
```

## 4. The dataset

- **10,919** JPG images, each **114 × 114 × 3** (RGB), organised in `Database\1..11`, every
  numbered folder containing an `Au\` (authentic) and `Sp\` (spliced) subfolder. Labels are
  derived from the folder names.
- **10 corrupt files** in `Database\9\Sp\` crash `imageDatastore`; `filterReadable.m`
  detects and drops them, leaving **10,909 usable** images — **Au: 5,598 / Sp: 5,311**.
- **Split:** `rng(123); splitEachLabel(IMDS, 0.8, 'randomized')` → 80% train / 20% validation.
  The seed is identical across every script, so all models share the same held-out set.
- **Augmentation:** random horizontal/vertical flips + small translations (±4 px) **only**.
  Rotation and scaling are deliberately avoided — their interpolation smooths over the fragile
  splicing cues the network must detect.

## 5. Step-by-step: how to execute the code

> All commands are run from `D:\Code_Paper3\DCNN_8.22.18`. The MATLAB executable is
> `F:\Program Files\MATLAB\R2022b\bin\matlab.exe`.

**Step 1 — Clone & open the project folder**
```powershell
git clone https://github.com/abhithakur25/ImageForgeryUsingDCNN.git
cd ImageForgeryUsingDCNN\DCNN_8.22.18
```

**Step 2 — Phase 1: evaluate the existing trained model** (no retraining)
```powershell
& "F:\Program Files\MATLAB\R2022b\bin\matlab.exe" -batch "Test_Existing_Model"
```
Loads `XONet_r003.mat`, classifies train & validation sets, and writes the full metric suite
to `Results\Existing_Model\`.

**Step 3 — Phase 2: train one model** (transfer learning example)
```powershell
& "F:\...\matlab.exe" -batch "BASENET='resnet50'; EPOCHS=30; LR0=1e-4; MBS=64; TAG='resnet50'; Train_Transfer"
```

**Step 4 — Run the full backbone sweep** (resnet50 → alexnet → vgg16, then auto-compare)
```powershell
powershell -ExecutionPolicy Bypass -File run_queue.ps1
```
This is **resumable**: any model whose `Results\Transfer_<Tag>\SUMMARY.txt` already exists is
skipped (use `-Force` to retrain). A live progress bar is written to
`Results\queue_status.txt`.

**Step 5 — Aggregate every result into one comparison table**
```powershell
& "F:\...\matlab.exe" -batch "Compare_Models"
```
Produces `Results\COMPARISON.csv`, `COMPARISON.txt`, and `COMPARISON_bar.png`.

## 6. Parameters explained

`Train_Transfer.m` reads these workspace variables (all optional; sensible defaults shown):

| Parameter | Default | Meaning |
|---|---|---|
| `BASENET` | `'resnet18'` | Pretrained backbone: `resnet18`, `resnet50`, `alexnet`, `vgg16`, `squeezenet`. |
| `EPOCHS` | `12` | Number of training epochs (full passes over the training set). |
| `LR0` | `1e-4` | Initial learning rate for the Adam optimiser. |
| `MBS` | `64` | Mini-batch size. Lower it for large nets on an 8 GB GPU (vgg16 uses `24`). |
| `TAG` | `=BASENET` | Output folder name suffix → `Results\Transfer_<TAG>\`. |

Other fixed training options inside the script: optimiser **Adam**; learning-rate schedule
**piecewise** (drop ×0.5 every 12 epochs); **validation patience 8**;
`OutputNetwork = 'best-validation-loss'` (keeps the best checkpoint, not just the last).

The from-scratch scripts (`Train_Optimized.m`, `Train_CustomV3.m`) read a `VARIANT` tag
(e.g. `'v1'`, `'v3'`) used purely to name the output folder.

## 7. Models used and how they work

**Transfer learning (the winning approach).** A backbone pretrained on ImageNet already
encodes rich, general visual features. `Train_Transfer.m`:
1. loads the backbone via `setupPretrained.m` (which adds the locally cached support packages
   to the MATLAB path);
2. converts it to a `layerGraph` and **replaces the classification head** — the final
   learnable layer (a fully-connected or 1×1 conv layer) is swapped for one with **2 outputs**
   (`Au`/`Sp`) at a **10× learning-rate factor**, and the classification layer is replaced;
3. fine-tunes the whole network on the forgery data with light augmentation.

Backbones benchmarked: **ResNet-18 / ResNet-50** (residual networks), **AlexNet**,
**VGG-16**, and **SqueezeNet**. ResNet-50's residual connections and depth gave it the best
generalisation on this task.

**From-scratch CNNs** (`Optimized_v1/v2/v3`) were trained as a baseline to show *why*
transfer learning is needed — they plateau around 77% because the dataset is too small to
learn good low-level features from random initialisation.

**The original baseline** `XONet_r003` is a small custom `SeriesNetwork` (input 114×114×3)
that is only *evaluated* here (Phase 1), not retrained.

## 8. Training time

Measured on the GTX 1070 (final elapsed time from each run log):

| Model | Epochs | Mini-batch | Training time |
|---|---|---|---|
| Transfer_squeezenet_best | 35 | 64 | ~18 min |
| Transfer_resnet18 | 30 | 64 | ~24 min |
| Transfer_alexnet | 30 | 64 | ~22 min |
| Transfer_resnet50 | 30 | 64 | **~1 h 18 min** |
| Transfer_vgg16 | 30 | 24 | **~8 h 20 min** |
| Optimized_v1 (scratch) | 30 | 64 | ~6 min |

The full 3-job sweep (resnet50 + alexnet + vgg16) driven by `run_queue.ps1` took
**10 h 07 m** end-to-end — dominated by VGG-16 (138 M parameters, small batch to fit 8 GB).

## 9. Comparison & results

Honest held-out validation, fixed `rng(123)` 80/20 split, sorted by validation accuracy
(source: `Results\COMPARISON.txt`):

| Model | Train Acc | **Val Acc** | Val Precision | Val Recall | Val F1 | Val AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Transfer_resnet50** | 0.9986 | **0.9940** | 0.9940 | 0.9941 | 0.9940 | 0.9999 |
| Transfer_resnet18 | 0.9985 | 0.9881 | 0.9881 | 0.9881 | 0.9881 | 0.9996 |
| Existing_Model (XONet)¹ | 0.9880 | 0.9849 | 0.9848 | 0.9850 | 0.9849 | 0.9909 |
| Transfer_alexnet | 0.9881 | 0.9748 | 0.9749 | 0.9747 | 0.9748 | 0.9975 |
| Transfer_vgg16 | 0.9807 | 0.9533 | 0.9532 | 0.9535 | 0.9532 | 0.9928 |
| Transfer_squeezenet_best | 0.9622 | 0.9358 | 0.9358 | 0.9361 | 0.9358 | 0.9893 |
| Transfer_squeezenet_long | 0.9290 | 0.9138 | 0.9139 | 0.9142 | 0.9138 | 0.9797 |
| Transfer_squeezenet | 0.8856 | 0.8607 | 0.8614 | 0.8613 | 0.8607 | 0.9496 |
| Optimized_v1 (scratch) | 0.8077 | 0.7741 | 0.7743 | 0.7734 | 0.7736 | 0.8678 |
| Optimized_v3 (scratch) | 0.8067 | 0.7699 | 0.7726 | 0.7685 | 0.7686 | 0.8631 |
| Optimized_v2 (scratch)² | 0.5131 | 0.5133 | 0.2566 | 0.5000 | 0.3392 | 0.5000 |

¹ In-sample / optimistic — see §10. ² Collapsed to one class (learning rate too high).

**Best model: `Transfer_resnet50` — 99.40% validation accuracy, F1 0.9940, AUC 0.9999**,
measured honestly on held-out data. The progression across iterations
(77% scratch → 86% → 91% → 93.6% squeezenet → 98.8% resnet18 → **99.4% resnet50**) shows that
pretrained features plus light, forgery-safe augmentation decisively win.

## 10. Methodology note (data leakage)

The original `TransferLearning3.m` calls `trainNetwork(IMDS, ...)` on the **entire** dataset
and then "validates" on a random subset of those *same* images. Every validation image was
seen in training, so its ~98.5% is an **in-sample** number, not a measure of generalisation.
All Phase-2 models here avoid this: train on 80%, test on a held-out 20% (fixed `rng(123)`).
**For publication, report the honest held-out numbers.** (A further refinement — a
*group-aware* split so rotation/colour variants of one source image can't straddle train and
val — is noted in `Results\FINAL_REPORT.md` but not applied, to match the original protocol.)

## 11. Output files

Each `Results\<ModelName>\` folder contains:
- `SUMMARY.txt` — headline metrics
- `confusion_training.png`, `confusion_validation.png` — confusion matrices
- `roc_curve.png` — ROC for train & validation with AUC
- `loss_curve.png`, `accuracy_curve.png` — training-vs-validation curves (Phase-2 models)
- `results_*.mat` — metrics struct + training `info`
- `net_*.mat` — the trained network (tracked only if < 100 MB)
- `run_log.txt` — full MATLAB console log

Cross-model: `Results\COMPARISON.{csv,txt}`, `COMPARISON_bar.png`, `FINAL_REPORT.md`.

## 12. What is / isn't tracked in this repo

**Tracked:** all executable code (`.m`, `.ps1`), the active CASIA dataset (~27 MB), every
result artifact (summaries, PNGs, logs, comparison), and trained models **under 100 MB**
(`net_resnet50` 86 MB, `net_resnet18` 41 MB, `net_squeezenet` 3 MB), plus this `README.md`,
`CLAUDE.md`, the `logs/` session log, and the `documentation/` write-up.

**Excluded** (see `.gitignore`): models **over GitHub's 100 MB limit**
(`net_vgg16.mat` 478 MB, `net_alexnet.mat` 204 MB — regenerable via `run_queue.ps1`),
redundant raw image-dataset dumps, and `.rar` / `.pdf` reference blobs.
