# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MATLAB deep-learning pipeline for **image-forgery detection** on the CASIA dataset:
binary classification of `Au` (authentic) vs `Sp` (spliced) JPGs. The original work
(`TransferLearning3.m` → `XONet_r003.mat`) has been rebuilt into a reproducible
train/evaluate/compare pipeline that honestly measures held-out performance and
produces a full metric suite (confusion matrices, P/R/F1, ROC/AUC, loss & accuracy
curves).

## Where the live code is

**`DCNN_8.22.18/` is the active working directory — do all work there.** The sibling
folders `DCNN_1/`, `DCNN_2/`, `DCNN_8.21.18/`, and `Review8.8.18/` are older dated
snapshots kept only for reference; their `.m` files (`DCNN_New_Final.m`,
`TransferLearning3.m`, `colImgSeg.m`, `imagePreProcessing.m`, `ReSize*.m`,
`colorIllumination2.m`) are legacy and hardcode `F:\College\Abhishek\...` paths. Do not
edit them unless explicitly asked.

## Environment & how to run

MATLAB **R2022b** is installed on `F:` (not C:): `F:\Program Files\MATLAB\R2022b\bin\matlab.exe`.
GPU: NVIDIA GTX 1070 (8 GB) — scripts use `ExecutionEnvironment 'auto'`.

Run any script non-interactively from the project dir:
```powershell
& "F:\Program Files\MATLAB\R2022b\bin\matlab.exe" -batch "ScriptName"
```
Pass parameters by setting variables before the script name in the same `-batch` string:
```powershell
& "F:\...\matlab.exe" -batch "BASENET='resnet50'; EPOCHS=30; LR0=1e-4; MBS=64; TAG='resnet50'; Train_Transfer"
```
Each script `cd`s to its own folder, mirrors stdout to `Results\<Model>\run_log.txt` via
`diary`, and prints a sentinel on success (`DONE_TEST_EXISTING`, `DONE_TRAIN_OPTIMIZED`,
`DONE_COMPARE`) that the queue runner greps for.

## Pipeline architecture

**Phase 1 — evaluate the existing model without retraining:** `Test_Existing_Model.m`
loads `XONet_r003.mat` (a `SeriesNetwork`, input 114×114×3, classes Au/Sp) and scores it.
Note: the original training discarded the train/val split and the training-info struct, so
loss curves cannot be reconstructed for this model and its "validation" numbers are
optimistic (see leakage caveat below).

**Phase 2 — train fresh models** (all produce the same outputs as Phase 1 *plus* real
train/val loss & accuracy curves from the captured `info` struct):
- `Train_Transfer.m` — transfer learning from a pretrained backbone (the winning
  approach). `BASENET` ∈ {resnet18, resnet50, alexnet, vgg16, squeezenet}; replaces the
  classification head for 2 classes. Backbones are loaded via `setupPretrained.m`, which
  `addpath`s locally-cached support packages from
  `F:\ProgramData\MATLAB\SupportPackages\R2022b\...` (googlenet is NOT installed).
- `Train_Optimized.m`, `Train_CustomV2.m`, `Train_CustomV3.m` — from-scratch CNNs. These
  plateau ~77% and are kept for comparison; transfer learning wins decisively.

**Comparison/aggregation:** `Compare_Models.m` scans `Results\**\results_*.mat` and writes
`Results\COMPARISON.{csv,txt}` + `COMPARISON_bar.png` (sorted by val accuracy).

**Queue runner:** `run_queue.ps1` runs a sweep of `Train_Transfer` jobs sequentially (one
fresh `matlab -batch` per backbone), is **resumable** (skips any job whose
`Results\Transfer_<Tag>\SUMMARY.txt` exists; `-Force` re-runs), writes a live ASCII
progress bar to `Results\queue_status.txt`, and calls `Compare_Models` at the end.
```powershell
powershell -ExecutionPolicy Bypass -File DCNN_8.22.18\run_queue.ps1
```

**Shared evaluation helpers** (reuse these; don't reinvent metrics):
`computeMetrics.m` (per-class + macro/weighted P/R/F1/accuracy), `printMetrics.m`,
`saveConfusion.m`, `plotROC.m` (positive class = last class = `Sp`), `writeSummary.m`,
`filterReadable.m`.

## Critical data conventions — follow exactly

- **Data lives at `DCNN_8.22.18\Database\1..11\{Au,Sp}\`** (~10.9k JPGs). Build datastores
  with `imageDatastore(dbPath,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames')`.
- **Always `filterReadable(IMDS)` before splitting.** 10 corrupt JPGs in `Database\9\Sp\`
  fail to read; this helper drops them (10909 readable of 10919).
- **Always split with `rng(123); splitEachLabel(IMDS,0.8,'randomized')`.** Every model uses
  this identical seed so comparisons are apples-to-apples.
- **Leakage caveat:** the original `XONet_r003` split was never saved, so under the fixed
  `rng(123)` split that model likely trained on ~80% of its current "val" set — its
  validation metrics are optimistic. Phase-2 models are measured honestly on held-out data;
  prefer those when reporting.
- **Augmentation: flip + small translation ONLY** (`RandXReflection`/`RandYReflection`,
  `RandX/YTranslation [-4 4]`). Do NOT add rotation or scaling — their interpolation
  introduces artifacts that mask the splicing cues being detected.

## Results layout

Per-model: `DCNN_8.22.18\Results\<ModelName>\` with `SUMMARY.txt`, `confusion_*.png`,
`roc_curve.png`, `loss_curve.png`, `accuracy_curve.png`, `results_*.mat`,
`net_*.mat` (the trained network), `run_log.txt`. Cross-model: `Results\COMPARISON.*`,
`COMPARISON_bar.png`, `FINAL_REPORT.md`.

Current best (honest held-out val, sorted): **Transfer_resnet50 99.40%** > Transfer_resnet18
98.81% > Existing_Model 98.49% (leaky) > alexnet 97.48% > vgg16 95.33% > squeezenet variants
> from-scratch Optimized_v1/v3 ~77% (Optimized_v2 collapsed to 51% — needs LR ≤ 3e-4).
vgg16 must use `MBS=24` to fit the 8 GB GPU.
