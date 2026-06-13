# Image Forgery Detection using a Deep CNN (CASIA, Au vs Sp)

A reproducible MATLAB pipeline that detects **spliced/forged images** by binary
classification of the CASIA dataset: `Au` (authentic) vs `Sp` (spliced). It rebuilds the
original one-off experiment (`TransferLearning3.m` → `XONet_r003.mat`) into a documented
**train → evaluate → compare** workflow that measures honest held-out performance and
emits a full metric suite (confusion matrices, precision/recall/F1, ROC/AUC, and training
vs validation loss & accuracy curves).

## Results (honest held-out validation, fixed `rng(123)` 80/20 split)

| Model                     | Train Acc | Val Acc | Val F1 | Val AUC |
|---------------------------|:---------:|:-------:|:------:|:-------:|
| **Transfer_resnet50**     | 0.9986    | **0.9940** | 0.9940 | 0.9999 |
| Transfer_resnet18         | 0.9985    | 0.9881  | 0.9881 | 0.9996  |
| Existing_Model (XONet)¹   | 0.9880    | 0.9849  | 0.9849 | 0.9909  |
| Transfer_alexnet          | 0.9881    | 0.9748  | 0.9748 | 0.9975  |
| Transfer_vgg16            | 0.9807    | 0.9533  | 0.9532 | 0.9928  |
| Transfer_squeezenet_best  | 0.9622    | 0.9358  | 0.9358 | 0.9893  |
| Optimized_v1 (scratch)    | 0.8077    | 0.7741  | 0.7736 | 0.8678  |

¹ The original model's train/val split was never saved, so under the fixed split it likely
trained on part of its current "validation" set — its numbers are **optimistic**. The
Transfer models are the honest, comparable results. Full table: `DCNN_8.22.18/Results/COMPARISON.txt`.

## Pipeline

All active code lives in **`DCNN_8.22.18/`** (sibling `DCNN_*` / `Review8.8.18` folders are
dated legacy snapshots).

- **Phase 1 — `Test_Existing_Model.m`**: evaluate the pretrained `XONet_r003.mat`
  (114×114×3 SeriesNetwork) without retraining.
- **Phase 2 — training:**
  - `Train_Transfer.m` — transfer learning from a pretrained backbone
    (`resnet18`, `resnet50`, `alexnet`, `vgg16`, `squeezenet`); swaps the classification
    head for 2 classes.
  - `Train_Optimized.m`, `Train_CustomV2.m`, `Train_CustomV3.m` — from-scratch CNNs (kept
    for comparison; they plateau ~77%).
- **`Compare_Models.m`** — aggregates every `Results/**/results_*.mat` into
  `COMPARISON.{csv,txt}` and a bar chart.
- **`run_queue.ps1`** — resumable backbone sweep (one `matlab -batch` per model; skips jobs
  whose `SUMMARY.txt` exists; live progress bar to `Results/queue_status.txt`).
- **Shared eval helpers**: `computeMetrics`, `printMetrics`, `saveConfusion`, `plotROC`,
  `writeSummary`, `filterReadable`, `setupPretrained`.

## How to run

MATLAB **R2022b** (Deep Learning Toolbox + GPU). Run non-interactively from
`DCNN_8.22.18/`:

```powershell
# Phase 1 — evaluate the existing model
& "F:\Program Files\MATLAB\R2022b\bin\matlab.exe" -batch "Test_Existing_Model"

# Phase 2 — train one backbone
& "F:\Program Files\MATLAB\R2022b\bin\matlab.exe" -batch "BASENET='resnet50'; EPOCHS=30; LR0=1e-4; MBS=64; TAG='resnet50'; Train_Transfer"

# Full backbone sweep + comparison
powershell -ExecutionPolicy Bypass -File run_queue.ps1
```

## Data conventions (important)

- Dataset: `DCNN_8.22.18/Database/1..11/{Au,Sp}/` (~10.9k JPGs).
- **Always `filterReadable(IMDS)` before splitting** — 10 corrupt JPGs in `Database/9/Sp/`
  are dropped (10909 readable of 10919).
- **Always split with `rng(123); splitEachLabel(IMDS,0.8,'randomized')`** so every model is
  compared on the same held-out set.
- **Augmentation: flips + small translations only** — rotation/scaling introduce
  interpolation artifacts that mask the splicing cues being detected.

## What is / isn't in the repo

Tracked: all executable code (`.m`, `.ps1`), the active CASIA dataset (≈27 MB), all
result artifacts (`SUMMARY.txt`, confusion/ROC/loss/accuracy PNGs, logs), and trained
models **under 100 MB** (`resnet50` 86 MB, `resnet18` 41 MB, `squeezenet`).

Excluded (see `.gitignore`): models **over GitHub's 100 MB limit** (`net_vgg16.mat` 478 MB,
`net_alexnet.mat` 204 MB — regenerable via `run_queue.ps1`), redundant raw image-dataset
dumps, and reference `.rar`/`.pdf` blobs.

See `EXECUTION_LOG.md` for the verification + commit session log, and `CLAUDE.md` for
detailed contributor guidance.
