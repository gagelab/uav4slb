# uav4slb — UAV-Based Deep Learning for Southern Leaf Blight Severity Prediction in Maize

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)

This repository contains the complete, reproducible codebase for:

> **Hammett, C. H., Rumley, K. R., Balint-Kurti, P. J., & Gage, J. L. (2026). Aerial imagery and deep learning accurately estimate maize foliar disease severity.** *The Plant Phenome Journal.*

The study benchmarks nine pretrained deep learning architectures under a leave-one-year-out cross-validation scheme (CV0) across three field seasons (2023–2025) in North Carolina. EVA-02-B achieved the highest aggregate test R² across folds (mean R² = 0.697; range 0.610–0.766) and is the primary model reported in the manuscript. All training code, evaluation scripts, covariate pipelines, and figure scripts required to reproduce every result and figure in the manuscript are provided. The raw RGB images and plot images are located at [DOI]

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Data](#data)
3. [Environment Setup](#environment-setup)
4. [Reproducing Results](#reproducing-results)
5. [Model Architectures](#model-architectures)
6. [Cross-Validation Design (CV0)](#cross-validation-design-cv0)
7. [Configuration System](#configuration-system)
8. [Checkpointing and Resumption](#checkpointing-and-resumption)
9. [Figure Reference](#figure-reference)
10. [Data Availability](#data-availability)
11. [Citation](#citation)

---

## Repository Structure

```
uav4slb/
├── configs/                     # OmegaConf YAML experiment configs (one per model)
│   ├── coatnet2_cv0.yaml
│   ├── convnextv2_cv0.yaml
│   ├── convnextv2_large_cv0.yaml
│   ├── dinov2_vitb14_cv0.yaml
│   ├── dinov2_vits14_cv0.yaml
│   ├── efficientnetv2_s_cv0.yaml
│   ├── eva02_base_cv0.yaml      # EVA-02-B (primary model)
│   ├── eva02_base_retrain.yaml  # Warm-start config for expanded-dataset retraining
│   ├── maxvit_small_cv0.yaml
│   ├── swinv2_base_cv0.yaml
│   └── weed_pressure_config.yaml
│
├── data/
│   ├── covariates/
│   │   ├── flight_covariates.csv    # Flight-level solar geometry + irradiance (28 rows)
│   │   ├── image_covariates.csv     # Per-image quality metrics (26,071 rows)
│   │   └── raw/                     # Source files for covariate scripts
│   │       ├── flight_times.csv
│   │       ├── frac_weed.csv
│   │       ├── gdd.csv
│   │       ├── inoculation_dates.csv
│   │       ├── irradiance_minute.csv
│   │       └── planting_dates.csv
│   ├── cv_splits/cv0/
│   │   ├── fold_2023/               # train / val / test CSVs; test year = 2023
│   │   ├── fold_2024/               # test year = 2024
│   │   └── fold_2025/               # test year = 2025
│   └── labels/
│       ├── full_dataset.csv         # Master label file (26,071 rows; see Data section)
│       └── long_format_ratings.csv  # Multi-rater scoring experiment (Fig. 5)
│
├── data_README.md               # Extended data dictionary
├── environment.yaml
│
├── experiments/                 # Training outputs (one directory per model)
│   └── <model>_cv0/
│       ├── config_resolved.yaml # Fully resolved config snapshot
│       ├── cv_results.csv       # Per-fold metrics table (3 rows × 16 cols)
│       ├── logs/                # Per-fold training logs
│       ├── predictions/         # Per-fold and combined prediction CSVs
│       └── summary.json         # Machine-readable aggregate metrics
│
├── figures/                     # Figure scripts and outputs
│   ├── fig02_dataset_composition/
│   ├── fig03_model_comparison/
│   ├── fig04_test_performance/
│   ├── fig05_rater_agreement/
│   ├── fig06_flight_covariates/
│   ├── fig07_image_noise_examples/
│   ├── fig08_image_covariates/
│   ├── fig09_temporal_misalignment/
│   ├── fig10_mislabelled_examples/
│   ├── style.py                 # Shared style (batlow colormap, rcParams)
│   └── supplemental/
│       ├── figS1_flight_timeline/
│       ├── figS2_score_distributions/
│       ├── figS3_fold_r2/
│       └── figS5_temporal_alignment/
│
├── results/
│   └── cv0_aggregate_results.csv    # 28 rows × 12 cols; all 9 models × 3 folds
│
├── scripts/                     # Executable pipeline scripts
│   ├── build_flight_covariates.py
│   ├── build_image_covariates.py
│   ├── create_cv_splits.py
│   ├── evaluate_cv0.py          # Post-training evaluation and table generation
│   ├── predict_cv0.py           # Reproduce predictions / run inference on new data
│   ├── run_all_models.sh        # Sequential launcher for all 9 models
│   ├── run_evaluate_cv0.sh
│   ├── run_predict_cv0.sh
│   ├── run_train_cv0.sh         # HPC launcher for train_cv0.py
│   ├── train_cv0.py             # Unified training entry point (all 9 architectures)
│   └── weed_pressure_pipeline.py
│
└── src/                         # Importable library
    ├── data/
    │   ├── augmentation.py      # Albumentations augmentation pipeline
    │   ├── dataset.py           # PyTorch Dataset for plot images + labels
    │   ├── image_validation.py
    │   └── preprocessing.py     # GeoTIFF and standard image loading
    ├── evaluation/
    │   ├── evaluator.py
    │   └── metrics.py           # R², Pearson r, Spearman ρ, RMSE, MAE
    ├── models/
    │   ├── base_model.py        # Abstract base class for all model wrappers
    │   ├── coatnet2.py
    │   ├── convnextv2.py
    │   ├── convnextv2_large.py
    │   ├── dinov2_vitb14.py
    │   ├── dinov2_vits14.py
    │   ├── efficientnetv2s.py
    │   ├── eva02_base.py
    │   ├── maxvit_small.py
    │   └── swinv2_base.py
    └── utils/
        ├── config.py
        ├── cv_utils.py
        ├── gpu_utils.py
        ├── reproducibility.py
        └── solar.py             # pvlib solar geometry helpers
```

---

## Data

### Image Data

Plot-level UAV images (~26,000 JPEG files) are stored separately from this repository due to their size and are available from the USDA Ag Data Commons (see [Data Availability](#data-availability)). Image filenames follow the convention:

```
YYYYMMDD_FIELD_{PLOT:05d}.jpg
```

where `FIELD` is one of: `G3`, `I3B`, `B7A`, `C6B`, `C10`, `C7B`. The zero-padded five-digit plot number used to join to visual scores. 

All scripts that require images accept an `--image-dir` argument (or read `data.image_dir` from the YAML config) pointing to the directory of sliced plot images. Update this path to wherever you have downloaded the images before running training.

---

### data/labels/full_dataset.csv

The master label file: **26,071 rows × 8 columns**. One row per scored plot image.

| Column | Type | Description |
|--------|------|-------------|
| `image_filename` | str | Image filename (`YYYYMMDD_FIELD_{PLOT:05d}.jpg`); primary join key |
| `plot` | int | Plot number |
| `field` | str | Field site (`G3`, `I3B`, `B7A`, `C6B`, `C10`, `C7B`) |
| `year` | int | Field season (2023, 2024, or 2025) |
| `date` | str | Visual scoring date (YYYY-MM-DD) |
| `score` | float | SLB severity score (1–9 scale; half-integer increments) |
| `flight_date` | str | UAV flight date matched to scoring event (YYYY-MM-DD) |
| `signed_days_diff` | int | Signed days between scoring and flight (−3 to +3; negative = flight before scoring) |

Score distribution by year: 2023 mean = 6.11 (n = 2,601); 2024 mean = 5.55 (n = 8,290); 2025 mean = 5.32 (n = 15,180).

---

### data/labels/long_format_ratings.csv

Multi-rater scoring experiment used in the rater variability analysis (Fig. 5). **498 rows × 3 columns**: 5 raters × ~100 plots, with one row per rater–plot pair.

| Column | Type | Description |
|--------|------|-------------|
| `plot` | int | Plot number (462–562) |
| `rater` | int | Rater index (1–5) |
| `score` | float | SLB severity score (1–9 scale; half-integer increments) |

---

### data/cv_splits/cv0/

Pre-generated CV0 split files. Each fold subdirectory contains `train.csv`, `val.csv`, and `test.csv`. All three files share the same **7-column schema**: `plot`, `field`, `year`, `date`, `score`, `flight_date`, `image_filename`. The `image_filename` column is the primary join key to the image directory and to all covariate files.


| Fold | Test year | n_train | n_val | n_test |
|------|-----------|---------|-------|--------|
| `fold_2023` | 2023 | 19,949 | 3,521 | 2,601 |
| `fold_2024` | 2024 | 15,113 | 2,668 | 8,290 |
| `fold_2025` | 2025 | 9,257 | 1,634 | 15,180 |

To regenerate splits from `full_dataset.csv`:

```bash
python scripts/create_cv_splits.py \
    --labels-csv data/labels/full_dataset.csv \
    --output-dir data/cv_splits/cv0 \
    --seed 42
```

The training script verifies split integrity via an MD5 hash only when resuming with `--resume` (which also requires `--fold`); regenerated splits must match the original hash to resume a checkpoint this way. To load pre-trained weights onto a new or regenerated dataset, use `--warm-start` instead, which intentionally skips hash verification.

---

### data/covariates/flight_covariates.csv

Flight-level environmental and solar covariates: **28 rows × 16 columns** (one row per flight).

| Column | Type | Description |
|--------|------|-------------|
| `flight_key` | str | Join key (`YYYYMMDD_FIELD`) |
| `flight_date` | str | Flight date (YYYY-MM-DD) |
| `field` | str | Field site identifier |
| `year` | int | Season (2023–2025) |
| `flight_start` | str | Flight start time (datetime string) |
| `flight_end` | str | Flight end time (datetime string) |
| `sza` | float | Solar zenith angle at flight midpoint (°) |
| `sun_azimuth` | float | Solar azimuth at flight midpoint (°) |
| `abs_time_from_noon` | float | Absolute minutes from solar noon |
| `irradiance_mean` | float | Mean shortwave irradiance during flight window (W m⁻²) |
| `irradiance_cv` | float | Coefficient of variation of irradiance (dimensionless) |
| `irradiance_n` | int | Number of minute-resolution irradiance observations in the flight window |
| `dpi` | int | Days post-inoculation at flight date |
| `agdd` | float | Accumulated growing degree days from planting (base 50 °F) |
| `row_azimuth` | float | Field row orientation (°) |
| `shadow_row_angle` | float | Angle between sun azimuth and row azimuth (°); proxy for shadow geometry |

To regenerate from raw sources in `data/covariates/raw/`:

```bash
python scripts/build_flight_covariates.py
```

---

### data/covariates/image_covariates.csv

Per-image quality metrics: **26,071 rows × 7 columns** (one row per plot image).

| Column | Type | Description |
|--------|------|-------------|
| `image_filename` | str | Primary join key |
| `flight_id` | str | Flight identifier (`YYYYMMDD_FIELD`) |
| `field` | str | Field site identifier |
| `frac_weed` | float | Fraction of image pixels classified as weed (0–1); 158 nulls (~0.6%) |
| `sf_illuminorm` | float | Illumination-normalized shadow fraction (0–1) |
| `mean_brightness` | float | Mean pixel brightness (0–255) |
| `contrast_rms` | float | RMS contrast (pixel standard deviation) |

To regenerate:

```bash
python scripts/build_image_covariates.py --image-dir /path/to/plot/images
```

---

### data/covariates/raw/

Source files consumed by the covariate scripts. These are committed for reproducibility and should not need to be modified.

| File | Rows | Description |
|------|------|-------------|
| `flight_times.csv` | 27 | Manual flight start/end times per flight |
| `frac_weed.csv` | 26,197 | Raw weed fractions per image (pre-deduplication) |
| `gdd.csv` | 324 | Daily GDD values (date, gdd) for AGDD computation |
| `inoculation_dates.csv` | 3 | Inoculation date per season year |
| `irradiance_minute.csv` | 19,066 | Minute-resolution EcoNet station data (radiation W m⁻², PAR µmol m⁻² s⁻¹); includes `QCF` quality-control flags |
| `planting_dates.csv` | 6 | Planting date per field (used for AGDD base date) |

---

### Regenerating data/covariates/raw/frac_weed.csv

`frac_weed.csv` records the fraction of weed-height pixels in each plot image, derived from per-flight Canopy Height Models (CHM = DSM − DTM). It is produced by [scripts/weed_pressure_pipeline.py](scripts/weed_pressure_pipeline.py) and archived here as the raw input to `build_image_covariates.py`.

**The per-flight DSMs and DTMs are not distributed with this repository or the Ag Data Commons data deposit.** They must be regenerated from the raw UAV imagery using the Metashape photogrammetry processing pipeline maintained in a separate repository: [nirwan1265/metashape](https://github.com/nirwan1265/metashape). Run that pipeline first to produce the `*_dsm.tif` / `*_dtm.tif` outputs for each flight, then point `weed_pressure_pipeline.py` at the resulting directory tree:

```bash
python scripts/weed_pressure_pipeline.py \
    --config configs/weed_pressure_config.yaml \
    --base-dir /path/to/dsm_dtm/tree \
    --shapefile-dir /path/to/plot_outline/shapefiles \
    --output-dir data/covariates/raw
```

See the module docstring in `scripts/weed_pressure_pipeline.py` for the expected `{YEAR}/{FIELD}/{FLIGHT_FOLDER}/` directory layout, the DTM reference strategy (per-flight vs. earliest-flight-in-season), and the height classification thresholds.

---

### experiments/\<model\>\_cv0/cv_results.csv

Per-fold training and test metrics: **3 rows × 16 columns** (one row per fold). Written by `train_cv0.py` at the end of each fold.

| Column | Description |
|--------|-------------|
| `fold` | Fold index (1, 2, 3) |
| `fold_name` | Fold label (`fold_23`, `fold_24`, `fold_25`) |
| `train_samples` | Training set size |
| `val_samples` | Validation set size |
| `test_samples` | Test set size |
| `best_val_loss` | Best MSE loss on the validation set |
| `best_val_r2` | Validation R² at the best-loss checkpoint |
| `final_train_loss` | Training loss at the final epoch |
| `final_train_r2` | Training R² at the final epoch |
| `final_val_loss` | Validation loss at the final epoch |
| `final_val_r2` | Validation R² at the final epoch |
| `test_loss` | MSE loss on the test set (best checkpoint) |
| `test_r2` | Test R² (best checkpoint) |
| `test_rmse` | Test RMSE |
| `test_mae` | Test MAE |
| `epochs_trained` | Total epochs completed before early stopping |

---

### experiments/\<model\>\_cv0/history/\<fold\>\_history.csv

Epoch-level training curves. Columns vary slightly by model but the standard schema is:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number (1-indexed) |
| `train_loss` | Training MSE loss |
| `train_r2` | Training R² |
| `val_loss` | Validation MSE loss |
| `val_r2` | Validation R² |
| `val_rmse` | Validation RMSE |
| `val_mae` | Validation MAE |

---

### experiments/\<model\>\_cv0/predictions/

**Per-fold prediction files** (`fold_2023_test_predictions.csv`, etc.): **n_test rows × 5 columns**.

| Column | Description |
|--------|-------------|
| `image_filename` | Primary join key |
| `actual` | Ground-truth SLB score (from test split) |
| `predicted` | Model output (continuous regression value) |
| `signed_error` | `predicted − actual` (positive = overprediction) |
| `abs_error` | `|predicted − actual|` |

**Combined prediction file** (`cv0_combined_predictions.csv`): **26,071 rows × 11 columns**. Concatenates all three folds and adds context columns.

| Column | Description |
|--------|-------------|
| `model` | Model identifier string (e.g., `eva02_base`) |
| `fold` | Fold label (e.g., `fold_25`) |
| `test_year` | Four-digit test year (2023, 2024, or 2025) |
| `image_filename` | Primary join key |
| `date` | Flight date (YYYY-MM-DD) |
| `field` | Field site identifier |
| `plot` | Plot number |
| `actual` | Ground-truth score |
| `predicted` | Model prediction |
| `signed_error` | `predicted − actual` |
| `abs_error` | `|predicted − actual|` |

---

### results/cv0_aggregate_results.csv

Long-format summary of test metrics for all nine models across all three folds: **27 rows × 12 columns**. This is the primary input for Fig. 3 and the manuscript results table.

| Column | Description |
|--------|-------------|
| `model` | Internal model identifier |
| `display_name` | Human-readable name (e.g., `EVA-02-B`) |
| `family` | Architecture family (`CNN`, `ViT`, `Hybrid`) |
| `fold` | Fold index (1, 2, 3) |
| `fold_name` | Fold label (`fold_23`, `fold_24`, `fold_25`) |
| `test_year` | Held-out test year (2023, 2024, or 2025) |
| `n_train` | Training set size |
| `n_val` | Validation set size |
| `n_test` | Test set size |
| `r2` | Test R² |
| `mae` | Test MAE |
| `rmse` | Test RMSE |

---

## Environment Setup

Requires **conda** (Miniforge or Anaconda). GPU training requires a CUDA-capable GPU; see the PyTorch/CUDA note in `environment.yaml` for the correct install command for your CUDA version.

```bash
# 1. Clone the repository
git clone https://github.com/GageLab/uav4slb.git
cd uav4slb

# 2. Create the conda environment (CPU-only by default)
conda env create -f environment.yaml
conda activate uav4slb

# 3. [GPU users] Replace PyTorch with the appropriate CUDA build.
#    Example for CUDA 12.4 (used in the original study on NC State's Sunny HPC):
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

**Python version:** 3.11 (developed and tested). Python 3.10+ should work but is untested.

**Key dependencies:**

| Package | Version | Role |
|---------|---------|------|
| `torch` | 2.5.1 | Deep learning framework |
| `timm` | 1.0.22 | Pretrained model registry (all 9 architectures) |
| `albumentations` | 2.0.8 | Train-time image augmentation |
| `omegaconf` | 2.3.0 | YAML config management |
| `rasterio` | 1.4.4 | GeoTIFF image loading |
| `geopandas` | 1.1.2 | Shapefile I/O |
| `pvlib` | 0.15.0 | Solar position (SPA algorithm) |

See `environment.yaml` for the full pinned dependency list with rationale comments.

---

## Reproducing Results

All scripts write outputs to deterministic paths. Running every step in order will reproduce the numbers reported in the manuscript. Steps 1 and 2 can be skipped if using the pre-computed files already committed to the repository.

### Step 1: Prepare Cross-Validation Splits *(optional)*

Pre-generated splits are committed at `data/cv_splits/cv0/`. To regenerate:

```bash
python scripts/create_cv_splits.py \
    --labels-csv data/labels/full_dataset.csv \
    --output-dir data/cv_splits/cv0 \
    --seed 42
```

### Step 2: Build Covariates *(optional)*

Pre-computed covariate files are committed at `data/covariates/`. To regenerate from raw sources:

```bash
# Flight-level solar geometry and irradiance
python scripts/build_flight_covariates.py

# Per-image weed fraction (requires DSMs/DTMs — see "Regenerating
# data/covariates/raw/frac_weed.csv" below)
python scripts/weed_pressure_pipeline.py --config configs/weed_pressure_config.yaml

# Per-image quality metrics (requires image directory)
python scripts/build_image_covariates.py --image-dir /path/to/plot/images
```

### Step 3: Train All Models

Each model is trained with `scripts/train_cv0.py` using its corresponding YAML config. The script iterates over all three CV0 folds automatically.

**Train a single model (all three folds):**

```bash
python scripts/train_cv0.py --config configs/eva02_base_cv0.yaml
```

**Train all nine models sequentially:**

```bash
bash scripts/run_all_models.sh
```

**Train a single fold only** (useful for cluster submission):

```bash
python scripts/train_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --fold fold_2025
```

**Resume an interrupted run** (restores full optimizer, scheduler, AMP, and epoch state):

```bash
python scripts/train_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --fold fold_2025 \
    --resume experiments/eva02_base_cv0/checkpoints/fold_2025/checkpoint_latest.pt
```

**HPC/SLURM users:** See `scripts/run_train_cv0.sh` for a SLURM template. The original study used NC State's Sunny HPC with one NVIDIA A100 (80 GB) per job.

**Mixed precision:** EVA-02-B uses bfloat16 AMP; all other models use float16 with GradScaler. The precision mode is set automatically from the config.

**Training outputs** are written to `experiments/<experiment.name>/`:

| Path | Description |
|------|-------------|
| `checkpoints/<fold>/checkpoint_best.pt` | Best-validation-loss weights |
| `checkpoints/<fold>/checkpoint_latest.pt` | Last-epoch state (preemption recovery; gitignored) |
| `predictions/<fold>_test_predictions.csv` | Per-fold test predictions (5 cols) |
| `predictions/cv0_combined_predictions.csv` | All folds combined (11 cols) |
| `history/<fold>_history.csv` | Epoch-level training curves |
| `logs/cv0_<fold>.log` | Full training log |
| `cv_results.csv` | Per-fold metrics (3 rows × 16 cols) |
| `summary.json` | Machine-readable aggregate metrics |
| `config_resolved.yaml` | Fully resolved OmegaConf snapshot |

### Step 4: Generate Predictions

To reproduce the paper's test-set predictions from committed checkpoints:

```bash
python scripts/predict_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --weights-dir experiments/eva02_base_cv0/checkpoints
```

To run inference on new images with a specific fold's checkpoint:

```bash
python scripts/predict_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --weights-dir experiments/eva02_base_cv0/checkpoints \
    --fold fold_2025 \
    --image-dir /path/to/new/plot_images \
    --labels-csv /path/to/new_labels.csv \
    --output predictions_new_season.csv
```

The output CSV follows the 5-column per-fold schema: `image_filename`, `actual`, `predicted`, `signed_error`, `abs_error`. If `--labels-csv` is not provided, `actual` and `signed_error` will be null.

### Step 5: Evaluate and Aggregate Results

**Single model evaluation** (scatter plots + metrics tables):

```bash
python scripts/evaluate_cv0.py \
    --experiment-dir experiments/eva02_base_cv0
```

Outputs are written to `results/eva02_base_cv0/evaluation/` and include per-fold and summary metrics CSVs (`metrics_per_fold.csv`, `metrics_summary.csv`, `metrics_aggregate.json`) and figure panels.

**Cross-model comparison table** (reproduces manuscript Table):

```bash
python scripts/evaluate_cv0.py \
    --compare-dirs experiments/eva02_base_cv0 experiments/convnextv2_base_cv0 \
    --model-labels "EVA-02-B" "ConvNeXt V2-B"
```

**Aggregate results table** (primary input for Fig. 3):

```bash
python scripts/evaluate_cv0.py --aggregate-table
# Output: results/cv0_aggregate_results.csv (27 rows × 12 cols)
```

### Step 6: Reproduce Figures

Each figure has a self-contained script in `figures/`. All scripts share the style module at `figures/style.py` and write paired PDF + PNG outputs at 150 DPI.

```bash
# Single figure
python figures/fig03_model_comparison/plot_model_comparison.py

# All figures
for script in figures/fig*/plot_*.py figures/supplemental/fig*/plot_*.py; do
    python "$script"
done
```

---

## Model Architectures

All nine architectures are accessed via [timm](https://github.com/huggingface/pytorch-image-models) (v1.0.22). Each is wrapped in a common interface defined in `src/models/base_model.py` that adds a regression head (global average pool → dropout(0.3) → linear → scalar) on top of the pretrained backbone.

| Model | Family | Params | Input | Pretraining | timm ID |
|-------|--------|--------|-------|-------------|---------|
| **EVA-02-B** *(primary)* | ViT | ~86 M | 448 × 448 | CLIP-guided MIM (merged-38M)<sup>a | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` |
| DINOv2 ViT-B/14 | ViT | ~86 M | 518 × 518<sup>b | Discriminative SSL (DINO + iBOT, LVD-142M) | `vit_base_patch14_dinov2.lvd142m` |
| DINOv2 ViT-S/14 | ViT | ~22 M | 448 × 448 | Discriminative SSL (DINO + iBOT, LVD-142M) | `vit_small_patch14_dinov2.lvd142m` |
| ConvNeXt V2-B | CNN | ~89 M | 224 × 224 | FCMAE self-supervised | `convnextv2_base.fcmae_ft_in22k_in1k` |
| ConvNeXt V2-L | CNN | ~198 M | 224 × 224 | FCMAE self-supervised | `convnextv2_large.fcmae_ft_in22k_in1k` |
| EfficientNet V2-S | CNN | ~22 M | 384 × 384 | Supervised IN-21K → IN-1K| `tf_efficientnetv2_s.in21k_ft_in1k` |
| MaxViT-S | Hybrid | ~69 M | 224 × 224 | Supervised IN-1k | `maxvit_small_tf_224.in1k` |
| SwinV2-B | Hybrid | ~88 M | 256 × 256 | SimMIM → Supervised IN-22K → IN-1K | `swinv2_base_window12to16_192to256.ms_in22k_ft_in1k` |
| CoAtNet-2 | Hybrid | ~75 M | 224 × 224 | Supervised IN-12k<sup>c | `coatnet_2_rw_224.sw_in12k_ft_in1k` |

**EVA-02-B vs. DINOv2 ViT-B/14** is the controlled pairwise comparison that holds backbone size (~86 M parameters, ViT-B/14 patch size) constant while varying pretraining objective (CLIP-guided MIM vs. self-distillation) and positional encoding (RoPE vs. learned absolute).

<sup>a</sup> MIM pretraining with a CLIP vision encoder as teacher

<sup>b</sup> This input size is the timm default for this model ID, not a paper-specified training resolution

<sup>c</sup>
This is a timm pretrained weight, not the original paper's setup

**Shared training hyperparameters:**

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Weight decay | 5 × 10⁻² |
| Backbone LR | 2 × 10⁻⁵ |
| Head LR | 2 × 10⁻⁴ |
| LR schedule | Cosine decay with linear warmup |
| Loss function | MSE |
| Early stopping patience | 15 epochs |
| Head dropout | 0.3 |

See individual configs in `configs/` for model-specific batch sizes and input resolutions.

### Architecture Citations

| Model | Citation |
|-------|----------|
| **EVA-02-B** | Fang, Y., Sun, Q., Wang, X., Huang, T., Wang, X., & Cao, Y. (2024). EVA-02: A Visual Representation for Neon Genesis. *Image and Vision Computing*, 149, 105171. https://doi.org/10.1016/j.imavis.2024.105171 |
| **DINOv2 ViT-B/14, DINOv2 ViT-S/14** | Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. https://doi.org/10.48550/arXiv.2304.07193 |
| **ConvNeXt V2-B, ConvNeXt V2-L** | Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., & Xie, S. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders (arXiv:2301.00808). arXiv. https://doi.org/10.48550/arXiv.2301.00808 |
| **EfficientNet V2-S** | Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training (arXiv:2104.00298). arXiv. https://doi.org/10.48550/arXiv.2104.00298 |
| **MaxViT-S** | Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., & Li, Y. (2022). MaxViT: Multi-Axis Vision Transformer (arXiv:2204.01697). arXiv. https://doi.org/10.48550/arXiv.2204.01697 |
| **SwinV2-B** | Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., … Guo, B. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution (arXiv:2111.09883). arXiv. https://doi.org/10.48550/arXiv.2111.09883 |
| **CoAtNet-2** | Dai, Z., Liu, H., Le, Q. V., & Tan, M. (2021). CoAtNet: Marrying Convolution and Attention for All Data Sizes (arXiv:2106.04803). arXiv. https://doi.org/10.48550/arXiv.2106.04803 |

---

## Cross-Validation Design (CV0)

CV0 is a **leave-one-year-out** scheme. Each fold holds out one complete field season as the test set; the remaining two seasons form the train/val pool. This design directly evaluates temporal generalization — the capacity of a model trained on prior seasons to predict disease severity in an unseen future season with different genotypic populations, field sites, and environmental conditions.

| Fold | Train years | Val years | Test year | n_test |
|------|-------------|-----------|-----------|--------|
| `fold_2023` | 2024, 2025 | 2024, 2025 | 2023 | 2,601 |
| `fold_2024` | 2023, 2025 | 2023, 2025 | 2024 | 8,290 |
| `fold_2025` | 2023, 2024 | 2023, 2024 | 2025 | 15,180 |

Note that `fold_2023` has the smallest test set (2,601 images; only two field sites: G3 and I3B) and is the chronologically earliest fold. The asymmetric fold sizes reflect the expanding experimental design across seasons.

Aggregate test metrics are reported as mean ± SD across the three folds. Per-fold results for EVA-02-B: fold_2023 R² = 0.610, fold_2024 R² = 0.713, fold_2025 R² = 0.766.

---

## Configuration System

All experiment-variable parameters are stored in YAML configs under `configs/`. The codebase uses [OmegaConf](https://omegaconf.readthedocs.io/) (v2.3.0) for loading and resolution. A fully resolved snapshot is written to `experiments/<name>/config_resolved.yaml` at the start of each training run for exact reproducibility.

**The one value you must update** before running training is `data.image_dir` in each config (or pass `--image-dir` on the command line). Everything else reproduces manuscript results as-is.

```yaml
# Minimal config structure (eva02_base_cv0.yaml shown)
experiment:
  name: "eva02_base_cv0"

model:
  name: "eva02_base"
  architecture: "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
  pretrained: true
  dropout_rate: 0.3

data:
  image_dir: "/path/to/your/plot/images"   # ← update this
  labels_csv: "data/labels/full_dataset.csv"
  cv_splits_dir: "data/cv_splits/cv0"
```

---

## Checkpointing and Resumption

`train_cv0.py` saves two checkpoint files per fold:

- **`checkpoint_best.pt`** — best validation-loss weights; committed to `experiments/` and used by all downstream scripts.
- **`checkpoint_latest.pt`** — full training state saved every epoch (model weights, optimizer, scheduler, AMP GradScaler, epoch counter, patience counter, and an MD5 hash of the split CSV). Gitignored due to size (~700 MB for EVA-02-B).

To resume an interrupted run:

```bash
python scripts/train_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --fold fold_2025 \
    --resume experiments/eva02_base_cv0/checkpoints/fold_2025/checkpoint_latest.pt
```

For **incremental retraining** on an expanded dataset, use `--warm-start` with `configs/eva02_base_retrain.yaml`. This loads only model weights from a prior `checkpoint_best.pt` and begins training with fresh optimizer/scheduler state and halved learning rates:

```bash
python scripts/train_cv0.py \
    --config configs/eva02_base_retrain.yaml \
    --warm-start experiments/eva02_base_cv0/checkpoints/fold_2025/checkpoint_best.pt
```

> **Do not** use `eva02_base_cv0.yaml` with `--warm-start`. The learning rates in that config are calibrated for training from ImageNet weights, not from a domain-adapted checkpoint.

---

## Figure Reference

All scripts write paired PDF + PNG outputs (150 DPI, `pdf.fonttype=42` for editable PDF text). Run from the repository root. The [batlow](https://www.fabiocrameri.ch/batlow/) perceptually uniform colormap is used throughout (colorblind-safe).

| Figure | Description | Script | Key inputs |
|--------|-------------|--------|-----------|
| Fig. 2 | Dataset composition, score distribution, and label heaping analysis | `fig02_dataset_composition/plot_dataset_composition.py` | `data/labels/full_dataset.csv` → `fig02_half_integer_stats.csv` |
| Fig. 3 | Model comparison bar chart (mean test R² across folds) | `fig03_model_comparison/plot_model_comparison.py` | `results/cv0_aggregate_results.csv` |
| Fig. 4 | EVA-02-B test performance panels (actual vs. predicted violin; signed error) | `fig04_test_performance/plot_test_performance.py` | `experiments/eva02_base_cv0/predictions/cv0_combined_predictions.csv` → `fig04_metrics_summary.csv` |
| Fig. 5 | Rater vs. model agreement (Pearson r comparison across rater pairs) | `fig05_rater_agreement/plot_rater_agreement.py` | `data/labels/long_format_ratings.csv` → `fig05_rater_agreement_stats.csv` |
| Fig. 6 | Flight-level covariate Spearman correlations with EVA-02-B MAE | `fig06_flight_covariates/plot_flight_covariates.py` | `data/covariates/flight_covariates.csv` + `experiments/eva02_base_cv0/` → `fig06_flight_covariates_stats.csv`, `tableS_flight_covariates.csv` |
| Fig. 7 | Image quality noise example panels (shadow, weed, blur) | `fig07_image_noise_examples/plot_image_noise_examples.py` | Image directory + `data/covariates/image_covariates.csv` |
| Fig. 8 | Image-level covariate scatter plots (Spearman ρ with abs/signed error) | `fig08_image_covariates/plot_image_covariates.py` | `data/covariates/image_covariates.csv` + `experiments/eva02_base_cv0/` → `fig08_image_covariates_stats.csv` |
| Fig. 9 | Temporal misalignment: error by signed days between scoring and flight | `fig09_temporal_misalignment/plot_temporal_misalignment.py` | `data/labels/full_dataset.csv` + `experiments/eva02_base_cv0/` → `fig09_temporal_misalignment_stats.csv` |
| Fig. 10 | Likely mislabelled image examples (high-residual plots) | `fig10_mislabelled_examples/plot_mislabelled_examples.py` | Image directory + `experiments/eva02_base_cv0/predictions/` |
| Fig. S1 | Rating and flight timeline by season and field | `supplemental/figS1_flight_timeline/plot_flight_timeline.py` | `data/labels/full_dataset.csv` |
| Fig. S2 | Score distributions by year | `supplemental/figS2_score_distributions/plot_score_distributions.py` | `data/labels/full_dataset.csv` → `figS2_score_distributions_stats.csv` |
| Fig. S3 | Per-fold R² bar chart for all 9 models | `supplemental/figS3_fold_r2/` | `results/cv0_aggregate_results.csv` |
| Fig. S5 | Temporal alignment: distribution of signed days-diff by year | `supplemental/figS5_temporal_alignment/plot_temporal_alignment.py` | `data/labels/full_dataset.csv` → `figS5_temporal_alignment_stats.csv` |

---

## Data Availability

Raw UAV images, processed plot images, and trained model checkpoints are deposited at:

> **USDA Ag Data Commons:** [DOI to be inserted upon acceptance]

The deposit includes all 27 flight images (all years and sites), the ~26,000 sliced plot images used for training and evaluation, and trained `checkpoint_best.pt` files for EVA-02-B across all three CV0 folds. Image filenames in the deposit match `data/labels/full_dataset.csv` exactly.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{hammett2025uav4slb,
  title   = {Aerial imagery and deep learning accurately estimate maize foliar
             {disease severity},
  author  = {Hammett, Cole and Rumley, Katelyn and Balint-Kurti, Peter and Gage, Joseph L.},
  journal = {The Plant Phenome Journal},
  year    = {2026},
  doi     = {to be inserted upon acceptance}
}
```

**Corresponding author:** Joseph L. Gage (jlgage@ncsu.edu), Department of Crop & Soil Sciences, NC State University.

---

## License

This repository is released under the [MIT License](LICENSE). Model weights downloaded by timm are subject to their respective upstream licenses (see timm documentation).
