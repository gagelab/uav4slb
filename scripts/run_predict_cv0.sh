#!/usr/bin/env bash
# =============================================================================
# scripts/run_predict_cv0.sh
# =============================================================================
# Batch-generate predictions for all 9 CV0 model checkpoints.
#
# This script assumes:
#   1. train_cv0.py has already been run for every architecture and the
#      best checkpoints are at:
#        experiments/<model>_cv0/checkpoints/fold_2{3,4,5}/checkpoint_best.pt
#   2. The published CV0 splits are at data/cv_splits/cv0/
#   3. The conda environment uav_for_slb is active.
#
# Usage
# -----
#   bash scripts/run_predict_cv0.sh               # all 9 models
#   MODEL=eva02_base_cv0 bash scripts/run_predict_cv0.sh   # single model
#
# Output
# ------
#   experiments/<model>_cv0/predictions/fold_*_test_predictions.csv
#   experiments/<model>_cv0/predictions/cv0_all_folds_predictions.csv
# =============================================================================

set -euo pipefail

CONDA_ENV="uav_for_slb"
SCRIPT="scripts/predict_cv0.py"

# Architectures to predict (name matches config filename stem and
# experiments/<name>/checkpoints directory)
declare -a MODELS=(
    "coatnet2_cv0"
    "convnextv2_base_cv0"
    "convnextv2_large_cv0"
    "dinov2_vitb14_cv0"
    "dinov2_vits14_cv0"
    "efficientnetv2_s_cv0"
    "eva02_base_cv0"
    "maxvit_small_cv0"
    "swinv2_base_cv0"
)

# If MODEL env var is set, run only that one
if [[ -n "${MODEL:-}" ]]; then
    MODELS=("$MODEL")
fi

# Activate conda environment if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

echo "=========================================="
echo "  CV0 Prediction — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Models: ${MODELS[*]}"
echo "=========================================="

for MODEL_NAME in "${MODELS[@]}"; do
    CONFIG="configs/${MODEL_NAME}.yaml"
    WEIGHTS_DIR="experiments/${MODEL_NAME}/checkpoints"

    echo ""
    echo "------------------------------------------"
    echo "  Model  : $MODEL_NAME"
    echo "  Config : $CONFIG"
    echo "  Weights: $WEIGHTS_DIR"
    echo "------------------------------------------"

    if [[ ! -f "$CONFIG" ]]; then
        echo "  [WARN] Config not found: $CONFIG — skipping."
        continue
    fi
    if [[ ! -d "$WEIGHTS_DIR" ]]; then
        echo "  [WARN] Checkpoint dir not found: $WEIGHTS_DIR — skipping."
        continue
    fi

    python "$SCRIPT" \
        --config      "$CONFIG" \
        --weights-dir "$WEIGHTS_DIR" \
        --num-workers 4

    echo "  [OK] $MODEL_NAME predictions complete."
done

echo ""
echo "=========================================="
echo "  All predictions complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
