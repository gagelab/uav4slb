#!/usr/bin/env bash
# ==============================================================================
# run_evaluate_cv0.sh
# ===================
# Shell wrapper for evaluate_cv0.py.
#
# Activates the conda environment, then runs the evaluation script in either
# single-experiment mode or cross-model comparison mode depending on the
# arguments provided.
#
# Usage — single experiment
# -------------------------
#   bash scripts/run_evaluate_cv0.sh --experiment-dir experiments/eva02_base_cv0
#
#   # With custom output directory
#   bash scripts/run_evaluate_cv0.sh \
#       --experiment-dir experiments/eva02_base_cv0 \
#       --output-dir .results/eva02_eval
#
# Usage — cross-model comparison (all nine architectures)
# -------------------------------------------------------
#   bash scripts/run_evaluate_cv0.sh \
#       --compare-dirs \
#           experiments/efficientnetv2_s_cv0 \
#           experiments/dinov2_vits14_cv0 \
#           experiments/dinov2_vitb14_cv0 \
#           experiments/maxvit_small_cv0 \
#           experiments/swinv2_base_cv0 \
#           experiments/coatnet2_cv0 \
#           experiments/convnextv2_base_cv0 \
#           experiments/convnextv2_large_cv0 \
#           experiments/eva02_base_cv0 \
#       --model-labels \
#           "EfficientNet V2-S" "DINOv2 ViT-S/14" "DINOv2 ViT-B/14" \
#           "MaxViT-S" "SwinV2-B" "CoAtNet-2" \
#           "ConvNeXt V2-B" "ConvNeXt V2-L" "EVA-02-B" \
#       --output-dir .results/model_comparison
#
#   # Re-evaluate from raw prediction CSVs (skip cached metrics_aggregate.json)
#   bash scripts/run_evaluate_cv0.sh \
#       --compare-dirs experiments/* \
#       --force-recompute
#
# All arguments are forwarded verbatim to evaluate_cv0.py.
# ==============================================================================

set -euo pipefail

# ── Auto-detect project root from script location ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-uav_for_slb}"

# ── Activate conda ────────────────────────────────────────────────────────────
for conda_init in \
    ~/miniforge3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh \
    ~/miniconda3/etc/profile.d/conda.sh; do
    [[ -f "$conda_init" ]] && source "$conda_init" && break
done
conda activate "${CONDA_ENV}" 2>/dev/null || true

cd "${PROJECT_ROOT}"

echo ""
echo "============================================================"
echo "  evaluate_cv0.py"
echo "  Args: $*"
echo "============================================================"
echo ""

# ── Run the evaluation script with all forwarded arguments ────────────────────
python scripts/evaluate_cv0.py "$@"

echo ""
echo "============================================================"
echo "  Evaluation complete."
echo "============================================================"
echo ""
