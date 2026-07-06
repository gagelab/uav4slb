#!/usr/bin/env bash
# ==============================================================================
# run_all_models.sh
# =================
# Launch CV0 training for all nine SLB architectures, one model at a time.
#
# Each model is launched in its own named tmux session via run_train_cv0.sh.
# Models run sequentially by default (the script waits for you to launch each
# one); pass --parallel to launch all sessions immediately without confirmation.
#
# Usage
# -----
#   # Interactive: confirm before launching each model
#   bash scripts/run_all_models.sh
#
#   # Non-interactive: launch all models immediately in separate tmux sessions
#   bash scripts/run_all_models.sh --parallel
#
#   # Forward GPU or AMP options to every model
#   bash scripts/run_all_models.sh --parallel --gpu 0
#   bash scripts/run_all_models.sh --parallel --amp-dtype float16
#
#   # Smoke-test all models (1 fold, 2 epochs each)
#   bash scripts/run_all_models.sh --parallel --max-folds 1 --max-epochs 2
#
# Architecture order
# ------------------
# Models are launched in increasing parameter count so that smaller models
# (faster to train) complete early and can be spot-checked before committing
# GPU time to the larger ones.
#
# Monitor sessions
# ----------------
#   tmux ls                                 # list all sessions
#   tmux attach -t efficientnetv2_s_cv0     # attach to a specific session
#   tensorboard --logdir experiments/ --port 6006   # compare all models
#
# After all models complete, run the cross-model comparison:
#   python scripts/evaluate_cv0.py \
#       --compare-dirs experiments/efficientnetv2_s_cv0 \
#                      experiments/dinov2_vits14_cv0 \
#                      experiments/dinov2_vitb14_cv0 \
#                      experiments/maxvit_small_cv0 \
#                      experiments/swinv2_base_cv0 \
#                      experiments/coatnet2_cv0 \
#                      experiments/convnextv2_base_cv0 \
#                      experiments/convnextv2_large_cv0 \
#                      experiments/eva02_base_cv0 \
#       --model-labels "EfficientNet V2-S" "DINOv2 ViT-S/14" "DINOv2 ViT-B/14" \
#                      "MaxViT-S" "SwinV2-B" "CoAtNet-2" \
#                      "ConvNeXt V2-B" "ConvNeXt V2-L" "EVA-02-B" \
#       --output-dir results/model_comparison
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Architecture order (ascending parameter count, then family) ───────────────
# Config filenames map directly to the MODEL_REGISTRY keys in train_cv0.py.
CONFIGS=(
    "configs/efficientnetv2_s_cv0.yaml"    # CNN  |  ~21 M  | 384×384 | supervised IN-21k
    "configs/dinov2_vits14_cv0.yaml"        # ViT  |  ~22 M  | 448×448 | DINOv2 self-supervised
    "configs/maxvit_small_cv0.yaml"         # Hybrid | ~69 M | 224×224 | supervised IN-1k
    "configs/swinv2_base_cv0.yaml"          # Hybrid | ~88 M | 256×256 | supervised IN-21k
    "configs/coatnet2_cv0.yaml"             # Hybrid | ~75 M | 224×224 | supervised IN-12k
    "configs/convnextv2_cv0.yaml"           # CNN  |  ~88 M  | 224×224 | FCMAE self-supervised
    "configs/convnextv2_large_cv0.yaml"     # CNN  |  ~197 M | 224×224 | FCMAE self-supervised
    "configs/dinov2_vitb14_cv0.yaml"               # ViT  |  ~86 M  | 518×518 | DINOv2 self-supervised
    "configs/eva02_base_cv0.yaml"           # ViT  |  ~86 M  | 448×448 | CLIP-MIM merged-38M
)

# ── Parse script-level flags (everything else forwarded to run_train_cv0.sh) ──
PARALLEL=false
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --parallel) PARALLEL=true ;;
        *)          PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

# ── Print plan ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  UAV-SLB  |  CV0 All-Models Training"
echo "============================================================"
echo "  Mode    : $(if $PARALLEL; then echo 'parallel (all launched immediately)'; else echo 'interactive (confirm each model)'; fi)"
echo "  Models  : ${#CONFIGS[@]}"
echo "  Extra   : ${PASSTHROUGH_ARGS[*]:-none}"
echo "============================================================"
echo ""

# ── Verify all config files exist before launching anything ───────────────────
MISSING_CONFIGS=()
for cfg in "${CONFIGS[@]}"; do
    [[ -f "${SCRIPT_DIR}/../${cfg}" ]] || MISSING_CONFIGS+=("$cfg")
done
if [[ ${#MISSING_CONFIGS[@]} -gt 0 ]]; then
    echo "❌  The following config files were not found:"
    for cfg in "${MISSING_CONFIGS[@]}"; do echo "    $cfg"; done
    echo ""
    echo "    Create or populate configs/ before running this script."
    exit 1
fi

# ── Launch each model ─────────────────────────────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
    # Derive model label for display (strip path and _cv0.yaml suffix)
    model_label="$(basename "${cfg}" _cv0.yaml)"

    echo "──────────────────────────────────────────────────────────"
    echo "  Model: ${model_label}"
    echo "  Config: ${cfg}"

    if ! $PARALLEL; then
        # Interactive mode: give the user a chance to skip or abort
        read -r -p "  Launch this model? [y/n/q (quit)]: " resp
        case "${resp}" in
            y|Y) ;;
            n|N) echo "  Skipping ${model_label}."; continue ;;
            q|Q) echo "  Aborting."; exit 0 ;;
            *)   echo "  Please enter y, n, or q."; continue ;;
        esac
    fi

    bash "${SCRIPT_DIR}/run_train_cv0.sh" \
        --config "${cfg}" \
        "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

    # Brief pause between launches to avoid tmux socket race conditions
    sleep 2
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All models launched."
echo "============================================================"
echo ""
echo "  Monitor sessions:"
echo "    tmux ls"
echo "    tensorboard --logdir experiments/ --port 6006"
echo ""
echo "  After training completes, run evaluation:"
echo "    python scripts/evaluate_cv0.py \\"
echo "        --compare-dirs \\"
for cfg in "${CONFIGS[@]}"; do
    model="$(basename "${cfg}" _cv0.yaml)"
    echo "            experiments/${model}_cv0 \\"
done
echo "        --output-dir results/model_comparison"
echo ""
