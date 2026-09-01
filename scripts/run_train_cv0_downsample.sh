#!/usr/bin/env bash
# ==============================================================================
# run_train_cv0.sh
# ================
# Launch CV0 training for any architecture in a detached tmux session.
#
# This single script replaces the nine model-specific shell scripts from the
# legacy repository.  The model is determined entirely by the --config argument
# passed to train_cv0.py; no architecture-specific logic lives here.
#
# Usage — standard training
# ─────────────────────────
#   bash scripts/run_train_cv0_downsample.sh --config configs_downsample/efficientnetv2_s_cv0.yaml
#   bash scripts/run_train_cv0_downsample.sh --config configs_downsample/eva02_base_cv0.yaml --gpu 2
#   bash scripts/run_train_cv0_downsample.sh --config configs_downsample/eva02_base_cv0.yaml \
#       --max-folds 1 --max-epochs 2   # smoke-test
#
# Usage — resume an interrupted run (same data, full state restore)
# ───────────────────────────────────────────────────────────────── 
#   bash scripts/run_train_cv0_downsample.sh \
#       --config configs_downsample/eva02_base_cv0.yaml \
#       --resume results/eva02_base_cv0/checkpoints/fold_2025/checkpoint_latest.pt \
#       --fold fold_2025
#
#   Notes:
#     - --fold is required with --resume to target the correct split CSV and
#       verify the stored data integrity hash.
#     - The training split hash in the checkpoint must match the current split
#       CSV.  If the data has changed since the checkpoint was saved, the
#       script will abort with an informative error.  Use --warm-start instead.
#
# Usage — warm-start retraining on expanded dataset (EVA-02-B only)
# ─────────────────────────────────────────────────────────────────
#   bash scripts/run_train_cv0_downsample.sh \
#       --config configs_downsample/eva02_base_retrain.yaml \
#       --warm-start results/eva02_base_cv0/checkpoints/fold_2025/checkpoint_best.pt
#
#   Notes:
#     - Uses eva02_base_retrain.yaml, which specifies lower LRs and a shorter
#       schedule relative to the original training config.
#     - Optimizer, scheduler, and AMP scaler are freshly initialised from the
#       config — only the model weights are loaded from the checkpoint.
#     - No split hash verification is performed; it is expected and correct
#       that the data has changed (new field season appended).
#     - --fold is optional: omit to retrain all three folds from the same
#       source checkpoint, or specify --fold fold_2025 to target one fold.
#
# Environment variables
# ─────────────────────
#   PROJECT_ROOT   Absolute path to the repository root (auto-detected)
#   CONDA_ENV      Conda environment name (default: uav_for_slb)
#
# Notes
# ─────
#   - If a tmux session with the same name already exists, the script prompts
#     to kill and restart it, or attaches to the running session.
#   - Splits are auto-generated if cv0/ is not found under cfg.cv.splits_dir.
#   - All stdout + stderr is tee'd to <output_dir>/train_stdout.log.
#   - When --gpu is given, CPU cores are auto-pinned via scripts/select_cpu_affinity.py:
#     it snapshots live idle% and NUMA free memory for that GPU's NUMA node and
#     binds the job with numactl+taskset, so it doesn't spread across the whole
#     shared box or land on a memory-starved node. Skipped if --gpu is omitted
#     (GPU not known yet — train_cv0.py auto-selects it) or if you pass --no-affinity.
# ==============================================================================

set -euo pipefail

# ── Project root (auto-detect from script location) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# ── Configurable defaults ─────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-uav_for_slb}"

# ── Parse known flags; forward everything else to train_cv0.py ───────────────
# We extract --config to derive the session name and log directory.
# --resume, --warm-start, and --fold are passed through transparently to
# train_cv0.py — they do not require special handling in this launcher.
CONFIG=""
GPU_ID=""
NO_AFFINITY=false
REMAINING=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#--config=}"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            REMAINING+=("--gpu" "$2")
            shift 2
            ;;
        --gpu=*)
            GPU_ID="${1#--gpu=}"
            REMAINING+=("$1")
            shift
            ;;
        --no-affinity)
            NO_AFFINITY=true
            shift
            ;;
        *)
            REMAINING+=("$1")
            shift
            ;;
    esac
done

set -- "${REMAINING[@]}"

if [[ -z "$CONFIG" ]]; then
    echo "❌  --config is required."
    echo ""
    echo "Standard training:"
    echo "    bash scripts/run_train_cv0.sh --config configs/eva02_base_cv0.yaml"
    echo ""
    echo "Resume interrupted run:"
    echo "    bash scripts/run_train_cv0.sh \\"
    echo "        --config configs/eva02_base_cv0.yaml \\"
    echo "        --resume results/eva02_base_cv0/checkpoints/fold_2025/checkpoint_latest.pt \\"
    echo "        --fold fold_2025"
    echo ""
    echo "Warm-start on expanded dataset (EVA-02-B only):"
    echo "    bash scripts/run_train_cv0.sh \\"
    echo "        --config configs/eva02_base_retrain.yaml \\"
    echo "        --warm-start results/eva02_base_cv0/checkpoints/fold_2025/checkpoint_best.pt"
    exit 1
fi

# ── Derive model name and session / log names from the config filename ────────
CONFIG_BASENAME="$(basename "${CONFIG}" .yaml)"          # e.g. eva02_base_cv0
MODEL_NAME="${CONFIG_BASENAME%_cv0}"                      # e.g. eva02_base
# For retrain configs (e.g. eva02_base_retrain), strip _retrain suffix too
MODEL_NAME="${MODEL_NAME%_retrain}"

SESSION_NAME="${SESSION_NAME:-${CONFIG_BASENAME}}"        # full config name as session
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/downsample_results/${CONFIG_BASENAME}}"

# ── Helper ────────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ── Pre-flight: tmux ──────────────────────────────────────────────────────────
print_header "${CONFIG_BASENAME}"

if ! command -v tmux &>/dev/null; then
    echo "❌  tmux not found.  Install with:  sudo apt install tmux"
    exit 1
fi

# ── Activate conda ────────────────────────────────────────────────────────────
for conda_init in \
    ~/miniforge3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh \
    ~/miniconda3/etc/profile.d/conda.sh; do
    [[ -f "$conda_init" ]] && source "$conda_init" && break
done
conda activate "${CONDA_ENV}" 2>/dev/null || true

# ── Pick NUMA-local, currently-idle CPU cores for the target GPU ──────────────
# Snapshot-based: samples live CPU idle% and NUMA free memory right before
# launch, rather than a hardcoded core list, so it stays correct as other
# users' jobs come and go on the shared box.
AFFINITY_PREFIX=""
AFFINITY_OMP_THREADS=1
if [[ "${NO_AFFINITY}" == true ]]; then
    echo "CPU affinity  : disabled (--no-affinity)"
elif [[ -z "${GPU_ID}" ]]; then
    echo "CPU affinity  : skipped (no --gpu given; train_cv0.py will auto-select a GPU, so its NUMA node isn't known yet)"
else
    eval "$(python3 "${SCRIPT_DIR}/select_cpu_affinity.py" --gpu "${GPU_ID}")"
    if [[ -n "${AFFINITY_PREFIX}" ]]; then
        echo "CPU affinity  : ${AFFINITY_PREFIX}"
    else
        echo "CPU affinity  : skipped (could not determine — see warning above, if any)"
    fi
fi

# ── Print run configuration ────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
echo "Project root  : ${PROJECT_ROOT}"
echo "Conda env     : ${CONDA_ENV}"
echo "Config        : ${CONFIG}"
echo "Model name    : ${MODEL_NAME}"
echo "tmux session  : ${SESSION_NAME}"
echo "Output dir    : ${LOG_DIR}"
echo "Extra args    : $*"
echo ""

# ── Detect checkpoint mode from extra args for user-facing display ────────────
CKPT_MODE="fresh training from pretrained weights"
for arg in "$@"; do
    case "$arg" in
        --resume)       CKPT_MODE="RESUME (full state restore)" ;;
        --warm-start)   CKPT_MODE="WARM-START (weights only, fresh optimizer)" ;;
    esac
done
echo "Checkpoint mode: ${CKPT_MODE}"
echo ""

# ── Verify CV0 splits exist; generate them if not ─────────────────────────────
SPLITS_DIR=$(python3 -c "
import sys, yaml
try:
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)
    print(cfg.get('cv', {}).get('splits_dir', 'data/cv_splits'))
except Exception:
    print('data/cv_splits')
" "${CONFIG}" 2>/dev/null)

[[ "${SPLITS_DIR}" != /* ]] && SPLITS_DIR="${PROJECT_ROOT}/${SPLITS_DIR}"
CV0_DIR="${SPLITS_DIR}/cv0"

if [[ ! -d "${CV0_DIR}" ]]; then
    echo "⚠️   CV0 splits not found at: ${CV0_DIR}"
    echo "    Generating splits now ..."
    echo ""
    python3 scripts/create_cv_splits.py \
        --labels-csv "${PROJECT_ROOT}/data/labels/full_dataset.csv" \
        --output-dir "${SPLITS_DIR}" \
        --image-source-dir "${PROJECT_ROOT}/data/images" \
        --image-id-column image_filename \
        --target-column score \
        --year-column year \
        --cv-strategies cv0 \
        --auto-confirm-missing
    echo "✓  Splits generated."
else
    FOLD_COUNT=$(find "${CV0_DIR}" -maxdepth 1 -type d -name "fold_*" 2>/dev/null | wc -l)
    echo "✓  CV0 splits found: ${FOLD_COUNT} fold(s) in ${CV0_DIR}"
fi
echo ""

# ── Build the command to run inside the tmux session ─────────────────────────
EXTRA_ARGS="$*"

TRAIN_CMD=$(cat <<TMUXCMD
source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh  2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

conda activate ${CONDA_ENV}
cd ${PROJECT_ROOT}
mkdir -p ${LOG_DIR}

echo '============================================================'
echo '  ${CONFIG_BASENAME}'
echo '  Mode: ${CKPT_MODE}'
echo "  Started: \$(date)"
echo '============================================================'

OMP_NUM_THREADS=${AFFINITY_OMP_THREADS} MKL_NUM_THREADS=${AFFINITY_OMP_THREADS} \
${AFFINITY_PREFIX} python scripts/train_cv0.py \
    --config ${CONFIG} \
    --output-dir ${LOG_DIR} \
    ${EXTRA_ARGS} 2>&1 | tee ${LOG_DIR}/train_stdout.log

echo ''
echo '============================================================'
echo "  Training complete: \$(date)"
echo "  Results: ${LOG_DIR}"
echo '  Next step:'
echo "    python scripts/evaluate_cv0.py --experiment-dir ${LOG_DIR}"
echo '============================================================'
TMUXCMD
)

# ── Launch or reattach the tmux session ───────────────────────────────────────
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo ""
    echo "⚠️   Session '${SESSION_NAME}' already exists."
    echo "    To attach : tmux attach -t ${SESSION_NAME}"
    echo "    To kill   : tmux kill-session -t ${SESSION_NAME}"
    echo ""
    read -r -p "Kill existing session and restart? [y/N]: " confirm
    if [[ "${confirm}" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "${SESSION_NAME}"
    else
        echo "Attaching to existing session ..."
        tmux attach -t "${SESSION_NAME}"
        exit 0
    fi
fi

tmux new-session -d -s "${SESSION_NAME}" -x 220 -y 50
tmux send-keys -t "${SESSION_NAME}" "${TRAIN_CMD}" Enter

# ── Post-launch instructions ──────────────────────────────────────────────────
echo ""
echo "✅  Training launched in tmux session: ${SESSION_NAME}"
echo ""
echo "  Attach  :  tmux attach -t ${SESSION_NAME}"
echo "  Detach  :  Ctrl-b d"
echo "  Kill    :  tmux kill-session -t ${SESSION_NAME}"
echo "  Logs    :  tail -f ${LOG_DIR}/train_stdout.log"
echo "  TB      :  tensorboard --logdir ${LOG_DIR} --port 6006"
echo ""
echo "  Compare original vs retrained:"
echo "  tensorboard --logdir results/eva02_base_cv0:results/eva02_base_retrain --port 6007"
echo ""
