#!/bin/bash
# End-to-end v2 run on a machine with a real GPU: attention temporal pooling plus
# chroma-key signer augmentation.
#
#   bash run_v2.sh                 # 40 epochs
#   bash run_v2.sh 70              # 70 epochs
#   SKIP_SYNTH=1 bash run_v2.sh    # reuse synthetic clips from a previous run
#
# Steps 1-2 are idempotent and skip themselves when their outputs are already
# present, so re-running after an interrupted training does not redo them.
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS="${1:-40}"
CONFIG=configs/attn_pool_aug.yaml
PY="${PYTHON:-python}"
SYNTH_ROOT=dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px-synth

# MPS lacks a CTC kernel; harmless on CUDA, required on Apple Silicon.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=== 1/4  Annotations ==="
# Rebuilds {split}_info.npy, gloss_dict.npy and the ground-truth STMs from the
# CSVs. fix_annotations.py is not optional: the evaluator reads the STMs it
# writes into evaluation/slr_eval/, and if those still describe an older split
# every WER is measured against the wrong reference.
if [ -f preprocess/CESLR/gloss_dict.npy ] && [ -z "${FORCE_PREP:-}" ]; then
    echo "    present (FORCE_PREP=1 to rebuild)"
else
    (cd preprocess && "$PY" dataset_preprocess.py --input-res 256x256px)
    "$PY" preprocess/fix_annotations.py
fi

echo
echo "=== 2/4  Synthetic clips (chroma-key, CPU) ==="
# Three appearance-shifted copies per training clip. The signer is matted off the
# green screen and recomposited, so geometry is never resampled and the signing
# stays pixel-exact -- the gloss label still holds.
if [ -n "${SKIP_SYNTH:-}" ]; then
    echo "    skipped (SKIP_SYNTH set)"
elif [ -d "$SYNTH_ROOT/train" ] && [ -z "${FORCE_SYNTH:-}" ]; then
    echo "    present: $(find "$SYNTH_ROOT/train" -maxdepth 1 -type d | wc -l) variants"
    echo "    (FORCE_SYNTH=1 to regenerate)"
else
    "$PY" preprocess/generate_synthetic_clips.py --split train --variants 3
fi

echo
echo "=== 3/4  Train ==="
WORK_DIR=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['work_dir'])")
# Resume from the newest checkpoint if one exists, so an interrupted run picks up
# where it stopped rather than restarting from epoch 0.
RESUME=""
LATEST=$(ls -t "${WORK_DIR}"*.pt 2>/dev/null | head -1 || true)
[ -n "$LATEST" ] && RESUME="--load-checkpoints $LATEST" && echo "    resuming from $LATEST"

"$PY" main.py --config "$CONFIG" --device "${DEVICE:-0}" \
    --num-epoch "$EPOCHS" --keep-last "${KEEP_LAST:-3}" $RESUME

echo
echo "=== 4/4  Evaluate + fairness ==="
BEST=$(ls "${WORK_DIR}"dev_*_model.pt 2>/dev/null | sort -t_ -k2 -n | head -1)
if [ -z "$BEST" ]; then
    echo "    no checkpoint found; training produced nothing"
    exit 1
fi
echo "    best checkpoint: $BEST"
"$PY" main.py --config "$CONFIG" --device 0 --phase test --load-weights "$BEST"

for MODE in dev test; do
    echo
    echo "--- $MODE ---"
    "$PY" evaluation/fairness_report.py --work-dir "$WORK_DIR" --mode "$MODE"
done

echo
echo "=== Done. Results in ${WORK_DIR} ==="
