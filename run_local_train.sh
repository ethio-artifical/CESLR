#!/bin/bash
# Local CESLR training test on macOS (Apple Silicon / MPS).
#
#   bash run_local_train.sh          # 3 epochs (default)
#   bash run_local_train.sh 10       # 10 epochs
#
# NOTE ON batch_size: it is 1 here on purpose. On MPS, a batch of 2 containing a
# ~280-frame clip blows past unified-memory and the step goes from ~2s to ~83s.
# batch_size 1 keeps every step at ~1.5s. On Kaggle (dedicated 16GB T4) use
# configs/baseline.yaml, which keeps batch_size 2.
set -e
cd "$(dirname "$0")"
EPOCHS="${1:-3}"

source ../CorrNet/.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 1. Preprocess. Two independent outputs, checked separately: the resized frames
#    (slow) and the annotations -- info npys / gloss dict / STMs (fast). Checking
#    only the frames would silently skip regenerating missing annotations.
RESIZE_FLAGS=""
if [ ! -d "dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px/train" ]; then
    echo "=== Frames not resized yet: will resize to 256x256 ==="
    RESIZE_FLAGS="--process-image --multiprocessing"
else
    echo "=== 256x256 frames present (delete the folder to force a rebuild) ==="
fi

if [ -n "$RESIZE_FLAGS" ] || [ ! -f "preprocess/CESLR/gloss_dict.npy" ]; then
    echo "=== Generating annotations (gloss dict, info npys, ground-truth STMs) ==="
    (cd preprocess && python dataset_preprocess.py $RESIZE_FLAGS)
    python preprocess/fix_annotations.py
else
    echo "=== Annotations present ==="
fi

# 2. Config for the local run
cat > configs/local_test.yaml <<EOF
feeder: dataset.dataloader_video.BaseFeeder
phase: train
dataset: CESLR
num_epoch: ${EPOCHS}
work_dir: ./work_dir/local_test/
batch_size: 1
random_seed: 0
test_batch_size: 1
num_worker: 4
device: 0
log_interval: 40
eval_interval: 1
save_interval: 1
evaluate_tool: python
loss_weights:
  SeqCTC: 1.0

optimizer_args:
  optimizer: Adam
  base_lr: 0.0001
  step: [ 20, 35]
  learning_ratio: 1
  weight_decay: 0.0001
  start_epoch: 0
  nesterov: False

feeder_args:
  mode: 'train'
  datatype: 'video'
  num_gloss: -1
  drop_ratio: 1.0

model: slr_network.SLRModel
decode_mode: beam
model_args:
  num_classes: 13  # overridden at runtime from gloss_dict
  c2d_type: resnet18
  conv_type: 2
  use_bn: 1
  share_classifier: False
  weight_norm: False
EOF

# 3. Train (per-epoch dev WER, checkpoint saved every epoch)
echo "=== Training ${EPOCHS} epoch(s) on MPS -- roughly 5 min/epoch ==="
python main.py --config configs/local_test.yaml --device 0

echo
echo "=== Done. Checkpoints + logs in work_dir/local_test/ ==="
grep -E "Dev WER" work_dir/local_test/log.txt | tail -20
