# End-to-End Continuous Ethiopian Sign Language Recognition (CESLR)

Recognises continuous Ethiopian Sign Language from video and outputs a sequence of
Amharic glosses. The model is a ResNet18 visual encoder → 1D temporal convolution →
BiLSTM, trained with CTC loss, following the
[VAC](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)
/ [CorrNet](https://arxiv.org/abs/2303.03202) line of work.

- **Input:** a clip of one signed sentence (folder of frames, or a video file)
- **Output:** a gloss sequence, e.g. `አባቴ ዘመድ ይወዳል`
- **Metric:** Word Error Rate (WER), lower is better

---

## Contents

1. [Requirements](#1-requirements)
2. [Setup by operating system](#2-setup-by-operating-system)
3. [Dataset layout](#3-dataset-layout)
4. [Preprocessing](#4-preprocessing)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Inference on a single clip](#7-inference-on-a-single-clip)
8. [Web demo](#8-web-demo)
9. [Training on Kaggle](#9-training-on-kaggle)
10. [Configuration reference](#10-configuration-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Repository layout](#12-repository-layout)

---

## 1. Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.9 | 3.9 and 3.10 are both known to work |
| RAM | 8 GB | 16 GB+ comfortable |
| GPU | optional | NVIDIA (CUDA), Apple Silicon (MPS), or CPU |
| Disk | ~5 GB | dataset + checkpoints |

The device is selected automatically at runtime: **CUDA → MPS → CPU**. You do not
need to change any code to move between machines.

### Python packages

| Package | Version used | Why |
|---|---|---|
| torch, torchvision | 2.8.0 / 0.23.0 | model + ImageNet-pretrained ResNet18 |
| numpy | ≥ 1.26 (2.x fine) | see the pyctcdecode note below |
| opencv-python | 4.11+ | frame reading and resizing |
| pandas, PyYAML, scipy, six, tqdm, matplotlib | recent | data handling, configs, progress |
| pyctcdecode + pygtrie | 0.5.0 / 2.5.0 | CTC beam-search decoding |
| gradio | 4.44.1 | optional, web demo only |
| pydantic | **2.10.6** | optional, pinned for gradio (see below) |
| eva-decord (macOS) / decord (Linux) | 0.6.1 | optional, video-file input only |

> **Do not install `ctcdecode`.** The upstream repos use it, but it only builds on
> Linux with an older toolchain. This fork uses **pyctcdecode**, which is pure
> Python and works everywhere. `utils/decode.py` falls back to greedy decoding
> automatically if pyctcdecode is missing.

> **Install pyctcdecode with `--no-deps`.** It declares `numpy<2.0.0`, which makes
> pip downgrade numpy and break any environment built on numpy 2 (notably Kaggle).
> That pin is stale — pyctcdecode works fine on numpy 2.

---

## 2. Setup by operating system

Pick your platform. All three end with the same working environment.

<details open>
<summary><b>Linux (NVIDIA GPU or CPU)</b></summary>

```bash
git clone https://github.com/ethio-artifical/CESLR.git
cd CESLR

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.1 build; see https://pytorch.org for other CUDA versions.
# For CPU-only, replace the index URL with https://download.pytorch.org/whl/cpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install numpy opencv-python pandas PyYAML scipy six tqdm matplotlib
pip install --no-deps pyctcdecode pygtrie

# Optional extras
pip install decord                            # video-file input
pip install gradio==4.44.1 pydantic==2.10.6   # web demo
```

Verify:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

</details>

<details>
<summary><b>macOS (Apple Silicon — M1/M2/M3/M4/M5)</b></summary>

macOS has no CUDA; PyTorch uses the **MPS** Metal backend instead.

```bash
git clone https://github.com/ethio-artifical/CESLR.git
cd CESLR

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install torch torchvision          # MPS support is included by default
pip install numpy opencv-python pandas PyYAML scipy six tqdm matplotlib
pip install --no-deps pyctcdecode pygtrie

# Optional extras
pip install eva-decord                        # video input (NOT plain `decord` on macOS)
pip install gradio==4.44.1 pydantic==2.10.6   # web demo
```

Verify:

```bash
python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
```

**Always export this before training or inference:**

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

CTC loss and a few other operators are not implemented for MPS; this lets them
fall back to CPU instead of crashing. `run_local_train.sh` sets it for you.

**Use `batch_size: 1` on macOS.** MPS shares unified memory with the OS, and a
batch of 2 containing a ~280-frame clip pushes past it — a step jumps from ~2 s to
~83 s. Batch 1 keeps every step near 1.2 s.

</details>

<details>
<summary><b>Windows</b></summary>

Two options. **WSL2 is strongly recommended** — it gives you the Linux path above,
including CUDA if you have an NVIDIA GPU.

**Option A — WSL2 (recommended)**

```powershell
wsl --install -d Ubuntu
```

Then open Ubuntu and follow the **Linux** instructions verbatim. Keep the repo
inside the WSL filesystem (`~/CESLR`), not on `/mnt/c/...` — filesystem calls
across that boundary are far slower, which matters when reading thousands of frames.

**Option B — native Windows**

```powershell
git clone https://github.com/ethio-artifical/CESLR.git
cd CESLR

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python pandas PyYAML scipy six tqdm matplotlib
pip install --no-deps pyctcdecode pygtrie
pip install gradio==4.44.1 pydantic==2.10.6   # optional demo
```

Native Windows caveats:

- `decord` has no official Windows wheel — use folders of frames rather than
  video files, or use WSL2.
- The WER evaluator shells out to `bash`, `cat`, `sort` and `sed`. Run training
  from **Git Bash** so those exist. WSL2 avoids this entirely.
- Set `num_worker: 0` in the config if you hit dataloader multiprocessing errors.

</details>

<details>
<summary><b>Conda (any OS)</b></summary>

```bash
conda create -n ceslr python=3.9 -y
conda activate ceslr

pip install torch torchvision           # add --index-url for a specific CUDA build
pip install numpy opencv-python pandas PyYAML scipy six tqdm matplotlib
pip install --no-deps pyctcdecode pygtrie
```

</details>

> **Paths with spaces:** if your checkout lives somewhere like
> `/Users/you/My Projects/CESLR`, always quote paths in shell commands
> (`cd "/Users/you/My Projects/CESLR"`). Unquoted paths silently split into
> multiple arguments.

---

## 3. Dataset layout

The code expects this structure under the repository root:

```
dataset/CESLR/CESLR-multisigner/
├── annotations/
│   └── manual/
│       ├── train.corpus.csv
│       ├── dev.corpus.csv
│       └── test.corpus.csv
└── features/
    └── fullFrame-256x256px/          # fullFrame-210x260px if not yet resized
        ├── train/
        │   └── <clip_id>/1/*.png     # frames; filename order = temporal order
        ├── dev/
        └── test/
```

Each CSV has a single header column, `id|folder|signer|annotation`, and
pipe-separated rows:

```
id|folder|signer|annotation
01day_2016_sentence_0|01day_2016_sentence_0/1/*.png|Signer01|አባቴ ዘመድ ይወዳል
```

| Field | Meaning |
|---|---|
| `id` | unique clip id; must match the folder name |
| `folder` | glob for that clip's frames, relative to the split directory |
| `signer` | signer id — used to build signer-independent splits |
| `annotation` | space-separated Amharic glosses (the training target) |

If your dataset lives elsewhere, symlink it rather than copying:

```bash
ln -s /path/to/CESLR-multisigner dataset/CESLR/CESLR-multisigner
```

---

## 4. Preprocessing

This generates `gloss_dict.npy`, the `{split}_info.npy` files, and the ground-truth
`.stm` files the evaluator reads — and optionally resizes frames to 256×256.

**If your frames are still at the source resolution (e.g. 210×260):**

```bash
cd preprocess
python dataset_preprocess.py --process-image --multiprocessing
cd ..
python preprocess/fix_annotations.py
```

**If your frames are already 256×256:**

```bash
cd preprocess
python dataset_preprocess.py --input-res 256x256px
cd ..
python preprocess/fix_annotations.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--input-res` | `210x260px` | resolution folder the source frames live in |
| `--output-res` | `256x256px` | resize target |
| `--process-image` | off | actually resize (omit to only regenerate annotations) |
| `--multiprocessing` | off | use 10 worker processes for resizing |

`fix_annotations.py` normalises Amharic spelling variants, rebuilds the gloss
dictionary, and writes the STM files to **both** `preprocess/CESLR/` and
`evaluation/slr_eval/`. The evaluator reads the latter — if the two disagree, your
WER is measured against the wrong ground truth. **Always run it after
`dataset_preprocess.py`.**

> Resizing writes into the dataset tree by substituting `--input-res` for
> `--output-res` in each path. Passing equal values would overwrite your originals
> in place, so the script refuses that combination. Back up raw frames anyway.

Verify the result:

```bash
python -c "
import numpy as np
gd = np.load('preprocess/CESLR/gloss_dict.npy', allow_pickle=True).item()
print('glosses:', len(gd), '-> num_classes:', len(gd)+1)
for s in ['train','dev','test']:
    info = np.load(f'preprocess/CESLR/{s}_info.npy', allow_pickle=True).item()
    print(s, sum(1 for v in info.values() if isinstance(v, dict)), 'clips')
"
```

---

## 5. Training

`num_classes` is set automatically from the gloss dictionary — the value in the
config file is ignored.

**Linux / Windows (single GPU):**

```bash
python main.py --config configs/baseline.yaml --device 0
```

**Linux (multi-GPU):**

```bash
python main.py --config configs/baseline.yaml --device 0,1
```

**macOS:**

```bash
bash run_local_train.sh 40        # argument = number of epochs
```

That script preprocesses if needed, sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, writes a
macOS-appropriate config (batch size 1), and trains. To run `main.py` directly on
macOS instead:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config configs/baseline.yaml --device 0
```

**Useful overrides** (command line beats the config file):

```bash
python main.py --config configs/baseline.yaml --device 0 \
    --num-epoch 40 --batch-size 2 --work-dir ./work_dir/my_run/
```

### Outputs

Everything lands in `work_dir/` (`baseline_res18/` by default):

| File | Contents |
|---|---|
| `dev_<WER>_epoch<N>_model.pt` | checkpoint, named by dev WER |
| `log.txt` | full training log |
| `dev.txt` / `test.txt` | per-epoch WER |
| `output-hypothesis-dev.ctm` | predicted glosses |

### Resuming

Checkpoints store model, optimizer, scheduler and RNG state:

```bash
python main.py --config configs/baseline.yaml --device 0 \
    --load-checkpoints work_dir/baseline_res18/dev_45.20_epoch12_model.pt
```

Use `--load-weights` instead to load only the weights (e.g. fine-tuning from a
different dataset), which starts training from epoch 0.

### Expected speed

Measured on ~1,000 training clips:

| Hardware | Per step | Per epoch |
|---|---|---|
| Kaggle T4, batch 2 | ~2 s | ~16 min |
| Apple M5 (MPS), batch 1 | ~1.2 s | ~20 min |
| CPU only | very slow | not recommended |

40 epochs ≈ 11 h on one T4. Checkpoints save every epoch, so a run can span
several sessions.

---

## 6. Evaluation

Evaluates the given checkpoint on **both** dev and test:

```bash
python main.py --config configs/baseline.yaml --device 0 \
    --phase test --load-weights work_dir/baseline_res18/dev_45.20_epoch12_model.pt
```

Two WER numbers are printed per split. The one that matters is the line
`Epoch ..., dev  XX.XX%` — that is the BiLSTM branch. The other, much larger number
is the auxiliary convolutional branch, which is untrained unless `ConvCTC` is
enabled in `loss_weights` (see [§10](#10-configuration-reference)).

To read predictions side by side with the ground truth:

```bash
python show_predictions.py --work-dir ./work_dir/baseline_res18/ --mode test
python show_predictions.py --work-dir ./work_dir/baseline_res18/ --mode test --all
```

---

## 7. Inference on a single clip

```bash
python test_one_video.py \
    --model_path work_dir/baseline_res18/dev_45.20_epoch12_model.pt \
    --video_path dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px/test/<clip_id>/1
```

`--video_path` accepts a folder of frames or a video file (`.mp4`, `.avi`, `.mov`,
`.mkv`; video input needs decord). Other flags: `--dict_path`, `--device`,
`--max_frames_num`.

> Frames should be **256×256**. The script centre-crops to 224 but does not
> resize, so other resolutions crop a different region than the model saw in
> training.

---

## 8. Web demo

```bash
pip install gradio==4.44.1 pydantic==2.10.6
python demo.py --model_path work_dir/baseline_res18/dev_45.20_epoch12_model.pt
```

Open <http://127.0.0.1:7863>. Upload the frames of one clip, or a video, and press
**Run**. Flags: `--port`, `--share` (public link), `--max_frames_num`.

> **Pin `pydantic==2.10.6`.** gradio 4.44 crashes on pydantic ≥ 2.11 with
> `TypeError: argument of type 'bool' is not iterable` while building its API schema.

---

## 9. Training on Kaggle

Free 2×T4 GPUs, 30 GPU-hours per week, ~12 h per session. Use
[`ceslr_kaggle_train.ipynb`](ceslr_kaggle_train.ipynb).

1. Resize frames to 256×256 locally ([§4](#4-preprocessing)).
2. Upload a Kaggle Dataset containing both `features/fullFrame-256x256px/` and
   `annotations/manual/`.
3. New Notebook → **File → Import Notebook** → upload the `.ipynb`.
4. Settings → Accelerator **GPU T4 ×2**, Internet **On**.
5. **Add Input** → your dataset.
6. Run all cells.

The notebook clones the repo, installs pyctcdecode with `--no-deps`, locates your
dataset wherever Kaggle mounted it, regenerates annotations from your CSVs, runs
consistency checks, trains with auto-resume, then evaluates on test.

**Spanning sessions:** `/kaggle/working` is wiped between sessions. To continue a
run, attach the previous notebook version's **Output** as an extra input — the
training cell picks up those checkpoints automatically.

---

## 10. Configuration reference

Two files, both required:

**`configs/CESLR.yaml`** — paths:

```yaml
dataset_root: ./dataset/CESLR/CESLR-multisigner
dict_path: ./preprocess/CESLR/gloss_dict.npy
evaluation_dir: ./evaluation/slr_eval
evaluation_prefix: CESLR-groundtruth
```

**`configs/baseline.yaml`** — training. Key entries:

| Key | Default | Notes |
|---|---|---|
| `num_epoch` | 70 | 40 is usually enough |
| `batch_size` | 2 | **use 1 on macOS/MPS** |
| `test_batch_size` | 2 | lower if evaluation runs out of memory |
| `num_worker` | 4 | match your CPU count; 0 on native Windows |
| `device` | `0,1` | overridden by `--device` |
| `save_interval` | 1 | save every epoch |
| `evaluate_tool` | `python` | `sclite` needs a separate Kaldi install |
| `optimizer_args.base_lr` | 0.0001 | Adam |
| `optimizer_args.step` | `[20, 35]` | LR decay epochs |
| `model_args.num_classes` | — | **ignored**, set from the gloss dict |

### Loss weights

```yaml
loss_weights:
  SeqCTC: 1.0
  # ConvCTC: 1.0
  # Dist: 25.0
```

Only `SeqCTC` is active by default, i.e. plain CTC. Uncommenting `ConvCTC` and
`Dist` enables the full **VAC** objective this codebase was built around; those
auxiliary losses exist specifically to help the visual encoder train on limited
data, and are worth enabling. They also make the conv-branch WER meaningful.

---

## 11. Troubleshooting

**`zsh: command not found: python`**
Activate the venv (`source .venv/bin/activate`), or call it explicitly with
`.venv/bin/python`.

**`ModuleNotFoundError: No module named 'ctcdecode'`**
You are on an older revision. This fork uses pyctcdecode:
`pip install --no-deps pyctcdecode pygtrie`.

**pip downgrades numpy and everything breaks**
You installed pyctcdecode without `--no-deps`. Reinstall numpy
(`pip install "numpy>=2"`), then `pip install --no-deps pyctcdecode pygtrie`.

**`RuntimeError: MPS backend out of memory`, or steps suddenly take ~80 s**
Set `batch_size: 1` on macOS. Long clips at batch 2 exceed unified memory.

**Training steps get slower and slower within an epoch (macOS)**
Already handled — `seq_scripts.py` releases tensors and drains the MPS cache each
step. If you still see it, pull the latest revision.

**Every epoch reports exactly `100.00%` WER**
Evaluation is failing and being reported as 100%. Current code prints the real
traceback. Historically this was caused by the evaluator invoking a bare `python`
that did not exist — fixed by using `sys.executable`.

**`WER_primary: 500%` on the conv branch**
Expected when `ConvCTC` is disabled: that branch is untrained. Read the
`Epoch ..., dev  XX.XX%` line instead.

**`sh: python: command not found` during evaluation**
Older revision — pull the latest.

**gradio demo: `TypeError: argument of type 'bool' is not iterable`**
`pip install pydantic==2.10.6`.

**`kenlm python bindings are not installed`**
Harmless. Decoding runs without a language model. Silenced in current revisions.

**Dataloader hangs or crashes with multiprocessing errors**
Set `num_worker: 0`.

**Predictions are empty or a single repeated gloss**
Normal early in training — CTC learns to emit blanks first. Train longer; the WER
curve in `work_dir/*/dev.txt` should still be falling.

---

## 12. Repository layout

```
CESLR/
├── main.py                     # entry point: train / test / features
├── slr_network.py              # SLRModel: ResNet18 + TemporalConv + BiLSTM + CTC
├── seq_scripts.py              # train / eval loops
├── demo.py                     # Gradio web UI
├── test_one_video.py           # single-clip inference
├── show_predictions.py         # predictions vs ground truth
├── run_local_train.sh          # macOS one-command training
├── ceslr_kaggle_train.ipynb    # Kaggle notebook
├── configs/
│   ├── CESLR.yaml              # dataset paths
│   └── baseline.yaml           # training hyperparameters
├── modules/                    # BiLSTM, temporal conv, CTC losses
├── utils/
│   ├── decode.py               # CTC decoding (pyctcdecode beam / greedy)
│   ├── device.py               # CUDA -> MPS -> CPU selection
│   ├── video_augmentation.py   # crop / flip / temporal rescale
│   └── ...
├── dataset/dataloader_video.py # frame loading, batching, padding
├── preprocess/
│   ├── dataset_preprocess.py   # resize + generate annotations
│   └── fix_annotations.py      # normalise Amharic, rebuild dict + STMs
└── evaluation/slr_eval/        # WER calculation + ground-truth STMs
```

---

## Citation

This work builds on VAC (ICCV 2021) and CorrNet (CVPR 2023):

```bibtex
@inproceedings{hu2023continuous,
  title={Continuous Sign Language Recognition with Correlation Network},
  author={Hu, Lianyu and Gao, Liqing and Liu, Zekang and Feng, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
