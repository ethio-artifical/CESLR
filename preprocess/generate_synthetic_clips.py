"""Render appearance-shifted copies of clips, holding the signing fixed.

    # 3 training variants per train clip
    python preprocess/generate_synthetic_clips.py --split train --variants 3

    # held-out appearance conditions on val-signer poses
    python preprocess/generate_synthetic_clips.py --split dev \
        --prompts preprocess/prompts_val.yaml --out-suffix -cond

Stable Diffusion 1.5 with ControlNet-openpose, conditioned on the skeletons
written by extract_pose.py. ControlNet pins the pose, so the generated signer
makes exactly the same hand and body movements as the source clip -- only
identity, clothing and scene change. The gloss content is therefore preserved by
construction: this augmentation cannot silently produce the wrong sign, only the
right sign performed by someone who looks different.

That is the point. With 22 signers the cheapest way for the model to fit the
training data is to memorise who is signing rather than what is signed. Training
on many appearances per motion removes that shortcut.

Frames are written in the same layout as the real ones, so the dataloader reads
them with no special casing:

    <out-root>/<split>/<clip_id>_v<k>/1/*.png

Generation runs at --gen-res (512 suits SD 1.5) and is written at --out-res
(256, matching the real frames, which training centre-crops to 224).

Resumable: a clip whose output folder already holds the expected frame count is
skipped, so an interrupted run continues where it stopped.
"""
import argparse
import glob
import hashlib
import os

import numpy as np
import yaml
from PIL import Image


def pick_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_pipe(args, device):
    import torch
    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetPipeline)

    dtype = torch.float16 if device == 'cuda' else torch.float32
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None)

    if args.lcm:
        # LCM-LoRA gets usable images in 4 steps instead of ~20. The quality it
        # gives up is photorealistic texture, which this never needed -- what
        # matters is that appearance varies, not that it is convincing.
        pipe.load_lora_weights(args.lcm_lora)
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == 'cuda':
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def load_prompts(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    axes = cfg['axes']
    return cfg.get('negative', ''), list(axes.keys()), axes


def variant_prompt(axes, axis_names, clip_id, k):
    """Variant k draws from axis k, choosing within it by a hash of the clip id.

    Hashing rather than random keeps the run reproducible and lets a resumed run
    give a clip the same appearance it would have had on the first pass.
    """
    axis = axis_names[k % len(axis_names)]
    options = axes[axis]
    h = int(hashlib.md5(f'{clip_id}|{k}'.encode()).hexdigest()[:8], 16)
    return axis, options[h % len(options)]


def clip_seed(clip_id, k):
    """One seed per (clip, variant), shared by every frame of that clip.

    Same seed plus same prompt means the same initial noise, which keeps the
    person and background recognisably the same across the clip while ControlNet
    moves the pose. Per-frame seeds would reshuffle the scene every frame.
    """
    return int(hashlib.md5(f'{clip_id}|{k}|seed'.encode()).hexdigest()[:8], 16)


def generate_clip(pipe, skeletons, prompt, negative, seed, args, device):
    import torch
    out = []
    for i in range(0, len(skeletons), args.batch_size):
        chunk = skeletons[i:i + args.batch_size]
        gen = [torch.Generator(device=device).manual_seed(seed) for _ in chunk]
        images = pipe(
            prompt=[prompt] * len(chunk),
            negative_prompt=[negative] * len(chunk),
            image=chunk,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.control_scale,
            generator=gen,
            height=args.gen_res,
            width=args.gen_res,
        ).images
        out.extend(images)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    p.add_argument('--variants', type=int, default=3)
    p.add_argument('--prompts', default='preprocess/prompts_train.yaml')
    p.add_argument('--pose-dir', default='./pose')
    p.add_argument('--info-dir', default='./preprocess/CESLR')
    p.add_argument('--out-root',
                   default='./dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px-synth')
    p.add_argument('--out-suffix', default='',
                   help='appended to out-root, e.g. -cond for held-out conditions')
    p.add_argument('--model', default='runwayml/stable-diffusion-v1-5')
    p.add_argument('--controlnet', default='lllyasviel/control_v11p_sd15_openpose')
    p.add_argument('--lcm-lora', default='latent-consistency/lcm-lora-sdv1-5')
    p.add_argument('--lcm', action='store_true', default=True)
    p.add_argument('--no-lcm', dest='lcm', action='store_false')
    p.add_argument('--steps', type=int, default=4, help='4 with LCM, ~20 without')
    p.add_argument('--guidance', type=float, default=1.5, help='~1.5 with LCM, ~7.5 without')
    p.add_argument('--control-scale', type=float, default=1.0)
    p.add_argument('--gen-res', type=int, default=512, help='must be a multiple of 8')
    p.add_argument('--out-res', type=int, default=256, help='match the real frames')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int, default=None, help='stop after N clips (smoke test)')
    p.add_argument('--device', default=None)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.gen_res % 8:
        p.error(f'--gen-res must be a multiple of 8, got {args.gen_res}')

    device = args.device or pick_device()
    out_root = args.out_root + args.out_suffix
    negative, axis_names, axes = load_prompts(args.prompts)

    info = np.load(os.path.join(args.info_dir, f'{args.split}_info.npy'),
                   allow_pickle=True).item()
    clips = [v for v in info.values() if isinstance(v, dict)]
    if args.limit:
        clips = clips[:args.limit]

    print(f'device={device}  split={args.split}  clips={len(clips)}  '
          f'variants={args.variants}  steps={args.steps}  gen={args.gen_res}px '
          f'-> out={args.out_res}px')
    print(f'prompts: {args.prompts}  axes={axis_names}')
    print(f'output : {out_root}\n')
    pipe = build_pipe(args, device)

    done = skipped = 0
    for n, fi in enumerate(clips, 1):
        clip_id = fi['fileid']
        skel_paths = sorted(glob.glob(os.path.join(
            args.pose_dir, args.split, clip_id, 'skeleton', '*.png')))
        if not skel_paths:
            raise SystemExit(
                f'no skeletons for {clip_id}. Run:\n'
                f'    python preprocess/extract_pose.py --split {args.split}')
        skeletons = [Image.open(sp).convert('RGB').resize(
            (args.gen_res, args.gen_res), Image.BICUBIC) for sp in skel_paths]

        for k in range(args.variants):
            dst = os.path.join(out_root, args.split, f'{clip_id}_v{k}', '1')
            if (not args.overwrite and os.path.isdir(dst)
                    and len(glob.glob(os.path.join(dst, '*.png'))) == len(skeletons)):
                skipped += 1
                continue

            axis, prompt = variant_prompt(axes, axis_names, clip_id, k)
            images = generate_clip(pipe, skeletons, prompt, negative,
                                   clip_seed(clip_id, k), args, device)
            os.makedirs(dst, exist_ok=True)
            for i, im in enumerate(images):
                im.resize((args.out_res, args.out_res), Image.LANCZOS).save(
                    os.path.join(dst, f'{i:05d}.png'))
            done += 1

        if n % 10 == 0 or n == len(clips):
            print(f'  {n}/{len(clips)} clips  ({done} variants written, '
                  f'{skipped} already present)')

    print(f'\n{args.split}: {done} variants written, {skipped} skipped -> {out_root}')
    print('Sanity-check before training on it:')
    print(f'    python preprocess/verify_synthetic.py --synth-root {out_root} '
          f'--split {args.split}')


if __name__ == '__main__':
    main()
