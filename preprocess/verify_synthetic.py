"""Contact sheet of generated clips, to eyeball before spending hours on the rest.

    python preprocess/verify_synthetic.py --split train --clips 6

Writes verify_synthetic.png: one row per (clip, variant), frames sampled evenly
along the clip, with the real source row on top for comparison.

Worth the two minutes. The failure modes here are silent -- hands that dissolve
mid-clip, a signer whose identity flickers frame to frame, a prompt that fights
the pose conditioning -- and none of them raise an error. They just quietly
produce training data that teaches the wrong thing. Run this on a handful of
clips before generating the rest.
"""
import argparse
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw


def sample_frames(folder, n):
    paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not paths:
        return []
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    return [paths[i] for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train')
    p.add_argument('--synth-root',
                   default='./dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px-synth')
    p.add_argument('--dataset-root', default='./dataset/CESLR/CESLR-multisigner')
    p.add_argument('--pose-dir', default='./pose')
    p.add_argument('--res', default='256x256px')
    p.add_argument('--clips', type=int, default=6, help='how many clips to show')
    p.add_argument('--frames', type=int, default=8, help='frames per row')
    p.add_argument('--variants', type=int, default=3)
    p.add_argument('--thumb', type=int, default=128)
    p.add_argument('--out', default='verify_synthetic.png')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    split_dir = os.path.join(args.synth_root, args.split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f'nothing at {split_dir}; run generate_synthetic_clips.py first')

    clip_ids = sorted({os.path.basename(d).rsplit('_v', 1)[0]
                       for d in glob.glob(os.path.join(split_dir, '*_v*'))})
    if not clip_ids:
        raise SystemExit(f'no <clip_id>_v<k> folders under {split_dir}')
    random.Random(args.seed).shuffle(clip_ids)
    clip_ids = clip_ids[:args.clips]

    rows = []       # (label, [frame paths])
    for cid in clip_ids:
        real = sample_frames(
            os.path.join(args.dataset_root, 'features', f'fullFrame-{args.res}',
                         args.split, cid, '1'), args.frames)
        if real:
            rows.append((f'{cid}  REAL', real))
        skel = sample_frames(
            os.path.join(args.pose_dir, args.split, cid, 'skeleton'), args.frames)
        if skel:
            rows.append((f'{cid}  pose', skel))
        for k in range(args.variants):
            syn = sample_frames(
                os.path.join(split_dir, f'{cid}_v{k}', '1'), args.frames)
            if syn:
                rows.append((f'{cid}  v{k}', syn))

    if not rows:
        raise SystemExit('found folders but no frames inside them')

    t, pad, label_w = args.thumb, 4, 190
    width = label_w + args.frames * (t + pad)
    height = len(rows) * (t + pad) + pad
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)

    for r, (label, paths) in enumerate(rows):
        y = pad + r * (t + pad)
        draw.text((6, y + t // 2 - 6), label, fill='black')
        for c, fp in enumerate(paths):
            im = Image.open(fp).convert('RGB').resize((t, t), Image.LANCZOS)
            sheet.paste(im, (label_w + c * (t + pad), y))

    sheet.save(args.out)
    print(f'wrote {args.out}  ({len(rows)} rows x {args.frames} frames)')
    print('\nLook for:')
    print('  - hands intact and in the same place as the pose row')
    print('  - identity and background steady across a row, not flickering')
    print('  - variants clearly different from each other and from REAL')


if __name__ == '__main__':
    main()
