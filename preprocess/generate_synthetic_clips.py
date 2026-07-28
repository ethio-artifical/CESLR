"""Render appearance-shifted copies of clips, holding the signing pixel-exact.

    # 3 training variants per train clip
    python preprocess/generate_synthetic_clips.py --split train --variants 3

    # held-out appearance conditions on dev-signer clips
    python preprocess/generate_synthetic_clips.py --split dev \
        --recipes preprocess/recipes_val.yaml --out-suffix -cond

The corpus is shot against a green screen, which makes appearance editing a
matting problem rather than a generation problem. The signer is separated by
chroma key and recomposited over a different background, optionally with a hue
and lightness shift applied to their clothing. The signer's pixels are either
copied through untouched or moved in colour space; their geometry is never
resampled, so the signing is preserved exactly and the gloss label still holds.

Why not diffusion. Stable Diffusion with ControlNet-openpose was tried first and
measured on a real clip:

    text2img            ignores the source framing entirely -- generates a studio
                        portrait at a different scale and crop than the green-screen
                        original, which is a domain shift, not an appearance shift.
    img2img str 0.45    signing preserved, but only 3.0% different from the real
                        frames -- less than the 4.9% between two different real
                        signers, i.e. barely an augmentation at all.
    img2img str 0.65    4.3% different, and hands begin to smear.

One `strength` knob controls preservation and variation together, and sign
language needs preservation strict enough that no useful setting exists. Chroma
key gives 25-30% difference with zero risk to the hands, and runs on CPU in
minutes rather than hours on a rented GPU.

What this does not do is vary signer identity -- body shape and face are carried
through unchanged. Neither did diffusion, at any setting that kept the hands.

Frames are written in the layout the dataloader already reads:

    <out-root>/<split>/<clip_id>_v<k>/1/*.png
"""
import argparse
import glob
import hashlib
import os

import cv2
import numpy as np
import yaml


def load_recipes(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['key'], cfg['axes']


def foreground_alpha(bgr, key):
    """Soft alpha for the signer: 1 on the signer, 0 on the green screen.

    Returned as float so the composite has a feathered edge; a hard mask leaves a
    green fringe that reads as a strong, learnable cue.

    Skin is forced opaque. Hands held near the screen pick up enough green bounce
    to land inside the key's hue range, and keying them out punches holes in
    exactly the region that carries the gloss. Skin never belongs to the
    background, so excluding it costs nothing and removes that failure entirely.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bg = cv2.inRange(hsv,
                     np.array(key['lower'], np.uint8),
                     np.array(key['upper'], np.uint8))
    # The screen is not uniform: it falls off in brightness and saturation toward
    # the frame edges and in shadow, and a fixed HSV window leaves those pixels
    # behind as a teal fringe -- which the clothing hue shift then turns orange.
    # Greenness (how far G exceeds both other channels) separates screen from
    # signer regardless of exposure, so it catches what the window misses.
    b, g, r = bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
    greenness = g - np.maximum(b, r)
    bg = bg | ((greenness > key.get('greenness', 18)).astype(np.uint8) * 255)
    k = np.ones((key['open_px'], key['open_px']), np.uint8)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, k)
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, k)
    blur = key['feather_px'] | 1                      # GaussianBlur needs odd
    bg = cv2.GaussianBlur(bg, (blur, blur), 0)
    alpha = 1.0 - bg.astype(np.float32) / 255.0       # alpha on the signer
    if key.get('skin'):
        alpha = np.maximum(alpha, skin_mask(bgr, key['skin']))
    # Close any residual pinholes inside the silhouette. A hole in a hand is a
    # missing finger as far as the model is concerned.
    fill = key.get('fill_px', 7) | 1
    solid = cv2.morphologyEx((alpha > 0.5).astype(np.uint8) * 255,
                             cv2.MORPH_CLOSE, np.ones((fill, fill), np.uint8))
    return np.maximum(alpha, solid.astype(np.float32) / 255.0 *
                      (cv2.GaussianBlur(solid, (blur, blur), 0) / 255.0))


def despill(bgr, alpha, amount):
    """Remove green light bounced onto the signer from the screen.

    Without this the signer keeps a green rim and colour cast that survives
    compositing, which is exactly the cue the augmentation is meant to remove.
    """
    if amount <= 0:
        return bgr
    out = bgr.astype(np.float32)
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    cap = np.maximum(b, r)
    spill = np.clip(g - cap, 0, None) * amount * alpha
    out[..., 1] = g - spill
    return np.clip(out, 0, 255).astype(np.uint8)


def skin_mask(bgr, skin):
    """Soft mask over skin, so recolouring can leave hands and face alone.

    Rotating the hue of the whole foreground turns the signer's face green or
    purple. That is not a plausible appearance -- it is an artefact, and a model
    can learn 'unnatural skin hue' as a marker of a synthetic clip, which is the
    opposite of what this augmentation is for. Hands especially must keep their
    real colour: they carry the lexical content.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(skin['lower'], np.uint8), np.array(skin['upper'], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    if skin.get('wrap_upper'):                 # skin hue wraps past 180 in OpenCV
        m |= cv2.inRange(hsv, np.array(skin['wrap_lower'], np.uint8),
                         np.array(skin['wrap_upper'], np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    blur = skin.get('feather_px', 7) | 1
    return cv2.GaussianBlur(m, (blur, blur), 0).astype(np.float32) / 255.0


def shift_foreground(bgr, alpha, hue, sat, val, skin=None):
    """Recolour the signer's clothing. Geometry is untouched; signing is unaffected."""
    if hue == 0 and sat == 1.0 and val == 1.0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    a = alpha.copy()
    if skin is not None:
        a *= 1.0 - skin_mask(bgr, skin)
    a = a[..., None]
    return (shifted * a + bgr * (1 - a)).astype(np.uint8)


def make_background(spec, h, w, rng):
    """Solid, vertical gradient, or textured noise. BGR."""
    kind = spec.get('kind', 'solid')
    if kind == 'solid':
        bg = np.zeros((h, w, 3), np.float32)
        bg[:] = spec['color'][::-1]                   # yaml is RGB, cv2 is BGR
    elif kind == 'gradient':
        top = np.array(spec['top'][::-1], np.float32)
        bot = np.array(spec['bottom'][::-1], np.float32)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        bg = top * (1 - t) + bot * t
        bg = np.repeat(bg, w, axis=1)
    elif kind == 'texture':
        base = np.array(spec['color'][::-1], np.float32)
        noise = rng.normal(0, spec.get('sigma', 18), (h // 8, w // 8, 3))
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        bg = np.clip(base + noise, 0, 255)
    else:
        raise ValueError(f'unknown background kind {kind!r}')
    return bg.astype(np.float32)


def render_frame(bgr, variant, key, bg):
    """Composite one frame over a background that is shared by the whole clip.

    The background must be built once per clip, not once per frame: a texture
    regenerated each frame shimmers, and a shimmering background is a temporal
    signal in a model whose entire input is a temporal sequence.
    """
    alpha = foreground_alpha(bgr, key)
    fg = despill(bgr, alpha, key.get('despill', 0.8))
    fg = shift_foreground(fg, alpha,
                          variant.get('hue', 0),
                          variant.get('sat', 1.0),
                          variant.get('val', 1.0),
                          skin=key.get('skin'))
    a = alpha[..., None]
    return (fg.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def variant_spec(axes, axis_names, clip_id, k):
    """Variant k draws from axis k; the option within it is chosen by clip hash.

    Hashing rather than sampling keeps a resumed run reproducible: a clip gets the
    same appearance it would have had on the first pass.
    """
    axis = axis_names[k % len(axis_names)]
    options = axes[axis]
    h = int(hashlib.md5(f'{clip_id}|{k}'.encode()).hexdigest()[:8], 16)
    return axis, options[h % len(options)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    p.add_argument('--variants', type=int, default=3)
    p.add_argument('--recipes', default='preprocess/recipes_train.yaml')
    p.add_argument('--info-dir', default='./preprocess/CESLR')
    p.add_argument('--dataset-root', default='./dataset/CESLR/CESLR-multisigner')
    p.add_argument('--res', default='256x256px')
    p.add_argument('--out-root',
                   default='./dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px-synth')
    p.add_argument('--out-suffix', default='',
                   help='appended to out-root, e.g. -cond for held-out conditions')
    p.add_argument('--limit', type=int, default=None, help='stop after N clips (smoke test)')
    p.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    key, axes = load_recipes(args.recipes)
    axis_names = list(axes)
    out_root = args.out_root + args.out_suffix

    info = np.load(os.path.join(args.info_dir, f'{args.split}_info.npy'),
                   allow_pickle=True).item()
    clips = [v for v in info.values() if isinstance(v, dict)]
    if args.limit:
        clips = clips[:args.limit]

    print(f'split={args.split}  clips={len(clips)}  variants={args.variants}')
    print(f'recipes: {args.recipes}  axes={axis_names}')
    print(f'output : {out_root}\n')

    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for fi in clips:
        for k in range(args.variants):
            jobs.append((fi['fileid'], fi['folder'], k))

    done = skipped = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_run_one, j, args, key, axes, axis_names, out_root)
                for j in jobs]
        for n, f in enumerate(futs, 1):
            if f.result():
                done += 1
            else:
                skipped += 1
            if n % 200 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)} variants  ({done} written, {skipped} present)')

    print(f'\n{args.split}: {done} written, {skipped} skipped -> {out_root}')
    print('Sanity-check before training on it:')
    print(f'    python preprocess/verify_synthetic.py --synth-root {out_root} '
          f'--split {args.split}')


def _run_one(job, args, key, axes, axis_names, out_root):
    clip_id, folder, k = job
    dst = os.path.join(out_root, args.split, f'{clip_id}_v{k}', '1')
    src = sorted(glob.glob(os.path.join(
        args.dataset_root, 'features', f'fullFrame-{args.res}', folder)))
    if not src:
        raise SystemExit(f'no frames for {clip_id} at {folder}')
    if (not args.overwrite and os.path.isdir(dst)
            and len(glob.glob(os.path.join(dst, '*.png'))) == len(src)):
        return False

    axis, variant = variant_spec(axes, axis_names, clip_id, k)
    os.makedirs(dst, exist_ok=True)
    # One background for the whole clip, seeded per (clip, variant) so a resumed
    # run reproduces it exactly.
    rng = np.random.default_rng(
        int(hashlib.md5(f'{clip_id}|{k}'.encode()).hexdigest()[:8], 16))
    bg = None
    for i, sp in enumerate(src):
        img = cv2.imread(sp)
        if img is None:
            raise SystemExit(f'could not read {sp}')
        if bg is None:
            bg = make_background(variant['background'], *img.shape[:2], rng)
        cv2.imwrite(os.path.join(dst, f'{i:05d}.png'),
                    render_frame(img, variant, key, bg))
    return True


if __name__ == '__main__':
    main()
