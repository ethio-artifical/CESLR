"""Extract whole-body pose from every clip, for ControlNet conditioning.

    python preprocess/extract_pose.py --split train
    python preprocess/extract_pose.py --split all --detector dwpose

Writes two things per clip:

    pose/<split>/<clip_id>/keypoints.npy   (T, K, 3) -- x, y, confidence
    pose/<split>/<clip_id>/skeleton/*.png  rendered skeletons, ControlNet input

The keypoints are not needed by generate_synthetic_clips.py, which reads the
rendered skeletons. They are saved because a pose stream is the obvious next
encoder to try after this, and re-running detection over ~180k frames to get them
later would be wasteful.

Output is written per split and the generator only ever reads one split at a
time. A synthetic clip inherits its motion from a real signer, and how a person
moves is nearly as identifying as how they look -- so a synthetic clip belongs to
whichever split its source signer is in. Keeping the pose caches separate makes
crossing that boundary take deliberate effort rather than a wrong path.

Detector backends, in order of hand accuracy:
    dwpose    -- 133 COCO-WholeBody keypoints, best hands. Needs onnxruntime.
    openpose  -- 18 body + 21/hand + 70 face. Pure torch, installs anywhere.
Both come from controlnet_aux and render in the exact format the matching
ControlNet checkpoint expects, so the skeleton style never has to be guessed.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def load_detector(name, device):
    """Return (detector, needs_hand_face_flags). Raises with install hints."""
    try:
        import controlnet_aux
    except ImportError:
        raise SystemExit(
            'controlnet_aux is required:\n'
            '    pip install controlnet_aux\n'
            'For --detector dwpose also:  pip install onnxruntime-gpu')

    if name == 'dwpose':
        try:
            from controlnet_aux import DWposeDetector
        except ImportError:
            raise SystemExit(
                'This controlnet_aux build has no DWposeDetector. Either upgrade\n'
                '    pip install -U controlnet_aux onnxruntime-gpu\n'
                'or fall back to  --detector openpose  (fewer hand keypoints).')
        det = DWposeDetector()
        if hasattr(det, 'to'):
            det = det.to(device)
        return det, False

    from controlnet_aux import OpenposeDetector
    det = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
    if hasattr(det, 'to'):
        det = det.to(device)
    return det, True


def detect(det, needs_flags, img):
    """Run the detector, returning (skeleton PIL image, keypoints array or None).

    controlnet_aux detectors return the rendered skeleton by default and the raw
    pose only when asked, and the exact keyword differs between versions, so try
    the richer call first and degrade to the plain one.
    """
    kw = dict(hand_and_face=True) if needs_flags else {}
    try:
        out = det(img, output_type='pil', **kw)
    except TypeError:
        out = det(img, **kw)

    if isinstance(out, tuple):
        skeleton, pose = out[0], out[1]
        return skeleton, _pose_to_array(pose)
    return out, None


def _pose_to_array(pose):
    """Best-effort flatten of a controlnet_aux pose object to (K, 3)."""
    if pose is None:
        return None
    if isinstance(pose, np.ndarray):
        return pose
    pts = []
    for attr in ('body', 'left_hand', 'right_hand', 'face'):
        part = getattr(pose, attr, None)
        if part is None:
            continue
        kps = getattr(part, 'keypoints', part)
        if kps is None:
            continue
        for kp in kps:
            if kp is None:
                pts.append((0.0, 0.0, 0.0))
            else:
                pts.append((getattr(kp, 'x', 0.0), getattr(kp, 'y', 0.0),
                            getattr(kp, 'score', 1.0) or 1.0))
    return np.asarray(pts, dtype=np.float32) if pts else None


def clip_frames(dataset_root, res, folder):
    """Frame paths for one clip, in temporal order (filename order is time order)."""
    return sorted(glob.glob(os.path.join(
        dataset_root, 'features', f'fullFrame-{res}', folder)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='all', choices=['train', 'dev', 'test', 'all'])
    p.add_argument('--detector', default='dwpose', choices=['dwpose', 'openpose'])
    p.add_argument('--dataset-root', default='./dataset/CESLR/CESLR-multisigner')
    p.add_argument('--info-dir', default='./preprocess/CESLR')
    p.add_argument('--res', default='256x256px')
    p.add_argument('--out-dir', default='./pose')
    p.add_argument('--device', default=None, help='cuda | mps | cpu (auto by default)')
    p.add_argument('--stride', type=int, default=1,
                   help='keep every Nth frame; 2 halves generation cost later')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.device is None:
        import torch
        args.device = ('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f'detector: {args.detector}   device: {args.device}')
    det, needs_flags = load_detector(args.detector, args.device)

    splits = ['train', 'dev', 'test'] if args.split == 'all' else [args.split]
    for split in splits:
        info = np.load(os.path.join(args.info_dir, f'{split}_info.npy'),
                       allow_pickle=True).item()
        clips = [v for v in info.values() if isinstance(v, dict)]
        done = skipped = frames = 0

        for n, fi in enumerate(clips, 1):
            clip_dir = os.path.join(args.out_dir, split, fi['fileid'])
            skel_dir = os.path.join(clip_dir, 'skeleton')
            if os.path.isdir(skel_dir) and os.listdir(skel_dir) and not args.overwrite:
                skipped += 1
                continue

            paths = clip_frames(args.dataset_root, args.res, fi['folder'])
            if not paths:
                raise SystemExit(f"no frames for {fi['fileid']}")
            paths = paths[::args.stride]

            os.makedirs(skel_dir, exist_ok=True)
            kps = []
            for i, fp in enumerate(paths):
                skeleton, kp = detect(det, needs_flags, Image.open(fp).convert('RGB'))
                skeleton.save(os.path.join(skel_dir, f'{i:05d}.png'))
                if kp is not None:
                    kps.append(kp)

            if kps:
                np.save(os.path.join(clip_dir, 'keypoints.npy'),
                        np.stack(kps).astype(np.float32))
            done += 1
            frames += len(paths)
            if n % 25 == 0 or n == len(clips):
                print(f'  {split}: {n}/{len(clips)} clips, {frames} frames')

        print(f'{split}: wrote {done}, skipped {skipped} existing '
              f'-> {os.path.join(args.out_dir, split)}\n')


if __name__ == '__main__':
    main()
