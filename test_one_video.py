"""Run CESLR inference on a single clip (a folder of frames, or a video file).

    python test_one_video.py --model_path work_dir/local_test/dev_58.75_epoch4_model.pt \
                             --video_path dataset/CESLR/CESLR-multisigner/features/fullFrame-256x256px/test/<clip_id>/1

Adapted from CorrNet's test_one_video.py for this repo: CESLR's model is built
with weight_norm/share_classifier off and trained with SeqCTC only, and reads the
Amharic gloss dict. Device is chosen automatically (CUDA -> MPS -> CPU).
"""
import argparse
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

import utils
from slr_network import SLRModel
from utils import video_augmentation

VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']


def load_frames_from_dir(path):
    """Frames sorted by name -- order is the temporal order, so it must be stable."""
    files = sorted(f for f in os.listdir(path)
                   if os.path.splitext(f)[-1].lower() in IMAGE_FORMATS)
    if not files:
        # A clip dir may hold the frames one level down (…/<clip_id>/1/*.png).
        nested = sorted(glob.glob(os.path.join(path, '*', '*')))
        files = [os.path.relpath(f, path) for f in nested
                 if os.path.splitext(f)[-1].lower() in IMAGE_FORMATS]
    if not files:
        raise ValueError(f'No images found in {path}')
    return [cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB)
            for f in files]


def load_frames_from_video(path, max_frames_num):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    idx = np.linspace(0, total - 1, min(total, max_frames_num), dtype=int).tolist()
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in vr.get_batch(idx).asnumpy()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True, help='path to a trained .pt checkpoint')
    p.add_argument('--video_path', required=True,
                   help='a folder of frames, or a video file')
    p.add_argument('--dict_path', default='./preprocess/CESLR/gloss_dict.npy')
    p.add_argument('--device', default=0, help='gpu id; ignored on MPS/CPU')
    p.add_argument('--max_frames_num', type=int, default=360)
    args = p.parse_args()

    gloss_dict = np.load(args.dict_path, allow_pickle=True).item()

    ext = os.path.splitext(args.video_path)[-1].lower()
    if os.path.isdir(args.video_path):
        img_list = load_frames_from_dir(args.video_path)
    elif ext in VIDEO_FORMATS:
        img_list = load_frames_from_video(args.video_path, args.max_frames_num)
    else:
        raise ValueError(f'{args.video_path} is neither a folder of frames nor '
                         f'one of {VIDEO_FORMATS}')
    print(f'loaded {len(img_list)} frames from {args.video_path}')

    # Same transform the dataloader uses for dev/test.
    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(224),
        video_augmentation.ToTensor(),
    ])
    vid, _ = transform(img_list, None, None)
    vid = (vid.float() / 127.5 - 1).unsqueeze(0)

    # Pad to the temporal receptive field of the 1D conv stack, exactly as
    # BaseFeeder.collate_fn does -- otherwise vid_lgt disagrees with the model.
    left_pad, last_stride, total_stride = 0, 1, 1
    for ks in ['K5', 'P2', 'K5', 'P2']:
        if ks[0] == 'K':
            left_pad = left_pad * last_stride + (int(ks[1]) - 1) // 2
        else:
            last_stride = int(ks[1])
            total_stride *= last_stride
    max_len = vid.size(1)
    video_length = torch.LongTensor(
        [int(np.ceil(max_len / total_stride)) * total_stride + 2 * left_pad])
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    padded_len = max_len + left_pad + right_pad
    vid = torch.cat((
        vid[0, 0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0, -1][None].expand(padded_len - vid.size(1) - left_pad, -1, -1, -1),
    ), dim=0).unsqueeze(0)

    device = utils.GpuDataParallel()
    device.set_device(args.device)

    # Must match how the checkpoint was trained (see configs/baseline.yaml).
    model = SLRModel(num_classes=len(gloss_dict) + 1, c2d_type='resnet18',
                     conv_type=2, use_bn=1, gloss_dict=gloss_dict,
                     loss_weights={'SeqCTC': 1.0},
                     weight_norm=False, share_classifier=False)
    state_dict = torch.load(args.model_path, map_location='cpu',
                            weights_only=False)['model_state_dict']
    state_dict = OrderedDict([(k.replace('.module', ''), v)
                              for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device.output_device)
    model.eval()

    with torch.no_grad():
        ret = model(device.data_to_device(vid),
                    device.data_to_device(video_length))

    glosses = [g for g, _ in ret['recognized_sents'][0]]
    print(f'device       : {device.output_device}')
    print(f'raw output   : {ret["recognized_sents"][0]}')
    print(f'\npredicted    : {" ".join(glosses) if glosses else "(no glosses predicted)"}')


if __name__ == '__main__':
    main()
