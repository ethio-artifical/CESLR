"""Gradio UI for CESLR — upload frames or a video, get Amharic glosses.

    python demo.py --model_path work_dir/local_test/dev_58.75_epoch4_model.pt

Then open the printed URL (default http://127.0.0.1:7863).

Adapted from CorrNet's demo.py for this repo: CESLR's model is built with
weight_norm/share_classifier off and trained with SeqCTC only, reads the Amharic
gloss dict, and picks its device automatically (CUDA -> MPS -> CPU).
"""
import argparse
import os
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch

import utils
from slr_network import SLRModel
from utils import video_augmentation

os.environ.setdefault('GRADIO_TEMP_DIR', 'gradio_temp')
import gradio as gr  # noqa: E402  (must follow GRADIO_TEMP_DIR)

warnings.filterwarnings('ignore')

VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

model = None
device = None
args = None


def as_path(x):
    """Gradio hands back plain paths on some versions and file objects on others."""
    return x if isinstance(x, str) else getattr(x, 'name', str(x))


def load_video(path, max_frames_num=360):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    idx = np.linspace(0, total - 1, min(total, max_frames_num), dtype=int).tolist()
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in vr.get_batch(idx).asnumpy()]


def to_model_input(img_list):
    """Crop/normalise, then pad to the conv stack's receptive field (as collate_fn does)."""
    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(224),
        video_augmentation.ToTensor(),
    ])
    vid, _ = transform(img_list, None, None)
    vid = (vid.float() / 127.5 - 1).unsqueeze(0)

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
    return vid, video_length


def run_inference(inputs):
    if inputs is None or (isinstance(inputs, list) and not inputs):
        return 'Nothing uploaded yet.'

    if isinstance(inputs, list):  # frame sequence
        paths = sorted((as_path(x) for x in inputs), key=os.path.basename)
        paths = [p for p in paths
                 if os.path.splitext(p)[-1].lower() in IMAGE_FORMATS]
        if not paths:
            return f'No images found. Supported: {", ".join(IMAGE_FORMATS)}'
        img_list = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    else:  # single video file
        path = as_path(inputs)
        if os.path.splitext(path)[-1].lower() not in VIDEO_FORMATS:
            return f'Unsupported file. Supported: {", ".join(VIDEO_FORMATS)}'
        try:
            img_list = load_video(path, args.max_frames_num)
        except Exception as e:
            return f'Could not read the video: {e}'

    if len(img_list) < 8:
        return (f'Only {len(img_list)} frame(s). A sign needs a sequence — '
                f'upload the whole clip.')

    vid, video_length = to_model_input(img_list)
    with torch.no_grad():
        ret = model(device.data_to_device(vid),
                    device.data_to_device(video_length))

    glosses = [g for g, _ in ret['recognized_sents'][0]]
    if not glosses:
        return f'No glosses predicted ({len(img_list)} frames processed).'
    return (f'{" ".join(glosses)}\n\n'
            f'({len(glosses)} gloss(es) from {len(img_list)} frames)')


def show_gallery(inputs):
    if not inputs:
        return []
    return [as_path(x) for x in inputs]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True, help='path to a trained .pt checkpoint')
    p.add_argument('--dict_path', default='./preprocess/CESLR/gloss_dict.npy')
    p.add_argument('--device', default=0)
    p.add_argument('--max_frames_num', type=int, default=360)
    p.add_argument('--port', type=int, default=7863)
    p.add_argument('--share', action='store_true', help='create a public gradio link')
    return p.parse_args()


def main():
    global model, device, args
    args = parse_args()

    gloss_dict = np.load(args.dict_path, allow_pickle=True).item()
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
    print(f'loaded {args.model_path} on {device.output_device} '
          f'({len(gloss_dict)} glosses)')

    with gr.Blocks(title='Ethiopian Sign Language Recognition') as demo:
        gr.Markdown('<center><font size=5>Continuous Ethiopian Sign Language '
                    'Recognition</font></center>')
        gr.Markdown('Upload a **sequence of frames** or a **video** of one signed '
                    'sentence to get the recognized Amharic glosses.')
        with gr.Tab('Frames'):
            with gr.Row():
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label='Input frames', height=220, columns=6)
                    frames_input = gr.UploadButton(
                        label='Upload frames of one clip',
                        file_types=IMAGE_FORMATS, file_count='multiple')
                    frames_button = gr.Button('Run', variant='primary')
                with gr.Column(scale=1):
                    frames_output = gr.Textbox(label='Predicted glosses', lines=4)
        with gr.Tab('Video'):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(sources=['upload'], label='Upload a video')
                    video_button = gr.Button('Run', variant='primary')
                with gr.Column(scale=1):
                    video_output = gr.Textbox(label='Predicted glosses', lines=4)

        # Preview as soon as files land, so a wrong upload is obvious before running.
        frames_input.upload(show_gallery, inputs=frames_input, outputs=gallery)
        frames_button.click(run_inference, inputs=frames_input, outputs=frames_output)
        video_button.click(run_inference, inputs=video_input, outputs=video_output)

    demo.launch(share=args.share, server_name='127.0.0.1', server_port=args.port)


if __name__ == '__main__':
    main()
