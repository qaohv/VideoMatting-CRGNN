import argparse
import os
import sys
sys.path.append('models/')
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.measure import label

from hlmobilenetv2 import hlmobilenetv2


OUTPUT_ALPHA_DIR = './result-real/alpha'  # output image path
OUTPUT_FG_DIR = './result-real/fg'  # output fg path


def image_alignment(img, trimap, output_stride):
    imsize = np.asarray(img.shape[:2], dtype=np.float)
    new_imsize = np.ceil(imsize / output_stride) * output_stride

    h, w = int(new_imsize[0]), int(new_imsize[1])

    img_resized = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    trimap_resized = cv2.resize(trimap, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    if len(trimap_resized.shape) < 3:
        trimap_resized = np.expand_dims(trimap_resized, axis=2)
    img_trimap = np.concatenate((img_resized, trimap_resized), axis=2)

    return img_trimap


def image_read(name, img_path, trimap_path, output_stride):
    img_file = os.path.join(img_path, name)
    trimap_file = os.path.join(trimap_path, name[:-3] + 'png')

    trimap = np.array(Image.open(trimap_file)).astype(np.float32)
    img = np.array(Image.open(img_file)).astype(np.float32)
    image_trimap = image_alignment(img, trimap, output_stride)

    img = image_trimap[:, :, 0:3]
    trimap = image_trimap[:, :, 3]

    img = torch.from_numpy(img)
    trimap = torch.from_numpy(trimap)

    img = img.permute(2, 0, 1).unsqueeze(dim=0).cuda()
    trimap = trimap.view(1, 1, trimap.shape[0], trimap.shape[1]).cuda()

    return img, trimap


def image_save(out_path, out_path_fg, name, trimap, pred, pred_fg):
    trimap = trimap.squeeze(dim=2)

    pred[trimap == 255] = 1
    pred[trimap == 0] = 0

    out_name = os.path.join(out_path, name[:-3] + 'png')
    out_name_fg = os.path.join(out_path_fg, name[:-3] + 'png')
    pred_out = pred.squeeze().detach().cpu().numpy()

    # refine alpha with connected component
    labels = label((pred_out > 0.05).astype(int))
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    pred_out = pred_out * largestCC
    cv2.imwrite(out_name, np.uint8(pred_out * 255))

    pred_fg = pred_fg.squeeze().detach().cpu().numpy()
    pred_out_fg = np.expand_dims(pred_out, axis=0) * pred_fg
    pred_out_fg = pred_out_fg.transpose(1, 2, 0) * 255
    pred_out_fg = pred_out_fg.clip(0, 255).round()
    pred_out_fg = cv2.cvtColor(np.uint8(pred_out_fg), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_name_fg, pred_out_fg)


def main():
    parser = argparse.ArgumentParser(description="Transformer Network")

    parser.add_argument('--data-root', type=str, required=True, help="Path to data root dir")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model weights")

    parser.add_argument('--output_stride', type=int, default=8, help="output stirde of the network")
    parser.add_argument('--crop_size', type=int, default=320, help="crop size of input image")
    parser.add_argument('--conv_operator', type=str, default='std_conv', help=" ")
    parser.add_argument('--decoder', type=str, default='indexnet', help=" ")
    parser.add_argument('--decoder_kernel_size', type=int, default=5, help=" ")
    parser.add_argument('--indexnet', type=str, default='depthwise', help=" ")
    parser.add_argument('--index_mode', type=str, default='m2o', help=" ")
    parser.add_argument('--use_nonlinear', type=str, default=True, help=" ")
    parser.add_argument('--use_context', type=str, default=True, help=" ")
    parser.add_argument('--apply_aspp', type=str, default=True, help=" ")
    parser.add_argument('--sync_bn', type=str, default=False, help=" ")

    args = parser.parse_args()

    model = hlmobilenetv2(
        pretrained=True,
        freeze_bn=True,
        output_stride=args.output_stride,
        input_size=args.crop_size,
        apply_aspp=args.apply_aspp,
        conv_operator=args.conv_operator,
        decoder=args.decoder,
        decoder_kernel_size=args.decoder_kernel_size,
        indexnet=args.indexnet,
        index_mode=args.index_mode,
        use_nonlinear=args.use_nonlinear,
        use_context=args.use_context,
        sync_bn=args.sync_bn
    )

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    model.load_state_dict(torch.load(args.checkpoint), strict=True)
    ck_split = Path(args.checkpoint).stem

    data_root_path = Path(args.data_root)
    image_root_dir, trimap_root_dir = data_root_path / 'image', data_root_path / 'trimap'

    videos = sorted(os.listdir(trimap_root_dir))
    for i, video in enumerate(videos):
        video_frames_path, video_trimap_path = os.path.join(image_root_dir, video), os.path.join(trimap_root_dir, video)

        out_path, out_path_fg = os.path.join(OUTPUT_ALPHA_DIR, ck_split, video), \
                                os.path.join(OUTPUT_FG_DIR, ck_split, video)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if not os.path.exists(out_path_fg):
            os.makedirs(out_path_fg)

        video_frames = sorted(os.listdir(video_frames_path))

        pred_p = {}
        pred_fg_p = {}
        for j in range(1, len(video_frames) - 1):
            current_frame_name = video_frames[j]
            current_frame, current_trimap = \
                image_read(current_frame_name, video_frames_path, video_trimap_path, args.output_stride)

            prev_frame_name = video_frames[j - 1]
            prev_frame, prev_trimap = \
                image_read(prev_frame_name, video_frames_path, video_trimap_path, args.output_stride)

            next_frame_name = video_frames[j + 1]
            next_frame, next_trimap = \
                image_read(next_frame_name, video_frames_path, video_trimap_path, args.output_stride)

            with torch.no_grad():
                prev_alpha, prev_fg, current_alpha, current_fg, next_alpha, next_fg = \
                    model(prev_frame.clone(), prev_trimap.clone(), current_frame.clone(), current_trimap.clone(),
                          next_frame.clone(), next_trimap.clone())
            current_trimap = current_trimap.squeeze(dim=2)

            if j == 1:
                pred_p[j - 1] = prev_alpha
                pred_p[j] = current_alpha
                pred_p[j + 1] = next_alpha

                pred_fg_p[j - 1] = prev_fg
                pred_fg_p[j] = current_fg
                pred_fg_p[j + 1] = next_fg

            else:
                pred_p[j - 1] += prev_alpha
                pred_p[j] += current_alpha
                pred_p[j + 1] += next_alpha

                pred_fg_p[j - 1] += prev_fg
                pred_fg_p[j] += current_fg
                pred_fg_p[j + 1] += next_fg

            _, _, h_align, w_align = current_frame.shape
            pred_p[j + 2] = torch.zeros((1, 1, h_align, w_align)).to(current_alpha.device)
            pred_fg_p[j + 2] = torch.zeros((1, 3, h_align, w_align)).to(current_alpha.device)

            if j == 1:
                pred_ = pred_p[j - 1]
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j - 1]
            if j == 2:
                pred_ = pred_p[j - 1] / 2
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j - 1] / 2
            if j >= 3:
                pred_ = pred_p[j - 1] / 3
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j - 1] / 3

            image_save(out_path, out_path_fg, prev_frame_name, prev_trimap, pred_, pred_fg_)

            # save the last 2 images
            if j == len(video_frames) - 2:
                pred_ = pred_p[j] / 2
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j] / 2
                image_save(out_path, out_path_fg, current_frame_name, current_trimap, pred_, pred_fg_)

                pred_ = pred_p[j + 1]
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j + 1]
                image_save(out_path, out_path_fg, next_frame_name, next_trimap, pred_, pred_fg_)

            del pred_p[j - 1]
            del pred_fg_p[j - 1]
            del pred_
            del pred_fg_
            del prev_alpha
            del prev_fg
            del current_alpha
            del current_fg
            del next_alpha
            del next_fg


if __name__ == "__main__":
    main()
