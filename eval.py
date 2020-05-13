from __future__ import print_function, division, absolute_import

import argparse
import glob
import math
import time

import numpy as np
import scipy.io as sio
import torch

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--device", default="cuda", help="device to use, e.g. 'cpu', 'cuda' or 'cuda:0'")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


opt = parser.parse_args()
device = torch.device(opt.device)

torch.set_grad_enabled(False)

model = torch.load(opt.model, map_location=device)["model"]
model.to(device)

image_list = glob.glob("./testsets/{}/*.*".format(opt.dataset))

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0

for image_name in image_list:
    print("Processing ", image_name)
    im_gt_y = sio.loadmat(image_name)['im_gt_y']
    im_b_y = sio.loadmat(image_name)['im_b_y']
    im_l = sio.loadmat(image_name)['im_l']

    im_gt_y = im_gt_y.astype(float)
    im_b_y = im_b_y.astype(float)
    im_l = im_l.astype(float)

    psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)
    avg_psnr_bicubic += psnr_bicubic

    im_input = torch.from_numpy(im_l).permute(2, 0, 1).to(device)
    im_input = im_input.float().div(255).unsqueeze(0)

    start_time = time.time()
    HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    im_h = HR_4x.squeeze().cpu().numpy()
    im_h = np.clip(im_h, 0, 1) * 255
    im_h = im_h.transpose(1, 2, 0)

    # im_h_ycbcr = rgb2ycbcr(im_h)
    # im_h_y = im_h_ycbcr[:, :, 0]
    im_h_y = im_h.dot(np.array([65.481, 128.553, 24.966]) / 255.) + 16

    psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=opt.scale)
    avg_psnr_predicted += psnr_predicted

print("Scale=", opt.scale)
print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
print("PSNR_bicubic=", avg_psnr_bicubic / len(image_list))
print("It takes average {:.3f} ms for processing".format(avg_elapsed_time / len(image_list) * 1e3))
