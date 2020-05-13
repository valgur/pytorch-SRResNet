from __future__ import print_function, division, absolute_import

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from hubconf import SRResNet

parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--device", default="cuda", help="device to use, e.g. 'cpu', 'cuda' (default) or 'cuda:0'")
parser.add_argument("--model", type=str, help="local model path (optional)")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, default: 4")


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

if opt.model:
    model = torch.load(opt.model, map_location=device)["model"]
else:
    model = SRResNet(pretrained=True, map_location=device)
model.to(device)

im_gt = sio.loadmat("testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_gt']
im_b = sio.loadmat("testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_b']
im_l = sio.loadmat("testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_l']

im_gt = im_gt.astype(float).astype(np.uint8)
im_b = im_b.astype(float).astype(np.uint8)
im_l = im_l.astype(float).astype(np.uint8)

im_input = torch.from_numpy(im_l).permute(2, 0, 1).to(device)
im_input = im_input.float().div(255).unsqueeze(0)

start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

im_h = out.cpu().squeeze().numpy()
im_h = np.clip(im_h, 0, 1) * 255
im_h = im_h.transpose(1, 2, 0)

print("Dataset=", opt.dataset)
print("Scale=", opt.scale)
print("It takes {:.3f} ms for processing".format(elapsed_time * 1e3))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(Bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h.astype(np.uint8))
ax.set_title("Output(SRResNet)")
plt.show()
