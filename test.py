#!/usr/local/bin/python
import sys
sys.path.insert(1, '/data/pylib')
import torch
import torch.nn as nn
import time
import os
import datetime
import random
import numpy as np
import argparse
import model.SICNet as SICNet
from config import config_1
import model.Teacher.StereoWarp as StereoWarp
from torch.utils.data import DataLoader, DistributedSampler
from myUtils import build_dataset, cal_psnr,cal_ssim
from pathlib import Path
import myUtils
import logging
from PIL import Image
import torchvision.utils as vutils
from einops import rearrange
import math
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def save(img,path,flag,mode):
    img = rearrange(img, 'b C H W -> b H W C')
    img = torch.clamp(img.squeeze(), min=0, max=1)
    img = (img.cpu().numpy() * 255.0).round()
    img = Image.fromarray(np.uint8(img), mode=mode)
    cnt = path.split("\\")[-1].replace("L",flag)
    img.save(f"images\\out\\{cnt}")

def run(args, ckp):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SICNet.SICNet(channels = args.MODEL.channels)
    checkpoint = torch.load(f"ckp\\{ckp}")
    net = checkpoint['model']
    net = {key.replace("module.", ""): val for key, val in net.items()}
    model.load_state_dict(net)
    model.to(device)
    #----------------------------
    if args.TRAIN.Teacher != "":
        teacher = StereoWarp.StereoWarp(channels = 64)
        CheckPoint = torch.load(args.TRAIN.Teacher)
        CheckPoint = CheckPoint['model']
        CheckPoint = {key.replace("module.", ""): val for key, val in CheckPoint.items()}
        teacher.load_state_dict(CheckPoint)
        teacher.to(device)
    #--------------------------------
    dataset_test = build_dataset(args, mode='test')
    n = len(dataset_test)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = DataLoader(dataset_test, batch_size=1,sampler=sampler_test, num_workers=1,pin_memory=True)
    start = time.time()
    psnr = 0
    ssim = 0
    with torch.no_grad():
        model.eval()
        teacher.eval()
        for _, samples in enumerate(dataloader_val):
            L = samples[0]['Left'].cuda()
            R = samples[0]['Right'].cuda()
            T = samples[0]['True'].cuda()
            TR = samples[0]['TR'].cuda()
            R = nn.functional.interpolate(R, scale_factor=4, mode='bicubic', align_corners=True)
            left_pre, right_pre, M, output = model(left = L, right = R)
            psnr = psnr + cal_psnr(output, T)
            ssim = ssim + cal_ssim(output, T)
            save(output,samples[1][0],"out","RGB")

    end = time.time()
    gap = round(end * 1000) - round(start * 1000)
    print(f"time_avg: {gap/n:.4f}ms")
    print(f"psnr: {psnr/n:.4f}")
    print(f"ssim: {ssim/n:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default = -1)
    parser.add_argument("--config", default = 1, type=int)
    parser.add_argument("--ckp")
    args = parser.parse_args()
    if args.config == 1:
        cfg = config_1.get_config()
    run(cfg, args.ckp)
