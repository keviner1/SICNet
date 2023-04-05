#!/usr/local/bin/python
import sys
import os
import time
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sys
src = str(os.path.split(os.path.realpath(__file__))[0]).replace("\\","/")
from PIL import Image
from PIL import ImageDraw
import glob
import math
from torchvision.transforms import ToTensor
import cv2
import logging
from einops import rearrange
#-------------------------------------dataLoad------------------------------------------
class My_Dataset(Dataset):
    def __init__(self,args,mode='train'):
        Train_len = 888888
        Val_len = 888888
        Test_len = 888888
        if mode == "train":
            self.left_images = glob.glob(args.SERVER.TRAIN_DATA+"/*_L.png")    #  left mono
            self.Right_images = glob.glob(args.SERVER.TRAIN_DATA+"/*_R.png")   #  right color
            self.Target_images = glob.glob(args.SERVER.TRAIN_DATA+"/*_T.png")  #  left gt
            self.TargetR_images = glob.glob(args.SERVER.TRAIN_DATA+"/*_HR.png")#  right gt
            self.left_images = sorted(self.left_images)[:Train_len]
            self.Right_images = sorted(self.Right_images)[:Train_len]
            self.Target_images = sorted(self.Target_images)[:Train_len]
            self.TargetR_images = sorted(self.TargetR_images)[:Train_len]
        elif mode == "val":
            self.left_images = glob.glob(args.SERVER.VAL_DATA+"/*_L.png")
            self.Right_images = glob.glob(args.SERVER.VAL_DATA+"/*_R.png")
            self.Target_images = glob.glob(args.SERVER.VAL_DATA+"/*_T.png")
            self.TargetR_images = glob.glob(args.SERVER.VAL_DATA+"/*_HR.png")
            self.left_images = sorted(self.left_images)[:Val_len]
            self.Right_images = sorted(self.Right_images)[:Val_len]
            self.Target_images = sorted(self.Target_images)[:Val_len]
            self.TargetR_images = sorted(self.TargetR_images)[:Val_len]
        elif mode == "test":
            self.left_images = glob.glob(args.SERVER.TEST_DATA+"/*_L.png")
            self.Right_images = glob.glob(args.SERVER.TEST_DATA+"/*_R.png")
            self.Target_images = glob.glob(args.SERVER.TEST_DATA+"/*_T.png")
            self.TargetR_images = glob.glob(args.SERVER.TEST_DATA+"/*_HR.png")
            self.left_images = sorted(self.left_images)[:Test_len]
            self.Right_images = sorted(self.Right_images)[:Test_len]
            self.Target_images = sorted(self.Target_images)[:Test_len]
            self.TargetR_images = sorted(self.TargetR_images)[:Test_len]

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, index):
        L_img = Image.open(self.left_images[index])
        R_img = Image.open(self.Right_images[index])
        T_img = Image.open(self.Target_images[index])
        TR_img = Image.open(self.TargetR_images[index])
        L = np.array(L_img, dtype=np.uint8).astype(np.float32) / 255.0
        R = np.array(R_img, dtype=np.uint8).astype(np.float32) / 255.0
        T = np.array(T_img, dtype=np.uint8).astype(np.float32) / 255.0
        TR = np.array(TR_img, dtype=np.uint8).astype(np.float32) / 255.0
        L = ToTensor()(L)
        R = ToTensor()(R)
        T = ToTensor()(T)
        TR = ToTensor()(TR)
        # print("------totensor-------",L.size())    #'C H W'
        samples = {
            "Left": L,
            "Right": R,
            "True": T,
            "TR": TR,
        }
        return samples, self.left_images[index]

def Noise_Add(imgL, imgR, SETUP):  # for createing noise dataset
    if SETUP == 2:
        R_std = 0.03
    elif SETUP == 3:
        R_std = 0.07
    noiseL = np.random.normal(0, 0.01, size=imgL.shape).astype(np.float32)
    imgL = noiseL + imgL
    noiseR = np.random.normal(0, R_std, size=imgR.shape).astype(np.float32)
    imgR = noiseR + imgR
    return imgL.clip(0, 1), imgR.clip(0, 1)

def build_dataset(args, mode):
    return My_Dataset(args,mode)

#-------------------------------------Trian/validation one epoch ------------------------------------------
def train_one_epoch(args, teacher, model, criterion, dataloader_train, optimizer, lr_scheduler, epoch, logger):
    if teacher != 0:
        teacher.eval()
    model.train()
    criterion.eval()
    optimizer.zero_grad()
    num_steps = len(dataloader_train)
    start = time.time()
    totalloss = 0
    for _, samples in enumerate(dataloader_train):
        L = samples[0]['Left'].cuda()
        R = samples[0]['Right'].cuda()
        T = samples[0]['True'].cuda()
        TR = samples[0]['TR'].cuda()
        #==========================================================================
        R = nn.functional.interpolate(R, scale_factor=4, mode='bicubic', align_corners=True)
        left_pre, right_pre, M, output = model(left = L, right = R)
        #========================Lout + Llp + Lrp=================================
        if args.TRAIN.lamda1_step != 0:
            pre_weight = args.TRAIN.lamda1 / math.pow(10,math.floor(epoch/args.TRAIN.lamda1_step))
        else:
            pre_weight = args.TRAIN.lamda1
        loss = criterion(output, T) + criterion(left_pre, T)*pre_weight + criterion(right_pre, TR)*pre_weight
        #========================parallax attention distillation===================
        M_, output_ = teacher(left_gray = 0, left_color = T, right_color = TR, mode = "teacher")
        if args.TRAIN.lamda2_step != 0:
            kd_weight = args.TRAIN.lamda2 / math.pow(10,math.floor(epoch/args.TRAIN.lamda2_step))
        else:
            kd_weight = args.TRAIN.lamda2
        loss = loss + criterion(M_, M)*kd_weight
        #===========================================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)
        cur_lr=optimizer.param_groups[-1]['lr']
        totalloss = totalloss + loss.item()
        if _ % args.TRAIN.PRINT_FREQ == 0 and (args.distributed == False or torch.distributed.get_rank() == 0):
            logger.info(f"||train -- epoch:{epoch} / {args.TRAIN.EPOCHS} --step:{_+1} / {num_steps} --lr:{cur_lr:.7f}-----curloss: {loss.item():.5f}")
    loss_avg = totalloss / num_steps
    epoch_time = time.time() - start 
    if args.distributed == False or torch.distributed.get_rank() == 0:
        Logger_msg = f":-------------------------Train: Epoch:{epoch:04}, timespend:{datetime.timedelta(seconds=int(epoch_time))}, avg_loss:{loss_avg:.5f}"
        logger.info(Logger_msg)
        

@torch.no_grad()
def validate(args, teacher, model, criterion, dataloader_val, epoch, logger, dataset_val_len):
    if teacher != 0:
        teacher.eval()
    model.eval()
    criterion.eval()
    num_steps = len(dataloader_val)
    start = time.time()
    totalloss = 0
    psnr_x = 0 
    psnr_l = 0
    psnr_r = 0
    ssim_x = 0
    for _, samples in enumerate(dataloader_val):
        if _ % args.TRAIN.PRINT_FREQ == 0 and (args.distributed == False or torch.distributed.get_rank() == 0):
            logger.info(f"|| Val -- epoch:{epoch} / {args.TRAIN.EPOCHS} ------step:{_+1} / {num_steps}")
        L = samples[0]['Left'].cuda()
        R = samples[0]['Right'].cuda()
        T = samples[0]['True'].cuda()
        TR = samples[0]['TR'].cuda()
        #==========================================================================
        R = nn.functional.interpolate(R, scale_factor=4, mode='bicubic', align_corners=True)
        left_pre, right_pre, M, output = model(left = L, right = R)
        #==========================================================================
        if args.TRAIN.lamda1_step != 0:
            pre_weight = args.TRAIN.lamda1 / math.pow(10,math.floor(epoch/args.TRAIN.lamda1_step))
        else:
            pre_weight = args.TRAIN.lamda1
        loss = criterion(output, T) + criterion(left_pre, T)*pre_weight + criterion(right_pre, TR)*pre_weight
        #==========================================================================
        psnr_x = psnr_x + cal_psnr(output, T)
        ssim_x = ssim_x + cal_ssim(output, T)
        psnr_l = psnr_l + cal_psnr(left_pre, T)
        psnr_r = psnr_r + cal_psnr(right_pre, TR)
        #==========================================================================
        totalloss = totalloss + loss.item()
    loss_avg = totalloss / num_steps
    psnr_l = psnr_l / num_steps
    psnr_r = psnr_r / num_steps
    psnr_x = psnr_x / num_steps 
    ssim_x = ssim_x / num_steps
    epoch_time = time.time() - start 
    #
    if args.distributed == False or torch.distributed.get_rank() == 0:
        Logger_msg = f":Validation: Epoch:{epoch:04}, timespend:{datetime.timedelta(seconds=int(epoch_time))}, loss_avg:{loss_avg:.4f}, psnr_l:{psnr_l:.4f},psnr_r:{psnr_r:.4f},psnr_x:{psnr_x:.4f}, ssim_x:{ssim_x:.4f} "
        logger.info(Logger_msg)
    return {"PSNR":psnr_x,"SSIM":ssim_x}

#-------------------------------------metrics ------------------------------------------
def cal_psnr(imag1, imag2):
    psnr_sum = 0
    for i in range(imag1.shape[0]):
        im1 = imag1[i]
        im2 = imag2[i]
        im1 = im1.cpu().numpy()
        im2 = im2.cpu().numpy()
        im1 = im1 * 255.0
        im2 = im2 * 255.0
        mse = (np.abs(im1 - im2) ** 2).mean()
        if mse == 0:
            return 100
        psnr = 10 * np.log10(255 * 255 / mse)
        psnr_sum = psnr_sum + psnr
    psnr_sum = psnr_sum / imag1.shape[0]
    return psnr_sum

def cal_ssim(imag1,imag2):    #"b C H W"
    ssim_sum = 0
    for i in range(imag1.shape[0]):
        im1 = imag1[i].permute(1,2,0)  #to 'H W C'
        im2 = imag2[i].permute(1,2,0)  #to 'H W C'
        im1 = im1.cpu().numpy()
        im2 = im2.cpu().numpy()
        im1 = im1 * 255.0
        im2 = im2 * 255.0
        ssim = calculate_ssim(im1,im2)
        #ssim = structural_similarity(im1, im2, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        ssim_sum = ssim_sum + ssim
    ssim_sum = ssim_sum / imag1.shape[0]
    return ssim_sum

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


#-------------------------------------log ------------------------------------------
def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=True):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root,"log/"+'SICNet_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

#-------------------------------------log ------------------------------------------


if __name__ == '__main__':
    pass

