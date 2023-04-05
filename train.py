#!/usr/local/bin/python
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import time
import os
import datetime
import random
import numpy as np
import argparse
import model.SICNet as SICNet
import model.Teacher.StereoWarp as StereoWarp
from config import config_1
from torch.utils.data import DataLoader
from myUtils import build_dataset, train_one_epoch, validate
from pathlib import Path
import myUtils as myUtils
import logging
import math

def run(args):
    #-----------------------------------log--------------------------------
    myUtils.setup_logger('base', args.SERVER.OUTPUT, level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    # ---------------------------------DDP--------------------------------
    seed = 1
    rank = 0
    world_size = 1
    args.distributed = False
    local_rank = 0
    logger.info("no distributed")
    logger.info(f"cur: {torch.cuda.get_device_name(0)}--{local_rank}; cuda: {torch.cuda.is_available()}; gpus: {torch.cuda.device_count()}")


    # device = torch.device("cuda")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    #---------------------------------Teacher_model----------------------------
    teacher = 0
    if args.TRAIN.Teacher != "":
        teacher = StereoWarp.StereoWarp(channels = 64)
        CheckPoint = torch.load(args.TRAIN.Teacher)
        CheckPoint = CheckPoint['model']
        CheckPoint = {key.replace("module.", ""): val for key, val in CheckPoint.items()}
        if args.distributed == False or torch.distributed.get_rank() == 0:
            logger.info('teacher model:  %s' % args.TRAIN.Teacher)
        teacher.load_state_dict(CheckPoint)
        teacher.cuda(local_rank)
        for p in teacher.parameters():
            p.requires_grad = False
        n_parameters = sum(p.numel() for p in teacher.parameters())
        if args.distributed == False or torch.distributed.get_rank() == 0:
            logger.info('number of teacher params: %.2f M' % (n_parameters / 1024 / 1024))
    # ---------------------------------main_model--------------------------------
    model = SICNet.SICNet(channels = args.MODEL.channels)
    model.cuda(local_rank)
    # ----------------------------criterion-----optimizer-----LR_Schedule------------------------------
    criterion = torch.nn.L1Loss().cuda(local_rank)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.TRAIN.BASE_LR)
    if args.TRAIN.LR_MODE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.TRAIN.LR_STEP, gamma=args.TRAIN.LR_DECAY)
    elif args.TRAIN.LR_MODE == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.TRAIN.MILESTONES, gamma=args.TRAIN.LR_DECAY)
    # ---------------------------------dataloader--------------------------------
    dataset_train = build_dataset(args, mode='train')
    dataset_val = build_dataset(args, mode='val')
    dataset_val_len = len(dataset_val)
    dataset_train_len = len(dataset_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.TRAIN.BATCH_SIZE, drop_last=True)
    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  num_workers=args.TRAIN.NUM_WORKERS, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.TRAIN.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.TRAIN.NUM_WORKERS,
                                pin_memory=True)
    # ---------------------------------start--------------------------------
    start_time = time.time()
    best_PSNR = 0
    best_SSIM = 0
    start_epoch = 1
    #-----------------------------------params-------------------------------
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.distributed == False or torch.distributed.get_rank() == 0:
        logger.info('number of model_main params: %.2f M' % (n_parameters / 1024 / 1024))
        logger.info(f"----start training-----{args.TRAIN.lamda1}--{args.TRAIN.lamda1_step}-{args.TRAIN.lamda2}--{args.TRAIN.lamda2_step}---")
    # ---------------------------------RESUME--------------------------------
    if args.TRAIN.RESUME != "":
        checkpoint = torch.load(args.TRAIN.RESUME)
        net = checkpoint['model']
        net = {key.replace("module.", ""): val for key, val in net.items()}
        if args.distributed == False or torch.distributed.get_rank() == 0:
            logger.info('resume from %s' % args.TRAIN.RESUME)
        model.load_state_dict(net)
        start_epoch = checkpoint['epoch']+1
    # ---------------------------------Train and Validate--------------------------------
    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        #train
        train_one_epoch(args, teacher, model, criterion, dataloader_train, optimizer, lr_scheduler, epoch, logger)
        #validation
        eval_status = validate(args, teacher, model, criterion, dataloader_val, epoch, logger, dataset_val_len)
        cur_PSNR = eval_status['PSNR']
        cur_SSIM = eval_status['SSIM']
        if cur_PSNR > best_PSNR or cur_SSIM > best_SSIM:
            #save 
            #distributed
            Path(args.SERVER.OUTPUT+"/checkpoints").mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(args.SERVER.OUTPUT+f"/checkpoints", f'checkpoint_{epoch:04}_{cur_PSNR:.6}_{cur_SSIM:.6}.pth')
            best_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(best_checkpoint, checkpoint_path)
            #log
            if args.distributed == False or torch.distributed.get_rank() == 0:
                Logger_msg = f"----------------CheckPoint:  curEpoch:{epoch:04},  PSNR:{cur_PSNR:.6},  SSIM:{cur_SSIM:.6}"
                logger.info(Logger_msg)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.distributed == False or  torch.distributed.get_rank() == 0:
        Logger_msg = "---------------Training time {}".format(total_time_str)
        logger.info(Logger_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default = -1)
    parser.add_argument("--config", default = 1, type=int)
    args = parser.parse_args()
    if args.config == 1:
        run(config_1.get_config())

