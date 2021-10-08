from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import datetime
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
        if len(cfg.BASE) > 0:
            cfg.merge_from_file(cfg.BASE)
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    loss_postfix = f'{cfg.MODEL.ID_LOSS_TYPE}'
    if cfg.MODEL.ID_LOSS_TYPE == 'arcface':
        loss_postfix += f'scale{cfg.SOLVER.COSINE_SCALE}margin{cfg.SOLVER.COSINE_MARGIN}'
    if cfg.MODEL.NECK_BIAS:
        neck_bias_postfix = 'neckbias'
    else:
        neck_bias_postfix = 'necknobias'
    if len(cfg.DATALOADER.CACHE_LIST) > 0:
        cache_info = cfg.DATALOADER.CACHE_LIST.split('.pkl')[0]
    else:
        cache_info = 'nocache'
    aug_info = f'color{cfg.INPUT.COLOR_PROB}_affine{cfg.INPUT.RANDOM_AFFINE_PROB}'
    output_dir = cfg.OUTPUT_DIR + f'_input{cfg.INPUT.SIZE_TRAIN[0]}_bs{cfg.SOLVER.IMS_PER_BATCH}_loss{loss_postfix}_opt{cfg.SOLVER.OPTIMIZER_NAME}_lr{cfg.SOLVER.BASE_LR}_wd{cfg.SOLVER.WEIGHT_DECAY}_warm{cfg.SOLVER.WARMUP_EPOCHS}_ep{cfg.SOLVER.MAX_EPOCHS}_sche{cfg.SOLVER.WARMUP_METHOD}_drop{cfg.MODEL.CNN_DROPOUT}_re{cfg.INPUT.RE_PROB}_smooth{cfg.MODEL.IF_LABELSMOOTH}_sampler{cfg.DATALOADER.SAMPLER}_pad{cfg.INPUT.PADDING}_{neck_bias_postfix}_{cache_info}_{aug_info}'
    if output_dir and not os.path.exists(output_dir) and args.local_rank == 0:
        os.makedirs(output_dir)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    # logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://',
                                             timeout=datetime.timedelta(1800))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)

    if cfg.MODEL.NUM_CLASSES > 0:
        num_classes = 203094
    model = make_model(cfg, num_class=num_classes)

    loss_func = make_loss(cfg, num_classes=num_classes)

    optimizer = make_optimizer(cfg, model)
    args.sched = cfg.SOLVER.WARMUP_METHOD
    if args.sched == 'cosine':
        print('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        print('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
