import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, CBIR_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def compute_reg_loss(model, weight_decay):
    """Compute reg loss."""

    l2_lambda = weight_decay / 2
    l2_reg = torch.tensor(0., device=model.device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            l2_reg += torch.norm(param)
    l2_reg = l2_lambda * l2_reg
    return l2_reg

def do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    # device = "cuda"
    device = local_rank
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    if dist.get_rank() == 0:
        logger.info("Running with config:\n{}".format(cfg))
    logger.info('start training')
    model.to(local_rank)
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    trp_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.TEST.CBIR_METRIC:
        evaluator = CBIR_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    """
    for epoch in range(1, epochs + 1):
        if cfg.MODEL.DIST_TRAIN and cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader.sampler.set_epoch(epoch)
        print(f"Epoch: {epoch}")
        for n_iter, (img, vid) in enumerate(train_loader):
            # if n_iter == 0 and dist.get_rank() == 0:
            if n_iter % 10 == 0 and dist.get_rank() == 0:
                unique_vid, counts = torch.unique(vid, return_counts=True)
                print(f'Epoch {epoch}, Iter: {n_iter}, '
                      f'unique_id_num: {len(unique_vid)}, '
                      f'unique_id: {unique_vid}, unique_counts: {counts}')
    """
    """
    for epoch in range(1, epochs + 1):
        total_iter = 1544
        for n_iter in range(total_iter):
            accurate_epoch = epoch+float(n_iter + 1)/float(total_iter) - 1
            scheduler.step(accurate_epoch)    # this will slow down a little
            cur_lr = scheduler.get_lr()
            if n_iter % 100 == 0:
                print(epoch, accurate_epoch, cur_lr)
    """
    # train
    for epoch in range(1, epochs + 1):
        # NOTE: very important to shuffle data
        # Ref: https://zhuanlan.zhihu.com/p/97115875
        #      https://github.com/pytorch/pytorch/issues/31771
        if cfg.MODEL.DIST_TRAIN and cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        loss_meter.reset()
        cls_loss_meter.reset()
        trp_loss_meter.reset()
        reg_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.SOLVER.FP16_ENABLED:
                with amp.autocast(enabled=True):    # with fp16
                    score, feat = model(img, target)
                    loss, cls_loss, trp_loss, ver_loss = loss_fn(score, feat, target)
            else:
                score, feat = model(img, target)
                loss, cls_loss, trp_loss, ver_loss = loss_fn(score, feat, target)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            # reg_loss = compute_reg_loss(model, cfg.SOLVER.WEIGHT_DECAY)

            loss_meter.update(loss.item(), img.shape[0])
            cls_loss_meter.update(cls_loss.item(), img.shape[0])
            trp_loss_meter.update(trp_loss.item(), img.shape[0])
            # reg_loss_meter.update(reg_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            assert cfg.SOLVER.WARMUP_METHOD == 'cosine'
            if (n_iter + 1) % log_period == 0 and dist.get_rank() == 0:
                cur_lr = scheduler.get_lr()
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, CLS Loss: {:.3f}, TRP Loss: {:.3f}, REG Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, cls_loss_meter.avg, trp_loss_meter.avg, reg_loss_meter.avg, acc_meter.avg, cur_lr))
                loss_meter.reset()
                cls_loss_meter.reset()
                trp_loss_meter.reset()
                reg_loss_meter.reset()
                acc_meter.reset()
            accurate_epoch = epoch+float(n_iter + 1)/len(train_loader) - 1
            scheduler.step(accurate_epoch)    # this will slow down a little

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0:
            reg_loss = compute_reg_loss(model, 1e-4)
            logger.info("Reg loss {:.4f}".format(reg_loss.cpu().item()))
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_sampler.batch_size / time_per_batch))
        elif not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            model.eval()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                    h,w = cfg.INPUT.SIZE_TRAIN
                    dummy_input = torch.randn(1, 3, h, w, device='cuda')
                    try:
                        torch.onnx.export(model.module, dummy_input,
                                          "%s/reid_%s_%d.onnx" % (cfg.OUTPUT_DIR, cfg.MODEL.NAME, epoch),
                                          verbose=False)
                    except:
                        print('fail to export onnx')
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            # No hang: https://github.com/pytorch/pytorch/issues/54059
            if cfg.MODEL.DIST_TRAIN:
                model.eval()
                if dist.get_rank() == 0:
                    for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            feat = model.module(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")

    if cfg.TEST.CBIR_METRIC:
        print('eval for cbir')
        evaluator = CBIR_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        print('common rank-1 mAP eval')
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


