import os
import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .bases import ImageDataset
from .preprocessing import RandomErasing, RandomPatch
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP, ImageUniformSampler_DDP, GLDSampler_DDP
import torch.distributed as dist
from .ourapi import OURAPI

from .augmix import AugMix
from .detaug import DetColorJitter

__factory = {
    'ourapi': OURAPI,
}

class ShortDistributedSampler(DistributedSampler):
    def __init__(self, dataset, **kwargs):
        DistributedSampler.__init__(self, dataset, **kwargs)

    def __len__(self):
        # Control how many iterations
        # return 1000 iter * batch_size * 8
        return self.num_samples


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

def make_dataloader(cfg):
    logger = logging.getLogger("reid_baseline.check")
    trans_list = [T.RandomHorizontalFlip(p=cfg.INPUT.PROB), T.Pad(cfg.INPUT.PADDING), T.RandomCrop(cfg.INPUT.SIZE_TRAIN)]
    if cfg.INPUT.DET_AUG_PROB > 0:
        trans_list += [DetColorJitter(prob = cfg.INPUT.DET_AUG_PROB)]
        logger.info('detection aug is used: %f.' %cfg.INPUT.DET_AUG_PROB)
    # norm with mean and std
    if cfg.INPUT.COLOR_PROB > 0:
        trans_list += [T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)], p=cfg.INPUT.COLOR_PROB)]
        logger.info('color aug is used: %f.' %cfg.INPUT.COLOR_PROB)
    if cfg.INPUT.AUGMIX_PROB > 0:
        trans_list += [AugMix(prob = cfg.INPUT.AUGMIX_PROB)]
        logger.info('augmix aug is used: %f.' %cfg.INPUT.AUGMIX_PROB)
    if cfg.INPUT.RANDOM_PATCH_PROB > 0:
        trans_list += [RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16),]
        logger.info('randompatch aug is used: %f.' %cfg.INPUT.RANDOM_PATCH_PROB)
    if cfg.INPUT.RANDOM_AFFINE_PROB > 0:
        trans_list += [T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),]
        logger.info('randomaffine aug is used: %f.' %cfg.INPUT.RANDOM_AFFINE_PROB)
    trans_list += [T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD), RandomErasing(probability=cfg.INPUT.RE_PROB, sh=0.2, mean=[0.0, 0.0, 0.0])]
    train_transforms = T.Compose(trans_list)
    """
    train_transforms = T.Compose([
            # T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            RandomErasing(probability=cfg.INPUT.RE_PROB, sh=0.2, mean=[0.0, 0.0, 0.0])
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    """

    val_transforms = T.Compose([
        # T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, config=cfg)
    dataset = __factory[cfg.DATASETS.NAMES](root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)

    train_set = ImageDataset(dataset.train, train_transforms, cfg.INPUT.SIZE_TRAIN)
    # train_set_normal = ImageDataset(dataset.train, val_transforms, cfg.INPUT.SIZE_TRAIN)
    num_classes = dataset.num_train_pids

    if cfg.DATALOADER.SAMPLER == 'id_uniform':
        print('using id_uniform sampler')
        if cfg.MODEL.DIST_TRAIN:
            # print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                # batch_size=mini_batch_size,
                # sampler=data_sampler,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,    # TODO
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn,drop_last=True
            )
    elif cfg.DATALOADER.SAMPLER == 'gld':
        print('using gld sampler')
        if cfg.MODEL.DIST_TRAIN:
            # print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH
            data_sampler = GLDSampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                # batch_size=mini_batch_size,
                # sampler=data_sampler,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,    # TODO
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn,drop_last=True
            )
    elif cfg.DATALOADER.SAMPLER == 'img_uniform':
        print('using img_uniform sampler')
        if cfg.MODEL.DIST_TRAIN:
            # print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH
            data_sampler = ImageUniformSampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                # batch_size=mini_batch_size,
                # sampler=data_sampler,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,    # TODO
                sampler=ImageUniformSampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn,drop_last=True
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        if cfg.MODEL.DIST_TRAIN:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = dist.get_world_size()
            # datasampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, seed=cfg.SOLVER.SEED)
            datasampler = ShortDistributedSampler(train_set, num_replicas=world_size, rank=local_rank, seed=0)
            train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, sampler=datasampler,
                                      num_workers=num_workers, collate_fn=train_collate_fn)
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
    else:
        print('unsupported sampler! expected id_uniform, img_uniform, softmax but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.INPUT.SIZE_TEST)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    # train_loader_normal = DataLoader(
    #     train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
    #     collate_fn=val_collate_fn
    # )
    return train_loader, val_loader, len(dataset.query), num_classes
