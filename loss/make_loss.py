# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = torch.nn.CrossEntropyLoss()
        print("label smooth off, numclasses:", num_classes)

    if 'triplet' not in cfg.MODEL.METRIC_LOSS_TYPE:
        def loss_func(score, feat, target):
            ID_LOSS = xent(score, target)
            DUMMY_LOSS = torch.tensor([0.0], device=score.get_device())
            return [ID_LOSS, ID_LOSS, DUMMY_LOSS, 0.0]
    else:
        def loss_func(score, feat, target):
            ID_LOSS = xent(score, target)
            TRI_LOSS = triplet(feat, target)[0]
            return [cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                    cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, ID_LOSS, TRI_LOSS, 0.0]

    return loss_func


