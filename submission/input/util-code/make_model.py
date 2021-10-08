"""Make model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.resnet import ResNet, Bottleneck
from backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a,resnet50_ibn_a_LeakyRelu
from backbones.resnest import resnest101, resnest50, resnest200, resnest269
from backbones.efficientnet import EfficientNet
from backbones.resnext_ibn import resnext101_ibn_a
from backbones.se_resnet_ibn_a import se_resnet101_ibn_a

class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, model_name, num_classes):
        super(Backbone, self).__init__()
        last_stride = 1
        self.cos_layer = True
        self.gem_pooling = True
        if self.gem_pooling:
            print('using GeM pooling')
            self.gem = GeM()

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=-1)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride, frozen_stages=-1)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet152_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet152_ibn_a(last_stride, frozen_stages=-1)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'efficientnet-b0':
            self.in_planes = 1280
            self.base = EfficientNet.from_name(model_name)
            # self.base = EfficientNet.from_name(model_name, drop_connect_rate=None)
            print('using efficientnet-b0 as a backbone')
        elif model_name == 'efficientnet-b1':
            self.in_planes = 1280
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b1 as a backbone')
        elif model_name == 'efficientnet-b2':
            self.in_planes = 1408
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b2 as a backbone')
        elif model_name == 'efficientnet-b3':
            self.in_planes = 1536
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b3 as a backbone')
        elif model_name == 'efficientnet-b4':
            self.in_planes = 1792
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b4 as a backbone')
        elif model_name == 'efficientnet-b5':
            self.in_planes = 2048
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b5 as a backbone')
        elif model_name == 'efficientnet-b6':
            self.in_planes = 2304
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b6 as a backbone')
        elif model_name == 'efficientnet-b7':
            self.in_planes = 2560
            self.base = EfficientNet.from_name(model_name)
            print('using efficientnet-b7 as a backbone')
        elif model_name == 'resnest50':
            self.in_planes = 2048
            self.base = resnest50(last_stride)
            print('using resnest50 as a backbone')
        elif model_name == 'resnest101':
            self.in_planes = 2048
            self.base = resnest101(last_stride)
            print('using resnest101 as a backbone')
        elif model_name == 'resnest200':
            self.in_planes = 2048
            self.base = resnest200(last_stride)
            print('using resnest200 as a backbone')
        elif model_name == 'resnest269':
            self.in_planes = 2048
            self.base = resnest269(last_stride)
            print('using resnest269 as a backbone')
        elif model_name == 'resnext101_ibn_a':
            self.in_planes = 2048
            self.base = resnext101_ibn_a()
            print('using resnext101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck_planes = 512
        self.dropout_rate = 0.7

        if self.neck_planes > 0:
            # self.pre_bn = nn.BatchNorm1d(self.in_planes, affine=False)
            # self.pre_bn.apply(weights_init_kaiming)
            self.fcneck = nn.Linear(self.in_planes, self.neck_planes, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.fcneck_bn = nn.BatchNorm1d(self.neck_planes)
            # self.fcneck_bn.bias.requires_grad_(False)
            self.fcneck_bn.apply(weights_init_kaiming)
            self.in_planes = self.neck_planes
            # print('fcneck is used.')

            self.relu = nn.ReLU(inplace=True)
        else:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            # print('dropout is used: %f.' %self.dropout_rate)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        if self.gem_pooling:
            global_feat = self.gem(x)
        else:
            global_feat = self.gap(x) + self.gmp(x)  # (b, 2048, 1, 1)
        # global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])    # used to cope with Transreid
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck_planes > 0:
            # global_feat = self.pre_bn(global_feat)
            global_feat = self.fcneck(global_feat)
            global_feat = self.fcneck_bn(global_feat)
            #global_feat = self.relu(global_feat)
            if self.dropout_rate > 0:
                global_feat_cls = self.dropout(global_feat)
            else:
                global_feat_cls = global_feat

            if self.training:
                if self.cos_layer:
                    cls_score = self.classifier(global_feat_cls, label)
                else:
                    cls_score = self.classifier(global_feat_cls)
                return cls_score, global_feat  # global feature for triplet loss
            else:
                return global_feat
        else:
            feat = self.bottleneck(global_feat)
            if self.training:
                if self.cos_layer:
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat
            else:
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                self.state_dict()[i.replace('module.', '')]
                print(f'Fail to set {i}')
        print('Loading pretrained model from {}'.format(trained_path))


def make_model(model_name, num_classes=751):
    """Make model."""

    model = Backbone(model_name, num_classes)
    return model
