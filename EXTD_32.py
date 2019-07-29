'''
EXTD Copyright (c) 2019-present NAVER Corp. MIT License
'''
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from layers.modules.l2norm import L2Norm
from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from torch.autograd import Variable

from layers import *
from data.config import cfg
import numpy as np

import mobileFacenet_32_PReLU


def upsample(in_channels, out_channels): # should use F.inpterpolate
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                  stride=1, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class EXTD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.


    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, head, num_classes):
        super(EXTD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        '''
        self.priorbox = PriorBox(size,cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        '''
        # SSD network
        self.base = nn.ModuleList(base)

        self.upfeat = []

        for it in range(5):
            self.upfeat.append(upsample(in_channels=32, out_channels=32))

        self.upfeat = nn.ModuleList(self.upfeat)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(8):
            x = self.base[k](x)

        s1 = x

        # apply vgg up to fc7
        for k in range(2, 8):
            x = self.base[k](x)

        s2 = x

        #x = self.base[8](x)
        #sources.append(x)

        for k in range(2, 8):
            x = self.base[k](x)

        s3 = x

        #x = self.base[9](x)
        #sources.append(x)

        for k in range(2, 8):
            x = self.base[k](x)

        s4 = x

        #x = self.base[10](x)
        #sources.append(x)

        for k in range(2, 8):
            x = self.base[k](x)

        s5 = x

        #x = self.base[11](x)
        #sources.append(x)

        for k in range(2, 8):
            x = self.base[k](x)

        s6 = x

        #x = self.base[12](x)
        sources.append(s6)

        u1 = self.upfeat[0](F.interpolate(s6, size=(s5.size()[2], s5.size()[3]), mode='bilinear')) + s5 # 10x10
        sources.append(u1)

        u2 = self.upfeat[1](F.interpolate(u1, size=(s4.size()[2], s4.size()[3]), mode='bilinear')) + s4 # 20x20
        sources.append(u2)

        u3 = self.upfeat[2](F.interpolate(u2, size=(s3.size()[2], s3.size()[3]), mode='bilinear')) + s3  # 40x40
        sources.append(u3)

        u4 = self.upfeat[3](F.interpolate(u3, size=(s2.size()[2], s2.size()[3]), mode='bilinear')) + s2  # 80x80
        sources.append(u4)

        u5 = self.upfeat[4](F.interpolate(u4, size=(s1.size()[2], s1.size()[3]), mode='bilinear')) + s1  # 160x160
        sources.append(u5)

        sources = sources[::-1]  # reverse order


        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        '''
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        '''

        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]

        self.priorbox = PriorBox(size, features_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if torch.is_tensor(m.bias):
                m.bias.data.zero_()



def mobileFacenet():
    net = mobileFacenet_32_PReLU.Net()
    #print(net)
    return nn.ModuleList(net.features)


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(cfg[k + 1]), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def add_extras_mobileFace(in_channel=32):
    layers = []
    channels = [in_channel]
    for v in channels:
        layers += [
            nn.Conv2d(in_channels=in_channel, out_channels=v, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(v),
            nn.ReLU(inplace=True)]

    return layers


def add_extras_dwc(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, in_channels,
                                     kernel_size=3, stride=2, padding=1, bias=False, groups=in_channels),
                           nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm2d(cfg[k + 1]), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def multibox(base, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    net_source = [4, 4, 4, 4]
    feature_dim = []

    feature_dim += [base[4].conv[-3].out_channels]
    for idx in net_source:
        feature_dim += [base[idx].conv[-3].out_channels]

    #print(feature_dim)

    loc_layers += [nn.Conv2d(feature_dim[0], 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(feature_dim[0], 3 + (num_classes - 1), kernel_size=3, padding=1)]

    for k, v in enumerate(net_source, 1):
        loc_layers += [nn.Conv2d(feature_dim[k], 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(feature_dim[k], num_classes, kernel_size=3, padding=1)]

    # for k, v in enumerate(extra_layers[3::6], 2):

    for v in [0]:
        print(extra_layers[v].out_channels)
        loc_layers += [nn.Conv2d(extra_layers[v].out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra_layers[v].out_channels,
                                  num_classes, kernel_size=3, padding=1)]

    return base[:8], extra_layers, (loc_layers, conf_layers)


def build_extd(phase, num_classes=2):
    base_, extras_, head_ = multibox(
        mobileFacenet(), add_extras_mobileFace(in_channel=32), num_classes)

    return EXTD(phase, base_, head_, num_classes)


if __name__ == '__main__':
    net = build_extd('train', num_classes=2)
    inputs = Variable(torch.randn(4, 3, 640, 640))
    output = net(inputs)
    # print(output)

