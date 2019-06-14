from __future__ import division

"""
 * 
 *  .--,       .--,
 * ( (  \.---./  ) )
 *  '.__/o   o\__.'
 *     {=  ^  =}
 *      >  -  <
 *     /       \
 *    //       \\
 *   //|   .   |\\
 *   "'\       /'"_.-~^`'-.
 *      \  _  /--'         `
 *    ___)( )(___
 *   (((__) (__)))    高山仰止,景行行止.虽不能至,心向往之。
 */ author:ydc  date:20190530
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import torch.utils.model_zoo as model_zoo
import pysnooper

__all__ = ['glore_resnet18_112']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

#---------

class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x).permute(0, 2, 1)
        return self.conv1d_2(h).permute(0, 2, 1)


class GloRe(nn.Module): #28            16        12
    def __init__(self, in_channels, mid_channels, N):    #(28,16,12) batch_size=4
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.N = N

        self.phi = nn.Conv2d(in_channels, mid_channels, 1)      #(28,16)
        self.theta = nn.Conv2d(in_channels, N, 1)               #(28,12)
        self.gcn = GCN(N, mid_channels)                         #(12,16)
        self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)  #(16,28)

    def forward(self, x):   #4,28,6,6
        batch_size, in_channels, h, w = x.shape
        mid_channels = self.mid_channels
        N = self.N

        B = self.theta(x).view(batch_size, N, -1)           #(4,12,7)
        x_reduced = self.phi(x).view(batch_size, mid_channels, h * w)  #(4,16,7)
        x_reduced = x_reduced.permute(0, 2, 1)        #(4,7,16)
        v = B.bmm(x_reduced)              #(4,12,16)

        z = self.gcn(v)                   #(12,16)
        y = B.permute(0, 2, 1).bmm(z).permute(0, 2, 1)    #()
        y = y.view(batch_size, mid_channels, h, w)
        x_res = self.phi_inv(y)

        return x + x_res

def glore(in_channels=4, mid_channels=4, N=4):

    model = GloRe(in_channels, mid_channels, N)
    return model


#==========

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000,
                 extra_conv=False, dropout=0, glore=None):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(ResNet, self).__init__()
        # if bottleneck:
        #   block = Bottleneck
        # else:
        #   block = BasicBlock
        self.glore = None
        self.inplanes = baseWidth  # default 64
        self.extra_conv = extra_conv
        self.dropout_ratio = dropout
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth)
        else:
            self.conv1 = nn.Conv2d(3, baseWidth // 2, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth // 2)
            self.conv2 = nn.Conv2d(baseWidth // 2, baseWidth // 2, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(baseWidth // 2)
            self.conv3 = nn.Conv2d(baseWidth // 2, baseWidth, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(baseWidth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = [self._make_layer(block, baseWidth, layers[0])]
        for idx, layer in enumerate(layers[1:]):
            if idx == 1:       #position of glore
                self.glore = glore

            self.layers.append(self._make_layer(block, baseWidth * pow(2, idx + 1), layer, 2, self.glore))
        self.reslayers = nn.Sequential(*self.layers)

        if self.extra_conv:
            self.extra_conv_layer = nn.Conv2d(baseWidth * 8, 512, 1, 1, 0, bias=False)
            self.bn_extra_conv = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.extra_conv:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(baseWidth * pow(2, len(layers) - 1) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, glore=None):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        if glore is not None:
            layers.append(glore(planes,planes//2,planes//2))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.reslayers(x)
        if self.extra_conv:
            x = self.extra_conv_layer(x)
            x = self.bn_extra_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_ratio > 0:
            x = self.dropout(x)
        x = self.fc(x)

        return x

def glore_resnet18_112(**kwargs):
    num_classes = 1000
    drop_out = 0
    glore = None
    if 'num_classes' in kwargs:
        num_classes = kwargs['num_classes']
    if 'dropout' in kwargs:
        drop_out = kwargs['dropout']
    if 'glore' in kwargs:
        glore = kwargs['glore']
    model = ResNet(block=BasicBlock, baseWidth=64, head7x7=False, layers=(2, 2, 2, 2), num_classes=num_classes,
                       extra_conv=False, dropout=drop_out, glore=glore)
    return model




