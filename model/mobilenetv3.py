
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

""" 
Creates a MobileNet Model as defined in: 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
author: (c) ydc, 2019/5/10
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from torch.autograd.function import Function
import pysnooper

__all__ = ['mobilenet_v3_large','mobilenet_v3_small']

# Inherit from Function
class SEFuction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, se):
        ctx.save_for_backward(input, se)
        se = se.view(se.size(0), se.size(1), 1, 1)
        output = input * se.expand_as(input)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, se = ctx.saved_tensors
        se_temp = se.view(se.size(0), se.size(1), 1, 1)
        grad_input = grad_output * se_temp.expand_as(input)
        grad_se = torch.sum(grad_output * input, [2, 3]).reshape(se.shape)
        return grad_input, grad_se


@torch._jit_internal.weak_script
def SE(input, se):
    return SEFuction.apply(input, se)

#hs
class hswish_op(nn.Module):
    def __init__(self, inplace = True):
        super(hswish_op, self).__init__()
        self.relu6 = nn.ReLU6(inplace = inplace)
 
    def forward(self, x):
        return x * self.relu6(x + 3)/6

class hsigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace = inplace)
 
    def forward(self, x):
        return self.relu6(x + 3)/6


class relu(nn.Module):
    def __init__(self, inplace = True):
        super(relu, self).__init__()
        self.relu = nn.ReLU(inplace = inplace)
 
    def forward(self, x):
        return self.relu(x)
    
def conv_bn_3x3_hs(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        hswish_op()
    )

def conv_1x1_hs(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        hswish_op()
    )

def conv_1x1_hs_NBN(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        hswish_op()
    )


class InvertedResidual(nn.Module):
    #@pysnooper.snoop()
    def __init__(self, inp, oup, stride, expand_ratio, nl, nk):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        hidden_dim = int(hidden_dim)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        if nl == 'HS':
            act = hswish_op()
        else:
            act = relu()

        if self.expand_ratio == 1:
            print('hidden_dim',hidden_dim)
            print('stride',stride)
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, nk, stride=stride, padding=nk//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
                nn.Conv2d(hidden_dim, hidden_dim, nk, stride=stride, padding=nk//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    #@pysnooper.snoop()
    def forward(self, x):
        out = self.conv(x)
        print('outr',out.size())

        if self.use_res_connect:
            return x + out
        else:
            return out

class SE_InvertedResidual(nn.Module):
    #@pysnooper.snoop()    
    def __init__(self, inp, oup, stride, expand_ratio, nl, nk):
        super(SE_InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        hidden_dim = int(hidden_dim)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.expand_ratio = expand_ratio

        if nl == 'HS':
            act = hswish_op()
        else:
            act = relu()

        if self.expand_ratio == 1:
            print('hidden_dim',hidden_dim)
            print('stride',stride)
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, nk, stride=stride, padding=nk//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                #SEModule(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
                nn.Conv2d(hidden_dim, hidden_dim, nk, stride=stride, padding=nk//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                #SEModule(oup),
            )
            
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(oup, oup // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(oup // 16, oup)
        self.sigmoid = hsigmoid()
        self.SE = SE
        #self.se = SEModule(oup)
            
    #@pysnooper.snoop()
    def forward(self, x):
        out = self.conv(x)
        #print('outse',out.size())
        se = self.global_avg(out)
        se = se.view(se.size(0), -1)
        #print('outse1',out.size())        
        se = self.fc1(se)
        se = self.relu(se)
        #print('outse2',out.size())        
        se = self.fc2(se)
        se = self.sigmoid(se)
        #print('outse3',out.size())
        #out = SE(out, se)
        out = self.SE(out,se)
        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV3_large(nn.Module):
    #@pysnooper.snoop()
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout=0):
        super(MobileNetV3_large, self).__init__()
        input_channel = 16
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s, se, nl, nk
            [1, 16, 1, 1, 0, 'RE', 3],
            [4, 24, 1, 2, 0, 'RE', 3],
            [3, 24, 1, 1, 0, 'RE', 3],
            [3, 40, 1, 2, 1, 'RE', 5],
            [3, 40, 1, 1, 1, 'RE', 5],
            [3, 40, 1, 1, 1, 'RE', 5],
            [3, 80, 1, 2, 0, 'HS', 3],
            [2.5, 80, 1, 1, 0, 'HS', 3],
            [2.3, 80, 1, 1, 0, 'HS', 3],
            [2.3, 80, 1, 1, 0, 'HS', 3],
            [6, 112, 1, 1, 1, 'HS', 3],
            [6, 112, 1, 1, 1, 'HS', 3],
            [6, 112, 1, 1, 1, 'HS', 5],
            [6, 160, 1, 2, 1, 'HS', 5],
            [6, 160, 1, 1, 1, 'HS', 5],
        ]

        # building first layer
        assert input_size % 16 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.features = [conv_bn_3x3_hs(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, se, nl, nk in interverted_residual_setting:
            output_channel = int(c * width_mult)

            if se == 1:
                block = SE_InvertedResidual
            else:
                block = InvertedResidual

            self.features.append(block(input_channel, output_channel, s, expand_ratio=t, nl=nl, nk=nk))
    
            input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_hs(input_channel, 960))
        self.features.append(nn.AvgPool2d(7))


        self.features.append(conv_1x1_hs_NBN(960, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            conv_1x1_hs_NBN(self.last_channel, n_class),
        )

        self._initialize_weights()
    #@pysnooper.snoop()
    def forward(self, x):
        print('input',x.size())
#        print('input',x)
        x = self.features(x)
        print('11',x.size())
        x = self.classifier(x)
        print('22',x.size())
        x = torch.squeeze(x)
        #x = x.mean(3).mean(2)
        print('33',x.size())        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

#    def _initialize_weights(self):
#        # weight initialization
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                if m.bias is not None:
#                    nn.init.zeros_(m.bias)
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#            elif isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight, 0, 0.01)
#                if m.bias is not None:
#                    nn.init.zeros_(m.bias)


class MobileNetV3_small(nn.Module):
    @pysnooper.snoop()
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout=0):
        super(MobileNetV3_small, self).__init__()
        input_channel = 16
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s, se, nl, nk
            [1, 16, 1, 2, 1, 'RE', 3],
            [4.5, 24, 1, 2, 0, 'RE', 3],
            [3.67, 24, 1, 1, 0, 'RE', 3],
            [4, 40, 1, 1, 1, 'HS', 5],
            [6, 40, 1, 1, 1, 'HS', 5],
            [6, 40, 1, 1, 1, 'HS', 5],
            [3, 48, 1, 1, 1, 'HS', 5],
            [3, 48, 1, 1, 1, 'HS', 5],
            [6, 96, 1, 2, 1, 'HS', 5],
            [6, 96, 1, 1, 1, 'HS', 5],
            [6, 96, 1, 1, 1, 'HS', 5],
        ]

        # building first layer
        assert input_size % 16 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.features = [conv_bn_3x3_hs(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, se, nl, nk in interverted_residual_setting:
            output_channel = int(c * width_mult)

            if se == 1:
                block = SE_InvertedResidual
            else:
                block = InvertedResidual

            self.features.append(block(input_channel, output_channel, s, expand_ratio=t, nl=nl, nk=nk))
    
            input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_hs(input_channel, 576))
        self.features.append(nn.AvgPool2d(7))


        self.features.append(conv_1x1_hs(576, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            conv_1x1_hs(self.last_channel, n_class),
        )

        self._initialize_weights()
    #@pysnooper.snoop()
    def forward(self, x):
        print('input',x.size())
 #       print('input',x)
        x = self.features(x)
        print('11',x.size())
        x = self.classifier(x)
        print('33',x.size())
        x = x.mean(3).mean(2)
        print('33',x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def mobilenet_v3_large(widen_factor=1.0, num_classes=1000, dropout=0):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    model = MobileNetV3_large(width_mult=widen_factor, n_class=num_classes, dropout=dropout)
    return model


def mobilenet_v3_small(widen_factor=1.0, num_classes=1000, dropout=0):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    model = MobileNetV3_small(width_mult=widen_factor, n_class=num_classes, dropout=dropout)
    return model








