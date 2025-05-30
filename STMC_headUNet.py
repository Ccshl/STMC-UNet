# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils import UpConvBlock, Upsample

from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        upsample_cfg = dict(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BasicConvBlock(in_size, out_size)
        self.up = Upsample(**upsample_cfg)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv(outputs)

        return outputs


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


@MODELS.register_module()
class UNetMEMCAMYFSKIPHead(BaseDecodeHead):
    def __init__(self,
                 in_filters=[64, 128, 256, 512],
                 num_classes=2,
                 in_channels=512,
                 channels=96,
                 init_cfg=None,
                 ekernel_sizes=[1, 3, 5, 7],
                 expansion_factor=6,
                 dw_parallel=True,
                 add=True,
                 lgag_ks=3,
                 activation='relu',
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False),
                 **kwargs
                 ):
        super().__init__(init_cfg=init_cfg, in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)
        out_filters01 = in_filters[2]
        out_filters02 = in_filters[1]
        out_filters03 = in_filters[0]

        in_filters01 = in_filters[3] + in_filters[2]
        in_filters02 = in_filters[2] + in_filters[1]
        in_filters03 = in_filters[1] + in_filters[0]

        self.up = Upsample(**upsample_cfg)

        self.mscb4 = MSCBLayer(in_filters[3], in_filters[3], n=1, stride=1, kernel_sizes=ekernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.mscb3 = MSCBLayer(in_filters[2], in_filters[2], n=1, stride=1, kernel_sizes=ekernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.mscb2 = MSCBLayer(in_filters[1], in_filters[1], n=1, stride=1, kernel_sizes=ekernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.mscb1 = MSCBLayer(in_filters[0], in_filters[0], n=1, stride=1, kernel_sizes=ekernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.cab4 = EMA(in_filters[3])
        self.cab3 = EMA(in_filters[2])
        self.cab2 = EMA(in_filters[1])
        self.cab1 = EMA(in_filters[0])

        self.sab = SAB()

        self.conv1 = ConvModule(
            in_channels=in_filters[3] + in_filters[2],
            out_channels=in_filters[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.conv2 = ConvModule(
            in_channels=in_filters[2] + in_filters[1],
            out_channels=in_filters[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.conv3 = ConvModule(
            in_channels=in_filters[1] + in_filters[0],
            out_channels=in_filters[0],
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.pwc3a = nn.Sequential(
            nn.Conv2d(in_filters[3], in_filters[2], kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.pwc3b = nn.Sequential(
            nn.Conv2d(in_filters[2], in_filters[3], kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.pwc2a = nn.Sequential(
            nn.Conv2d(in_filters[2], in_filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.pwc2b = nn.Sequential(
            nn.Conv2d(in_filters[1], in_filters[2], kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.pwc1a = nn.Sequential(
            nn.Conv2d(in_filters[1], in_filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.pwc1b = nn.Sequential(
            nn.Conv2d(in_filters[0], in_filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.sigmoid = nn.Sigmoid()
        self.stskip4 = CAB(in_filters[3] + in_filters[2] + in_filters[1] + in_filters[0])
        self.stskip3 = CAB(in_filters[3] + in_filters[2] + in_filters[1] + in_filters[0])
        self.stskip2 = CAB(in_filters[2] + in_filters[1] + in_filters[0])
        self.stskip1 = CAB(in_filters[1] + in_filters[0])


        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4] = inputs
        print(feat1.shape)
        print(feat2.shape)
        print(feat3.shape)
        print(feat4.shape)
        f1 = feat1
        f2 = feat2
        f3 = feat3
        f4 = feat4
        feat1_channels = feat1.size(1)
        feat2_channels = feat2.size(1)
        feat3_channels = feat3.size(1)
        feat4_channels = feat4.size(1)

        # f4
        f41 = self.pool(f1)
        f41 = self.pool(f41)
        f41 = self.pool(f41)
        f42 = self.pool(f2)
        f42 = self.pool(f42)
        f43 = self.pool(f3)
        f44 = f4
        fcat4 = torch.cat([f41, f42, f43, f44], 1)
        fcat4 = self.stskip4(fcat4) * fcat4
        fcat4 = self.sab(fcat4) * fcat4
        f44 = fcat4[:, feat1_channels + feat2_channels + feat3_channels:, :, :]
        f44 = f44 + feat4

        # f3
        f31 = self.pool(f1)
        f31 = self.pool(f31)
        f32 = self.pool(f2)
        f33 = f3
        f34 = self.up(f44)
        fcat3 = torch.cat([f31, f32, f33, f34], 1)
        fcat3 = self.stskip3(fcat3) * fcat3
        fcat3 = self.sab(fcat3) * fcat3
        f33 = fcat3[:, feat1_channels + feat2_channels:feat1_channels + feat2_channels + feat3_channels, :, :]
        f33 = f33 + feat3

        fb11 = fcat3[:, :feat1_channels, :, :]

        # f2
        f21 = self.pool(f1)
        f22 = f2
        f23 = self.up(f33)

        fcat2 = torch.cat([f21, f22, f23], 1)
        fcat2 = self.stskip2(fcat2) * fcat2
        fcat2 = self.sab(fcat2) * fcat2
        f22 = fcat2[:, feat1_channels:feat1_channels + feat2_channels, :, :]
        f22 = f22+feat2

        # f1
        f11 = f1
        f12 = self.up(f22)

        fcat1 = torch.cat([f11, f12], 1)
        fcat1 = self.stskip1(fcat1) * fcat1
        fcat1 = self.sab(fcat1) * fcat1
        f11 = fcat1[:, :feat1_channels, :, :]
        f11 = f11 + feat1



        da4 = self.cab4(f44)
        db4 = self.mscb4(f44)
        d4 = da4 * db4
        print(d4.shape)

        # up
        d3 = self.up(d4)

        # cat
        dd3 = self.pwc3a(d3)
        dd3 = self.sigmoid(dd3)
        ft3 = feat3 * dd3

        ftt3 = self.pwc3b(f33)
        ftt3 = self.sigmoid(ftt3)
        dt3 = d3 * ftt3

        d3 = torch.cat([ft3, dt3], dim=1)
        d3 = self.conv1(d3)


        da3 = self.cab3(d3)
        db3 = self.mscb3(d3)
        d3 = da3 * db3
        print(d3.shape)

        # up
        d2 = self.up(d3)

        # cat
        dd2 = self.pwc2a(d2)
        dd2 = self.sigmoid(dd2)
        ft2 = feat2 * dd2

        ftt2 = self.pwc2b(f22)
        ftt2 = self.sigmoid(ftt2)
        dt2 = d2 * ftt2

        d2 = torch.cat([ft2, dt2], dim=1)
        d2 = self.conv2(d2)

        da2 = self.cab2(d2)
        db2 = self.mscb2(d2)
        d2 = da2 * db2
        print(d2.shape)

        # up
        d1 = self.up(d2)

        # cat
        dd1 = self.pwc1a(d1)
        dd1 = self.sigmoid(dd1)
        ft1 = feat1 * dd1

        ftt1 = self.pwc1b(f11)
        ftt1 = self.sigmoid(ftt1)
        dt1 = d1 * ftt1

        d1 = torch.cat([ft1, dt1], dim=1)
        d1 = self.conv3(d1)


        da1 = self.cab1(d1)
        db1 = self.mscb1(d1)
        d1 = da1 * db1
        print(d1.shape)
        exit()

        output = self.cls_seg(d1)

        return output
