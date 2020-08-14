
import torch.nn as nn
from Tools import Conv2dDynamicSamePadding
from Tools import MemoryEfficientSwish
from Tools import drop_connect
from Tools import SE
from typing import TypeVar
import torch
import torch.nn.functional as F

__all__ = ["ConvBlock", "EncoderModule"]

Tensor = TypeVar("Tensor")

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.point_project = nn.Conv2d(planes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.act = MemoryEfficientSwish()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.point_project(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):

    """
    Mobile Inverted Residual Bottleneck Block.
    """

    def __init__(self, inChannels, outChannels, kernel, expand_ratio, drop_ratio,  skip = True, stride = 1):
        super().__init__()
        self._bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        self._bn_eps = 1e-3
        self.drop_ratio = drop_ratio

        # Expansion phase (Inverted Bottleneck)
        inp = inChannels  # number of input channels
        oup = inChannels * expand_ratio  # number of output channels
        self._expand_conv = Conv2dDynamicSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = kernel
        self.id_skip = skip
        s = stride
        self._depthwise_conv = Conv2dDynamicSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        self.se = SE(oup, oup, reduce_factor=4)

        # Pointwise convolution phase
        final_oup = outChannels
        self._project_conv = Conv2dDynamicSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish1 = MemoryEfficientSwish()
        self._swish2 = MemoryEfficientSwish()

    def forward(self, inputs):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this block after processing.
        """
        # Expansion and Depthwise Convolution
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish1(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish2(x)

        # attention
        x = self.se(x) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # drop connect
        if self.drop_ratio > 0:
            x = drop_connect(x, p=self.drop_ratio, training=self.training)

        # skip connection
        if self.id_skip :
            xCopy = inputs.clone()
            x = x + xCopy

        return x



class Blocks(nn.Module):

    def __init__(self, layers_num, inChannels: int, outChannels: int, expand_ratio: int,
                 drop_ratio = 0.2,  stride = 1):
        super(Blocks, self).__init__()
        layers = [ConvBlock(inChannels, outChannels, 3, expand_ratio, drop_ratio,
                            skip=False, stride=stride)]
        for _ in range(layers_num-1):
            layers.append(ConvBlock(outChannels, outChannels, 3,  expand_ratio, drop_ratio,
                                    skip=True, stride=1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class EncoderModule(nn.Module):

    def __init__(self, inChannels, enFC,drop_ratio = 0.2,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.):
        """
        :param inChannels: input image channels number
        :param enFC: encoder feature channels number

        """
        super(EncoderModule, self).__init__()
        rates = [1, 2, 3, 4]
        bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        bn_eps = 1e-3
        ### 128 * 512
        self.conv_stem = Conv2dDynamicSamePadding(inChannels, int(32 * channelsExpandRatio), kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=int(32 * channelsExpandRatio), momentum=bn_mom, eps=bn_eps)
        self.stem_swish = MemoryEfficientSwish()
        ### 64 * 256
        self.b1 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(32 * channelsExpandRatio),
                         outChannels=int(16 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=1)
        ### 32 * 128
        self.b2 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(16 * channelsExpandRatio),
                         outChannels=int(24 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        self.b2Trans = nn.Sequential(nn.Conv2d(int(24 * channelsExpandRatio),out_channels=1,kernel_size=1),
                                     nn.BatchNorm2d(1,momentum=bn_mom, eps=bn_eps),
                                     MemoryEfficientSwish())
        ### 16 * 64
        self.b3 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(24 * channelsExpandRatio),
                         outChannels=int(40 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        ### 8 * 32
        self.b4 = Blocks(layers_num=int(2 * layersExpandRatio), inChannels=int(40 * channelsExpandRatio),
                         outChannels=int(80 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        ### 8 * 32
        self.b5 = Blocks(layers_num=int(2 * layersExpandRatio), inChannels=int(80 * channelsExpandRatio),
                         outChannels=int(112 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=1)
        ### 8 * 32
        self.b6 = Blocks(layers_num=int(2 * layersExpandRatio), inChannels=int(112 * channelsExpandRatio),
                         outChannels=int(128 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio,stride=1)
        ### aspp
        self.aspp1 = ASPP_module(int(128 * channelsExpandRatio), int(32 * channelsExpandRatio), rate=rates[0])
        self.aspp2 = ASPP_module(int(128 * channelsExpandRatio), int(32 * channelsExpandRatio), rate=rates[1])
        self.aspp3 = ASPP_module(int(128 * channelsExpandRatio), int(32 * channelsExpandRatio), rate=rates[2])
        self.aspp4 = ASPP_module(int(128 * channelsExpandRatio), int(32 * channelsExpandRatio), rate=rates[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(int(128 * channelsExpandRatio),
                                                       int(32 * channelsExpandRatio), 1, stride=1, bias=False),
                                             nn.BatchNorm2d(int(32 * channelsExpandRatio)),
                                             MemoryEfficientSwish())

        self.b7 = nn.Sequential(nn.Conv2d(in_channels=int(160 * channelsExpandRatio)
                                          , out_channels=enFC, kernel_size=1),
                                nn.BatchNorm2d(enFC, momentum=1-0.99, eps=1e-3),
                                MemoryEfficientSwish())


    def forward(self,x):
        stem = self.stem_swish(self.bn0(self.conv_stem(x)))
        #print("stem shape: {}".format(stem.shape))
        b1 = self.b1(stem)
        #print("b1 shape: {}".format(b1.shape))
        b2 = self.b2(b1)
        #print("b2 shape: {}".format(b2.shape))
        b3 = self.b3(b2)
        #print("b3 shape: {}".format(b3.shape))
        b4 = self.b4(b3)
        #print("b4 shape: {}".format(b4.shape))
        b5 = self.b5(b4)
        #print("b5 shape: {}".format(b5.shape))
        b6 = self.b6(b5)
        #print("b6 shape: {}".format(b6.shape))
        x1 = self.aspp1(b6)
        x2 = self.aspp2(b6)
        x3 = self.aspp3(b6)
        x4 = self.aspp4(b6)
        x5 = self.global_avg_pool(b6)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        xCat = torch.cat([x1, x2, x3, x4, x5], dim=-3)
        #return self.b7(xCat)
        return self.b2Trans(b2), self.b7(xCat)







