
import torch.nn as nn
from Tools import Conv2dDynamicSamePadding
from Tools import MemoryEfficientSwish
from Tools import drop_connect
from Tools import SE
from typing import TypeVar


Tensor = TypeVar("Tensor")

class ConvBlock(nn.Module):

    """
    Mobile Inverted Residual Bottleneck Block.
    """

    def __init__(self, inChannels, outChannels, kernel, expand_ratio, drop_ratio, stride = 1):
        super().__init__()
        self._bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        self._bn_eps = 1e-3
        self.drop_ratio = drop_ratio
        self.stride = stride
        self.downSample = nn.Sequential(nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, stride=stride),
                                            nn.BatchNorm2d(num_features=outChannels, momentum=self._bn_mom, eps=self._bn_eps),
                                            MemoryEfficientSwish())
        # Expansion phase (Inverted Bottleneck)
        inp = inChannels  # number of input channels
        oup = inChannels * expand_ratio  # number of output channels
        self._expand_conv = Conv2dDynamicSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish1 = MemoryEfficientSwish()

        # Depthwise convolution phase
        k = kernel
        s = stride
        self._depthwise_conv = Conv2dDynamicSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish2 = MemoryEfficientSwish()

        self.se = SE(oup, oup, reduce_factor=4)

        # Pointwise convolution phase
        final_oup = outChannels
        self._project_conv = Conv2dDynamicSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)


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

        xCopy = self.downSample(inputs)
        x = x + xCopy
        return x



class Blocks(nn.Module):

    def __init__(self, layers_num, inChannels: int, outChannels: int, expand_ratio: int,
                 drop_ratio = 0.2,  stride = 1):
        super(Blocks, self).__init__()
        self.first = ConvBlock(inChannels, outChannels, 3, expand_ratio, drop_ratio, stride=stride)
        layers = []
        for _ in range(layers_num):
            layers.append(ConvBlock(outChannels, outChannels, 3,  expand_ratio, drop_ratio, stride=1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.seq(x)
        return x



class Encoder(nn.Module):

    def __init__(self, inChannels, encoderChannels ,drop_ratio = 0.2,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.):

        super(Encoder, self).__init__()
        bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        bn_eps = 1e-3
        ### 128 * 512
        self.conv_stem = Conv2dDynamicSamePadding(inChannels, int(32 * channelsExpandRatio), kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=int(32 * channelsExpandRatio), momentum=bn_mom, eps=bn_eps)
        self.stem_swish = MemoryEfficientSwish()
        ### 64 * 256
        self.b1 = Blocks(layers_num=int(1 * layersExpandRatio), inChannels=int(32 * channelsExpandRatio),
                         outChannels=int(32 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=1)
        ### 64 * 256
        self.b2 = Blocks(layers_num=int(2 * layersExpandRatio), inChannels=int(32 * channelsExpandRatio),
                         outChannels=int(32 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=1)
        ### 32 * 128
        self.b3 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(32 * channelsExpandRatio) ,
                         outChannels=int(40 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        ### 16 * 64
        self.b4 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(40 * channelsExpandRatio) ,
                         outChannels=int(56 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        ### 8 * 32
        self.b5 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(56 * channelsExpandRatio) ,
                         outChannels=int(64 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        self.Seq = nn.Sequential(nn.Flatten(),
                                 nn.Linear(int(64 * channelsExpandRatio) * 8 * 32, encoderChannels),
                                 nn.BatchNorm1d(encoderChannels))


    def forward(self,x):
        stem = self.stem_swish(self.bn0(self.conv_stem(x)))
        b1 = self.b1(stem)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        return self.Seq(b5)









