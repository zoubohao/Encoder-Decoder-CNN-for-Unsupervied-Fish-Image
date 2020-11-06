import torch.nn as nn
from Tools import Conv2dDynamicSamePadding
from Tools import MemoryEfficientSwish
from Tools import drop_connect
from typing import TypeVar
from Tools import ChannelsAttention
from Tools import SpatialAttention
import torch


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
                                        nn.GroupNorm(num_groups=8, num_channels=outChannels, eps=self._bn_eps),
                                        MemoryEfficientSwish())
        # Expansion phase (Inverted Bottleneck)
        inp = inChannels  # number of input channels
        oup = inChannels * expand_ratio  # number of output channels
        self._expand_conv = Conv2dDynamicSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        self._bn0 = nn.GroupNorm(num_groups=8, num_channels=oup, eps=self._bn_eps)
        self._swish1 = MemoryEfficientSwish()

        ## Attention
        self.channelsAttention = ChannelsAttention(oup, oup, reduce_factor=2)
        self.spatialAttention = SpatialAttention(in_channels=oup, reduce_factor=5)

        # Depthwise convolution phase
        k = kernel
        s = stride
        self._depthwise_conv = Conv2dDynamicSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.GroupNorm(num_channels=oup, eps=self._bn_eps, num_groups=8)
        self._swish2 = MemoryEfficientSwish()

        # Pointwise convolution phase
        final_oup = outChannels
        self._project_conv = nn.Conv2d(oup, final_oup, kernel_size=1, padding=0, stride=1, bias=True)
        self._bn2 = nn.GroupNorm(num_channels=final_oup, eps=self._bn_eps, num_groups=8)


    def forward(self, inputs):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this block after processing.
        """
        # Expansion
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish1(x)

        # Attention
        x = self.channelsAttention(x) * x
        x = self.spatialAttention(x) * x

        # Depthwise Convolution
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish2(x)

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
        self.first = nn.Sequential(nn.Conv2d(inChannels, outChannels, 3, padding=1, stride=stride,bias=True),
                                   nn.GroupNorm(num_channels=outChannels, eps=1-0.99, num_groups=8))
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
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.,
                 encoderImgHeight = 12, encoderImgWidth = 52):

        super(Encoder, self).__init__()
        self.encoderHeight = encoderImgHeight
        self.encoderWidth = encoderImgWidth
        bn_eps = 1e-3
        ### 96 * 416
        self.conv_stem = Conv2dDynamicSamePadding(inChannels, int(32 * channelsExpandRatio), kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.GroupNorm(num_channels=int(32 * channelsExpandRatio), num_groups=8, eps=bn_eps)
        self.stem_swish = MemoryEfficientSwish()
        ### 48
        self.b1 = Blocks(layers_num=int(2 * layersExpandRatio), inChannels=int(32 * channelsExpandRatio),
                         outChannels=int(32 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=1)
        ### 48
        self.b2 = Blocks(layers_num=int(2 * layersExpandRatio), inChannels=int(32 * channelsExpandRatio),
                         outChannels=int(48 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=1)
        ### 24
        self.b3 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(48 * channelsExpandRatio) ,
                         outChannels=int(64 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        ### 12
        self.b4 = Blocks(layers_num=int(3 * layersExpandRatio), inChannels=int(64 * channelsExpandRatio) ,
                         outChannels=int(80 * channelsExpandRatio),expand_ratio=int(1 * blockExpandRatio),
                         drop_ratio=drop_ratio, stride=2)
        self.enBox = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(80 * channelsExpandRatio) * self.encoderHeight * self.encoderWidth, encoderChannels * 2),
            nn.LayerNorm(encoderChannels * 2),
            nn.Linear(encoderChannels * 2, encoderChannels)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.08)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.08)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        stem = self.stem_swish(self.bn0(self.conv_stem(x)))
        b1 = self.b1(stem)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        return self.enBox(b4)

if __name__ == "__main__":
    testInput = torch.randn(size=[5, 3, 96, 416])
    testModule = Encoder(3, 128)
    print(testModule(testInput).shape)