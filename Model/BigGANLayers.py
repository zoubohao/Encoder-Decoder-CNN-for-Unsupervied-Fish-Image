import torch.nn as nn
from Tools import MemoryEfficientSwish
from Tools import Conv2dDynamicSamePadding
import torch
import torch.nn.functional as F

# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ClassConditionalBatchNorm(nn.Module):
    def __init__(self, output_size, input_size, eps=1e-5, momentum=0.01):
        super().__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = nn.Linear(input_size, output_size)
        self.bias = nn.Linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1. + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                            self.training, self.momentum, self.eps)
        return out * gain + bias


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 if_classBN = False, concatEmbeddingSize = 0):
        super(GBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation1 = MemoryEfficientSwish()
        self.activation2 = MemoryEfficientSwish()
        # upsample layers
        self.upsample2 = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.upBN = nn.BatchNorm2d(out_channels,eps=1e-3, momentum=1-0.99)
        # Conv layers
        self.conv1 = Conv2dDynamicSamePadding(out_channels,out_channels, kernel_size=3,bias=False)
        self.conv2 = Conv2dDynamicSamePadding(out_channels,out_channels, kernel_size=3,bias=False)
        if if_classBN:
            # Batchnorm layers
            self.bn1 = ClassConditionalBatchNorm(out_channels, concatEmbeddingSize)
            self.bn2 = ClassConditionalBatchNorm(out_channels, concatEmbeddingSize)
        else:
            # Batchnorm layers
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y = None):
        x = self.upsample2(x)
        if y is not None:
            h = self.activation1(self.bn1(x, y))
        else:
            h = self.activation1(self.bn1(x))
        x = self.upBN(x)
        h = self.conv1(h)
        if y is not None:
            h = self.activation2(self.bn2(h, y))
        else:
            h = self.activation2(self.bn2(h))
        h = self.conv2(h)
        return h + x


