import torch
import torch.nn as nn
from Tools import Bottleneck


class UpSampling(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpSampling, self).__init__()
        self.transpose = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                                       nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
        self.block = nn.Sequential(Bottleneck(out_channels, out_channels, stride=1),
                                   Bottleneck(out_channels, out_channels, stride=1))

    def forward(self,x):
        x = self.transpose(x)
        return self.block(x)


class Decoder (nn.Module):

    def __init__(self, encoderChannels, channelsExpandRatio):

        super(Decoder, self).__init__()
        self.channels = int(64 * channelsExpandRatio)
        self.upChannels = nn.Sequential(nn.Linear(encoderChannels, int(64 * channelsExpandRatio) * 8 * 32),
                                        nn.BatchNorm1d(int(64 * channelsExpandRatio) * 8 * 32))
        ###
        self.upSample1 = UpSampling(self.channels, 96 ,kernel_size=4, stride=2, padding=1)
        ###
        self.upSample2 = UpSampling(96, 64, kernel_size=4, stride=2, padding=1)
        ###
        self.upSample3 = UpSampling(64, 56, kernel_size=4, stride=2, padding=1)
        ###
        self.upSample4 = UpSampling(56, 48, kernel_size=4, stride=2, padding=1)
        ###
        self.imageConv = nn.Sequential(nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

    def forward(self,inputs):
        upChannels = self.upChannels(inputs)
        upChannels = torch.reshape(upChannels, [-1, self.channels, 8, 32])
        up2 = self.upSample1(upChannels)
        up3 = self.upSample2(up2)
        up4 = self.upSample3(up3)
        up5 = self.upSample4(up4)
        return self.imageConv(up5)


