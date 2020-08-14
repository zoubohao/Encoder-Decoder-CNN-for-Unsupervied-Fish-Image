import torch
import torch.nn as nn
from EncoderNet import ConvBlock
from Tools import MemoryEfficientSwish


class UpSampling(nn.Module):

    def __init__(self,in_channels, out_channels, blockExpandRatio = 2.):

        super(UpSampling, self).__init__()
        self._bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        self._bn_eps = 1e-3
        #self.upSampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.transChannels = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                                           nn.BatchNorm2d(out_channels, momentum=self._bn_mom, eps=self._bn_eps))
        self.residual = nn.Sequential(ConvBlock(out_channels, out_channels, kernel=5,
                                                expand_ratio=int(blockExpandRatio * 1),
                                                drop_ratio=0.),
                                      ConvBlock(out_channels, out_channels, kernel=3,
                                                expand_ratio=int(blockExpandRatio * 1),
                                                drop_ratio=0.))

    def forward(self,x):
        #x = self.upSampling(x)
        x = self.transChannels(x)
        return self.residual(x)

class DecoderModule(nn.Module):

    def __init__(self, encoder_in_channels,channelsExpandRatio = 1., blockExpandRatio = 2.):
        super(DecoderModule, self).__init__()
        self.up1 = UpSampling(encoder_in_channels,
                              int(encoder_in_channels * channelsExpandRatio), blockExpandRatio = blockExpandRatio)
        self.up2 = UpSampling(int(encoder_in_channels * channelsExpandRatio),
                              int(encoder_in_channels * channelsExpandRatio),blockExpandRatio = blockExpandRatio)
        self.low_feat_conv = nn.Sequential(nn.Conv2d(1,
                                        out_channels=int(encoder_in_channels * channelsExpandRatio), kernel_size=1),
                                           nn.BatchNorm2d(int(encoder_in_channels * channelsExpandRatio)),
                                           MemoryEfficientSwish())
        self.up3 = UpSampling(int(encoder_in_channels * channelsExpandRatio * 2),
                              int(encoder_in_channels * channelsExpandRatio), blockExpandRatio = blockExpandRatio)
        self.up4 = UpSampling(int(encoder_in_channels * channelsExpandRatio),
                              int(encoder_in_channels * channelsExpandRatio),blockExpandRatio = blockExpandRatio)
        self.imageConv = nn.Sequential(nn.Conv2d(int(encoder_in_channels * channelsExpandRatio),
                                                 int(32 * channelsExpandRatio), kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(int(32 * channelsExpandRatio)),
                                       MemoryEfficientSwish(),
                                       nn.Conv2d(int(32 * channelsExpandRatio),
                                                 int(8 * channelsExpandRatio), kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(int(8 * channelsExpandRatio)),
                                       nn.Conv2d(int(8 * channelsExpandRatio), 3, kernel_size=3, stride=1, padding=1))

    def forward(self,low_feat, encoderX):
        #print("encoder shape : ",encoderX.shape)
        up1 = self.up1(encoderX)
        #print(up1.shape)
        up2 = self.up2(up1)
        #print(up2.shape)
        low_feat = self.low_feat_conv(low_feat)
        addT = torch.cat([low_feat, up2], dim=1)
        up3 = self.up3(addT)
        #print(up3.shape)
        up4 = self.up4(up3)
        #print(up4.shape)
        return self.imageConv(up4)











