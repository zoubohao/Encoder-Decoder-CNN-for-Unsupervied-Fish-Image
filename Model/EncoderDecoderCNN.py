import torch
import torch.nn as nn
from Model.EncoderNet import Encoder
from Model.DecoderNet import Decoder

class EncoderDecoderNet(nn.Module):
    """
    The encoder will down sample the original image 4 times. eg. [128, 512] --> [8, 32]
    The decoder will up sample the encoded tensor 4 times and convert it to original size.
    """

    def __init__(self,inChannels, encoderChannels,drop_ratio = 0.2,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.):
        super(EncoderDecoderNet, self).__init__()
        self.encoderDim = encoderChannels
        self.encoder = Encoder(inChannels, encoderChannels, drop_ratio, layersExpandRatio,
                               channelsExpandRatio, blockExpandRatio)
        self.decoder = Decoder(encoderChannels, channelsExpandRatio)
        self._initialize_weights()

    def forward(self, img):
        encoded = self.encoder(img)
        reproduced = self.decoder(encoded)
        return reproduced, encoded.flatten(start_dim=1, end_dim=-1)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class EncoderDecoderMultiGPUs(nn.Module):

    def __init__(self,inChannels, encoderChannels,drop_ratio = 0.2,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.,
                 device0 = "cuda:0", device1 = "cuda:1", split_size = 20):
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        self.split_size = split_size
        self.encoderDim = encoderChannels
        self.encoder = Encoder(inChannels, encoderChannels, drop_ratio, layersExpandRatio,
                               channelsExpandRatio, blockExpandRatio).to(device1)
        self.decoder = Decoder(encoderChannels, channelsExpandRatio).to(device0)
        self._initialize_weights()

    def forward(self, img):
        """
        :param img: only input cpu tensor
        :return: the return tensor is on device 0, so loss is also in device 0
        """
        img = img.to(self.device1)
        splits = iter(torch.split(img, split_size_or_sections=self.split_size,
                                  dim=0))
        s_next = next(splits)
        s_prev = self.encoder(s_next).to(self.device0)
        ret = []

        for s_next in splits:
            ### decoder
            s_prev = self.decoder(s_prev)
            ret.append(s_prev)

            ### encoder
            s_prev = self.encoder(s_next).to(self.device0)

        s_prev = self.decoder(s_prev)
        ret.append(s_prev)


        return torch.cat(ret)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




