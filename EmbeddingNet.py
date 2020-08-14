from EncoderNet import EncoderModule
from DecoderNet import DecoderModule
import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):

    def __init__(self, in_channels, enFC, drop_ratio,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.):
        super(EmbeddingNet, self).__init__()
        self.encoder = EncoderModule(inChannels=in_channels, enFC=enFC,
                                     drop_ratio = drop_ratio,
                                     layersExpandRatio= layersExpandRatio,
                                     channelsExpandRatio = channelsExpandRatio,
                                     blockExpandRatio = blockExpandRatio)
        self.decoder = DecoderModule(enFC,
                                     channelsExpandRatio=channelsExpandRatio,
                                     blockExpandRatio=blockExpandRatio
                                     )
        self.flatten = nn.Flatten()

    def forward(self, x):
        low_feat , encoderTensor = self.encoder(x)
        originalImage = self.decoder(low_feat,encoderTensor)
        return torch.sigmoid(originalImage), \
               self.flatten(torch.cat([encoderTensor,torch.reshape(low_feat,encoderTensor.shape)], dim=1))

