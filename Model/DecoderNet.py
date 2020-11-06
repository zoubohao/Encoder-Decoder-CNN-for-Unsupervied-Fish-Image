import torch
import torch.nn as nn
from Model.BigGANLayers import GBlock
from Tools import Bottleneck
from Model.EncoderNet import ConvBlock
from Tools import DeformConv2d

class Decoder (nn.Module):

    def __init__(self, encodedDimension,ch = 8, if_add_plate_infor = True, encoderImgHeight = 12, encoderImgWidth = 52):

        super(Decoder, self).__init__()
        self.emcodedDimension = encodedDimension
        self.channels = 16 * ch
        self.encoderHeight = encoderImgHeight
        self.encoderWidth = encoderImgWidth
        if if_add_plate_infor:
            ## the embedding size is emcodedDimension
            self.embeddingPlate = nn.Sequential(nn.Linear(1, encodedDimension // 2, bias=True),
                                                nn.BatchNorm1d(encodedDimension // 2),
                                                nn.Linear(encodedDimension // 2, encodedDimension, bias=True),
                                                nn.BatchNorm1d(encodedDimension))
            self.deBox = nn.Sequential(
                nn.Linear(encodedDimension * 2, self.channels * self.encoderHeight * self.encoderWidth),
                nn.BatchNorm1d(self.channels * self.encoderHeight * self.encoderWidth))
        else:
            self.deBox = nn.Sequential(
                nn.Linear(encodedDimension, self.channels * self.encoderHeight * self.encoderWidth),
                nn.BatchNorm1d(self.channels * self.encoderHeight * self.encoderWidth))
        self.concatDimension = encodedDimension
        ### 24
        self.upSample1 = GBlock(self.channels, 8 * ch, if_classBN=if_add_plate_infor, concatEmbeddingSize=self.concatDimension)
        self.medium1 = nn.Sequential(
            ConvBlock(8 * ch, 8 * ch, 3, 2, 0.0),
            DeformConv2d(8 * ch, 8 * ch),
            nn.BatchNorm2d(8 * ch, eps=1e-3, momentum=1 - 0.99),
            Bottleneck(8 * ch, 8 * ch))
        ### 48
        self.upSample2 = GBlock(8 * ch , 4 * ch, if_classBN=if_add_plate_infor, concatEmbeddingSize=self.concatDimension)
        self.medium2 = nn.Sequential(
            Bottleneck(4 * ch, 4 * ch),
            DeformConv2d(4 * ch, 4 * ch),
            nn.BatchNorm2d(4 * ch, eps=1e-3, momentum=1 - 0.99),
            ConvBlock(4 * ch, 4 * ch, 3, 2, 0.0),
        )
        ### 96
        self.upSample3 = GBlock(4 * ch, 2 * ch, if_classBN=if_add_plate_infor, concatEmbeddingSize=self.concatDimension)
        self.medium3 = nn.Sequential(
            Bottleneck(2 * ch, 2 * ch),
            ConvBlock(2*ch, 2*ch, 3, 2, 0.0))
        ###
        self.imageConv = nn.Sequential(
            nn.Conv2d(2*ch, 2*ch, 3, 1, 1),
            nn.BatchNorm2d(2 * ch, eps=1e-3, momentum=1-0.99),
            nn.Conv2d(2 * ch, 3, kernel_size=1, stride=1, padding=0))

    def forward(self,imgEncoded, platesInfor = None):
        if platesInfor is None:
            linear = self.deBox(imgEncoded)
            linear = torch.reshape(linear, shape=[-1, self.channels, self.encoderHeight, self.encoderWidth])
            up1 = self.upSample1(linear, None)
            medium1 = self.medium1(up1)
            up2 = self.upSample2(medium1, None)
            medium2 = self.medium2(up2)
            up3 = self.upSample3(medium2, None)
            medium3 = self.medium3(up3)
        else:
            #print("Add plate infor.")
            platesEmbedding = self.embeddingPlate(platesInfor)
            linear = self.deBox(torch.cat([imgEncoded, platesEmbedding], dim=-1))
            linear = torch.reshape(linear, shape=[-1, self.channels, self.encoderHeight, self.encoderWidth])
            up1 = self.upSample1(linear, platesEmbedding)
            medium1 = self.medium1(up1)
            up2 = self.upSample2(medium1, platesEmbedding)
            medium2 = self.medium2(up2)
            up3 = self.upSample3(medium2, platesEmbedding)
            medium3 = self.medium3(up3)
        return self.imageConv(medium3)

if __name__ == "__main__":
    testPlates = torch.randn(size=[5,1])
    testEncode = torch.randn(size=[5, 128])
    testModule = Decoder(128,if_add_plate_infor = False)
    print(testModule(testEncode, testPlates).shape)

