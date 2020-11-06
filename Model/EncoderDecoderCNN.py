import torch
import torch.nn as nn
from Model.EncoderNet import Encoder
from Model.DecoderNet import Decoder

class EncoderDecoderNet(nn.Module):
    """
    The encoder will down sample the original image 4 times. eg. [128, 512] --> [8, 32]
    The decoder will up sample the encoded tensor 4 times and convert it to original size.
    """

    def __init__(self,inChannels, encodedDimension,drop_ratio = 0.2,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.,
                 encoderImgHeight = 12, encoderImgWidth = 52, ch = 8, if_add_plate_infor = True):
        super(EncoderDecoderNet, self).__init__()
        self.encoderDim = encodedDimension
        self.encoder = Encoder(inChannels, encodedDimension, drop_ratio, layersExpandRatio,
                               channelsExpandRatio, blockExpandRatio, encoderImgHeight=encoderImgHeight, encoderImgWidth=encoderImgWidth)
        self.decoder = Decoder(encodedDimension,ch = ch, if_add_plate_infor=if_add_plate_infor,
                               encoderImgHeight=encoderImgHeight, encoderImgWidth=encoderImgWidth)
        self._initialize_weights()

    def forward(self, img, platesInfor = None):
        """
        :param img: GPU tensor, [N, C, H, W]
        :param platesInfor: GPU tensor, [N]
        :return:
        """
        encoded = self.encoder(img)
        if platesInfor is not None:
            platesInfor = torch.unsqueeze(platesInfor, dim=-1)  ## [N, 1]
            reproduced = self.decoder(encoded, platesInfor)
        else:
            reproduced = self.decoder(encoded, None)
        return reproduced, encoded.flatten(start_dim=1, end_dim=-1)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")

class EncoderDecoderMultiGPUs(nn.Module):
    """
    The loss is on device0 gpu
    """

    def __init__(self,inChannels:int, encodedDimension:int,drop_ratio = 0.2,
                 layersExpandRatio= 1., channelsExpandRatio = 1., blockExpandRatio = 2.,
                 device0 = "cuda:0", device1 = "cuda:1", split_size = 20,
                 encoderImgHeight = 12, encoderImgWidth = 52, ch = 8, if_add_plate_infor = True):
        """
        The loss is on device0 gpu

        :param inChannels: input images channels
        :param encodedDimension: the dimension of encoded vector
        :param drop_ratio: the drop ratio of drop connection of one block
        :param layersExpandRatio: this parameter can adjust the layers of encoder module. the basic layer number will multiple this ratio
        :param channelsExpandRatio: this parameter can adjust the channels of encoder module. the basic channels number will multiple this ratio
        :param blockExpandRatio: this parameter can adjust the expand channels in the block.
        :param device0: GPU 0
        :param device1: GPU 1
        :param split_size: we will split the data. this parameter will control the batch size in one forward processing.
        :param encoderImgHeight: the height of original size // 8 (Down sampling 3 times)
        :param encoderImgWidth: the width of original size // 8 (Down sampling 3 times)
        :param if_add_plate_infor: if add plate classes information
        """
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        self.split_size = split_size
        self.encoderDim = encodedDimension
        self.encoder = Encoder(inChannels, encodedDimension, drop_ratio, layersExpandRatio,
                               channelsExpandRatio, blockExpandRatio, encoderImgHeight=encoderImgHeight, encoderImgWidth=encoderImgWidth).to(device1)
        self.decoder = Decoder(encodedDimension,ch = ch, if_add_plate_infor=if_add_plate_infor,
                               encoderImgHeight=encoderImgHeight, encoderImgWidth=encoderImgWidth).to(device0)
        self._initialize_weights()

    def forward(self, img, platesInfor = None):
        """
        :param img: only input cpu tensor
        :param platesInfor: [N], cpu tensor
        :return: the return tensor is on device 0, so loss is also in device 0
        """
        if platesInfor is not None:
            ### tensor to GPUs
            img = img.to(self.device1)
            platesInfor = torch.unsqueeze(platesInfor.to(self.device0),dim=-1)

            ### img splits
            splits = iter(torch.split(img, split_size_or_sections=self.split_size, dim=0))
            s_next = next(splits)

            ### plates infor splits
            #print("Plates Infor is {}".format(platesInfor))
            platesSplits = iter(torch.split(platesInfor, split_size_or_sections=self.split_size, dim=0))
            plates_next = next(platesSplits)
            #print("Plates Next is {}".format(plates_next))

            s_prev = self.encoder(s_next).to(self.device0)
            #print("encoderTensor is {}".format(s_prev))
            ret = []

            for s_next in splits:
                ### decoder
                s_prev = self.decoder(s_prev, plates_next)
                ret.append(s_prev)
                plates_next = next(platesSplits)
                #print("Plates Next is {}".format(plates_next))
                ### encoder
                s_prev = self.encoder(s_next).to(self.device0)
            s_prev = self.decoder(s_prev, plates_next)
            ret.append(s_prev)
            return torch.cat(ret)
        else:
            ### tensor to GPUs
            img = img.to(self.device1)
            ### img splits
            splits = iter(torch.split(img, split_size_or_sections=self.split_size, dim=0))
            s_next = next(splits)
            s_prev = self.encoder(s_next).to(self.device0)
            ret = []
            for s_next in splits:
                ### decoder
                s_prev = self.decoder(s_prev, None)
                ret.append(s_prev)
                ### encoder
                s_prev = self.encoder(s_next).to(self.device0)
            s_prev = self.decoder(s_prev, None)
            ret.append(s_prev)
            return torch.cat(ret)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")




