import torch.nn.functional as F
import torch.nn as nn

import torch
from torch.utils.tensorboard import SummaryWriter
from Model.EncoderDecoderCNN import EncoderDecoderNet
#
### TRAIN CONFIG
batchSize = 22
imageRootPath = "./ImagesResizeTrain"
drop_ratio = 0.075
LR = 1e-4
multiplier = 10
reg_lambda = 1e-5
epoch = 150
warmEpoch = 15
displayTimes = 4
inputImageSize = [128, 512]  ## height, width
targetSize = [128, 512]
randomCropSize = [90, 400]
reduction = "mean"
### ENCODER CONFIG
encoderChannels = 200  ## 200
layersExpandRatio = 2.5
channelsExpandRatio = 1.5
blockExpandRatio = 2
### LOSS CONFIG
loss_coefficient = 1
### PRETRAIN CONFIG
if_load_check_point = False
load_check_point_path = "./CheckPoint/20000Times.pth"
check_point_save_folder = "./CheckPoint"

### Model
model = EncoderDecoderNet(inChannels=3, encoderChannels=encoderChannels, drop_ratio=drop_ratio,
                          layersExpandRatio=layersExpandRatio,
                          channelsExpandRatio=channelsExpandRatio,
                          blockExpandRatio=blockExpandRatio)

writer = SummaryWriter()
model = model.train(True)
testInput = torch.randn(size=[5, 3, 128, 512]).float()
writer.add_graph(model,testInput)
writer.close()

# import torchvision as tv
# from PIL import Image
# imgPath = "testBilnear.jpg"
# img = Image.open(imgPath).convert("RGB")
# imag = torch.unsqueeze(tv.transforms.ToTensor()(img), dim=0)
# bImg = F.interpolate(imag, mode="bilinear", size=[128, 512], align_corners=True).squeeze(dim=0)
# bImg = tv.transforms.ToPILImage()(bImg)
# bImg.save("testBilinearOri.jpg")

# from Model.EncoderNet import EncoderModule
#
# testModel = EncoderModule(3, 16, )
# testInput = torch.randn(size=[5, 3, 128, 512]).float()
# print(testModel(testInput))
# writer = SummaryWriter()
# model = testModel.train(True)
# testInput = torch.randn(size=[5, 3, 128, 512]).float()
# writer.add_graph(model,testInput)
# writer.close()
#
# testInput = torch.randn(size=[5, 3, 128, 512]).float()
# print(torch.mean(testInput))

from Model.PretrainedModel import resnet18_pre_trained

model = resnet18_pre_trained("./resnet18.pth")
model = model.eval()
testInput = torch.randn(size=[5, 3, 128, 512]).float()
print(model(testInput).shape)

