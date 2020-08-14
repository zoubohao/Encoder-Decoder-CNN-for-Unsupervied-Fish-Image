import torch.nn.functional as F
import torch.nn as nn
from Tools import PixelAttention
from EncoderNet import EncoderModule
from EmbeddingNet import EmbeddingNet

import torch
from torch.utils.tensorboard import SummaryWriter
#
# ### config
# device = "cuda:1"
# batchSize = 4
# imageRootPath = "./ImagesResize"
# drop_ratio = 0.25
# drop_block_ratio = 0.15
# LR = 1e-5
# multiplier = 100
# reg_lambda = 5e-5
# epoch = 100
# warmEpoch = 10
# displayTimes = 5
# layersExpandRatio = 2
# channelsExpandRatio = 2
# blockExpandRatio = 6
# inputImageSize = [128, 512]  ## height, width
# randomCropSize = [90, 400]
#
# if_load_check_point = False
# load_check_point_path = ""
# check_point_save_folder = "./CheckPoint"
#
# ### Model
# model = EmbeddingNet(in_channels=3, enFC=32, drop_ratio=drop_ratio,
#                          layersExpandRatio= layersExpandRatio,
#                          channelsExpandRatio = channelsExpandRatio,
#                          blockExpandRatio = blockExpandRatio)
#
# writer = SummaryWriter()
#
# model = model.train(True)
# testInput = torch.randn(size=[5, 3, 128, 512]).float()
# writer.add_graph(model,testInput)
# writer.close()
# print(model(testInput)[0].shape)
# print(model(testInput)[1].shape)

# testInput = torch.randn(size=[5, 3, 64, 64]).float()
# testModule = nn.ConvTranspose2d(3, 16, kernel_size=6, stride=2, padding=2)
# print(testModule(testInput).shape)

import numpy as np
np.random.seed(19680801)
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    print(x.shape)
    print(y.shape)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()

testInput = torch.randn(size=[5, 3, 64, 64]).float()
testModule = nn.ConvTranspose2d(3, 16, kernel_size=4, stride=2, padding=1)
print(testModule(testInput).shape)



