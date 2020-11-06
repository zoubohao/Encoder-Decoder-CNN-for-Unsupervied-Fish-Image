import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.PretrainedModel import resnet18_pre_trained
import numpy as np


class HuberLoss(nn.Module):

    def __init__(self,delta = 0.5):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, inputs , targets):
        with torch.no_grad():
            absLoss = torch.abs(inputs - targets)
            lessDeltaMask = (absLoss <= self.delta).float()
            largeDeltaMask = 1. - lessDeltaMask
        lessDeltaLoss = F.mse_loss(inputs, targets ,reduction="none") * lessDeltaMask * 0.5
        largeDeltaLoss = F.l1_loss(inputs, targets, reduction="none") * largeDeltaMask * self.delta - 0.5 * self.delta ** 2
        tLoss = lessDeltaLoss + largeDeltaLoss
        return torch.mean(tLoss)

class PerceptionLoss(nn.Module):

    def __init__(self, pretrained_path):
        super().__init__()
        self.model = resnet18_pre_trained(pre_trained_path=pretrained_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.eval()

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, "The shape of inputs and targets are not same."
        perInput = self.model(inputs)
        perTarget = self.model(targets)
        loss = F.mse_loss(perInput, perTarget, reduction="mean")
        return loss

class BCELogitsWeightLoss(nn.Module):

    def __init__(self, threshold = 180., fixed_basic_coefficient = 1.2):
        super(BCELogitsWeightLoss, self).__init__()
        self.bceLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        self.threshold = threshold / 255.
        self.fixed = fixed_basic_coefficient

    def forward(self, inputs, targets):
        with torch.no_grad():
            meanTargets = torch.mean(targets, dim=1, keepdim=True)
            #print(meanTargets)
            whiteMask = (self.threshold <= meanTargets).float()
            blackMask = 1. - whiteMask
            #print(whiteMask)
            #print(blackMask)
            totalPixelsNum = np.prod(targets.shape)
            whiteWeight = torch.sum(whiteMask) / totalPixelsNum
            blackWeight = torch.sum(blackMask) / totalPixelsNum
        whitePixelsLoss = self.bceLogitsLoss(inputs, targets) * whiteMask * (blackWeight / whiteWeight)
        blackPixelsLoss = self.bceLogitsLoss(inputs, targets) * blackMask * self.fixed
        #print("white loss {}".format(torch.mean(whitePixelsLoss)))
        #print("black loss {}".format(torch.mean(blackPixelsLoss)))
        addLoss = whitePixelsLoss + blackPixelsLoss
        return torch.mean(addLoss)

if __name__ == "__main__":
    testLoss = BCELogitsWeightLoss()
    testInput = torch.rand(size=[5, 3, 128, 512])
    testTarget = torch.rand(size=[5, 3, 128, 512])
    print(testLoss(testInput, testTarget))