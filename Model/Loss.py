import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.PretrainedModel import resnet18_pre_trained


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



