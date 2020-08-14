from Dataset import FishDataset
import torch
import torch.nn as nn
import torch.optim as optim
from WarmScheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
import torchvision as tv
from EmbeddingNet import EmbeddingNet
import os


if __name__ == "__main__":

    ### config
    device = "cuda:0"
    batchSize = 2
    imageRootPath = "./ImagesResizeTrain"
    drop_ratio = 0.1
    LR = 1e-5
    multiplier = 100
    reg_lambda = 1e-5
    epoch = 150
    warmEpoch = 15
    displayTimes = 10
    layersExpandRatio = 2.5
    channelsExpandRatio = 2.5
    blockExpandRatio = 7
    inputImageSize = [128, 512] ## height, width
    randomCropSize = [90, 400]
    reduction = "mean"
    encoderChannels = 16 ## 32 x 8 x encoderChannels
    loss_coefficient = 6.
    if_load_check_point = False
    load_check_point_path = "./CheckPoint/6400Times.pth"
    check_point_save_folder = "./CheckPoint"

    ### Date set
    transforms = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p = 0.25),
        tv.transforms.RandomVerticalFlip(p = 0.25),
        tv.transforms.RandomApply([tv.transforms.RandomCrop(size=randomCropSize)],p = 0.5),
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=25)], p = 0.5),
        tv.transforms.ToTensor()
    ])
    fishDataset = FishDataset(imageRootPath,transforms=transforms)
    fishDataLoader = DataLoader(fishDataset, batch_size=batchSize, num_workers=2, pin_memory=True)

    ### Model
    model = EmbeddingNet(in_channels=3, enFC=encoderChannels, drop_ratio=drop_ratio,
                         layersExpandRatio= layersExpandRatio,
                         channelsExpandRatio = channelsExpandRatio,
                         blockExpandRatio = blockExpandRatio).to(device)

    if if_load_check_point:
        model.load_state_dict(torch.load(load_check_point_path))

    ### Loss
    lossCri = nn.MSELoss(reduction=reduction).to(device)

    ### Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=reg_lambda, nesterov=True)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,
                                       after_scheduler=cosine_scheduler)

    model = model.train(True)
    ### Train part
    currentTrainingTimes = 0
    for e in range(1, epoch + 1):
        for times , inputs in enumerate(fishDataLoader):
            inputsCuda = inputs.to(device)
            optimizer.zero_grad()
            netOutput, _ = model(inputsCuda)
            loss = lossCri(netOutput, inputsCuda)
            loss = torch.mul(loss, loss_coefficient)
            if torch.isnan(loss).tolist() is False:
                loss.backward()
                optimizer.step()
            currentTrainingTimes += 1
            if currentTrainingTimes % displayTimes == 0:
                with torch.no_grad():
                    print("######################")
                    print("Epoch : {} , Training time : {}".format(e, currentTrainingTimes))
                    print("Loss is {}".format(loss.item()))
                    print("Learning rate is {}".format( optimizer.state_dict()['param_groups'][0]["lr"]))
                    print("Input  {}".format(list(inputs.numpy()[0, 0, 1, 0:5])))
                    print("Net Output  {}".format(list(netOutput.detach().cpu().numpy()[0, 0, 1, 0:5])))
        torch.save(model.state_dict(), os.path.join(check_point_save_folder ,str(currentTrainingTimes) + "Times.pth"))
        scheduler.step()



