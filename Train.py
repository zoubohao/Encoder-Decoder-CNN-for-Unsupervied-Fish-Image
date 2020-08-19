from Dataset import FishDataset
import torch
import torch.nn as nn
import torch.optim as optim
from WarmScheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
import torchvision as tv
from Model.EncoderDecoderCNN import EncoderDecoderMultiGPUs
import os
import torch.cuda.amp as amp
import torch.nn.functional as F


if __name__ == "__main__":

    ### TRAIN CONFIG
    batchSize = 42
    imageRootPath = "./ImagesResizeTrain"
    drop_ratio = 0.005
    LR = 1e-4
    multiplier = 10
    reg_lambda = 1e-5
    epoch = 100
    warmEpoch = 8
    displayTimes = 5
    inputImageSize = [128, 512] ## height, width
    targetSize = [128, 512]
    randomCropSize = [90, 400]
    reduction = "mean"
    split_size = 21
    ### ENCODER CONFIG
    encoderChannels = 200  ## 128
    layersExpandRatio = 2
    channelsExpandRatio = 1.25
    blockExpandRatio = 2
    ### LOSS CONFIG
    loss_coefficient = 1
    ### PRETRAIN CONFIG
    if_load_check_point = False
    load_check_point_path = "./CheckPoint/20000Times.pth"
    check_point_save_folder = "./CheckPoint"
    ### Date set
    transforms = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p = 0.3),
        tv.transforms.RandomVerticalFlip(p = 0.3),
        tv.transforms.RandomApply([tv.transforms.RandomCrop(size=randomCropSize)],p = 0.3),
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=25)], p = 0.3),
        tv.transforms.ToTensor()
    ])
    fishDataset = FishDataset(imageRootPath,transforms=transforms)
    fishDataLoader = DataLoader(fishDataset, batch_size=batchSize, num_workers=2, pin_memory=True)

    ### Model
    model = EncoderDecoderMultiGPUs(inChannels = 3, encoderChannels = encoderChannels,drop_ratio = drop_ratio,
                              layersExpandRatio= layersExpandRatio,
                              channelsExpandRatio = channelsExpandRatio,
                              blockExpandRatio = blockExpandRatio, split_size=split_size)

    if if_load_check_point:
        model.load_state_dict(torch.load(load_check_point_path))

    ### Loss
    #lossCri = HuberLoss(delta=0.1).to(device)
    #lossCri = nn.MSELoss(reduction=reduction).to(device)
    lossCri = nn.BCEWithLogitsLoss(reduction=reduction).to("cuda:0")
    #lossCri = PerceptionLoss("./resnet18.pth").to(device)

    ### Optimizer
    optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay=reg_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,
                                       after_scheduler=cosine_scheduler)
    scaler = amp.GradScaler()
    model = model.train(True)
    ### Train part
    currentTrainingTimes = 0
    for e in range(1, epoch + 1):
        for times , inputs in enumerate(fishDataLoader):
            inputsCuda = inputs.to("cuda:0")
            optimizer.zero_grad()
            with amp.autocast():
                netOutput = model(inputs)
                target = F.interpolate(inputsCuda, size=targetSize, mode="bilinear", align_corners=True)
                #print(netOutput.shape)
                #print(target.shape)
                assert netOutput.shape == target.shape, "The shape of net output is not as same as target."
                loss = lossCri(netOutput, target)
                loss = torch.mul(loss, loss_coefficient)
            if torch.isnan(loss).tolist() is False:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if currentTrainingTimes % displayTimes == 0:
                with torch.no_grad():
                    with open("LossRecord.txt", mode="a") as wh:
                        wh.write(str(loss.item()) + "\n")
                    print("######################")
                    print("Epoch : {} , Training time : {}".format(e, currentTrainingTimes))
                    print("Loss is {}".format(loss.item()))
                    print("Learning rate is {}".format(optimizer.state_dict()['param_groups'][0]["lr"]))
                    print("Input  {}".format(list(inputs.numpy()[0, 0, 1, 0:5])))
                    print("Net Output  {}".format(list(netOutput.detach().cpu().numpy()[0, 0, 1, 0:5])))
            currentTrainingTimes += 1
        torch.save(model.state_dict(), os.path.join(check_point_save_folder ,str(currentTrainingTimes) + "Times.pth"))
        scheduler.step()


