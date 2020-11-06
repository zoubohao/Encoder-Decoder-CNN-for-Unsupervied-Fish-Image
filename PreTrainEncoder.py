from AutoEncoderDataset import FishDataset
import torch
import torch.optim as optim
from WarmScheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
import torchvision as tv
from Model.EncoderDecoderCNN import EncoderDecoderMultiGPUs
import os
from Model.Loss import BCELogitsWeightLoss



if __name__ == "__main__":
    device0 = "cuda:1"
    device1 = "cuda:2"
    ### TRAIN CONFIG
    batchSize = 14
    imageRootPath = "./CropResizeTrain"
    drop_ratio = 0.01
    LR = 1e-5
    multiplier = 50
    reg_lambda = 1e-5
    epoch = 100
    swa_start = 0
    warmEpoch = 10
    displayTimes = 5
    inputImageSize = [96, 416] ## height, width
    targetSize = [96, 416]
    randomCropSize = [86, 380]
    reduction = "mean"
    split_size = 7
    ### ENCODER CONFIG
    encodedDimension = 128  ## 128
    layersExpandRatio = 2.5
    channelsExpandRatio = 1
    blockExpandRatio = 2
    if_add_plate_information = True
    ### PRETRAIN CONFIG
    if_load_check_point = False
    load_check_point_path = "./CheckPoint/102800Times.pth"
    check_point_save_folder = "./CheckPoint"

    ### Date set
    transforms = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p = 0.3),
        tv.transforms.RandomVerticalFlip(p = 0.3),
        tv.transforms.RandomApply([tv.transforms.RandomCrop(size=randomCropSize)],p = 0.5),
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.ToTensor()
    ])
    fishDataset = FishDataset(imageRootPath,transforms=transforms,platesPath="./Name2Plate.txt")
    fishDataLoader = DataLoader(fishDataset, batch_size=batchSize, num_workers=2, pin_memory=True, shuffle=True,drop_last=True)

    ### Model
    model = EncoderDecoderMultiGPUs(inChannels = 3,
                                    encodedDimension = encodedDimension,
                                    drop_ratio = drop_ratio,
                                    layersExpandRatio= layersExpandRatio,
                                    channelsExpandRatio = channelsExpandRatio,
                                    blockExpandRatio = blockExpandRatio,
                                    split_size=split_size,
                                    device0=device0, device1 = device1,
                                    encoderImgHeight = 12,
                                    encoderImgWidth = 52,
                                    ch = 12,
                                    if_add_plate_infor = if_add_plate_information)
    if if_load_check_point:
        model.load_state_dict(torch.load(load_check_point_path))

    ### Loss
    #lossCri = HuberLoss(delta=0.1).to(device)
    #lossCri = nn.MSELoss(reduction=reduction).to(device)
    #lossCri = nn.BCEWithLogitsLoss(reduction=reduction).to(device0)
    #lossCri = PerceptionLoss("./resnet18.pth").to(device)
    lossCri = BCELogitsWeightLoss(threshold=170, fixed_basic_coefficient=1.185).to(device0)

    ### Optimizer
    optimizer = optim.AdamW(model.parameters(), lr = LR, weight_decay=reg_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,
                                       after_scheduler=cosine_scheduler)
    model = model.train(True)
    with open("LossRecord.txt", mode="w") as wh:
        wh.write( "\n")
    ### Train part
    currentTrainingTimes = 0
    for e in range(1, epoch + 1):
        for _ , (inputs, imagePlateID) in enumerate(fishDataLoader):
            inputsCuda = inputs.to(device0)
            optimizer.zero_grad()
            if if_add_plate_information:
                netOutput = model(inputs, imagePlateID.float())
            else:
                netOutput = model(inputs, None)
            # print(netOutput.shape)
            # print(inputsCuda.shape)
            assert netOutput.shape == inputsCuda.shape, "The shape of net output is not as same as target."
            loss = lossCri(netOutput, inputsCuda)
            if torch.isnan(loss).tolist() is False:
                loss.backward()
                optimizer.step()
            if currentTrainingTimes % displayTimes == 0:
                with torch.no_grad():
                    with open("LossRecord.txt", mode="a") as wh:
                        wh.write(str(loss.item()) + "\n")
                    print("######################")
                    print("Epoch : {} , Training time : {}".format(e, currentTrainingTimes))
                    print("Loss is {}".format(loss.item()))
                    print("Learning rate is {}".format(optimizer.state_dict()['param_groups'][0]["lr"]))
                    print("Input  {}".format(list(inputs.numpy()[0, 0, 1, 0:5])))
                    print("Net Output  {}".format(list(torch.sigmoid(netOutput).detach().cpu().numpy()[0, 0, 1, 0:5])))
            currentTrainingTimes += 1
        torch.save(model.state_dict(), os.path.join(check_point_save_folder ,str(currentTrainingTimes) + "Times.pth"))
        scheduler.step()
