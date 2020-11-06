import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torchvision as tv
from WarmScheduler import GradualWarmupScheduler
import sklearn.metrics as metrics
from MetaLearningDataset import MetaLearningDataset
from torch.optim.swa_utils import AveragedModel, SWALR

if __name__ == "__main__":
    ### general config
    imagePath = "D:\\ImageEncoder\\CropResizeTrain"
    datasetTxtPath = "./3dpfShuffle.txt"
    checkPoint = "./CheckPoint/classic_CNN_SWA_5dpf_test_val2.pth"
    testModelLoad = "./CheckPoint/classic_CNN_SWA_Resnet_0.62_val1_3dpf.pth"
    testSamplesPartition = 1 ### 5 fold cross validations, 0 means the first part as testing samples, [0 ~ 4]
    totalSamplesNumber = 320
    device = "cuda:0"
    inputImageSize = [96, 416] ## height, width
    randomCropSize = [75, 350]
    ### net config
    batch_size = 8
    LR = 1.15e-5
    multiplier = 6
    epoch = 565 ### 3dpf 1200-->0.62, 5dpf 650-->0.44, 600-->0.49, 550 --> 0.51, 500-->0.46
    swa_start = epoch // 4 * 3
    warmEpoch = 30
    displayTimes = 10
    ###
    trainOrTest = "test"
    if_random_labels = False

    ### partition training samples and testing samples
    onePartSamples = totalSamplesNumber // 5
    minIndex = onePartSamples * testSamplesPartition
    maxIndex = minIndex + onePartSamples
    print("Testing data set Min index {}, Max index {}".format(minIndex, maxIndex))
    testSamples = []
    testLabels = []
    trainSamples = []
    trainLabels = []
    with open(datasetTxtPath, mode="r") as rh:
        for i,line in enumerate(rh):
            oneLine = line.strip("\n").split("\t")
            name = oneLine[0]
            label = oneLine[1]
            #pers = int(name.split(".jpg")[0][-1])
            if minIndex <= i <= maxIndex:
                testSamples.append(name)
                testLabels.append(label)
            else:
                trainSamples.append(name)
                trainLabels.append(label)
    uniqueLabels = np.unique(trainLabels)
    label2Index = {}
    index2Label = {}
    for i, item in enumerate(uniqueLabels):
        label2Index[item] = i
        index2Label[i] = item
    intTrainLabels = []
    for item in trainLabels:
        intTrainLabels.append(label2Index[item])
    intTestLabels = []
    for item in testLabels:
        intTestLabels.append(label2Index[item])
    print("Label to int number: {}".format(label2Index))
    print("Training samples {}".format(trainSamples))
    print("Training labels {}".format(intTrainLabels))
    print("There are {} number of samples in training data set".format(len(trainSamples)))
    print("Testing samples {}".format(testSamples))
    print("Testing labels {}".format(intTestLabels))
    print("There are {} number of samples in testing data set".format(len(testSamples)))

    ### transforms
    trainTransforms = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p = 0.5),
        tv.transforms.RandomVerticalFlip(p = 0.5),
        tv.transforms.RandomApply([tv.transforms.RandomCrop(size=randomCropSize)], p=0.5),
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=60)], p=0.5),
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.ToTensor(),
        #tv.transforms.RandomErasing(p=0.2, scale=(0.1, 0.15), ratio=(0.1, 1.))
    ])
    testTransforms = tv.transforms.Compose([
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.ToTensor()
    ])
    model = tv.models.resnet50(num_classes=3).to(device)
    swa_model = AveragedModel(model)

    if trainOrTest.lower() == "train":
        ### Optimizer
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,
                                           after_scheduler=cosine_scheduler)
        swa_scheduler = SWALR(optimizer, swa_lr=LR, anneal_epochs=15, anneal_strategy="cos")
        ### Loss loss
        lossCri = nn.CrossEntropyLoss(reduction="sum")
        model = model.train()
        ### Data set
        fishDataset = MetaLearningDataset(trainSamples, intTrainLabels, trainTransforms, imgPath=imagePath)
        fishDataLoader =  DataLoader(fishDataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        trainingTimes = 0
        for e in range(epoch):
            for i, (inputs, labels) in enumerate(fishDataLoader):
                optimizer.zero_grad()
                inputsCuda = inputs.to(device)
                labelsCuda = labels.long().to(device)
                predict = model(inputsCuda)
                loss = lossCri(predict, labelsCuda)
                loss.backward()
                optimizer.step()
                if trainingTimes % displayTimes == 0:
                    print("###########")
                    print("Epoch: {}, Training times is {}, Loss is {}, Learning rate is {}".format(
                        e,trainingTimes, loss.item(), optimizer.state_dict()['param_groups'][0]["lr"]))
                    with torch.no_grad():
                        _, predicted = predict.max(1)
                        total = labelsCuda.size(0)
                        correct = predicted.eq(labelsCuda).sum().item()
                        print("The Predict is {}, label is {}, correct ratio {},".format(predict[0:5], labels[0:5], correct / total + 0.))
                trainingTimes += 1
            if e > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        swa_model.to("cpu")
        torch.optim.swa_utils.update_bn(fishDataLoader, swa_model)
        torch.save(swa_model.state_dict(), checkPoint)
    else:
        print("Test samples are {}".format(testSamples))
        print("No shuffle labels {}".format(intTestLabels))
        newTestLabel = []
        if if_random_labels:
            for i in range(len(uniqueLabels)):
                newTestLabel += [i for _ in range(batch_size)]
            dis = len(intTestLabels) - len(newTestLabel)
            for i in range(dis):
                newTestLabel.append(np.random.randint(0,len(uniqueLabels)))
            np.random.shuffle(newTestLabel)
        else:
            newTestLabel = intTestLabels
        print("After shuffle labels {}".format(newTestLabel))
        swa_model.load_state_dict(torch.load(testModelLoad))
        swa_model = swa_model.eval()
        predictList = []
        truthList = []
        k = 0
        fishDataset = MetaLearningDataset(testSamples, newTestLabel, testTransforms, imgPath=imagePath)
        fishDataLoader = DataLoader(fishDataset,batch_size=1, shuffle=False)
        for testImag, testTarget in fishDataLoader:
            #print(testImag.shape)
            predictTensor = swa_model(testImag.to(device)).cpu().detach().numpy()
            position = np.argmax(np.squeeze(predictTensor))
            print("##############" + str(k))
            print(position)
            print(testTarget.item())
            predictList.append(position)
            truthList.append(testTarget.item())
            k += 1
        acc = metrics.accuracy_score(y_true=truthList,y_pred=predictList)
        classifiedInfor = metrics.classification_report(y_true=truthList,y_pred=predictList)
        macroF1 = metrics.f1_score(y_true=truthList,y_pred=predictList,average="macro")
        microF1 = metrics.f1_score(y_true=truthList,y_pred=predictList,average="micro")
        print("The accuracy is : ",acc)
        print("The classified result is : ")
        print(classifiedInfor)
        print("The macro F1 is : ",macroF1)
        print("The micro F1 is : ",microF1)




