import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MetaLearningDataset import MetaLearningDataset
import torchvision as tv
from torch import optim
from WarmScheduler import GradualWarmupScheduler
import numpy as np
from tqdm import tqdm
from MetaLearningBatchSampler  import PrototypicalBatchSampler
from torch.optim.swa_utils import AveragedModel, SWALR


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors

    :param x:  N x D
    :param y:  M x D
    :return: N x M
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return torch.nonzero(target_cpu.eq(c), as_tuple=False)[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: torch.nonzero(target_cpu.eq(c), as_tuple=False)[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).sum()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val


if __name__ == "__main__":
    ### general config
    imagePath = "D:\\ImageEncoder\\CropResizeTrain"
    datasetTxtPath = "./5dpfShuffle.txt"
    checkPoint = "./CheckPoint/5dpf--perspective_13--validation_1_swa.pth"
    perspective = [1, 3] ### 1 - 4 perspectives
    testSamplesPartition = 0 ### 5 fold cross validations, 0 means the first part as testing samples, [0 ~ 4]
    totalSamplesNumber = 504
    device = "cuda:0"
    ### net config
    batch_size = 4
    n_support = 1
    LR = 5e-6
    multiplier = 10
    epoch = 54
    swa_start = epoch // 3 * 2
    trainingTimesInOneEpoch = 600
    warmEpoch = 25
    encodedDimension = 128  ## 128
    displayTimes = 10
    ###
    trainOrTest = "test"
    if_random_labels = True
    inputImageSize = [96, 416] ## height, width
    randomCropSize = [80, 360]

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
            pers = int(name.split(".jpg")[0][-1])
            if pers in perspective:
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
    # prevent some labels of test samples smaller than batch size
    testIntLabel2Num = {}
    for item in uniqueLabels:
        testIntLabel2Num[label2Index[item]] = 0
    for item in intTestLabels:
        a = testIntLabel2Num[item]
        a += 1
        testIntLabel2Num[item] = a
    for key, val in testIntLabel2Num.items():
        print("Int label: {}, Number of samples of this label: {}".format(key, val))
        if val < batch_size:
            for j in range(batch_size - val):
                trainIndex = intTrainLabels.index(key)
                testSamples.append(trainSamples[trainIndex])
                intTestLabels.append(key)
                testIntLabel2Num[key] = val + 1
                trainSamples.remove(trainSamples[trainIndex])
                intTrainLabels.remove(key)
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
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.ToTensor(),
    ])
    testTransforms = tv.transforms.Compose([
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.ToTensor()
    ])

    ### Model
    model = tv.models.resnet50(num_classes=encodedDimension).to(device)
    swa_model = AveragedModel(model)
    if trainOrTest.lower() == "train":
        ### Data set
        fishDataset = MetaLearningDataset(trainSamples, intTrainLabels, trainTransforms, imgPath=imagePath)
        sampler = PrototypicalBatchSampler(intTrainLabels, len(uniqueLabels), num_samples=batch_size, iterations=trainingTimesInOneEpoch)
        fishDataLoader = DataLoader(fishDataset, batch_sampler=sampler)

        ### Optimizer
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,after_scheduler=cosine_scheduler)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=20, anneal_strategy="cos")
        ### training
        model = model.train()
        trainingTimes = 0
        for e in range(epoch):
            tr_iter = iter(fishDataLoader)
            for batch in tqdm(tr_iter):
                optimizer.zero_grad()
                x, y = batch
                inputsCuda = x.to(device)
                netOutput = model(inputsCuda)
                ### Loss
                trainLoss, acc = prototypical_loss(netOutput, target=y, n_support=n_support)
                trainLoss.backward()
                optimizer.step()
                if trainingTimes % displayTimes == 0:
                    print("Epoch {}, Training times is {}, Loss is {}, Learning rate is {}, accuracy is {}.".format(e,
                        trainingTimes, trainLoss.item(), optimizer.state_dict()['param_groups'][0]["lr"], acc))
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
        swa_model.load_state_dict(torch.load(checkPoint))
        swa_model = swa_model.eval()
        ### Data set
        fishTestDataset = MetaLearningDataset(testSamples, newTestLabel, testTransforms, imgPath=imagePath)
        sampler = PrototypicalBatchSampler(newTestLabel, len(uniqueLabels), num_samples=batch_size, iterations=100)
        fishTestDataLoader = DataLoader(fishTestDataset, batch_sampler=sampler)
        avg_acc = list()
        for e in range(10):
            test_iter = iter(fishTestDataLoader)
            k = 0
            for batch in test_iter:
                x, y = batch
                with torch.no_grad():
                    xCuda = x.to(device)
                    model_output = swa_model(xCuda).detach().cpu()
                    _, acc = prototypical_loss(model_output, target=y, n_support=n_support)
                    print("Epoch {}, K {}, Acc is {}".format(e,k, acc))
                    avg_acc.append(acc.detach().cpu().item())
                k += 1
        avg_acc = np.mean(avg_acc)
        print("Test Acc: {}".format(avg_acc))




