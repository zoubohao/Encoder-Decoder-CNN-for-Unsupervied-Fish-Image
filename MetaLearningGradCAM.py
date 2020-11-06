import torch
import torch.nn.functional as F
import numpy as np
from Model.EncoderNet import Encoder
import cv2
import torchvision as tv
from MetaLearningBatchSampler import PrototypicalBatchSampler
from MetaLearningDataset import MetaLearningDataset
from torch.utils.data import DataLoader
from PIL import Image
import os

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


def prototypical_loss(inputs: torch.Tensor, targets: torch.Tensor, n_support = 1):
    """
    :param inputs: N x D matrix tensor,
    the first index of inputs is query sample and the others are support samples,
    :param targets: N dimension vector tensor
    :param n_support:
    :return:
    """
    ### select support samples
    support_targets = targets[1:]
    support_inputs = inputs[1:]
    classes = torch.unique(support_targets)
    print("unique classes {}".format(classes))
    def supp_idxs(c):
        return torch.nonzero(support_targets.eq(c), as_tuple=False)[:n_support].squeeze(1)
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([support_inputs[idx_list].mean(0) for idx_list in support_idxs])
    ### select query samples
    query_sample = torch.unsqueeze(inputs[0], dim=0)
    ### compute dists [1, classes_number] [[0, 1, 2, .... classes_number]]
    dists = euclidean_dist(query_sample, prototypes)
    return -dists, targets[0]


class FeatureExtractor:

    def __init__(self, model, target_layers_name = ""):
        assert target_layers_name != "", "It must has name of one module."
        self.model = model
        self.target_layers_name = target_layers_name
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def __call__(self, x):
        featureMap = None
        self.gradient = None
        for name, module in self.model.named_children():
            x = module(x)
            if name == self.target_layers_name:
                ### this will get gradient of x if used backward() function
                print("Name of module {}, targets name {}".format(name, self.target_layers_name))
                x.register_hook(self.save_gradient)
                featureMap = x.clone()
        if featureMap is None:
            raise ValueError("feature map or gradient is None.")
        return featureMap, x

class MetaLearningGradCAM(object):

    def __init__(self, model, target_layer_name, state_dic):
        super().__init__()
        self.model = model
        self.model.load_state_dict(torch.load(state_dic))
        self.model.eval()
        self.extractor = FeatureExtractor(self.model, target_layer_name)

    def __call__(self, inputs, targets, classes_num):
        """
        :param inputs[N, 3, H, W]: The first index of inputs is query sample. It is the object to get CAM.
        :param targets:
        :param classes_num:
        :return:
        """
        ### classify loss construct
        featureMap, embedding = self.extractor(inputs)
        p_y, correct_index = prototypical_loss(embedding, targets)
        correct_index = correct_index.item()
        detach_py = p_y.detach()
        predict_index = np.argmax(np.squeeze(detach_py.numpy()), axis=-1)
        print("predict index:{}, correct index: {}".format(predict_index, correct_index))
        if predict_index != correct_index:
            raise ValueError("The predict index is not equal with correct index.")
        ### The gradients are set to zero for all classes except the desired class
        one_hot = np.zeros((1, classes_num), dtype=np.float32)
        one_hot[0][correct_index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        print("One hot array {}, p_y {}".format(one_hot,p_y))
        one_hot = torch.sum(one_hot * p_y)
        self.model.zero_grad()
        one_hot.backward()
        ### get gradient [N,C,H,W], The first is query sample
        ### if you want to see cam of another samples, you can change the index
        grads_val = self.extractor.get_gradient()[0].numpy()
        featureMap = featureMap.detach()[0].numpy()
        print("Feature map shape is {}, grads shape is {}, embedding shape is {}".format(featureMap.shape,
                                                                                         grads_val.shape,
                                                                                         embedding.shape))
        ### build cam
        #print(grads_val)
        #print(featureMap)
        weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(featureMap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * featureMap[i, :, :]
        ### It is fine if we do not use ReLU function. The experiment can be seen in GradCAM.py file.
        minV = np.min(cam)
        maxV = np.max(cam)
        cam = (cam - minV) / (maxV - minV)
        #print(cam)
        cam = cv2.resize(cam, (inputs.shape[3], inputs.shape[2]))  ### resize the cam to original img
        return cam


def show_cam_on_image(img, mask, save_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_name, np.uint8(255 * cam))

if __name__ == "__main__":
    query_sample = "2019.09.13_TH244b_plate2_tail_W_C03_1_1.jpg"
    query_label = "homo"
    cam_name = "./GradCam/wt_2.jpg"
    ### general config
    imagePath = "D:\\ImageEncoder\\CropResizeTrain"
    modelWeightPath = "D:\\ImageEncoder/CheckPoint/5dpf--perspective_13--validation_3.pth"
    datasetTxtPath = "./5dpfShuffle.txt"
    perspective = [1, 3] ### 1 - 4 perspectives
    testSamplesPartition = 3 ### 5 fold cross validations, 0 means the first part as testing samples, [0 ~ 4]
    totalSamplesNumber = 520
    ###
    encodedDimension = 128  ## 128
    layersExpandRatio = 2.5
    channelsExpandRatio = 1
    blockExpandRatio = 2

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
        if val < 1:
            for j in range(1 - val):
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
    ### Model
    testTransform = tv.transforms.ToTensor()
    model = Encoder(inChannels=3, encoderChannels=encodedDimension, drop_ratio=0,
                                layersExpandRatio=layersExpandRatio,
                                channelsExpandRatio=channelsExpandRatio,
                                blockExpandRatio=blockExpandRatio)
    ### Data set, batch size is 1 and 10 iterations.
    fishDataset = MetaLearningDataset(trainSamples, intTrainLabels, testTransform, imgPath=imagePath)
    sampler = PrototypicalBatchSampler(intTrainLabels, len(uniqueLabels), num_samples=1, iterations=1)
    fishTestDataLoader = DataLoader(fishDataset, batch_sampler=sampler)

    ### CAM
    metaLearningCAM = MetaLearningGradCAM(model, "b4", modelWeightPath)

    for e in range(1):
        test_iter = iter(fishTestDataLoader)
        for batch in test_iter:
            x, y = batch
            query_img = testTransform(Image.open(os.path.join(imagePath, query_sample)).convert("RGB"))
            query_intLabel = torch.from_numpy(np.array([label2Index[query_label]])).type_as(y)
            x = torch.cat([torch.unsqueeze(query_img, dim=0), x], dim=0)
            y = torch.cat((query_intLabel, y), dim=0)
            print("Inputs shape {}".format(x.shape))
            print("Labels {}".format(y))
            print("Label 2 index dict {}".format(label2Index))
            cam = metaLearningCAM(x, y, classes_num=len(uniqueLabels))
            print("Grad CAM shape {}".format(cam.shape))
            img = cv2.imread(os.path.join(imagePath, query_sample), 1)
            img = np.float32(img) / 255
            show_cam_on_image(img, cam, cam_name)









