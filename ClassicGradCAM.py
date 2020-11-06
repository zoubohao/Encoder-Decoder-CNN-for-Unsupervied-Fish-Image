import torch
import numpy as np
import cv2
import torchvision as tv
from torch.optim.swa_utils import AveragedModel
from PIL import Image
import os


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
        for name, module in self.model.module.named_children():
            if name == self.target_layers_name:
                x = module(x)
                ### this will get gradient of x if used backward() function
                print("Name of module {}, targets name {}".format(name, self.target_layers_name))
                x.register_hook(self.save_gradient)
                featureMap = x.clone()
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        if featureMap is None:
            raise ValueError("feature map is None.")
        return featureMap, x

class GradCAM(object):

    def __init__(self, model, target_layer_name):
        super().__init__()
        self.model = model
        self.model.eval()
        self.extractor = FeatureExtractor(self.model, target_layer_name)

    def __call__(self, inputs, correct_index):
        """
        :param inputs[1, 3, H, W].
        :param correct_index: target index
        :return:
        """
        self.model.zero_grad()
        print("Inputs shape is {}".format(inputs.shape))
        featureMap, predict = self.extractor(inputs)
        index = np.argmax(predict.squeeze().cpu().data.numpy())
        print("predict index is {}, target index is {}".format(index, target_index))
        if index != correct_index:
            raise ValueError("Predict index is not as same as target index.")
        ### The gradients are set to zero for all classes except the desired class
        one_hot = np.zeros((1, predict.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * predict)
        self.model.zero_grad()
        one_hot.backward()
        print("Feature map shape is {}, grads shape is {}, embedding shape is {}".format(featureMap.shape,
                                                                                         self.extractor.get_gradient().shape,
                                                                                         predict.shape))
        grads_val = self.extractor.get_gradient()[0].numpy()
        featureMap = featureMap.detach()[0].numpy()
        print("Feature map shape is {}, grads shape is {}, embedding shape is {}".format(featureMap.shape,
                                                                                         grads_val.shape,
                                                                                         predict.shape))
        ### build cam
        weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(featureMap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * featureMap[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (inputs.shape[3], inputs.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
        ### It is fine if we do not use ReLU function. The experiment can be seen in GradCAM.py file.
        # minV = np.min(cam)
        # maxV = np.max(cam)
        # cam = (cam - minV) / (maxV - minV)
        # cam = cv2.resize(cam, (inputs.shape[3], inputs.shape[2]))  ### resize the cam to original img
        # return cam


def show_cam_on_image(img, mask, save_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_name, np.uint8(255 * cam))


if __name__ == '__main__':

    testModelLoad = "./CheckPoint/classic_CNN_SWA_Resnet_0.62_val1_3dpf.pth"
    img_path = "./CropResizeTrain/"
    inputImageSize = [96, 416]  ## height, width
    totalSamplesNumber = 320
    testSamplesPartition = 1
    datasetTxtPath = "./3dpfShuffle.txt"
    cam_output_path = "./GradCam/"

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
    print(label2Index)
    model = tv.models.resnet50(num_classes=3)
    swa_model = AveragedModel(model)
    swa_model.load_state_dict(torch.load(testModelLoad))
    swa_model = swa_model.eval()
    # for name, module in swa_model.module.named_children():
    #     print(name)
    #     print(module)
    #     module.zero_grad()
    grad_cam = GradCAM(model=swa_model,target_layer_name="layer3")
    testTransforms = tv.transforms.Compose([
        tv.transforms.Resize(size=inputImageSize),
        tv.transforms.ToTensor()
    ])
    for i,(testImgName, testLabel) in enumerate(zip(testSamples,testLabels)):
        imageTensor = torch.unsqueeze(testTransforms(Image.open(os.path.join(img_path, testImgName)).convert("RGB")), dim=0)
        target_index = label2Index[testLabel]
        try:
            for name, module in swa_model.module.named_children():
                module.zero_grad()
            mask = grad_cam(imageTensor, target_index)
            img = cv2.imread(os.path.join(img_path, testImgName), 1)
            img = np.float32(img) / 255.
            show_cam_on_image(img, mask, os.path.join(cam_output_path + "3dpf_" + testLabel +  str(i) + ".jpg"))
        except ValueError:
            pass