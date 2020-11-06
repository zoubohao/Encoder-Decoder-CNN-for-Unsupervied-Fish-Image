from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from Model.EncoderDecoderCNN import EncoderDecoderNet
import os
from PIL import Image
import torchvision as tv
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel


encodedDimension = 128
layersExpandRatio = 2.5
channelsExpandRatio = 1
blockExpandRatio = 2
if_add_plate_information = True

### Load plates information
platesPath = "./Name2Plate.txt"
img2Plates = {}
with open(platesPath, mode="r") as rh:
    for line in rh:
        oneLine = line.strip("\n")
        imgName, plateID = oneLine.split("\t")
        img2Plates[imgName] = int(plateID)
vocab_size = len(img2Plates.values())

def DecoderTensorWriting(model_weight_path, decoder_img_output_path, image_root_path, imageNames, if_swa = True):
    device = "cuda:1"
    model = EncoderDecoderNet(inChannels = 3, encodedDimension = encodedDimension,drop_ratio = 0,
                 layersExpandRatio= layersExpandRatio, channelsExpandRatio = channelsExpandRatio, blockExpandRatio = blockExpandRatio,
                 encoderImgHeight = 12, encoderImgWidth = 52, ch = 12, if_add_plate_infor = True).to(device)
    if if_swa:
        model = AveragedModel(model)
    model.load_state_dict(torch.load(model_weight_path))
    model = model.eval()
    transformer = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
    for i,nameD in enumerate(imageNames):
        imgD = Image.open(os.path.join(image_root_path, nameD)).convert("RGB")
        print("Decoder : ",i)
        print(nameD)
        tImg = transformer(imgD).unsqueeze(dim=0).to(device)
        if if_add_plate_information:
            decoderTensor, encoderT = model(tImg, torch.from_numpy(np.array([img2Plates[nameD]])).float().to(device))
        else:
            decoderTensor, encoderT = model(tImg, None)
        #print(encoderT)
        decoder = torch.sigmoid(decoderTensor).detach().cpu().squeeze(dim=0)
        decoderImg = tv.transforms.ToPILImage()(decoder)
        decoderImg.save(os.path.join(decoder_img_output_path, nameD))


def EncoderTensorWriting(model_weight_path, write_path,image_root_path, imageNames, if_swa = True):
    device = "cuda:1"
    model = EncoderDecoderNet(inChannels = 3, encodedDimension = encodedDimension,drop_ratio = 0,
                 layersExpandRatio= layersExpandRatio, channelsExpandRatio = channelsExpandRatio, blockExpandRatio = blockExpandRatio,
                 encoderImgHeight = 12, encoderImgWidth = 52, ch = 12, if_add_plate_infor = True).to(device)
    if if_swa:
        model = AveragedModel(model)
    model.load_state_dict(torch.load(model_weight_path))
    model = model.eval()
    transformer = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
    for i,nameE in enumerate(imageNames):
        imgE = Image.open(os.path.join(image_root_path, nameE)).convert("RGB")
        print("Encoder : ",i)
        print(nameE)
        tImg = transformer(imgE).unsqueeze(dim=0).to(device)
        if if_add_plate_information:
            _, encoderTensor = model(tImg, torch.from_numpy(np.array([img2Plates[nameE]])).float().to(device))
        else:
            _, encoderTensor = model(tImg, None)
        encoderTensor = encoderTensor.detach().cpu().numpy()
        encoderTensor = np.squeeze(encoderTensor,axis=0)
        np.save(os.path.join(write_path, nameE), encoderTensor)

def writeAsTxtFile(array, size, outputFile):
    sizeArray = array[0:size]
    n = sizeArray.shape[0]
    k = sizeArray.shape[1]
    with open(outputFile, mode="w") as wh:
        for i in range(n):
            print(i)
            for j in range(k):
                wh.write(str(sizeArray[i,j]) + "\t")
            wh.write("\n")

if __name__ == "__main__":
    labeledImgsWithoutNull = "2019.09.26_TH244b_5dpf.txt"
    labeledImgNames = []
    correspondLabels = []
    with open(labeledImgsWithoutNull, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            labeledImgNames.append(oneLine[0])
            correspondLabels.append(oneLine[1])
    uniqueLabels = np.unique(correspondLabels)
    print("There are {} number of unique labels.".format(len(uniqueLabels)))
    label2num = {}
    for i, label in enumerate(uniqueLabels):
        label2num[label] = i

    print("There are {} number of images which are labeled".format(len(labeledImgNames)))
    #DecoderTensorWriting("CheckPoint/102800Times.pth", "DecoderImgsTest" ,"LabeledCropResize", labeledImgNames, if_swa=False)
    #EncoderTensorWriting("CheckPoint/102800Times.pth" ,"EncoderTensorsTest", "LabeledCropResize", labeledImgNames, if_swa=False)
    imagePath = "LabeledCropResize"
    encoderTensorPath = "EncoderTensorsTest"
    if_perspective = True
    if_plate = False

    if if_perspective:
        ### images to np array
        perspective = [1,3]
        images = []
        imagesColor = []
        for i, imgName in enumerate(labeledImgNames):
            if int(imgName.split(".jpg")[0][-1]) in perspective:
                imgPIL = Image.open(os.path.join(imagePath, imgName)).convert("RGB")
                img = np.array(imgPIL).reshape([-1])
                images.append(img)
                imagesColor.append(label2num[correspondLabels[i]])
        oriImgArray = np.array(images) / 255.
        ### encoder to np array
        encoders = []
        encodersColor = []
        for i, name in enumerate(labeledImgNames):
            if int(name.split(".jpg")[0][-1]) in perspective:
                encoders.append(np.load(os.path.join(encoderTensorPath, name + ".npy")))
                encodersColor.append(label2num[correspondLabels[i]])

        encoderArray = np.array(encoders)
        print(oriImgArray.shape)
        print(encoderArray.shape)
        print(imagesColor)
        print(encodersColor)
        tsneOriImg = TSNE(n_components=2, perplexity=32, verbose=10, learning_rate=50,
                          n_iter=10000, n_iter_without_progress=10000, random_state=10)
        tsneEncoder = TSNE(n_components=2, perplexity=32, verbose=10, learning_rate=50,
                           n_iter=10000, n_iter_without_progress=10000, random_state=10)
        oriImgTsneResult = tsneOriImg.fit_transform(oriImgArray)
        encoderTsneResult = tsneEncoder.fit_transform(encoderArray)
        ### Draw plot
        fig, (axO, axE) = plt.subplots(1, 2)
        oriX = oriImgTsneResult[:, 0]
        oriY = oriImgTsneResult[:, 1]
        axO.scatter(oriX, oriY, c=imagesColor)
        axO.set_title("T-SNE for Original Images")
        ### Draw encoder
        encoderX = encoderTsneResult[:, 0]
        encoderY = encoderTsneResult[:, 1]
        axE.scatter(encoderX, encoderY, c=encodersColor)
        axE.set_title("T-SNE for all four perspective Encoders")
        plt.show()
    else:
        num2color = {1: "red", 2: "green", 3: "blue", 4: "yellow"}
        ### images to np array
        images = []
        imagesColor = []
        for i, imgName in enumerate(labeledImgNames):
            imgPIL = Image.open(os.path.join(imagePath, imgName)).convert("RGB")
            img = np.array(imgPIL).reshape([-1])
            images.append(img)
            if if_plate:
                imagesColor.append(img2Plates[imgName])
            else:
                imagesColor.append(num2color[int(imgName.split(".jpg")[0][-1])])
        oriImgArray = np.array(images) / 255.
        ### encoder to np array
        encoders = []
        encodersColor = []
        for i, name in enumerate(labeledImgNames):
            encoders.append(np.load(os.path.join(encoderTensorPath, name + ".npy")))
            if if_plate:
                encodersColor.append(img2Plates[name.strip(".npy")])
            else:
                encodersColor.append(num2color[int(name.split(".jpg")[0][-1])])

        encoderArray = np.array(encoders)
        print(oriImgArray.shape)
        print(encoderArray.shape)
        print(imagesColor)
        print(encodersColor)
        tsneOriImg = TSNE(n_components=2, perplexity=260., verbose=10, learning_rate=50,
                          n_iter=10000, n_iter_without_progress=10000, random_state=10)
        tsneEncoder = TSNE(n_components=2, perplexity=260., verbose=10, learning_rate=50,
                           n_iter=10000, n_iter_without_progress=10000, random_state=10)
        oriImgTsneResult = tsneOriImg.fit_transform(oriImgArray)
        encoderTsneResult = tsneEncoder.fit_transform(encoderArray)
        ### Draw plot
        fig, (axO, axE) = plt.subplots(1, 2)
        oriX = oriImgTsneResult[:, 0]
        oriY = oriImgTsneResult[:, 1]
        axO.scatter(oriX, oriY, c=imagesColor)
        axO.set_title("T-SNE for Original Images")
        ### Draw encoder
        encoderX = encoderTsneResult[:, 0]
        encoderY = encoderTsneResult[:, 1]
        axE.scatter(encoderX, encoderY, c=encodersColor)
        axE.set_title("T-SNE for Concat plates infor Encoders")
        plt.show()






