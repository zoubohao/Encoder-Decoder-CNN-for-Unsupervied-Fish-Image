from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from Model.EncoderDecoderCNN import EncoderDecoderNet
import os
from PIL import Image
import torchvision as tv
import torch
import numpy as np


encoderChannels = 200
layersExpandRatio = 2
channelsExpandRatio = 1.25
blockExpandRatio = 2

def DecoderTensorWriting(model_weight_path, decoder_img_output_path, image_root_path):
    device = "cuda:0"
    model = EncoderDecoderNet(inChannels = 3, encoderChannels = encoderChannels,drop_ratio =0.0,
                              layersExpandRatio= layersExpandRatio,
                              channelsExpandRatio = channelsExpandRatio,
                              blockExpandRatio = blockExpandRatio).to(device)
    model.load_state_dict(torch.load(model_weight_path))
    model = model.eval()
    imageNames = list(os.listdir(image_root_path))
    transformer = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
    for i,nameD in enumerate(imageNames):
        imgD = Image.open(os.path.join(image_root_path, nameD)).convert("RGB")
        print(i)
        print(nameD)
        tImg = transformer(imgD).unsqueeze(dim=0).to(device)
        decoderTensor, _ = model(tImg)
        decoder = torch.sigmoid(decoderTensor).detach().cpu().squeeze(dim=0)
        decoderImg = tv.transforms.ToPILImage()(decoder)
        decoderImg.save(os.path.join(decoder_img_output_path, nameD))


def EncoderTensorWriting(model_weight_path, image_root_path, write_path):
    device = "cuda:0"
    model = EncoderDecoderNet(inChannels = 3, encoderChannels = encoderChannels,drop_ratio =0.0,
                              layersExpandRatio= layersExpandRatio,
                              channelsExpandRatio = channelsExpandRatio,
                              blockExpandRatio = blockExpandRatio).to(device)
    model.load_state_dict(torch.load(model_weight_path))
    model = model.eval()
    imageNames = list(os.listdir(image_root_path))
    transformer = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
    for i,nameE in enumerate(imageNames):
        imgE = Image.open(os.path.join(image_root_path, nameE)).convert("RGB")
        print(i)
        print(nameE)
        tImg = transformer(imgE).unsqueeze(dim=0).to(device)
        _,  encoderTensor = model(tImg)
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
    #DecoderTensorWriting("CheckPoint/21400Times.pth", "DecoderImgsTest" ,"ImagesResizeTest")
    #EncoderTensorWriting("CheckPoint/21400Times.pth" ,"ImagesResizeTest", "EncoderTensorsTest")
    imagePath = "ImagesResizeTest"
    encoderTensorPath = "EncoderTensorsTest"

    num2color = {1:"red", 2:"yellow", 3:"blue", 4:"green"}
    ### images to np array
    imgNames= sorted(list(os.listdir(imagePath)))
    images = []
    imagesColor = []
    for imgName in imgNames:
        imgPIL = Image.open(os.path.join(imagePath, imgName)).convert("RGB")
        img = np.array(imgPIL).reshape([-1])
        images.append(img)
        imagesColor.append(num2color[int(imgName.split(".jpg")[0][-1])])
    oriImgArray = np.array(images) / 255.
    ### encoder to np array
    encoderNames = sorted(list(os.listdir(encoderTensorPath)))
    encoders = []
    encodersColor = []
    for name in encoderNames:
        encoders.append(np.load(os.path.join(encoderTensorPath, name)))
        encodersColor.append(num2color[int(name.split(".jpg")[0][-1])])
    encoderArray = np.array(encoders)
    print(oriImgArray.shape)
    print(encoderArray.shape)
    writeAsTxtFile(oriImgArray, 10, "oriImgArray.txt")
    writeAsTxtFile(encoderArray, encoderArray.shape[0], "encoderArray.txt")
    # np.save("OriImagesTSNEInputsFile.npy", oriImgArray)
    # np.save("EncodedVectorsTSNEInputsFile.npy", encoderArray)
    print(imagesColor)
    print(encodersColor)
    samplesNum = -1
    tsneOriImg = TSNE(n_components=2, perplexity=900., verbose=10,random_state=1024, learning_rate=100,
                      n_iter=10000, n_iter_without_progress=10000)
    tsneEncoder = TSNE(n_components=2, perplexity=900., verbose=10,random_state=1024, learning_rate=100,
                       n_iter=10000, n_iter_without_progress=10000)
    oriImgTsneResult = tsneOriImg.fit_transform(oriImgArray[0: samplesNum])
    encoderTsneResult = tsneEncoder.fit_transform(encoderArray[0: samplesNum])
    # oriImgTsneResult = tsne(oriImgArray).cpu().numpy()
    # encoderTsneResult = tsne(encoderArray).cpu().numpy()
    ### Draw plot
    fig, (axO, axE) = plt.subplots(1,2)
    #fig.suptitle("T-SNE for Original Images And Encoders")
    oriX = oriImgTsneResult[:,0]
    oriY = oriImgTsneResult[:,1]
    axO.scatter(oriX, oriY,c = imagesColor[0:samplesNum])
    axO.set_title("T-SNE for Original Images")
    ### Draw encoder
    encoderX = encoderTsneResult[:,0]
    encoderY = encoderTsneResult[:,1]
    axE.scatter(encoderX, encoderY, c=encodersColor[0:samplesNum])
    axE.set_title("T-SNE for Encoders")
    plt.show()









