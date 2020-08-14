from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from EmbeddingNet import EmbeddingNet
import os
from PIL import Image
import torchvision as tv
import torch
import numpy as np

enFC = 16
layersExpandRatio = 2.5
channelsExpandRatio = 2.5
blockExpandRatio = 7

def DecoderTensorWriting(model_weight_path, decoder_img_output_path, image_root_path):
    device = "cuda:1"
    model = EmbeddingNet(in_channels=3, enFC=enFC, drop_ratio=0.1,
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
        decoder = decoderTensor.detach().cpu().squeeze(dim=0)
        decoderImg = tv.transforms.ToPILImage()(decoder)
        decoderImg.save(os.path.join(decoder_img_output_path, nameD))


def EncoderTensorWriting(model_weight_path, image_root_path, write_path):
    device = "cuda:1"
    model = EmbeddingNet(in_channels=3, enFC=enFC, drop_ratio=0.1,
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


if __name__ == "__main__":
    #DecoderTensorWriting("CheckPoint/18300Times.pth", "DecoderImgsTest" ,"ImagesResizeTest")
    #EncoderTensorWriting("CheckPoint/18300Times.pth" ,"ImagesResizeTrain", "EncoderTensorsTrain")
    imagePath = "ImagesResizeTrain"
    encoderTensorPath = "EncoderTensorsTrain"

    num2color = {1:"red", 2:"green", 3:"red", 4:"green"}
    ### images to np array
    imgNames= list(os.listdir(imagePath))
    images = []
    imagesColor = []
    for imgName in imgNames:
        imgPIL = Image.open(os.path.join(imagePath, imgName)).convert("RGB")
        img = np.array(imgPIL).reshape([-1])
        images.append(img)
        imagesColor.append(num2color[int(imgName.split(".jpg")[0][-1])])
    oriImgArray = np.array(images) / 255.
    ### encoder to np array
    encoderNames = list(os.listdir(encoderTensorPath))
    encoders = []
    encodersColor = []
    for name in encoderNames:
        encoders.append(np.load(os.path.join(encoderTensorPath, name)))
        encodersColor.append(num2color[int(name.split(".jpg")[0][-1])])
    encoderArray = np.array(encoders)
    print(oriImgArray)
    print(encoderArray)
    print(imagesColor)
    print(encodersColor)
    tsneOriImg = TSNE(n_components=2, perplexity=120, init='pca',n_iter=10000, method="exact",
                      verbose=5, random_state=1024,n_iter_without_progress=5000)
    tsneEncoder = TSNE(n_components=2, perplexity=120, init='pca', n_iter=10000, method="exact",
                       verbose=5, random_state=1024,n_iter_without_progress=5000)
    oriImgTsneResult = tsneOriImg.fit_transform(oriImgArray)
    encoderTsneResult = tsneEncoder.fit_transform(encoderArray)
    ### Draw plot
    fig, (axO, axE) = plt.subplots(1,2)
    #fig.suptitle("T-SNE for Original Images And Encoders")
    oriX = oriImgTsneResult[:,0]
    oriY = oriImgTsneResult[:,1]
    axO.scatter(oriX, oriY,c = imagesColor)
    axO.set_title("T-SNE for Original Images")
    ### Draw encoder
    encoderX = encoderTsneResult[:,0]
    encoderY = encoderTsneResult[:,1]
    axE.scatter(encoderX, encoderY, c=encodersColor)
    axE.set_title("T-SNE for Encoders")
    plt.show()









