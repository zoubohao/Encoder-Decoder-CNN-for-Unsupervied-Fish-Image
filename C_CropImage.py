import  os
from PIL import Image
import numpy as np
import torchvision as tv


def judgeTube(colPixels: np.array, threshold = 30):
    """
    :param colPixels: [cols, 3]
    :param threshold: value to judge if is tube
    :return:
    """
    means = np.mean(colPixels, keepdims=False, axis=-1)
    tubeStart = 1000
    tubeEnd = 1000
    k = 1
    length = len(means)
    while k <= (length-1):
        currentWindow = means[k : k + 2]
        #print(np.mean(currentWindow))
        if (200 - np.mean(currentWindow)) >= threshold and tubeStart == 1000:
            tubeStart = k + 7
            k += 25
        elif (200 - np.mean(currentWindow)) >= threshold and tubeEnd == 1000 and (k - tubeStart) >= 40:
            tubeEnd = k + 10
            break
        k += 1
    return tubeStart, tubeEnd

def judgeFishLocation(img:np.array, threshold = 185):
    h, w, c = img.shape
    meImg = img[20: -20,:,:]
    start = 0
    for k in range(w):
        currentCol = meImg[:,k,:]
        currentColMean = np.mean(currentCol, keepdims=False, axis=-1)
        minMeanValue = np.min(currentColMean)
        #print(minMeanValue)
        if minMeanValue <= threshold:
            start = k
            break
    return start



if __name__ == "__main__":
    path = "./LabeledResizeImgs/"
    outputPath = "./LabeledCropImgs/"
    imgNames = os.listdir(path)
    for name in imgNames:
        print(name)
        imgPIL = Image.open(os.path.join(path, name)).convert("RGB")
        imgArray = np.array(imgPIL)
        h, w, c = imgArray.shape
        judgeTubeArray = imgArray[:, 0, :]
        emptyArea = np.where(np.mean(judgeTubeArray, keepdims=False, axis=-1) <= 190, 1, 0).sum()
        print("Empty rows {}".format(emptyArea))
        if emptyArea <= 30:
            start, end = judgeTube(imgArray[:, 0, :], threshold=30)
            print("tube start is {}".format(start))
            print("tube end is {}".format(end))
            ### select medium part
            medium_part = imgArray[start: end, :, :]
            headStart = judgeFishLocation(medium_part, threshold=150)
            print("head start is {}".format(headStart))
            newImg = Image.fromarray(imgArray[start - 5:end-5, headStart: headStart + 350, :]).save(os.path.join(outputPath, name))
        else:
            start, end = judgeTube(imgArray[:, -1, :], threshold=30)
            newImg = Image.fromarray(imgArray[start - 5:end-5, 0: 350, :]).save(os.path.join(outputPath, name))


    # imgPIL = Image.open(os.path.join(path)).convert("RGB")
    # imgArray = np.array(imgPIL)
    # h, w, c = imgArray.shape
    # judgeTubeArray = imgArray[:, 0, :]
    # emptyArea = np.where(np.mean(judgeTubeArray, keepdims=False, axis=-1) <= 190, 1, 0).sum()
    # print(emptyArea)
    # if emptyArea <= 30:
    #     start, end = judgeTube(imgArray[:, 0, :], threshold=30)
    #     print("tube start is {}".format(start))
    #     print("tube end is {}".format(end))
    #     ### select medium part
    #     medium_part = imgArray[start: end, :, :]
    #     headStart = judgeFishLocation(medium_part, threshold=150)
    #     print("head start is {}".format(headStart))
    #     print("head")
    #     newImg = Image.fromarray(imgArray[:, headStart: headStart + 320, :]).save(os.path.join(outputPath))
    # else:
    #     start, end = judgeTube(imgArray[:, -1, :], threshold=30)
    #     print("tube start is {}".format(start))
    #     print("tube end is {}".format(end))
    #     ### select medium part
    #     medium_part = imgArray[start: end, :, :]
    #     headStart = judgeFishLocation(medium_part, threshold=150)
    #     print("head start is {}".format(headStart))
    #     newImg = Image.fromarray(imgArray[:, headStart: headStart + 200, :]).save(os.path.join(outputPath))


