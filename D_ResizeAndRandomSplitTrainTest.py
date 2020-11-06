import os
import cv2
import numpy as np

if __name__ == "__main__":
    inputDir = "./LabeledCropImgs"
    outputDir1 = "./CropResizeTrain"
    outputDir2 = "./CropResizeTest"
    imagesName = list(os.listdir(inputDir))
    for name in imagesName:
        img = cv2.resize(cv2.imread(os.path.join(inputDir, name)), dsize=(416, 96))
        print(img.shape)
        if np.random.rand(1) <= 0.2:
            cv2.imwrite(os.path.join(outputDir1, name), img)
        else:
            cv2.imwrite(os.path.join(outputDir2, name), img)