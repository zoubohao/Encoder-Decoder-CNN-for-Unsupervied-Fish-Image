import os
import cv2

if __name__ == "__main__":
    inputDir = "./Testing"
    outputDir = "./ImagesResizeTest"
    imagesName = list(os.listdir(inputDir))
    for name in imagesName:
        img = cv2.resize(cv2.imread(os.path.join(inputDir, name)), dsize=(512, 128))
        print(img.shape)
        print(outputDir + name.split(".tif")[0] + ".jpg")
        cv2.imwrite(os.path.join(outputDir , name.split(".tif")[0] + ".jpg"), img)








