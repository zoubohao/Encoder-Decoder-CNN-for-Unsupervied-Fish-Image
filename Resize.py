import os
import cv2

if __name__ == "__main__":
    imagesName = list(os.listdir("./Imaging_Experiment_Ori"))
    print(imagesName)
    for name in imagesName:
        img = cv2.resize(cv2.imread("./Imaging_Experiment_Ori/" + name), dsize=(512, 128))
        print(img.shape)
        print("./ImagesResize/" + name.split(".tif")[0] + ".jpg")
        cv2.imwrite("./ImagesResize/" + name.split(".tif")[0] + ".jpg", img)








