from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

img1Path = "./ImagesResizeTrain/2019.09.12_Plate4_TH311_head_W_B04_1_3.jpg"
img2Path = "./DecoderImgsTrain/2019.09.12_Plate4_TH311_head_W_B04_1_3.jpg"

img1 = np.array(Image.open(img1Path).convert("RGB"))
img2 = np.array(Image.open(img2Path).convert("RGB"))


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_title("Train Original Image")
ax2.set_title("Train Reproduced Image")
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()







