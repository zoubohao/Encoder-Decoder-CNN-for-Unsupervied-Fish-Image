from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

img1Path = "./ImagesResizeTest/2019.09.21_TH324b_AlelleC_Plate2_tail_W_C07_1_1.jpg"
img2Path = "./DecoderImgsTest/2019.09.21_TH324b_AlelleC_Plate2_tail_W_C07_1_1.jpg"

img1 = np.array(Image.open(img1Path).convert("RGB"))
img2 = np.array(Image.open(img2Path).convert("RGB"))


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_title("Test Original Image")
ax2.set_title("Test Reproduced Image")
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

### LOSS
lossPath = "LossRecord.txt"

lossRecord = []
with open(lossPath, "r") as rh:
    for oneLine in rh:
        line = oneLine.strip("\n")
        lossRecord.append(float(line))
x = [i for i in range(len(lossRecord))]

fig, ax = plt.subplots()
ax.set_title("Loss Curve")
ax.plot(x, lossRecord)
plt.show()








