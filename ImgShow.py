from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

img1Path = "./LabeledCropResize/2019.09.11_plate1_head_W_C08_1_4.jpg"
img2Path = "./DecoderImgsTest/2019.09.11_plate1_head_W_C08_1_4.jpg"

img1 = np.array(Image.open(img1Path).convert("RGB"))
img2 = np.array(Image.open(img2Path).convert("RGB"))


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_title("Original Image")
ax2.set_title("Reproduced Image")
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




