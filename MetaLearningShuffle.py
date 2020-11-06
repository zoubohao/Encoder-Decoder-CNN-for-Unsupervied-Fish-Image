
import numpy as np

if __name__ == "__main__":
    filePath = "5dpf.txt"
    outputPath = "5dpfShuffle.txt"
    nameList = []
    name2Label = {}
    with open(filePath, "r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            name, label = oneLine
            nameList.append(name)
            name2Label[name] = label
    np.random.shuffle(nameList)
    np.random.shuffle(nameList)
    with open(outputPath, "w") as wh:
        for name in nameList:
            wh.write(name + "\t" + name2Label[name] + "\n")







