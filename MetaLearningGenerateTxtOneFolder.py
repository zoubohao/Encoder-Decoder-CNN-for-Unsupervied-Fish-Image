import os

if __name__ == "__main__":
    folderPath = ".\\LabeledImgsOriAndDic\\Exp2\\2019.09.26_TH244b_5dpf"
    name2labelFilePath = "./labeledImgs2LabelWithoutNull.txt"
    outputFilePath = "./2019.09.26_TH244b_5dpf.txt"

    name2label = {}
    with open(name2labelFilePath, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            name = oneLine[0]
            label = oneLine[1]
            name2label[name] = label

    with open(outputFilePath, mode="w") as wh:
        for root, dic, files in os.walk(folderPath):
            for file in files:
                if ".tiff" in file:
                    fileName = file.split(".tiff")[0] + ".jpg"
                    if fileName in name2label:
                        wh.write(fileName + "\t" + name2label[fileName] + "\n")


























