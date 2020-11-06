import os
import cv2

if __name__ == "__main__":
    ### resize labeled img to fixed size.
    # inputDir = "LabeledCropImgs"
    # outputDir = "LabeledCropResize"
    # imgNames = sorted(list(os.listdir(inputDir)))
    # for name in imgNames:
    #     img = cv2.resize(cv2.imread(os.path.join(inputDir, name)), dsize=(416, 96))
    #     cv2.imwrite(os.path.join(outputDir,name),img)
    imgsNameTxt = "Name2PlateLabeled.txt"
    exp1LabeledTxt = "D:\\ImageEncoder\\LabeledImgsOriAndDic\\Exp1\\Exp1GenoTypes.txt"
    exp2LabeledTxt = "D:\\ImageEncoder\\LabeledImgsOriAndDic\\Exp2\\Exp2GenoTypes.txt"
    imgs2Label = "labeledImgs2LabelWithoutNull.txt"
    ####
    exp1Plate1Name2Label = {}
    exp1Plate2Name2Label = {}
    exp2Name2Label = {}
    with open(exp1LabeledTxt,mode="r") as exp1RH, open(exp2LabeledTxt, mode="r") as exp2RH:
        k = 0
        for line in exp1RH:
            if k >= 1:
                oneLine = line.strip("\n").split("\t")
                name = oneLine[1]
                geneType = oneLine[2]
                if len(name) == 2:
                    name = name[0] + "0" + name[1]
                if oneLine[0] == "1":
                    exp1Plate1Name2Label[name] = geneType
                else:
                    exp1Plate2Name2Label[name] = geneType
            k += 1
        k = 0
        for line in exp2RH:
            if k >= 1:
                oneLine = line.strip("\n").split("\t")
                name = oneLine[0]
                geneType = oneLine[1]
                if len(name) == 2:
                    name = name[0] + "0" + name[1]
                exp2Name2Label[name] = geneType
            k += 1
    print(exp1Plate1Name2Label)
    print(exp1Plate2Name2Label)
    print(exp2Name2Label)
    with open(imgsNameTxt, mode="r") as rh, open(imgs2Label, mode="w") as wh:
        for oneLine in rh:
            line = oneLine.strip("\n").split("\t")
            name = line[0].split("_")
            idInfor = name[-3]
            if "plate1" in name:
                try:
                    label = exp1Plate1Name2Label[idInfor].lower()
                    wh.write(line[0] + "\t" + label + "\n")
                except KeyError:
                    pass
            elif "plate2" in name:
                try:
                    label = exp1Plate2Name2Label[idInfor].lower()
                    wh.write(line[0] + "\t" + label + "\n")
                except KeyError:
                    pass
            else:
                try:
                    label = exp2Name2Label[idInfor].lower()
                    wh.write(line[0] + "\t" + label + "\n")
                except KeyError:
                    pass

















