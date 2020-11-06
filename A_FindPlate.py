import os
import re
from shutil import copyfile

if __name__ == "__main__":
    path = "./LabeledImgsOriAndDic"
    outputPath = "./Name2PlateLabeled.txt"
    copyPath = "./LabeledImgsOri"

    ### tiff images
    regex = re.compile(r".*\.tiff$")
    plates = {}
    fileName2PlateID = {}
    k = 0
    for root, dirt, files in os.walk(path):
        for file in files:
            matched = regex.match(file)
            if matched is not None:
                subDateFileName = re.sub(r"\d{4}\.*(\d{2}|\w.*?)\.*\d{2}","", file)
                if "head" in subDateFileName:
                    plateID = subDateFileName.split("head")[0].strip("_").lower()
                else:
                    plateID = subDateFileName.split("tail")[0].strip("_").lower()
                if plateID not in plates:
                    plates[plateID] = k
                    fileName2PlateID[file.split(".tiff")[0] +".jpg"] = k
                    k += 1
                else:
                    fileName2PlateID[file.split(".tiff")[0] +".jpg"] = plates[plateID]
                print("##################")
                print(root)
                print(file)
                print(subDateFileName)
                print(plateID)
                copyfile(os.path.join(root,file), os.path.join(copyPath, file.split(".tiff")[0] + ".jpg"))
    print(plates)
    with open(outputPath, mode="w") as wh:
        for key, val in fileName2PlateID.items():
            wh.write(key + "\t" + str(val) + "\n")







