


if __name__ == "__main__":
    noLabelPlateInfor = "Name2Plate.txt"
    labeledPlateInfor = "Name2PlateLabeled.txt"

    noLabelPlatesList = []
    with open(noLabelPlateInfor, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n")
            noLabelPlatesList.append(int(oneLine.split("\t")[1]))
    maxValue = max(noLabelPlatesList) + 1
    with open(labeledPlateInfor, mode="r") as rh:
        with open(noLabelPlateInfor, mode="a") as ah:
            for line in rh:
                oneLine = line.strip("\n")
                name, plate = oneLine.split("\t")
                ah.write(name + "\t" + str(int(plate) + maxValue) + "\n")











