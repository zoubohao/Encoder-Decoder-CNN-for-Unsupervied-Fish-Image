import torch
from collections import OrderedDict



if __name__ == "__main__":
    autoEncoderPath = "D:\\ImageEncoder\\CheckPoint\\102800Times.pth"
    encoderParameterPath = "D:\\ImageEncoder\\CheckPoint\\pre-train-Encoder.pth"
    modelStateDic = torch.load(autoEncoderPath)
    newDic = OrderedDict()
    for key, value in modelStateDic.items():
        print("#########")
        print(key)
        if "encoder" in key:
            newDic[key] = value
    torch.save(newDic,encoderParameterPath)














