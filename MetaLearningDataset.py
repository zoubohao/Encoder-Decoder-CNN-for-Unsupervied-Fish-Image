from PIL import Image
from torch.utils.data import Dataset
import os


class MetaLearningDataset(Dataset):

    def __init__(self, dataset:list, datasetLabels:list, transforms, imgPath:str):
        self.root = imgPath
        self.datas = dataset
        self.transforms = transforms
        self.labels = datasetLabels

    def __getitem__(self, item):
        imgName = self.datas[item]
        imgLabel = self.labels[item]
        img_path = os.path.join(self.root, imgName)
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            imgTrans = self.transforms(img)
        else:
            imgTrans = img
        return imgTrans, imgLabel

    def __len__(self):
        return len(self.datas)








