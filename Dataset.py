from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision as tv

class FishDataset(Dataset):

    def __init__(self, root, transforms = None):
        super(FishDataset, self).__init__()
        self.root = root
        self.imgs = list(os.listdir(root))
        self.transforms = transforms
        self.toTensor = tv.transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            imgTrans = self.transforms(img)
        else:
            imgTrans = img
        return imgTrans

    def __len__(self):
        return len(self.imgs)



if __name__ == "__main__":
    testTrans = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p = 0.5),
        tv.transforms.RandomVerticalFlip(p = 0.5),
        tv.transforms.RandomApply([tv.transforms.RandomCrop(size=[90, 400])],p = 0.5),
        tv.transforms.Resize(size=[128, 512]),
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=25)], p = 0.5),
        tv.transforms.ToTensor()
    ])
    testDataSet = FishDataset("./ImagesResizeTrain", testTrans)
    testTransImg= testDataSet.__getitem__(5)
    pilTestTransImg = tv.transforms.ToPILImage()(testTransImg)
    pilTestTransImg.save("testTransImg.jpg")
    print(testTransImg)
    print(testTransImg.shape)





