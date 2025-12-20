from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MonetDataset(Dataset):
    def __init__(self, monet_files, photo_files, transform=None, random_pairing=True):
        self.monet_files = monet_files
        self.photo_files = photo_files
        self.transform = transform
        self.random_pairing = random_pairing
        self.length = max(len(self.monet_files), len(self.photo_files))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.random_pairing:
            monet_path = np.random.choice(self.monet_files)
            photo_path = np.random.choice(self.photo_files)
        else:
            monet_path = self.monet_files[idx % len(self.monet_files)]
            photo_path = self.photo_files[idx % len(self.photo_files)]

        monet_img = np.array(Image.open(monet_path).convert("RGB"))
        photo_img = np.array(Image.open(photo_path).convert("RGB"))

        if self.transform:
            monet_img = self.transform(image=monet_img)["image"]
            photo_img = self.transform(image=photo_img)["image"]

        return photo_img, monet_img

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2()
    ])
