from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageOnlyDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.file_paths[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img

def get_eval_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ToTensorV2()
    ])
