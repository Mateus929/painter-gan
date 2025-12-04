from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import glob

class MonetDataset(Dataset):
    def __init__(self, monet_dir, photo_dir, transform=None):
        self.monet_files = glob.glob(f'{monet_dir}/*.jpg')
        self.photo_files = glob.glob(f'{photo_dir}/*.jpg')
        self.transform = transform
        
        self.length = max(len(self.monet_files), len(self.photo_files))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        monet_path = self.monet_files[idx % len(self.monet_files)]
        photo_path = self.photo_files[idx % len(self.photo_files)]
        
        monet_img = np.array(Image.open(monet_path).convert('RGB'))
        photo_img = np.array(Image.open(photo_path).convert('RGB'))
        
        if self.transform:
            monet_img = self.transform(image=monet_img)['image']
            photo_img = self.transform(image=photo_img)['image']
        
        return photo_img, monet_img

def get_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])