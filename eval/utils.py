"""
Utility Functions
Image generation, loading, and feature extraction helpers
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from .inception import InceptionV3Features


class ImageDataset(Dataset):
    """Simple dataset for loading images from folder"""
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # Get all image files
        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {folder_path}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def generate_images(model, input_dir, output_dir, device='cuda', 
                    batch_size=8, image_size=256):
    """
    Generate images using trained model
    
    Args:
        model: Generator model
        input_dir: Input photos directory
        output_dir: Where to save generated images
        device: 'cuda' or 'cpu'
        batch_size: Batch size
        image_size: Image size (model input)
    
    Returns:
        num_generated: Number of images generated
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Transform for model input (normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    image_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating images"):
            batch = batch.to(device)
            generated = model(batch)
            
            # Denormalize from [-1, 1] to [0, 1]
            generated = (generated + 1) / 2.0
            generated = torch.clamp(generated, 0, 1)
            
            # Save each image
            for img in generated:
                img_pil = transforms.ToPILImage()(img.cpu())
                img_pil.save(os.path.join(output_dir, f'{image_idx:05d}.jpg'))
                image_idx += 1
    
    return image_idx


def extract_features(images_dir, device='cuda', batch_size=50):
    """
    Extract InceptionV3 features from all images in directory
    
    Args:
        images_dir: Directory containing images
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing
    
    Returns:
        features: (N, 2048) numpy array
    """
    # Transform for InceptionV3 (299x299, [0, 1] range)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Initialize feature extractor
    feature_extractor = InceptionV3Features().to(device)
    
    # Extract features
    features_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            batch = batch.to(device)
            features = feature_extractor(batch)
            features_list.append(features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    return features