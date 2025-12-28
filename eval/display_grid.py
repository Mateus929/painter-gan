import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


def display_grid(model, input_dir, device='cuda'):
    """
    Generate and display 20 images in 4x5 grid
    
    Args:
        model: Trained Generator model
        input_dir: Directory with input photos
        device: 'cuda' or 'cpu'
    """
    
    model.eval()
    model.to(device)
    
    # Get first 20 image files
    image_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:20]
    
    print(f"Generating {len(image_files)} images...")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Generate images
    generated_images = []
    
    with torch.no_grad():
        for filename in image_files:
            # Load image
            img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Generate
            gen_tensor = model(img_tensor)
            
            # Denormalize [-1, 1] -> [0, 1]
            gen_tensor = (gen_tensor + 1) / 2.0
            gen_tensor = torch.clamp(gen_tensor, 0, 1)
            
            # Convert to PIL
            gen_img = transforms.ToPILImage()(gen_tensor.squeeze(0).cpu())
            generated_images.append(gen_img)
    
    # Display in 4x5 grid
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < len(generated_images):
            ax.imshow(generated_images[i])
        ax.axis('off')
    
    plt.suptitle('Generated Monet Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Displayed {len(generated_images)} images")