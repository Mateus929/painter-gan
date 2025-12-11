import torch
import os
from eval.mifid import mifid

def evaluate_model(generator, config, epoch, eval_images_dir, device):
    """
    Evaluate generator using MiFID metric
    
    Args:
        generator: Generator model
        config: Configuration dict
        epoch: Current epoch
        eval_images_dir: Directory to save generated images
        device: Device to run on
    
    Returns:
        dict: Evaluation results
    """
    generator.eval()
    
    # Create epoch-specific directory for generated images
    epoch_eval_dir = os.path.join(eval_images_dir, f'epoch_{epoch+1}')
    os.makedirs(epoch_eval_dir, exist_ok=True)
    
    with torch.no_grad():
        result = mifid(
            model=generator,
            real_images_dir=config.get('photo_dir', "/content/painter-gan/data/photo_jpg"),
            generated_images_dir=epoch_eval_dir,
            training_images_dir=config.get('monet_dir', "/content/painter-gan/data/monet_jpg"),
            device=device,
            batch_size=50,
            image_size=config['image_size'],
            generate=True,
            cosine_threshold=config.get('cosine_threshold', 0.5),
            penalty_weight=config.get('penalty_weight', 1000.0)
        )
    
    return result