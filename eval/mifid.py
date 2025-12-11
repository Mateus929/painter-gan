"""
MiFID (Memorization-informed Fréchet Inception Distance)
Main evaluation interface
"""

import os
from .utils import generate_images, extract_features
from .fid import calculate_fid
from .memorization import detect_memorization


def mifid(model,
          real_images_dir,
          generated_images_dir=None,
          training_images_dir=None,
          device='cuda',
          batch_size=50,
          image_size=256,
          generate=True,
          cosine_threshold=0.5,
          penalty_weight=1000.0,
          verbose = False):
    """
    Calculate MiFID score for a CycleGAN generator
    
    MiFID = FID + penalty_weight * memorization_ratio
    
    Args:
        model: Trained generator (nn.Module)
        real_images_dir: Real photos directory (for generating images)
        generated_images_dir: Pre-generated images directory (optional)
        training_images_dir: Training Monet images (for memorization check)
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing
        image_size: Model input size (default 256)
        generate: If True, generate images from model
        cosine_threshold: Memorization threshold (0.3-0.7)
        penalty_weight: Weight for memorization penalty
        verbose = default False
    
    Returns:
        dict: {
            'mifid': MiFID score,
            'fid': FID score,
            'memorization_penalty': Penalty value,
            'memorization_ratio': Ratio of memorized images
        }
    """
    if verbose:
        print("\n" + "="*70)
        print("MiFID EVALUATION")
        print("="*70)
    
    model.eval()
    model.to(device)
    
    # Step 1: Generate images if needed
    if generate or generated_images_dir is None:
        if generated_images_dir is None:
            generated_images_dir = '/content/tmp/mifid_generated'
        os.makedirs(generated_images_dir, exist_ok=True)
        if verbose:
            print("\n[1/3] Generating images...")
        num_gen = generate_images(
            model=model,
            input_dir=real_images_dir,
            output_dir=generated_images_dir,
            device=device,
            batch_size=8,
            image_size=image_size
        )
        if verbose:
            print(f"  ✓ Generated {num_gen} images")
    elif verbose:
        print(f"\n[1/3] Using pre-generated images from {generated_images_dir}")
    
    # Step 2: Calculate FID
    if verbose:
        print("\n[2/3] Calculating FID...")
    
    if verbose:
        print("  - Extracting features from real images...")
    real_features = extract_features(real_images_dir, device, batch_size)
    if verbose:
        print(f"    ✓ Extracted {len(real_features)} features")
    if verbose:
        print("  - Extracting features from generated images...")
    gen_features = extract_features(generated_images_dir, device, batch_size)
    if verbose:
        print(f"    ✓ Extracted {len(gen_features)} features")
    
    fid_score = calculate_fid(real_features, gen_features)
    if verbose:
        print(f"  ✓ FID Score: {fid_score:.4f}")
    
    # Step 3: Calculate memorization penalty
    memorization_penalty = 0.0
    memorization_ratio = 0.0
    
    if training_images_dir is not None:
        if verbose:
            print("\n[3/3] Calculating memorization penalty...")
            print("  - Extracting features from training images...")
        train_features = extract_features(training_images_dir, device, batch_size)
        if verbose:
            print(f"    ✓ Extracted {len(train_features)} features")
        
        memorization_ratio = detect_memorization(
            gen_features, train_features, cosine_threshold
        )
        memorization_penalty = memorization_ratio * penalty_weight
        
        if verbose:
            print(f"  ✓ Memorization Ratio: {memorization_ratio*100:.2f}%")
            print(f"  ✓ Memorization Penalty: {memorization_penalty:.4f}")
    elif verbose:
        print("\n[3/3] Skipping memorization (no training_images_dir)")
    
    # Calculate MiFID
    mifid_score = fid_score + memorization_penalty
    
    # Results
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"FID Score:              {fid_score:.4f}")
        print(f"Memorization Penalty:   {memorization_penalty:.4f}")
        print(f"MiFID Score:            {mifid_score:.4f}")
        print("="*70 + "\n")
    
    return {
        'mifid': mifid_score,
        'fid': fid_score,
        'memorization_penalty': memorization_penalty,
        'memorization_ratio': memorization_ratio
    }