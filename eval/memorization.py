"""
Memorization Detection
Detects if generated images are too similar to training images
"""

import numpy as np


def detect_memorization(generated_features, training_features, threshold=0.5):
    """
    Calculate memorization ratio using cosine similarity
    
    For each generated image, find most similar training image.
    If distance < threshold, consider it memorized.
    
    Args:
        generated_features: (N, 2048) features from generated images
        training_features: (M, 2048) features from training images
        threshold: Distance threshold (default 0.5)
            Lower = stricter, Higher = more lenient
    
    Returns:
        memorization_ratio: float in [0, 1]
    """
    # Normalize features for cosine similarity
    gen_norm = generated_features / (np.linalg.norm(generated_features, axis=1, keepdims=True) + 1e-8)
    train_norm = training_features / (np.linalg.norm(training_features, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity matrix (N, M)
    cosine_sim = np.dot(gen_norm, train_norm.T)
    
    # Cosine distance = 1 - similarity
    cosine_dist = 1 - cosine_sim
    
    # Find minimum distance for each generated image
    min_distances = np.min(cosine_dist, axis=1)
    
    # Count memorized images (distance < threshold)
    memorized_count = np.sum(min_distances < threshold)
    memorization_ratio = memorized_count / len(generated_features)
    
    return memorization_ratio