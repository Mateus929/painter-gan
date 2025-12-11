"""
FID (Fréchet Inception Distance) Calculation
Measures distance between real and generated image distributions
"""

import numpy as np
from scipy import linalg


def calculate_fid(real_features, generated_features):
    """
    Calculate FID score from features
    
    FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    
    Args:
        real_features: (N, 2048) numpy array
        generated_features: (M, 2048) numpy array
    
    Returns:
        fid_score: float (lower is better)
    """
    # Calculate statistics
    mu1, sigma1 = calculate_activation_statistics(real_features)
    mu2, sigma2 = calculate_activation_statistics(generated_features)
    
    # Calculate Fréchet distance
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid_score


def calculate_activation_statistics(features):
    """
    Calculate mean and covariance
    
    Args:
        features: (N, D) numpy array
    
    Returns:
        mu: Mean vector (D,)
        sigma: Covariance matrix (D, D)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Compute Fréchet distance between two Gaussians
    
    Args:
        mu1, mu2: Mean vectors
        sigma1, sigma2: Covariance matrices
    
    Returns:
        distance: float
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    # Squared difference of means
    diff = mu1 - mu2
    
    # sqrt(sigma1 * sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    # FID formula
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid