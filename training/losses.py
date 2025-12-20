import torch
import torch.nn.functional as F

# --- Hyperparameters (Standard defaults from the CycleGAN paper) ---
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 5.0


def adversarial_loss(predictions, target_is_real):
    """
    Implements the Least Squares GAN (LSGAN) loss.
    
    The target is 1.0 for real images (or generated images trying to be real)
    and 0.0 for fake images (or real images trying to be identified as fake).
    
    Args:
        predictions (torch.Tensor): The output scores from the Discriminator.
        target_is_real (bool): Flag indicating if the target label should be 1.0 (real) or 0.0 (fake).
        
    Returns:
        torch.Tensor: The mean squared error loss.
    """
    if target_is_real:
        target = torch.ones_like(predictions, device=predictions.device)
    else:
        target = torch.zeros_like(predictions, device=predictions.device)
        
    return F.mse_loss(predictions, target)

def cycle_consistency_loss(real_images, reconstructed_images, lambda_cycle=LAMBDA_CYCLE):
    """
    Cycle consistency loss: ||F(G(X)) - X|| + ||G(F(Y)) - Y||
    
    Penalizes the difference between the original image and the image
    reconstructed after a full forward and backward translation cycle.
    L1 loss (Mean Absolute Error) is preferred for image tasks as it encourages
    less blurring than L2 loss.
    
    Args:
        real_images (torch.Tensor): The original input images (e.g., X).
        reconstructed_images (torch.Tensor): The images after the full cycle (e.g., F(G(X))).
        lambda_cycle (float): Weight for the cycle loss.
        
    Returns:
        torch.Tensor: The weighted L1 loss.
    """
    return lambda_cycle * F.l1_loss(reconstructed_images, real_images)

def identity_loss(real_images, same_images, lambda_identity=LAMBDA_IDENTITY):
    """
    Identity loss: ||G(Y) - Y|| or ||F(X) - X||
    
    Ensures that when a generator is fed an image already from its target domain,
    it acts as an identity function (i.e., it preserves the color, composition, and identity).
    
    Args:
        real_images (torch.Tensor): The original image from the target domain (e.g., Y).
        same_images (torch.Tensor): The output of the generator when given the target image (e.g., G(Y)).
        lambda_identity (float): Weight for the identity loss.
        
    Returns:
        torch.Tensor: The weighted L1 loss.
    """
    return lambda_identity * F.l1_loss(same_images, real_images)

def compute_contrastive_loss(feat_q_list, feat_k_list, netF_list, nce_loss, lambda_NCE=1.0):
    """
    Compute contrastive loss across multiple layers
    
    Args:
        feat_q_list: List of query features from generated image (multiple layers)
        feat_k_list: List of key features from input image (multiple layers)
        netF_list: List of PatchSampleF networks for each layer
        nce_loss: PatchNCELoss module
        lambda_NCE: Weight for NCE loss
    """
    total_nce_loss = 0.0
    num_layers = len(feat_q_list)
    
    for feat_q, feat_k, netF in zip(feat_q_list, feat_k_list, netF_list):
        feat_q_proj = netF(feat_q)
        feat_k_proj = netF(feat_k)
        
        loss = nce_loss(feat_q_proj, feat_k_proj)
        total_nce_loss += loss
    
    return (total_nce_loss / num_layers) * lambda_NCE