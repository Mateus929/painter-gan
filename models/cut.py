import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class CUTGenerator(nn.Module):
    """Generator with multiple output points for PatchNCE loss"""
    def __init__(self, num_residual_blocks=9, num_downsampling=2):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        in_channels = 64
        for i in range(num_downsampling):
            out_channels = in_channels * 2
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i in range(num_downsampling):
            out_channels = in_channels // 2
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels
        
        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )
        
    def forward(self, x, encode_only=False):
        """
        Args:
            x: Input image
            encode_only: If True, return intermediate features for contrastive learning
        Returns:
            If encode_only=False: Generated image
            If encode_only=True: List of feature maps from multiple layers
        """
        features = []
        
        # Initial conv - layer 0
        out = self.initial(x)
        if encode_only:
            features.append(out)
        
        # Downsampling - layers 1, 2
        for i, down_block in enumerate(self.down_blocks):
            out = down_block(out)
            if encode_only:
                features.append(out)
        
        # Residual blocks - layers 3-11 (for 9 residual blocks)
        for i, res_block in enumerate(self.res_blocks):
            out = res_block(out)
            if encode_only:
                features.append(out)
        
        if encode_only:
            return features
        
        # Upsampling
        for up_block in self.up_blocks:
            out = up_block(out)
        
        # Output
        out = self.output(out)
        return out


class PatchSampleF(nn.Module):
    """MLP for patch-wise feature projection"""
    def __init__(self, in_channels, out_channels=256, use_mlp=True):
        super().__init__()
        self.use_mlp = use_mlp
        
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, out_channels)
            )
        else:
            self.mlp = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        x_reshape = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        x_proj = self.mlp(x_reshape)  # (B*H*W, out_channels)
        return x_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_channels, H, W)


class PatchNCELoss(nn.Module):
    """Patch-based contrastive loss"""
    def __init__(self, temperature=0.07, num_patches=256):
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: Query features (B, C, H, W) - from generated image
            feat_k: Key features (B, C, H, W) - from input image
        """
        B, C, H, W = feat_q.shape
        
        # Sample random patches
        num_patches = min(self.num_patches, H * W)
        
        # Reshape to (B, C, H*W) and permute to (B, H*W, C)
        feat_q = feat_q.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        feat_k = feat_k.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        
        # Sample patches
        if num_patches < H * W:
            patch_ids = torch.randperm(H * W, device=feat_q.device)[:num_patches]
            feat_q = feat_q[:, patch_ids, :]  # (B, num_patches, C)
            feat_k = feat_k[:, patch_ids, :]  # (B, num_patches, C)
        
        # Normalize features
        feat_q = F.normalize(feat_q, dim=2)
        feat_k = F.normalize(feat_k, dim=2)
        
        # Compute similarity matrix
        # For each query patch, compute similarity with all key patches
        loss = 0
        for i in range(B):
            q = feat_q[i]  # (num_patches, C)
            k = feat_k[i]  # (num_patches, C)
            
            # Compute cosine similarity
            logits = torch.mm(q, k.transpose(0, 1)) / self.temperature  # (num_patches, num_patches)
            
            # Diagonal elements are the positive pairs
            labels = torch.arange(num_patches, device=logits.device)
            
            loss += self.cross_entropy(logits, labels)
        
        return loss / B


class CUTDiscriminator(nn.Module):
    """PatchGAN discriminator (same as CycleGAN)"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)