import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from tqdm import tqdm

# ---------------- Inception Feature Extractor ----------------
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.inception_v3(pretrained=True,transform_input=False)
        net.fc = nn.Identity()  # remove final classifier
        self.net = net.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)

# ---------------- Image preprocessing ----------------
def preprocess_for_inception(x):
    # x: (B,3,H,W) in [-1,1] or [0,1]
    if x.min() < 0:
        x = (x + 1) / 2
    x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None,:,None,None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None,:,None,None]
    return (x - mean) / std


# ---------------- Feature extraction ----------------
@torch.no_grad()
def extract_features(dataloader, model, device):
    feats = []
    for x in tqdm(dataloader, desc="Extracting features"):
        x = preprocess_for_inception(x.to(device))
        f = model(x)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------- FID computation ----------------
def compute_fid(real_feats, gen_feats):
    mu_r = real_feats.mean(axis=0)
    mu_g = gen_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)



# ---------------- Memorization distance ----------------
def compute_dthr(gen_feats, train_feats, epsilon=0.1):
    gen_feats = gen_feats / np.linalg.norm(gen_feats, axis=1, keepdims=True)
    train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    vals = []
    for g in gen_feats:
        d = 1.0 - np.dot(train_feats, g)
        d_min = d.min()
        vals.append(1.0 if d_min > epsilon else d_min)
    return float(np.mean(vals))

# ---------------- MiFID evaluation function ----------------
def evaluate_mifid(
    generator,
    photo_loader,        # input photos
    monet_val_loader,    # Monet validation set (FID)
    device,
    epsilon=0.5
):
    generator.eval()
    feat_net = InceptionFeatureExtractor().to(device)

    # ----- Generate Monet images -----
    gen_imgs = []
    with torch.no_grad():
        for x in tqdm(photo_loader, desc="Generating images"):
            x = x.to(device)
            y = generator(x)
            gen_imgs.append(y.cpu())
    gen_imgs = torch.cat(gen_imgs, dim=0)
    gen_loader = torch.utils.data.DataLoader(gen_imgs, batch_size=32, shuffle=False)

    # ----- Extract features -----
    gen_feats   = extract_features(gen_loader, feat_net, device)
    real_feats  = extract_features(monet_val_loader, feat_net, device)
    
    # ----- Compute metrics -----
    fid = compute_fid(real_feats, gen_feats)
    d_thr = compute_dthr(gen_feats, real_feats, epsilon)
    mifid = fid / d_thr

    return {
        "FID": float(fid),
        "d_thr": float(d_thr),
        "MiFID": float(mifid)
    }