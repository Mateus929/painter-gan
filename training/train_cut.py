import wandb
import torch
import uuid
from torch.utils.data import DataLoader
from models.cut import CUTGenerator, CUTDiscriminator, PatchSampleF, PatchNCELoss
from training.checkpoint_manager import CheckpointManager
from training.losses import adversarial_loss, compute_contrastive_loss, identity_loss
from training.random_monet_dataset import MonetDataset, get_train_transforms, get_val_transforms
from training.image_folder_dataset import ImageOnlyDataset, get_eval_transforms
from eval.main_eval import evaluate_mifid
import glob
from sklearn.model_selection import train_test_split


def split_domains(monet_dir, photo_dir, val_ratio=0.3, seed=42):
    monet_files = glob.glob(f"{monet_dir}/*.jpg")
    photo_files = glob.glob(f"{photo_dir}/*.jpg")

    monet_train, monet_val = train_test_split(
        monet_files, test_size=val_ratio, random_state=seed
    )
    photo_train, photo_val = train_test_split(
        photo_files, test_size=val_ratio, random_state=seed
    )

    return monet_train, monet_val, photo_train, photo_val


def train_cut(config):
    verbose = config.get("verbose", 1)

    def vprint(msg, level=1):
        if verbose >= level:
            print(msg)

    # ------------------ W&B ------------------
    run_id = config.get("run_id", str(uuid.uuid4().hex[:8]))
    wandb.init(
        project="monet-cut",
        name=config["run_name"],
        config=config,
        id=run_id,
        resume="allow"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vprint(f"Using device: {device}")

    # ------------------ Models ------------------
    # Generator (only one direction: photo -> monet)
    G = CUTGenerator(
        num_residual_blocks=config["num_residual_blocks"],
        num_downsampling=config.get("num_downsampling", 2)
    ).to(device)
    
    # Discriminator for generated monet images
    D = CUTDiscriminator().to(device)
    
    # Determine feature channels based on architecture
    # For standard config: initial=64, after 2 downsamples=256, residual blocks=256
    num_downsampling = config.get("num_downsampling", 2)
    initial_channels = 64
    channels_after_downsample = initial_channels * (2 ** num_downsampling)
    
    # Map layer indices to channels
    # Layer 0: initial conv (64 channels)
    # Layers 1 to num_downsampling: downsampling layers
    # Remaining layers: residual blocks (all same channel count)
    def get_channel_for_layer(layer_idx):
        if layer_idx == 0:
            return initial_channels
        elif layer_idx <= num_downsampling:
            return initial_channels * (2 ** layer_idx)
        else:
            return channels_after_downsample
    
    # PatchNCE networks for multiple layers
    nce_layers = config.get("nce_layers", [0, 4, 8, 12, 16])
    
    netF_list = torch.nn.ModuleList([
        PatchSampleF(
            in_channels=get_channel_for_layer(layer),
            out_channels=config.get("nce_dim", 256),
            use_mlp=config.get("use_mlp", True)
        ).to(device)
        for layer in nce_layers
    ])
    
    # PatchNCE loss
    nce_loss = PatchNCELoss(
        temperature=config.get("nce_temperature", 0.07),
        num_patches=config.get("num_patches", 256)
    )

    # ------------------ Optimizers ------------------
    g_optimizer = torch.optim.Adam(
        list(G.parameters()) + list(netF_list.parameters()),
        lr=config["lr"], betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        D.parameters(), 
        lr=config["lr"], betas=(0.5, 0.999)
    )

    # ------------------ Data ------------------
    monet_train, monet_val, photo_train, photo_val = split_domains(
        config.get("monet_dir", '/content/painter-gan/data/monet_jpg'),
        config.get("photo_dir", '/content/painter-gan/data/photo_jpg'),
        val_ratio=config.get("val_ratio", 0.1)
    )

    train_dataset = MonetDataset(
        monet_train, photo_train,
        transform=get_train_transforms(config["image_size"]),
        random_pairing=True
    )

    val_dataset = MonetDataset(
        monet_val, photo_val,
        transform=get_val_transforms(config["image_size"]),
        random_pairing=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ------------------ Checkpoints ------------------
    base_dir = config.get("base_dir", "/content/drive/MyDrive/paint-gan-checkpoints")
    checkpoint_manager = CheckpointManager(
        base_dir=base_dir,
        run_id=run_id,
        max_checkpoints=3
    )

    start_epoch = 0
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint and config.get('resume_training', False):
        vprint(f"Resuming from checkpoint: {latest_checkpoint}", level=1)
        models = {'G': G, 'D': D, 'netF': netF_list}
        optimizers = {'G_opt': g_optimizer, 'D_opt': d_optimizer}
        start_epoch, _ = checkpoint_manager.load_checkpoint(latest_checkpoint, models, optimizers)
        start_epoch += 1
        vprint(f"Resuming from epoch {start_epoch}", level=1)

    # ------------------ Training ------------------
    vprint(f"\nStarting training for {config['num_epochs'] - start_epoch} epochs...\n", level=1)
    
    for epoch in range(start_epoch, config["num_epochs"]):
        G.train()
        D.train()
        for netF in netF_list:
            netF.train()

        epoch_g = 0.0
        epoch_d = 0.0
        epoch_nce = 0.0
        epoch_gan = 0.0
        epoch_idt = 0.0

        for step, (real_X, real_Y) in enumerate(train_loader):
            real_X = real_X.to(device)  # Photo (source)
            real_Y = real_Y.to(device)  # Monet (target)

            # =================== GENERATOR ===================
            g_optimizer.zero_grad()

            # Generate fake monet from photo
            fake_Y = G(real_X, encode_only=False)
            
            # Extract features for contrastive loss
            # Query: features from GENERATED image (fake_Y)
            # Key: features from INPUT image (real_X)
            # These must be from the SAME spatial locations to enforce correspondence
            feat_q = G(fake_Y, encode_only=True)
            feat_k = G(real_X, encode_only=True)
            
            # Select only the layers we want for NCE loss
            feat_q_selected = [feat_q[i] for i in nce_layers if i < len(feat_q)]
            feat_k_selected = [feat_k[i] for i in nce_layers if i < len(feat_k)]
            
            # Adversarial loss (fool discriminator)
            loss_gan = adversarial_loss(D(fake_Y), True)
            
            # Contrastive loss
            loss_nce = compute_contrastive_loss(
                feat_q_selected, feat_k_selected, netF_list, nce_loss,
                lambda_NCE=config.get("lambda_NCE", 1.0)
            )
            
            # Optional identity loss (G(Y) = Y)
            loss_idt = torch.tensor(0.0, device=device)
            if config.get("lambda_identity", 0.0) > 0:
                loss_idt = identity_loss(
                    real_Y, G(real_Y, encode_only=False),
                    lambda_idt=config.get("lambda_identity", 0.0)
                )
            
            # Total generator loss
            loss_g = loss_gan + loss_nce + loss_idt
            
            loss_g.backward()
            g_optimizer.step()

            # =================== DISCRIMINATOR ===================
            d_optimizer.zero_grad()
            
            # Real loss
            loss_d_real = adversarial_loss(D(real_Y), True)
            # Fake loss
            loss_d_fake = adversarial_loss(D(fake_Y.detach()), False)
            # Total discriminator loss
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            
            loss_d.backward()
            d_optimizer.step()

            # =================== ACCUMULATE ===================
            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            epoch_nce += loss_nce.item()
            epoch_gan += loss_gan.item()
            epoch_idt += loss_idt.item()

            # =================== BATCH LOGGING ===================
            if step % 50 == 0:
                wandb.log({
                    "batch/g_loss": loss_g.item(),
                    "batch/d_loss": loss_d.item(),
                    "batch/nce_loss": loss_nce.item(),
                    "batch/gan_loss": loss_gan.item(),
                    "batch/idt_loss": loss_idt.item(),
                    "epoch": epoch
                })

                vprint(
                    f"[Epoch {epoch+1} | Step {step}/{len(train_loader)}] "
                    f"G: {loss_g.item():.3f} "
                    f"D: {loss_d.item():.3f} "
                    f"NCE: {loss_nce.item():.3f}",
                    level=2
                )

        # =================== TRAIN EPOCH METRICS ===================
        train_metrics = {
            "train/g_loss": epoch_g / len(train_loader),
            "train/d_loss": epoch_d / len(train_loader),
            "train/nce_loss": epoch_nce / len(train_loader),
            "train/gan_loss": epoch_gan / len(train_loader),
            "train/idt_loss": epoch_idt / len(train_loader),
        }

        # =================== VALIDATION ===================
        G.eval()
        for netF in netF_list:
            netF.eval()
        
        val_nce = 0.0
        val_idt = 0.0

        with torch.no_grad():
            for real_X, real_Y in val_loader:
                real_X = real_X.to(device)
                real_Y = real_Y.to(device)

                fake_Y = G(real_X, encode_only=False)
                
                # Contrastive loss on validation
                feat_q = G(fake_Y, encode_only=True)
                feat_k = G(real_X, encode_only=True)
                
                feat_q_selected = [feat_q[i] for i in nce_layers if i < len(feat_q)]
                feat_k_selected = [feat_k[i] for i in nce_layers if i < len(feat_k)]
                
                val_nce += compute_contrastive_loss(
                    feat_q_selected, feat_k_selected, netF_list, nce_loss,
                    lambda_NCE=config.get("lambda_NCE", 1.0)
                ).item()
                
                if config.get("lambda_identity", 0.0) > 0:
                    val_idt += identity_loss(
                        real_Y, G(real_Y, encode_only=False),
                        lambda_idt=config.get("lambda_identity", 0.0)
                    ).item()

        val_metrics = {
            "val/nce_loss": val_nce / len(val_loader),
            "val/idt_loss": val_idt / len(val_loader) if config.get("lambda_identity", 0.0) > 0 else 0.0,
        }

        # =================== WANDB EPOCH LOG ===================
        wandb.log({
            "epoch": epoch,
            **train_metrics,
            **val_metrics
        })

        # =================== PRINT (LEVEL 1) ===================
        vprint(
            f"Epoch [{epoch+1}/{config['num_epochs']}] | "
            f"Train G: {train_metrics['train/g_loss']:.4f}, "
            f"D: {train_metrics['train/d_loss']:.4f}, "
            f"NCE: {train_metrics['train/nce_loss']:.4f} || "
            f"Val NCE: {val_metrics['val/nce_loss']:.4f}",
            level=1
        )

        # =================== SAVE ===================
        if (epoch + 1) % config.get("save_every", 10) == 0:
            models = {'G': G, 'D': D, 'netF': netF_list}
            optimizers = {'G_opt': g_optimizer, 'D_opt': d_optimizer}
            checkpoint_manager.save_checkpoint(epoch, models, optimizers, {})

    # =================== FINAL EVALUATION ===================
    print("\n" + "="*80)
    print("🔍 Running final MiFID evaluation")
    print("="*80)

    eval_transform = get_eval_transforms(config["image_size"])

    photo_val_loader = DataLoader(
        ImageOnlyDataset(photo_val, eval_transform),
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    monet_val_loader = DataLoader(
        ImageOnlyDataset(monet_val, eval_transform),
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    mifid_result = evaluate_mifid(
        generator=G,
        photo_loader=photo_val_loader,
        monet_val_loader=monet_val_loader,
        device=device,
        epsilon=config.get("cosine_threshold", 0.5)
    )

    print(f"FID:   {mifid_result['FID']:.4f}")
    print(f"dThr:  {mifid_result['d_thr']:.4f}")
    print(f"MiFID: {mifid_result['MiFID']:.4f}")

    wandb.log({
        "final/FID": mifid_result["FID"],
        "final/dThr": mifid_result["d_thr"],
        "final/MiFID": mifid_result["MiFID"]
    })

    wandb.finish()