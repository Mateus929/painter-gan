import wandb
import torch
import uuid
from torch.utils.data import Dataset, DataLoader
from models.cyclegan import Generator, Discriminator
from training.checkpoint_manager import CheckpointManager
from training.losses import identity_loss, cycle_consistency_loss, adversarial_loss
from training.random_monet_dataset import MonetDataset, get_train_transforms, get_val_transforms
import glob
from sklearn.model_selection import train_test_split
from training.image_folder_dataset import ImageOnlyDataset, get_eval_transforms
from eval.main_eval import evaluate_mifid

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


def train_cyclegan(config):
    verbose = config.get("verbose", 1)

    def vprint(msg, level=1):
        if verbose >= level:
            print(msg)

    # ------------------ W&B ------------------
    run_id = config.get("run_id", uuid.uuid4().hex[:8])
    wandb.init(
        project="monet-cyclegan",
        name=config["run_name"],
        config=config,
        id=run_id,
        resume="allow"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vprint(f"Using device: {device}")

    # ------------------ Models ------------------
    G_XtoY = Generator(num_residual_blocks=config["num_residual_blocks"]).to(device)
    G_YtoX = Generator(num_residual_blocks=config["num_residual_blocks"]).to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)

    g_optimizer = torch.optim.Adam(
        list(G_XtoY.parameters()) + list(G_YtoX.parameters()),
        lr=config["lr"], betas=(0.5, 0.999)
    )
    d_x_optimizer = torch.optim.Adam(D_X.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    d_y_optimizer = torch.optim.Adam(D_Y.parameters(), lr=config["lr"], betas=(0.5, 0.999))

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
    if latest_checkpoint and config['resume_training']:
        vprint(f"Resuming from checkpoint: {latest_checkpoint}", level=1)
        models = {'G_XtoY': G_XtoY, 'G_YtoX': G_YtoX, 'D_X': D_X, 'D_Y': D_Y}
        optimizers = {'G_opt': g_optimizer, 'D_X_opt': d_x_optimizer, 'D_Y_opt': d_y_optimizer}
        start_epoch, _ = checkpoint_manager.load_checkpoint(latest_checkpoint, models, optimizers)
        start_epoch += 1
        vprint(f"Resuming from epoch {start_epoch}", level=1)

    # ------------------ Training ------------------
    vprint(f"\nStarting training for {config['num_epochs'] - start_epoch} epochs...\n", level=1)
    for epoch in range(start_epoch, config["num_epochs"]):
        G_XtoY.train()
        G_YtoX.train()
        D_X.train()
        D_Y.train()

        epoch_g = 0.0
        epoch_d = 0.0
        epoch_cycle = 0.0
        epoch_identity = 0.0
        epoch_adv = 0.0

        for step, (real_X, real_Y) in enumerate(train_loader):
            real_X = real_X.to(device)
            real_Y = real_Y.to(device)

            # =================== GENERATORS ===================
            g_optimizer.zero_grad()

            fake_Y = G_XtoY(real_X)
            fake_X = G_YtoX(real_Y)

            loss_adv_XtoY = adversarial_loss(D_Y(fake_Y), True)
            loss_adv_YtoX = adversarial_loss(D_X(fake_X), True)

            loss_cycle_X = cycle_consistency_loss(
                real_X, G_YtoX(fake_Y), config["lambda_cycle"]
            )
            loss_cycle_Y = cycle_consistency_loss(
                real_Y, G_XtoY(fake_X), config["lambda_cycle"]
            )

            loss_id_X = identity_loss(
                real_X, G_YtoX(real_X), config["lambda_identity"]
            )
            loss_id_Y = identity_loss(
                real_Y, G_XtoY(real_Y), config["lambda_identity"]
            )

            loss_g = (
                loss_adv_XtoY + loss_adv_YtoX +
                loss_cycle_X + loss_cycle_Y +
                loss_id_X + loss_id_Y
            )

            loss_g.backward()
            g_optimizer.step()

            # =================== DISCRIMINATOR X ===================
            d_x_optimizer.zero_grad()
            loss_dx = 0.5 * (
                adversarial_loss(D_X(real_X), True) +
                adversarial_loss(D_X(fake_X.detach()), False)
            )
            loss_dx.backward()
            d_x_optimizer.step()

            # =================== DISCRIMINATOR Y ===================
            d_y_optimizer.zero_grad()
            loss_dy = 0.5 * (
                adversarial_loss(D_Y(real_Y), True) +
                adversarial_loss(D_Y(fake_Y.detach()), False)
            )
            loss_dy.backward()
            d_y_optimizer.step()

            # =================== ACCUMULATE ===================
            epoch_g += loss_g.item()
            epoch_d += (loss_dx.item() + loss_dy.item())
            epoch_cycle += (loss_cycle_X.item() + loss_cycle_Y.item())
            epoch_identity += (loss_id_X.item() + loss_id_Y.item())
            epoch_adv += (loss_adv_XtoY.item() + loss_adv_YtoX.item())

            # =================== BATCH LOGGING ===================
            if step % 50 == 0:
                wandb.log({
                    "batch/g_loss": loss_g.item(),
                    "batch/d_loss": loss_dx.item() + loss_dy.item(),
                    "batch/adv_loss": loss_adv_XtoY.item() + loss_adv_YtoX.item(),
                    "batch/cycle_loss": loss_cycle_X.item() + loss_cycle_Y.item(),
                    "batch/identity_loss": loss_id_X.item() + loss_id_Y.item(),
                    "epoch": epoch
                })

                vprint(
                    f"[Epoch {epoch+1} | Step {step}/{len(train_loader)}] "
                    f"G: {loss_g.item():.3f} "
                    f"D: {(loss_dx.item()+loss_dy.item()):.3f}",
                    level=2
                )

        # =================== TRAIN EPOCH METRICS ===================
        train_metrics = {
            "train/g_loss": epoch_g / len(train_loader),
            "train/d_loss": epoch_d / len(train_loader),
            "train/cycle_loss": epoch_cycle / len(train_loader),
            "train/identity_loss": epoch_identity / len(train_loader),
            "train/adv_loss": epoch_adv / len(train_loader),
        }

        # =================== VALIDATION ===================
        G_XtoY.eval()
        G_YtoX.eval()

        val_cycle = 0.0
        val_identity = 0.0

        with torch.no_grad():
            for real_X, real_Y in val_loader:
                real_X = real_X.to(device)
                real_Y = real_Y.to(device)

                fake_Y = G_XtoY(real_X)
                fake_X = G_YtoX(real_Y)

                val_cycle += (
                    cycle_consistency_loss(
                        real_X, G_YtoX(fake_Y), config["lambda_cycle"]
                    ) +
                    cycle_consistency_loss(
                        real_Y, G_XtoY(fake_X), config["lambda_cycle"]
                    )
                ).item()

                val_identity += (
                    identity_loss(
                        real_X, G_YtoX(real_X), config["lambda_identity"]
                    ) +
                    identity_loss(
                        real_Y, G_XtoY(real_Y), config["lambda_identity"]
                    )
                ).item()

        val_metrics = {
            "val/cycle_loss": val_cycle / len(val_loader),
            "val/identity_loss": val_identity / len(val_loader),
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
            f"Cycle: {train_metrics['train/cycle_loss']:.4f} || "
            f"Val Cycle: {val_metrics['val/cycle_loss']:.4f}, "
            f"Val Id: {val_metrics['val/identity_loss']:.4f}",
            level=1
        )

        # =================== SAVE ===================
        if (epoch + 1) % config.get("save_every", 10) == 0:
            models = {'G_XtoY': G_XtoY, 'G_YtoX': G_YtoX, 'D_X': D_X, 'D_Y': D_Y}
            optimizers = {'G_opt': g_optimizer, 'D_X_opt': d_x_optimizer, 'D_Y_opt': d_y_optimizer}
            # metrics = {'g_loss': avg_g_loss, 'd_loss': avg_d_loss}
            checkpoint_manager.save_checkpoint(epoch, models, optimizers, {})

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
        generator=G_XtoY,             
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
