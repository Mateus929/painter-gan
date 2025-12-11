import wandb
import torch
import uuid
import os
from torch.utils.data import Dataset, DataLoader
from models.cyclegan import Generator, Discriminator
from training.checkpoint_manager import CheckpointManager
from training.losses import identity_loss, cycle_consistency_loss, adversarial_loss
from training.monet_dataset import MonetDataset, get_transforms
from eval.eval import evaluate_model


def train_cyclegan(config):
    """
      Main training function for CycleGAN.

      Args:
          config (dict): Dictionary containing the hyperparameters and configuration for training.

      Configuration Dictionary (config):
          - 'num_residual_blocks' (int): Number of residual blocks in the generator.
          - 'lr' (float): Learning rate for the Adam optimizer.
          - 'batch_size' (int): Batch size for training.
          - 'image_size' (int): The target image size for resizing.
          - 'lambda_cycle' (float): Weight for the cycle consistency loss.
          - 'lambda_identity' (float): Weight for the identity loss.
          - 'num_epochs' (int): Total number of training epochs.
          - 'cosine_threashold' (defauls 0.5): Threashold for cosine similarity in mifid.
          - 'penalty_weight' (default 100): Penaly for memorization in mifid.
          - 'save_every' (int, default=10): Frequency (in epochs) to save checkpoints.
          - 'eval_every' (int, default=10): Frequency (in epochs) to eval models.
          - 'resume_training' (bool): Whether to resume training from the last checkpoint.
          - 'run_name' (str)
              Name of the current run that will be logged on wandb.
          - 'monet_dir' (str, optional, default='/content/painter-gan/data/monet_jpg'): 
              Path to Monet dataset directory.
          - 'photo_dir' (str, optional, default='/content/painter-gan/data/photo_jpg'): 
              Path to Photo dataset directory.
          - 'base_dir' (str, optional, default='/content/drive/MyDrive/paint-gan-checkpoints'):
              Base directory to store checkpoints.
          - 'run_id' (str, optional, default=random ID): 
              The run ID for W&B logging. If not provided, uses random ID. 
              If this is set, make sure resume_training is set as well.
          - 'verbose' (int, default=1): Verbosity level:
              * 0: No output
              * 1: Epoch-level output only
              * 2: Detailed output (batch-level information)
      
      Note:
          All parameters are required except 'monet_dir', 'photo_dir', and 'run_id' 
          which have default values.
      """
    
    verbose_level = config.get('verbose', 1)
    
    def vprint(message, level=1):
        if verbose_level >= level:
            print(message)

    # Initialize WANDB

    run_id = ""
    if "run_id" in config and config["run_id"]:
        run_id = config["run_id"]
    else:
        run_id = uuid.uuid4().hex[:8]

    wandb.init(project="monet-cyclegan", config=config, name=config["run_name"], id=run_id, resume="allow")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vprint(f"Using device: {device}", level=1)
    
    # Initialize models
    vprint("Initializing models...", level=2)
    G_XtoY = Generator(num_residual_blocks=config['num_residual_blocks']).to(device)
    G_YtoX = Generator(num_residual_blocks=config['num_residual_blocks']).to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)
    
    # Initialize optimizers
    vprint("Initializing optimizers...", level=2)
    g_optimizer = torch.optim.Adam(
        list(G_XtoY.parameters()) + list(G_YtoX.parameters()),
        lr=config['lr'],
        betas=(0.5, 0.999)
    )
    d_x_optimizer = torch.optim.Adam(D_X.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    d_y_optimizer = torch.optim.Adam(D_Y.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # Data loading
    vprint("Loading datasets...", level=2)
    dataset = MonetDataset(
        monet_dir=config.get("monet_dir", "/content/painter-gan/data/monet_jpg"),
        photo_dir=config.get("photo_dir", "/content/painter-gan/data/photo_jpg"),
        transform=get_transforms(config['image_size'])
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    
    # Checkpoint manager
    base_dir = config.get("base_dir", "/content/drive/MyDrive/paint-gan-checkpoints")
    checkpoint_manager = CheckpointManager(
        base_dir = base_dir,
        run_id=run_id,
        max_checkpoints=3
    )
    
    # Resume from checkpoint if exists
    start_epoch = 0
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint and config['resume_training']:
        vprint(f"Resuming from checkpoint: {latest_checkpoint}", level=1)
        models = {'G_XtoY': G_XtoY, 'G_YtoX': G_YtoX, 'D_X': D_X, 'D_Y': D_Y}
        optimizers = {'G_opt': g_optimizer, 'D_X_opt': d_x_optimizer, 'D_Y_opt': d_y_optimizer}
        start_epoch, _ = checkpoint_manager.load_checkpoint(latest_checkpoint, models, optimizers)
        start_epoch += 1
        vprint(f"Resuming from epoch {start_epoch}", level=1)
    
    
    best_mifid = float('inf')

    # Training loop
    vprint(f"\nStarting training for {config['num_epochs'] - start_epoch} epochs...\n", level=1)
    for epoch in range(start_epoch, config['num_epochs']):
        G_XtoY.train()
        G_YtoX.train()
        D_X.train()
        D_Y.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_loss_cycle = 0
        epoch_loss_identity = 0
        
        for i, (real_X, real_Y) in enumerate(dataloader):
            real_X = real_X.to(device)
            real_Y = real_Y.to(device)
            
            # ============ Train Generators ============
            g_optimizer.zero_grad()
            
            # Generate fake images
            fake_Y = G_XtoY(real_X)
            fake_X = G_YtoX(real_Y)
            
            # Adversarial loss
            pred_fake_Y = D_Y(fake_Y)
            pred_fake_X = D_X(fake_X)
            loss_G_XtoY = adversarial_loss(pred_fake_Y, True)
            loss_G_YtoX = adversarial_loss(pred_fake_X, True)
            
            # Cycle consistency loss
            reconstructed_X = G_YtoX(fake_Y)
            reconstructed_Y = G_XtoY(fake_X)
            loss_cycle_X = cycle_consistency_loss(real_X, reconstructed_X, config['lambda_cycle'])
            loss_cycle_Y = cycle_consistency_loss(real_Y, reconstructed_Y, config['lambda_cycle'])
            
            # Identity loss
            identity_X = G_YtoX(real_X)
            identity_Y = G_XtoY(real_Y)
            loss_identity_X = identity_loss(real_X, identity_X, config['lambda_identity'])
            loss_identity_Y = identity_loss(real_Y, identity_Y, config['lambda_identity'])
            
            # Total generator loss
            g_loss = (loss_G_XtoY + loss_G_YtoX + 
                     loss_cycle_X + loss_cycle_Y + 
                     loss_identity_X + loss_identity_Y)
            
            g_loss.backward()
            g_optimizer.step()
            
            # ============ Train Discriminators ============
            # Discriminator X
            d_x_optimizer.zero_grad()
            pred_real_X = D_X(real_X)
            loss_D_real_X = adversarial_loss(pred_real_X, True)
            pred_fake_X = D_X(fake_X.detach())
            loss_D_fake_X = adversarial_loss(pred_fake_X, False)
            d_x_loss = (loss_D_real_X + loss_D_fake_X) * 0.5
            d_x_loss.backward()
            d_x_optimizer.step()
            
            # Discriminator Y
            d_y_optimizer.zero_grad()
            pred_real_Y = D_Y(real_Y)
            loss_D_real_Y = adversarial_loss(pred_real_Y, True)
            pred_fake_Y = D_Y(fake_Y.detach())
            loss_D_fake_Y = adversarial_loss(pred_fake_Y, False)
            d_y_loss = (loss_D_real_Y + loss_D_fake_Y) * 0.5
            d_y_loss.backward()
            d_y_optimizer.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += (d_x_loss.item() + d_y_loss.item())
            epoch_loss_cycle += (loss_cycle_X.item() + loss_cycle_Y.item())
            epoch_loss_identity += (loss_identity_X.item() + loss_identity_Y.item())

            # Log to WANDB every N steps
            if i % 50 == 0:
                vprint(f"Batch [{i+1}/{len(dataloader)}] - G Loss: {g_loss.item():.4f}, D Loss: {(d_x_loss.item() + d_y_loss.item()):.4f}", level=2)
                wandb.log({
                    'batch_g_loss': g_loss.item(),
                    'batch_d_loss': (d_x_loss.item() + d_y_loss.item()),
                    'loss_G_XtoY': loss_G_XtoY.item(),
                    'loss_G_YtoX': loss_G_YtoX.item(),
                    'loss_cycle': (loss_cycle_X.item() + loss_cycle_Y.item()),
                    'loss_cycle_X': loss_cycle_X.item(),
                    'loss_cycle_Y': loss_cycle_Y.item(),
                    'loss_identity': (loss_identity_X.item() + loss_identity_Y.item()),
                    'loss_identity_X': loss_identity_X.item(),
                    'loss_identity_Y': loss_identity_Y.item(),
                    'loss_D_X_real': loss_D_real_X.item(),
                    'loss_D_X_fake': loss_D_fake_X.item(),
                    'loss_D_Y_real': loss_D_real_Y.item(),
                    'loss_D_Y_fake': loss_D_fake_Y.item(),
                })
        
        # Log epoch metrics
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_loss_cycle = epoch_loss_cycle / len(dataloader)
        avg_loss_identity = epoch_loss_identity / len(dataloader)
        
        vprint(f"Epoch [{epoch+1}/{config['num_epochs']}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}", level=1)
        
        wandb.log({
            'epoch': epoch,
            'epoch_g_loss': avg_g_loss,
            'epoch_d_loss': avg_d_loss,
            'epoch_loss_cycle' : avg_loss_cycle,
            'epoch_loss_identity' : avg_loss_identity
        })

        # ============ EVALUATION ============
        is_best = False
        eval_every = config.get('eval_every', 10)
        if (epoch + 1) % eval_every == 0:
            
            vprint("\n" + "="*80, level=1)
            vprint(f"🔍 EVALUATING MODEL AT EPOCH {epoch+1}", level=1)
            vprint("="*80, level=1)
            
            # Run evaluation
            eval_result = evaluate_model(
                generator=G_XtoY,
                config=config,
                epoch=epoch,
                eval_images_dir=f"tmp/{run_id}",
                device=device
            )
            
            # Log evaluation metrics to WandB
            wandb.log({
                'eval/mifid': eval_result['mifid'],
                'eval/fid': eval_result['fid'],
                'eval/memorization_penalty': eval_result['memorization_penalty'],
                'eval/memorization_ratio': eval_result['memorization_ratio'],
                'eval/epoch': epoch
            })
            
            if verbose_level >= 1:
                display_evaluation_results(eval_result, epoch)
            
            # Save best model
            if eval_result['mifid'] < best_mifid:
                best_mifid = eval_result['mifid']
                vprint(f"New best MiFID: {best_mifid:.4f} (saving checkpoint)", level=1)
                is_best = True
                wandb.log({'best_mifid': best_mifid})
            
            print("="*80 + "\n")
            
            # Return to training mode
            G_XtoY.train()
            G_YtoX.train()
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config.get('save_every', 10) == 0 or is_best:
            vprint(f"Saving checkpoint at epoch {epoch+1}...", level=1)
            models = {'G_XtoY': G_XtoY, 'G_YtoX': G_YtoX, 'D_X': D_X, 'D_Y': D_Y}
            optimizers = {'G_opt': g_optimizer, 'D_X_opt': d_x_optimizer, 'D_Y_opt': d_y_optimizer}
            metrics = {'g_loss': avg_g_loss, 'd_loss': avg_d_loss}
            checkpoint_manager.save_checkpoint(epoch, models, optimizers, metrics, is_best)
            
            # Log sample images
            vprint("Logging sample images to W&B...", level=2)
            with torch.no_grad():
                sample_fake_Y = G_XtoY(real_X[:4])
                wandb.log({
                    "generated_images": [wandb.Image(img) for img in sample_fake_Y]
                })
    
    vprint("\nTraining completed!", level=1)
    wandb.finish()

def display_evaluation_results(result, epoch):
    """
    Display evaluation results in a nice format
    
    Args:
        result: Evaluation result dict
        epoch: Current epoch
    """
    print(f"\n Evaluation Results (Epoch {epoch+1}):")
    print("─" * 60)
    print(f"  MiFID Score:           {result['mifid']:>10.4f}  {'(lower is better)':>20}")
    print(f"  FID Score:             {result['fid']:>10.4f}  {'(quality)':>20}")
    print(f"  Memorization Penalty:  {result['memorization_penalty']:>10.4f}  {'(penalty)':>20}")
    print(f"  Memorization Ratio:    {result['memorization_ratio']*100:>9.2f}%  {'(similarity)':>20}")
    print("─" * 60)
    
    # Interpretation
    if result['memorization_ratio'] > 0.3:
        print("Warning: High memorization detected (>30%)")
    elif result['memorization_ratio'] > 0.1:
        print("Info: Moderate memorization detected (10-30%)")
    else:
        print("Good: Low memorization (<10%)")
    
    if result['fid'] < 50:
        print("Excellent FID score!")
    elif result['fid'] < 100:
        print("Good FID score")
    else:
        print("FID could be improved")
    
    print()