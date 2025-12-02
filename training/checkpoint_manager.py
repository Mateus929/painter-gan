import os
import torch
import shutil

class CheckpointManager:
    def __init__(self, base_dir, run_id, max_checkpoints=3, best_model_filename='best_model.pth', verbose = False):
        self.checkpoint_dir = os.path.join(base_dir, run_id)
        
        self.max_checkpoints = max_checkpoints
        self.best_model_filename = best_model_filename
        self.verbose = verbose
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if verbose:
          print(f"Checkpoint directory initialized: {self.checkpoint_dir}")

    def save_checkpoint(self, epoch, models_dict, optimizers_dict, metrics, is_best=False):
        """ 
        Save checkpoint into the run-specific folder (base_dir/run_id).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics
        }
        
        for name, model in models_dict.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()
        
        for name, optimizer in optimizers_dict.items():
            checkpoint[f'{name}_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        if self.verbose:
          print(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            self._save_best_checkpoint(checkpoint, epoch)
        self._cleanup_old_checkpoints()

    def _save_best_checkpoint(self, checkpoint_data, epoch):
        """
        Saves the provided checkpoint data as the 'best' model, overwriting the previous best.
        """
        best_path = os.path.join(self.checkpoint_dir, self.best_model_filename)
        torch.save(checkpoint_data, best_path)
        if self.verbose:
          print(f"**New Best Model** saved for epoch {epoch} at: {best_path}")

    def load_checkpoint(self, checkpoint_path, models_dict, optimizers_dict):
        """ 
        Load checkpoint from a specific path. 
        """
        checkpoint = torch.load(checkpoint_path)
        for name, model in models_dict.items():
            model.load_state_dict(checkpoint[f'{name}_state_dict']) 
        for name, optimizer in optimizers_dict.items():
            optimizer.load_state_dict(checkpoint[f'{name}_state_dict'])
        if self.verbose:
          print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']

    def get_latest_checkpoint(self):
        """Find the most recent *regular* checkpoint."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def get_best_checkpoint(self):
        """Return the path to the best model checkpoint if it exists."""
        best_path = os.path.join(self.checkpoint_dir, self.best_model_filename)
        if os.path.exists(best_path):
            return best_path
        return None

    def _cleanup_old_checkpoints(self):
        """Keep only the most recent *regular* checkpoints."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for old_checkpoint in checkpoints[:-self.max_checkpoints]:
            os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))