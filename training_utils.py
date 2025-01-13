# training_utils.py
"""
Training Utilities for GAN Training

Provides utilities for GAN training:
1. EarlyStopping: Prevents overfitting via validation loss monitoring (MSE)
2. WarmupScheduler: Learning rate warmup with plateau reduction
3. Gradient Penalty: WGAN-GP penalty computation
4. Gradient Clipping: Clips gradients to prevent exploding gradients

"""

import torch
import torch.nn as nn
import torch.optim as optim
from gan_settings import *
import os
import logging
from datetime import datetime

def setup_logging(log_dir):
    """Set up logging configuration."""
    # Clear existing handlers
    logger = logging.getLogger()
    logger.handlers = []
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'gan_training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Suppress verbose logging from libraries
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def compute_composite_score(metrics):
    """Compute a composite score from multiple metrics.
    
    All metrics are converted to a "lower is better" format for consistent minimization:
    - For MSE: Used directly (naturally "lower is better")
    - For MS-SSIM: Negated since it's naturally "higher is better"
    
    Returns:
        float: Composite score (lower is better) or None if metrics are invalid
    """
    if not metrics or not isinstance(metrics, dict):
        return None
        
    if 'msssim' in metrics:  # Voxelwise image generation mode
        score = -metrics['msssim']  # Negate since MS-SSIM is "higher is better"
    elif 'mse' in metrics:  # Standard mode (including regional data)
        score = metrics['mse']  # MSE is naturally "lower is better"
    else:
        return None
        
    # Check for NaN or infinite values
    if torch.is_tensor(score):
        score = score.item()
    if not isinstance(score, (int, float)) or torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)):
        return None
        
    return score

class EarlyStopping:
    """Early stopping handler for model training.
    
    Supports different metrics based on model type:
    - Generator: Uses MSE or MS-SSIM (lower is better)
    - Discriminator: Uses Wasserstein distance for WGAN (higher is better) or classification accuracy for other GANs (higher is better)
    """
    def __init__(self, patience=50, min_delta=0.0001, verbose=False, model_name='model', min_epochs=100, metric_mode='min'):
        """
        Args:
            patience (int): How many epochs to wait before stopping after loss has stopped improving
            min_delta (float): Minimum change in monitored quantity to qualify as improvement
            verbose (bool): If True, prints a message for each improvement
            model_name (str): Name of the model for checkpoint saving
            min_epochs (int): Minimum number of epochs before early stopping can occur
            metric_mode (str): 'min' for metrics where lower is better (e.g., MSE), 'max' for metrics where higher is better (e.g., accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = os.path.join(CHECKPOINT_DIR, f'{model_name}_checkpoint.pt')
        self.model_name = model_name
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epoch = 0
        self.metric_mode = metric_mode

    def __call__(self, metrics, model, optimizer):
        """
        Call method to check for early stopping.
        
        Args:
            metrics (dict or float): If dict, contains validation metrics including MS-SSIM or MSE.
                                   If float, direct metric value.
            model (nn.Module): Model to save
            optimizer (optim.Optimizer): Optimizer to save
        """
        self.epoch += 1
        
        # Handle both dict and direct metric input
        if isinstance(metrics, dict):
            score = compute_composite_score(metrics)
        else:
            score = metrics
            
        # Handle invalid metrics
        if score is None:
            if self.verbose:
                print(f"Warning: Invalid metrics at epoch {self.epoch}. Skipping early stopping check.")
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer)
        else:
            if self.metric_mode == 'min':
                improved = score < self.best_score - self.min_delta
            else:  # metric_mode == 'max'
                improved = score > self.best_score + self.min_delta
                
            if improved:
                self.best_score = score
                self.save_checkpoint(score, model, optimizer)
                self.counter = 0
            else:
                if self.epoch >= self.min_epochs:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True

    def save_checkpoint(self, val_loss, model, optimizer):
        """
        Save model checkpoint.

        Args:
            val_loss (float): Validation loss.
            model (nn.Module): Model to save.
            optimizer (optim.Optimizer): Optimizer to save.
        """
        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'epoch': self.epoch,
            'best_score': self.best_score
        }, self.path)

class WarmupScheduler:
    """Warmup Scheduler class for learning rate warmup with optional attention layer handling."""
    def __init__(self, optimizer, warmup_epochs, warmup_lr_init, warmup_lr_final, is_attention_layer=False):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.warmup_lr_final = warmup_lr_final
        self.current_epoch = 0
        self.is_attention_layer = is_attention_layer
        
        # Store initial lr
        self.base_lrs = []
        for param_group in optimizer.param_groups:
            self.base_lrs.append(param_group['lr'])
        
        # Wrap scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    def step(self, metrics=None):
        """Step method to update learning rate with attention layer handling."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # During warmup phase
            progress = float(self.current_epoch) / float(max(1, self.warmup_epochs))
            
            # For attention layers, use different warmup schedule if enabled
            if self.is_attention_layer and USE_ATTENTION_WARMUP:
                warmup_factor = ATTENTION_WARMUP_FACTOR + progress * (ATTENTION_FINAL_FACTOR - ATTENTION_WARMUP_FACTOR)
                new_lr = self.warmup_lr_init + progress * (self.warmup_lr_final * warmup_factor - self.warmup_lr_init)
            else:
                new_lr = self.warmup_lr_init + progress * (self.warmup_lr_final - self.warmup_lr_init)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # After warmup phase
            if metrics is not None:
                self.scheduler.step(metrics)
    
    def get_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def create_attention_optimizer(model, base_lr):
    """Creates a separate optimizer for attention layers with its own learning rate."""
    attention_params = []
    non_attention_params = []
    
    for name, param in model.named_parameters():
        if 'attention' in name.lower():
            attention_params.append(param)
        else:
            non_attention_params.append(param)
    
    # Create optimizers with different initial learning rates
    attention_lr = base_lr * ATTENTION_WARMUP_FACTOR if USE_ATTENTION_WARMUP else base_lr
    attention_optimizer = optim.Adam(attention_params, lr=attention_lr, betas=(0.9, 0.999))
    main_optimizer = optim.Adam(non_attention_params, lr=base_lr, betas=(0.9, 0.999))
    
    return main_optimizer, attention_optimizer

def compute_gradient_penalty(D, real_samples, fake_samples, X, c=None):
    """
    Compute WGAN gradient penalty.

    Args:
        D (nn.Module): Discriminator model.
        real_samples (torch.Tensor): Real samples.
        fake_samples (torch.Tensor): Fake samples.
        X (torch.Tensor): Conditional input.
        c (torch.Tensor, optional): Conditional vector. Defaults to None since this is half-baked atm.

    Returns:
        torch.Tensor: Gradient penalty.
    """
    device = real_samples.device
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    if c is not None:
        d_interpolates = D(interpolates, X, c)
    else:
        d_interpolates = D(interpolates, X)
        
    fake = torch.ones(real_samples.size(0), 1, device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def clip_gradients(model, max_norm):
    """
    Clips gradients of a model to prevent exploding gradients.
    
    Args:
        model (nn.Module): PyTorch model
        max_norm (float): Maximum norm for gradient clipping
    """
    if max_norm <= 0:
        return
        
    # Compute total norm of gradients
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return
        
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)
