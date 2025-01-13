import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from gan_settings import VARIANCE_THRESHOLD

logger = logging.getLogger(__name__)

def init_weights(m: nn.Module, seed: Optional[int] = None) -> None:
    """Initialize model weights using Xavier initialization with optional seed"""
    if seed is not None:
        torch.manual_seed(seed)
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def print_model_info(model: nn.Module, name: str) -> None:
    """Print model architecture and parameter count"""
    logger.info(f"\n{name} Architecture:")
    logger.info(str(model))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters in {name}: {params:,}")

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filename: str,
    output_dir: str
) -> Optional[str]:
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> bool:
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return True
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return False

def validate_model_outputs(
    model: nn.Module,
    sample_input: Union[torch.Tensor, List[torch.Tensor]],
    device: torch.device
) -> bool:
    """Validate model outputs given sample input"""
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, list):
                outputs = model(*[x.to(device) for x in sample_input])
            else:
                outputs = model(sample_input.to(device))
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def get_batch_metrics(
    generator: nn.Module,
    discriminator: nn.Module,
    batch_data: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device
) -> Optional[Dict[str, float]]:
    """Get metrics for a single batch"""
    try:
        real_samples, conditions = batch_data
        
        # Log shapes for debugging
        logger.debug(f"Batch shapes - real_samples: {real_samples.shape}, conditions: {conditions.shape}")
        
        # Check for NaN/Inf values
        if torch.isnan(real_samples).any() or torch.isinf(real_samples).any():
            logger.warning("NaN/Inf values found in real samples")
            return None
        if torch.isnan(conditions).any() or torch.isinf(conditions).any():
            logger.warning("NaN/Inf values found in conditions")
            return None
            
        real_samples = real_samples.to(device)
        conditions = conditions.to(device)
        batch_size = real_samples.size(0)
        
        # Generate latent vector
        z = torch.randn(batch_size, generator.z_dim, device=device)
        
        # Generate fake samples
        fake_samples = generator(z, conditions)
        
        # Check generated samples for NaN/Inf
        if torch.isnan(fake_samples).any() or torch.isinf(fake_samples).any():
            logger.warning("NaN/Inf values found in generated samples")
            return None
        
        # Get discriminator outputs
        real_validity = discriminator(real_samples, conditions).mean().item()
        fake_validity = discriminator(fake_samples, conditions).mean().item()
        
        # Check validity scores
        if not (-1e6 <= real_validity <= 1e6) or not (-1e6 <= fake_validity <= 1e6):
            logger.warning(f"Invalid validity scores - real: {real_validity}, fake: {fake_validity}")
            return None
        
        return {
            'real_validity': real_validity,
            'fake_validity': fake_validity,
            'fake_samples': fake_samples  # Add fake samples to metrics
        }
        
    except Exception as e:
        logger.error(f"Error in get_batch_metrics: {str(e)}")
        return None

def validate_models(
    generator: nn.Module,
    discriminator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mode: str = "behavior_to_image"
) -> Dict[str, float]:
    """Validate models on validation dataset with mode-specific metrics"""
    generator.eval()
    discriminator.eval()
    
    total_real_validity = 0
    total_fake_validity = 0
    total_mse = 0
    num_samples = 0
    
    # Track generated samples for variance check
    all_fake_samples = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            real_samples, conditions = batch_data
            batch_size = real_samples.size(0)
            
            # Generate fake samples
            z = torch.randn(batch_size, generator.z_dim, device=device)
            fake_samples = generator(z, conditions.to(device))
            
            if mode == "image_to_behavior":
                # Calculate MSE for behavioral predictions
                mse = torch.nn.functional.mse_loss(fake_samples, real_samples.to(device))
                total_mse += mse.item() * batch_size
            else:  # behavior_to_image mode
                # Only calculate discriminator validity scores
                real_validity = discriminator(real_samples.to(device), conditions.to(device)).mean().item()
                fake_validity = discriminator(fake_samples, conditions.to(device)).mean().item()
                total_real_validity += real_validity * batch_size
                total_fake_validity += fake_validity * batch_size
            
            num_samples += batch_size
            all_fake_samples.append(fake_samples.cpu())
    
    # If no valid samples were processed, return default metrics
    if num_samples == 0:
        logger.warning("No valid samples in validation - returning default metrics")
        return {
            'val_real_validity': 0.0,
            'val_fake_validity': 0.0,
            'val_mse': float('inf'),
            'mode_collapse': True,
            'sample_variance': 0.0
        }
    
    # Check for mode collapse using variance threshold
    mode_collapse = False
    sample_variance = 0.0
    if all_fake_samples:
        fake_samples = torch.cat(all_fake_samples, dim=0)
        sample_variance = torch.var(fake_samples, dim=0).mean().item()
        mode_collapse = sample_variance < VARIANCE_THRESHOLD

    metrics = {
        'mode_collapse': mode_collapse,
        'sample_variance': sample_variance
    }
    
    if mode == "image_to_behavior":
        metrics['val_mse'] = total_mse / num_samples
        metrics['val_real_validity'] = 0.0  # Add these for compatibility
        metrics['val_fake_validity'] = 0.0
    else:  # behavior_to_image mode
        metrics['val_real_validity'] = total_real_validity / num_samples
        metrics['val_fake_validity'] = total_fake_validity / num_samples
        metrics['val_mse'] = 0.0  # Set to 0 since we don't use MSE in this mode
    
    return metrics 