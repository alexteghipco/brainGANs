import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
import time
import torch
from torchvision.utils import save_image
import math

logger = logging.getLogger(__name__)

# Store active training figures with their timestamps
active_training_figures = {}
MAX_OPEN_FIGURES = 5

class LossHistory:
    def __init__(self):
        self.d_losses = []
        self.g_losses = []
        self.batch_numbers = []  # Add batch number tracking
        self.current_batch = 0

    def add_losses(self, d_loss, g_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        self.batch_numbers.append(self.current_batch)
        self.current_batch += 1

# Global loss history for each fold
fold_loss_histories = {}

def create_managed_subplots(nrows: int = 1, ncols: int = 1, figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure and array of subplots while managing figure limits."""
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, axes

def manage_figure_limit():
    """Ensure no more than MAX_OPEN_FIGURES are open by closing the oldest ones"""
    global active_training_figures
    
    if len(active_training_figures) > MAX_OPEN_FIGURES:
        # Sort by timestamp and get the oldest ones to remove
        sorted_folds = sorted(active_training_figures.items(), key=lambda x: x[1][1])
        folds_to_remove = sorted_folds[:len(active_training_figures) - MAX_OPEN_FIGURES]
        
        for fold, (fig, _) in folds_to_remove:
            plt.close(fig)
            del active_training_figures[fold]

def save_average_test_images(
    true_images: np.ndarray,
    generated_images: np.ndarray,
    output_dir: str,
    mode: str,
    fold: Optional[int] = None,
    brain_mask: Optional[np.ndarray] = None,
    original_shape: Optional[Tuple[int, ...]] = None,
) -> None:
    """Save average test images for visualization with multiple slices per view."""
    # Convert numpy arrays to tensors if needed
    if isinstance(true_images, np.ndarray):
        true_images = torch.from_numpy(true_images)
    if isinstance(generated_images, np.ndarray):
        generated_images = torch.from_numpy(generated_images)

    # Calculate averages
    avg_true = true_images.mean(dim=0)
    avg_generated = generated_images.mean(dim=0)

    if original_shape is not None and brain_mask is not None:
        # Convert brain mask to tensor if needed
        if isinstance(brain_mask, np.ndarray):
            brain_mask = torch.from_numpy(brain_mask)

        # Create empty tensors of the full size
        full_true = torch.zeros(brain_mask.shape, dtype=avg_true.dtype)
        full_generated = torch.zeros(brain_mask.shape, dtype=avg_generated.dtype)

        # Fill in the values using the brain mask
        full_true[brain_mask.bool()] = avg_true
        full_generated[brain_mask.bool()] = avg_generated

        # Reshape to original 3D shape
        avg_true = full_true.reshape(original_shape)
        avg_generated = full_generated.reshape(original_shape)

        # Function to get slice indices at 25%, 50%, and 75%
        def get_slice_indices(dim_size):
            return [
                dim_size // 4,          # 25%
                dim_size // 2,          # 50%
                (3 * dim_size) // 4     # 75%
            ]

        # Get slice indices for each dimension
        x_slices = get_slice_indices(avg_true.shape[0])
        y_slices = get_slice_indices(avg_true.shape[1])
        z_slices = get_slice_indices(avg_true.shape[2])

        # Function to normalize and prepare slice for visualization
        def prepare_slice(slice_tensor):
            slice_np = slice_tensor.squeeze().cpu().numpy()
            p1, p99 = np.percentile(slice_np[slice_np != 0], [1, 99])
            normalized = np.clip(slice_np, p1, p99)
            normalized = (normalized - p1) / (p99 - p1)
            return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        # Function to create a comparison figure with multiple slices
        def save_comparison_view(true_vol, gen_vol, slice_indices, axis, view_name):
            plt.figure(figsize=(12, 4))
            
            for i, slice_idx in enumerate(slice_indices, 1):
                # True image
                plt.subplot(2, len(slice_indices), i)
                if axis == 0:
                    slice_data = true_vol[slice_idx, :, :]
                elif axis == 1:
                    slice_data = true_vol[:, slice_idx, :]
                else:
                    slice_data = true_vol[:, :, slice_idx]
                    
                plt.imshow(prepare_slice(slice_data.unsqueeze(0).unsqueeze(0)).squeeze(), cmap='gray')
                plt.title(f'True ({slice_idx})')
                plt.axis('off')
                
                # Generated image
                plt.subplot(2, len(slice_indices), i + len(slice_indices))
                if axis == 0:
                    slice_data = gen_vol[slice_idx, :, :]
                elif axis == 1:
                    slice_data = gen_vol[:, slice_idx, :]
                else:
                    slice_data = gen_vol[:, :, slice_idx]
                    
                plt.imshow(prepare_slice(slice_data.unsqueeze(0).unsqueeze(0)).squeeze(), cmap='gray')
                plt.title(f'Generated ({slice_idx})')
                plt.axis('off')
            
            plt.suptitle(f'{view_name} View')
            plt.tight_layout()
            
            # Save and close
            fold_str = f"_fold{fold}" if fold is not None else ""
            plt.savefig(os.path.join(output_dir, f'average_comparison_{mode}_{view_name.lower()}{fold_str}.png'))
            plt.close()

        # Save comparison views for each axis
        os.makedirs(output_dir, exist_ok=True)
        save_comparison_view(avg_true, avg_generated, x_slices, 0, 'Sagittal')
        save_comparison_view(avg_true, avg_generated, y_slices, 1, 'Coronal')
        save_comparison_view(avg_true, avg_generated, z_slices, 2, 'Axial')

    else:
        logger.warning("Original shape or brain mask not provided. Visualization may not be optimal.")
        # Fallback to basic reshaping if original_shape not provided
        total_elements = avg_true.numel()
        size = int(math.sqrt(total_elements))
        width = total_elements // size
        avg_true = avg_true.reshape(1, 1, size, width)
        avg_generated = avg_generated.reshape(1, 1, size, width)
        
        # Save a single comparison view in this case
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(avg_true.squeeze(), cmap='gray')
        plt.title('True')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(avg_generated.squeeze(), cmap='gray')
        plt.title('Generated')
        plt.axis('off')
        
        plt.suptitle('2D View (Original shape not provided)')
        plt.tight_layout()
        
        fold_str = f"_fold{fold}" if fold is not None else ""
        plt.savefig(os.path.join(output_dir, f'average_comparison_{mode}{fold_str}.png'))
        plt.close()

def save_generated_data_matrix(
    true_data: np.ndarray,
    generated_data: np.ndarray,
    fold: int,
    output_dir: str,
    mode: str,
    is_final_fold: bool = False,
    all_folds_data: Optional[List] = None
) -> None:
    """Save generated data matrix for a fold."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual fold data
    np.save(os.path.join(output_dir, f'true_data_fold{fold}.npy'), true_data)
    np.save(os.path.join(output_dir, f'generated_data_fold{fold}.npy'), generated_data)
    
    # If this is the final fold and we have all folds data, save combined results
    if is_final_fold and all_folds_data is not None:
        all_true_data = []
        all_generated_data = []
        for fold_data in all_folds_data:
            if fold_data is not None and isinstance(fold_data, (list, tuple)) and len(fold_data) == 2:
                true_fold = fold_data[0]
                gen_fold = fold_data[1]
                if isinstance(true_fold, np.ndarray) and true_fold.size > 0 and \
                   isinstance(gen_fold, np.ndarray) and gen_fold.size > 0:
                    all_true_data.append(true_fold)
                    all_generated_data.append(gen_fold)
        
        if all_true_data and all_generated_data:
            try:
                combined_true = np.concatenate(all_true_data, axis=0)
                combined_generated = np.concatenate(all_generated_data, axis=0)
                np.save(os.path.join(output_dir, 'combined_true_data.npy'), combined_true)
                np.save(os.path.join(output_dir, 'combined_generated_data.npy'), combined_generated)
            except Exception as e:
                logger.error(f"Error concatenating data: {str(e)}")

def save_comparison_images(
    original_data: np.ndarray,
    generated_data: np.ndarray,
    subject_idx: int,
    output_dir: str,
    average_generated: Optional[np.ndarray] = None
) -> None:
    """Save comparison images between real and generated data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to PyTorch tensors and reshape
    original_tensor = torch.from_numpy(original_data).float()
    generated_tensor = torch.from_numpy(generated_data).float()
    
    # Reshape tensors to (1, 1, 15, -1) format using reshape instead of view
    original_tensor = original_tensor.squeeze(2).reshape(1, 1, 15, -1)
    generated_tensor = generated_tensor.squeeze(2).reshape(1, 1, 15, -1)
    
    # Create comparison grid
    if average_generated is not None:
        avg_tensor = torch.from_numpy(average_generated).float()
        avg_tensor = avg_tensor.squeeze(2).reshape(1, 1, 15, -1)
        comparison = torch.cat([original_tensor, generated_tensor, avg_tensor], dim=3)
    else:
        comparison = torch.cat([original_tensor, generated_tensor], dim=3)
    
    # Save the comparison image
    save_image(comparison, os.path.join(output_dir, f'comparison_subject{subject_idx}.png'))

def plot_training_progress(
    fold: int,
    epoch: int,
    d_losses: List[float],
    g_losses: List[float],
    val_metrics_history: Dict[str, List[float]],
    output_dir: str,
    mode: str,
    save_plot: bool = True,
    is_final_epoch: bool = False,
) -> None:
    """Plot training progress including losses, validation metrics, and sample variance."""
    global active_training_figures, fold_loss_histories
    
    # Define the color scheme
    colors = {
        'd_loss': '#ffbe0b',  # Yellow
        'g_loss': '#fb5607',  # Orange
        'real_validity': '#ff006e',  # Pink
        'fake_validity': '#8338ec',  # Purple
        'variance': '#3a86ff',  # Blue
        'mse': '#2ec4b6'  # Teal
    }
    
    try:
        # Initialize or get existing loss history
        if fold not in fold_loss_histories:
            fold_loss_histories[fold] = LossHistory()
        
        history = fold_loss_histories[fold]
        
        # Safely add losses if available
        if isinstance(d_losses, (list, np.ndarray)) and len(d_losses) > 0 and \
           isinstance(g_losses, (list, np.ndarray)) and len(g_losses) > 0:
            history.add_losses(float(d_losses[-1]), float(g_losses[-1]))

        # Get or create figure for this fold
        if fold in active_training_figures:
            fig, _ = active_training_figures[fold]
            plt.figure(fig.number)
            plt.clf()  # Clear the figure
        else:
            fig = plt.figure(figsize=(15, 5))
            manage_figure_limit()  # Ensure we don't have too many open figures
            active_training_figures[fold] = (fig, time.time())

        # Plot 1: Training Losses
        plt.subplot(131)
        plt.plot(history.batch_numbers, history.d_losses, label='D Loss', color=colors['d_loss'])
        plt.plot(history.batch_numbers, history.g_losses, label='G Loss', color=colors['g_loss'])
        plt.title('Training Losses')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Validation Metrics
        plt.subplot(132)
        if mode == "image_to_behavior":
            if isinstance(val_metrics_history, dict) and 'val_mse' in val_metrics_history and val_metrics_history['val_mse']:
                plt.plot(val_metrics_history['val_mse'], label='MSE', color=colors['mse'])
                plt.title(f'MSE Loss (Fold {fold})')
        else:
            if isinstance(val_metrics_history, dict):
                if 'val_real_validity' in val_metrics_history and val_metrics_history['val_real_validity']:
                    plt.plot(val_metrics_history['val_real_validity'], label='Real Validity', color=colors['real_validity'])
                if 'val_fake_validity' in val_metrics_history and val_metrics_history['val_fake_validity']:
                    plt.plot(val_metrics_history['val_fake_validity'], label='Fake Validity', color=colors['fake_validity'])
            plt.title(f'Validation Metrics (Fold {fold})')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Sample Variance
        plt.subplot(133)
        if isinstance(val_metrics_history, dict) and 'sample_variance' in val_metrics_history and val_metrics_history['sample_variance']:
            plt.plot(val_metrics_history['sample_variance'], label='Sample Variance', color=colors['variance'])
        plt.title(f'Sample Variance (Fold {fold})')
        plt.xlabel('Epoch')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Update the display
        plt.draw()
        plt.pause(0.1)  # Longer pause to ensure display updates
        
        # Save if final epoch
        if is_final_epoch:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'training_progress_fold{fold}.png'))
            plt.close(fig)
            if fold in active_training_figures:
                del active_training_figures[fold]
    except Exception as e:
        logger.error(f"Error plotting training progress: {str(e)}")
        if fold in active_training_figures:
            fig, _ = active_training_figures[fold]
            plt.close(fig)
            del active_training_figures[fold] 