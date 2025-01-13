# gan_eval.py
"""
GAN Evaluation Module

This module evaluates GAN-generated data using a battery of metrics. We evaluate prediction accuracy but also quality.

Key Components:
--------------
1. Model Evaluation (evaluate_model)
   - Regression Metrics:
     * Mean Squared Error (MSE): Measures average squared difference between predictions and targets
     * Root Mean Squared Error (RMSE): Square root of MSE, provides error measure in original scale
     * Mean Absolute Error (MAE): Average absolute difference between predictions and targets
   
   - Distribution Comparison:
     * Kolmogorov-Smirnov Test: Assesses if predictions and targets come from same distribution
       - KS Statistic: Maximum distance between cumulative distribution functions
       - KS p-value: Statistical significance of distribution difference
     * Wasserstein Distance: Earth mover's distance between distributions
   
   - Correlation Analysis:
     * Spearman Correlation: Measures monotonic relationship between predictions and targets
     * Correlation p-value: Statistical significance of correlation

2. Result Aggregation (aggregate_and_save)
   - Combines results across all cross-validation folds
   - Calculates aggregate metrics using evaluate_model
   - Saves comprehensive results including:
     * All computed metrics
     * Raw predictions and targets
     * Results saved in both JSON and NumPy formats

Output Files:
------------
- test_results.json: Contains metrics and arrays in human-readable format
- test_predictions.npy: NumPy array of model predictions
- test_targets.npy: NumPy array of true target values

Dependencies:
------------
- NumPy: Array operations and numerical computations
- SciPy: Statistical tests (KS test, Spearman correlation)
- Scikit-learn: Regression metrics
- JSON: Results serialization

Usage:
------
1. For individual model evaluation:
   ```python
   metrics = evaluate_model(test_targets, test_predictions)
   ```

2. For aggregating results across folds:
   ```python
   metrics = aggregate_and_save(all_test_predictions, all_test_targets)
   ```

Note: All metrics are returned as float values to ensure JSON serialization compatibility.
"""
import numpy as np
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim  # Import ssim and ms_ssim
import logging

# Assuming RESULTS_DIR is defined in gan_settings.py, otherwise define it here
from gan_settings import RESULTS_DIR, LOGS_DIR, VARIANCE_THRESHOLD

# Set up logging
def setup_logging(output_dir):
    """Set up logging configuration."""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'gan_evaluation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

# Initialize logger
logger = setup_logging(LOGS_DIR)

def is_voxelwise(modality):
    """Determine if the modality is voxelwise or regional."""
    raw_3d_modalities = {'T1', 'fmri', 'fmrib', 'fa', 'md', 'cbf', 'lesion'}
    return modality in raw_3d_modalities

def compute_correlation_metric(predictions, targets):
    """Compute correlation between predicted and target images."""
    try:
        if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
            predictions = np.array(predictions)
            targets = np.array(targets)
            
        if predictions.size == 0 or targets.size == 0:
            return 0.0
            
        if len(predictions.shape) > len(targets.shape):
            correlations = []
            for pred in predictions:
                try:
                    if len(pred.shape) > 1 and len(targets.shape) > 1:
                        if pred.shape[0] != targets.shape[0]:
                            continue
                        pred_flat = pred.reshape(pred.shape[0], -1)
                        target_flat = targets.reshape(targets.shape[0], -1)
                    else:
                        pred_flat = pred.ravel()
                        target_flat = targets.ravel()
                    
                    batch_correlations = []
                    for p, t in zip(pred_flat, target_flat):
                        if np.all(np.isfinite(p)) and np.all(np.isfinite(t)):
                            try:
                                corr = pearsonr(p, t)[0]
                                if np.isfinite(corr):
                                    batch_correlations.append(corr)
                            except ValueError:
                                continue
                    
                    if batch_correlations:
                        correlations.append(np.mean(batch_correlations))
                        
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Warning in correlation computation: {str(e)}")
                    continue
                    
            return np.mean(correlations) if correlations else 0.0
        else:
            try:
                if len(predictions.shape) > 1 and len(targets.shape) > 1:
                    if predictions.shape[0] != targets.shape[0]:
                        return 0.0
                    predictions_flat = predictions.reshape(predictions.shape[0], -1)
                    targets_flat = targets.reshape(targets.shape[0], -1)
                else:
                    predictions_flat = predictions.ravel()
                    targets_flat = targets.ravel()
                
                correlations = []
                for p, t in zip(predictions_flat, targets_flat):
                    if np.all(np.isfinite(p)) and np.all(np.isfinite(t)):
                        try:
                            corr = pearsonr(p, t)[0]
                            if np.isfinite(corr):
                                correlations.append(corr)
                        except ValueError:
                            continue
                
                return np.mean(correlations) if correlations else 0.0
                
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Warning in correlation computation: {str(e)}")
                return 0.0
                
    except Exception as e:
        logger.error(f"Error in correlation metric computation: {str(e)}")
        return 0.0

def calculate_msssim(img1, img2, data_range=255, size_average=True, win_size=3):
    """Calculate MS-SSIM between two images."""
    try:
        # Convert to tensors if needed
        if not isinstance(img1, torch.Tensor):
            img1 = torch.tensor(img1)
        if not isinstance(img2, torch.Tensor):
            img2 = torch.tensor(img2)

        # Handle different input shapes
        if len(img1.shape) == 1:
            # For 1D inputs, reshape to square if possible
            size = int(np.sqrt(img1.shape[0]))
            if size * size != img1.shape[0]:
                # If not perfect square, pad to next perfect square
                next_square = int(np.ceil(np.sqrt(img1.shape[0]))) ** 2
                img1 = torch.nn.functional.pad(img1, (0, next_square - img1.shape[0]))
                img2 = torch.nn.functional.pad(img2, (0, next_square - img2.shape[0]))
                size = int(np.sqrt(next_square))
            
            img1 = img1.reshape(size, size)
            img2 = img2.reshape(size, size)

        # Verify minimum size requirements
        min_size = win_size + 1  # Minimum required size
        if img1.shape[0] < min_size or img1.shape[1] < min_size:
            # Upscale images if too small
            scale_factor = min_size / min(img1.shape[0], img1.shape[1])
            img1 = torch.nn.functional.interpolate(
                img1.unsqueeze(0).unsqueeze(0),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False
            ).squeeze()
            img2 = torch.nn.functional.interpolate(
                img2.unsqueeze(0).unsqueeze(0),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False
            ).squeeze()

        # Add batch and channel dimensions [H, W] -> [1, 1, H, W]
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        # Ensure images are float type
        img1 = img1.float()
        img2 = img2.float()
        
        # Normalize to [0, 1] range
        img1_min, img1_max = img1.min(), img1.max()
        img2_min, img2_max = img2.min(), img2.max()
        
        if img1_max - img1_min > 1e-8:
            img1 = (img1 - img1_min) / (img1_max - img1_min)
        else:
            logger.warning("First image has no variance")
            return float('nan')
            
        if img2_max - img2_min > 1e-8:
            img2 = (img2 - img2_min) / (img2_max - img2_min)
        else:
            logger.warning("Second image has no variance")
            return float('nan')

        # Calculate MS-SSIM
        ms_ssim_val = ms_ssim(img1, img2, data_range=1.0, size_average=size_average, win_size=win_size)
        
        return float(ms_ssim_val)
    except Exception as e:
        logger.error(f"Error in calculate_msssim: {str(e)}")
        logger.error(f"Input shapes - img1: {img1.shape if isinstance(img1, torch.Tensor) else 'not tensor'}, "
                    f"img2: {img2.shape if isinstance(img2, torch.Tensor) else 'not tensor'}")
        return float('nan')

def evaluate_model(targets, predictions, modality, mode):
    """Evaluates the model's performance using various metrics.
    
    Args:
        targets: Ground truth data
        predictions: Model predictions
        modality: The data modality (e.g., 'fmri', 'behavior')
        mode: The generation mode ('behavior_to_image' or 'image_to_behavior')
    """
    metrics = {}

    # Validate inputs
    if targets is None or predictions is None or len(targets) == 0 or len(predictions) == 0:
        logger.warning("Empty targets or predictions provided to evaluate_model")
        return {'error': 'Empty data'}

    # Check for mode collapse using variance threshold
    try:
        predictions_variance = np.var(predictions, axis=0).mean()
        metrics['variance'] = float(predictions_variance)
        metrics['mode_collapse'] = predictions_variance < VARIANCE_THRESHOLD
        if metrics['mode_collapse']:
            logger.warning(f"Mode collapse detected: variance {predictions_variance} below threshold {VARIANCE_THRESHOLD}")
    except Exception as e:
        logger.error(f"Error computing variance metrics: {str(e)}")
        metrics['variance'] = float('nan')
        metrics['mode_collapse'] = True

    # Calculate MS-SSIM only when generating images (behavior_to_image mode)
    if mode == 'behavior_to_image' and is_voxelwise(modality):
        try:
            msssim_values = []
            for idx, (t, p) in enumerate(zip(targets, predictions)):
                if t.size == 0 or p.size == 0:
                    logger.warning(f"Skipping empty array at index {idx}")
                    continue
                
                # Log shapes for debugging
                logger.debug(f"Processing pair {idx} - Target shape: {t.shape}, Prediction shape: {p.shape}")
                
                # Skip if arrays contain NaN or inf values
                if np.any(np.isnan(t)) or np.any(np.isnan(p)) or np.any(np.isinf(t)) or np.any(np.isinf(p)):
                    logger.warning(f"Skipping index {idx}: Contains NaN or inf values")
                    continue
                
                msssim_val = calculate_msssim(t, p)
                if not np.isnan(msssim_val):
                    msssim_values.append(msssim_val)
                    logger.debug(f"MS-SSIM for pair {idx}: {msssim_val}")
            
            if msssim_values:
                metrics['msssim'] = float(np.mean(msssim_values))
                logger.info(f"Final MS-SSIM (averaged over {len(msssim_values)} valid pairs): {metrics['msssim']}")
            else:
                metrics['msssim'] = float('nan')
                logger.warning("No valid MS-SSIM values calculated - all pairs were invalid")
                
        except Exception as e:
            logger.error(f"Error computing MS-SSIM: {str(e)}")
            metrics['msssim'] = float('nan')
    else:
        # Skip MS-SSIM for image_to_behavior mode or non-voxelwise modalities
        logger.debug(f"Skipping MS-SSIM calculation for mode={mode} and modality={modality}")
        metrics['msssim'] = float('nan')
    
    # Always calculate basic metrics (MSE, MAE) regardless of mode or data type
    try:
        valid_pairs = [(t, p) for t, p in zip(targets, predictions) 
                      if t.size > 0 and p.size > 0 and not np.any(np.isnan(t)) 
                      and not np.any(np.isnan(p)) and not np.any(np.isinf(t)) 
                      and not np.any(np.isinf(p))]
        
        if valid_pairs:
            t_valid, p_valid = zip(*valid_pairs)
            t_array = np.array(t_valid)
            p_array = np.array(p_valid)
            
            # Check for zero variance
            if np.all(t_array == t_array[0]) or np.all(p_array == p_array[0]):
                logger.warning("Target or prediction data has zero variance")
                metrics['mse'] = float('nan')
                metrics['mae'] = float('nan')
            else:
                metrics['mse'] = float(np.mean((t_array - p_array) ** 2))
                metrics['mae'] = float(np.mean(np.abs(t_array - p_array)))
                logger.info(f"Calculated metrics on {len(valid_pairs)} valid pairs: MSE = {metrics['mse']:.4f}, MAE = {metrics['mae']:.4f}")
        else:
            metrics['mse'] = float('nan')
            metrics['mae'] = float('nan')
            logger.warning("No valid pairs for computing metrics - all pairs contained invalid values")
            
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        metrics['mse'] = float('nan')
        metrics['mae'] = float('nan')
    
    return metrics

def aggregate_and_save(all_test_predictions, all_test_targets):
    """
    Aggregates test predictions and targets across all outer folds and saves them,
    including evaluation of KS statistics.
    """
    try:
        if not all_test_predictions or not all_test_targets:
            logger.warning("Warning: Empty prediction or target lists")
            return {'error': 'Empty prediction or target lists'}
        
        valid_predictions = []
        valid_targets = []
        for pred, target in zip(all_test_predictions, all_test_targets):
            if pred is not None and target is not None and pred.size > 0 and target.size > 0:
                valid_predictions.append(pred)
                valid_targets.append(target)
        
        if not valid_predictions or not valid_targets:
            logger.warning("Warning: No valid predictions or targets after filtering")
            return {'error': 'No valid predictions or targets'}
        
        test_predictions = np.concatenate(valid_predictions)
        test_targets = np.concatenate(valid_targets)
        
        metrics = evaluate_model(test_targets, test_predictions)
        
        results = {
            'metrics': metrics,
            'predictions': test_predictions.tolist(),
            'targets': test_targets.tolist()
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        results_file = os.path.join(RESULTS_DIR, 'test_results.json')
        predictions_file = os.path.join(RESULTS_DIR, 'test_predictions.npy')
        targets_file = os.path.join(RESULTS_DIR, 'test_targets.npy')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        np.save(predictions_file, test_predictions)
        np.save(targets_file, test_targets)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in aggregate_and_save: {str(e)}")
        return {'error': str(e)}
