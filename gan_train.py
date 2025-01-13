import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp
import torch.multiprocessing as tmp
from datetime import datetime
import optuna
from multiprocessing import Process, Manager
from sklearn.model_selection import KFold
import pandas as pd
from torchvision.utils import save_image
import shutil
import gc

from data_utils import (
    create_dataloaders,
    load_transform_info,
    get_brain_mask,
    apply_brain_mask,
    reconstruct_volume,
    save_transform_info
)
from visualization import (
    save_average_test_images,
    plot_training_progress,
    save_generated_data_matrix,
    save_comparison_images
)
from process_utils import (
    get_device_for_fold,
    clear_gpu_memory,
    run_fold_wrapper,
    run_fold_with_recovery,
    cleanup_processes,
    setup_process_logging,
    get_available_gpus,
    distribute_folds_to_gpus
)
from gan_arch import (
    Generator,
    Discriminator
)
from gan_settings import (
    OUTPUT_DIR,
    OUTER_FOLDS,
    INNER_FOLDS,
    MAT_DIR,
    DEBUG_MODE,
    DEBUG_SAMPLES,
    HYPERPARAMETER_SEARCH_SPACE,
    MODALITIES,
    TARGET_VARIABLES,
    G_PATIENCE,
    D_PATIENCE,
    G_MIN_DELTA,
    D_MIN_DELTA,
    MIN_EPOCHS,
    G_WARMUP_EPOCHS,
    D_WARMUP_EPOCHS,
    WARMUP_LR_INIT,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MIN_LR,
    LAMBDA_GP,
    N_CRITIC_WGAN,
    N_CRITIC_OTHER,
    CSV_PATH,
    MODE,
    BATCH_SIZE,
    INNER_FOLD_EARLY_STOPPING_SPLIT,
    OUTER_FOLD_EARLY_STOPPING_SPLIT,
    PLOT_INTERVAL,
    save_settings,
    CACHE_DIR,
    TRANSFORM_CACHE_PATH,
    CLEAN_PREVIOUS_FILES,
    CLEAN_CACHE,
    RAW_3D_MODALITIES
)
from hyperparameter_optimization import (
    optimize_hyperparameters
)
from seed_manager import (
    set_global_seed
)
from gan_trainer import GANTrainer
from model_utils import (
    init_weights, print_model_info, save_checkpoint,
    load_checkpoint, validate_models
)
from data_utils import (
    reshape_data, validate_input_data, debug_data,
    create_dataloaders, GANDataset, standardize_data
)
from visualization import (
    plot_training_progress, save_comparison_images, save_average_test_images
)
from process_utils import (
    get_device_for_fold,
    clear_gpu_memory,
    run_fold_wrapper,
    run_fold_with_recovery,
    cleanup_processes,
    setup_process_logging,
    get_available_gpus,
    distribute_folds_to_gpus
)
from gan_arch import Generator, Discriminator
from concat_mat_beh import extract_modalities

from gan_settings import (
    OUTPUT_DIR, OUTER_FOLDS, INNER_FOLDS,
    MAT_DIR, DEBUG_MODE, DEBUG_SAMPLES,
    HYPERPARAMETER_SEARCH_SPACE, MODALITIES,
    TARGET_VARIABLES, G_PATIENCE, D_PATIENCE,
    G_MIN_DELTA, D_MIN_DELTA, MIN_EPOCHS,
    G_WARMUP_EPOCHS, D_WARMUP_EPOCHS, WARMUP_LR_INIT,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
    MIN_LR, LAMBDA_GP, N_CRITIC_WGAN, N_CRITIC_OTHER,
    CSV_PATH, MODE, BATCH_SIZE,
    INNER_FOLD_EARLY_STOPPING_SPLIT, OUTER_FOLD_EARLY_STOPPING_SPLIT,
    PLOT_INTERVAL, save_settings
)

from hyperparameter_optimization import optimize_hyperparameters
from seed_manager import set_global_seed

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_best_trial(study: optuna.Study) -> optuna.Trial:
    """Get the best trial from an Optuna study"""
    return study.best_trial

def create_dataloader(X, y, c=None, batch_size=32, shuffle=True):
    """Create a dataloader + standardization."""
    X_std = standardize_data(X, [], is_behavioral=MODE=="image_to_behavior")[0]
    y_std = standardize_data(y, [], is_behavioral=MODE=="behavior_to_image")[0]
    
    # Conditional data is iffy...not debugged yet...
    c_std = None
    if c is not None:
        c_std = standardize_data(c, [], is_behavioral=True)[0]
    
    return DataLoader(
        GANDataset(X_std, y_std, c_std),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

def run_fold(
    outer_fold: int,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    X_np: np.ndarray,
    y_np: np.ndarray,
    c_np: Optional[Union[np.ndarray, List]],
    all_test_predictions: List[Optional[np.ndarray]],
    all_test_targets: List[Optional[np.ndarray]],
    modality: str,
    mode: str,
    global_seed: int,
    error_dict: Dict[int, str],
    all_folds_data: Optional[Dict[str, List[np.ndarray]]] = None,
    trial_config: Optional[Dict[str, Union[int, float, str]]] = None
) -> None:
    """Run a single fold of the training process"""
    try:
        # fold-specific output dirs
        os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'results'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'average_images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
        
        logger.info(f"Starting fold {outer_fold}")
        setup_process_logging(OUTPUT_DIR, outer_fold)
        
        # Fold seed based on global...
        fold_seed = global_seed + outer_fold
        
        # Set all random seeds for this fold...
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(fold_seed)
            torch.cuda.manual_seed_all(fold_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Check device again
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            # Current device
            current_device = int(str(device).split(':')[1])
            torch.cuda.set_device(current_device)
            logger.info(f"Using CUDA device {current_device}")
            
            # Clear memory to be safe
            torch.cuda.empty_cache()
        
        # First split: Separate early stopping data from the rest
        early_stopping_split = KFold(n_splits=int(1/OUTER_FOLD_EARLY_STOPPING_SPLIT), shuffle=True, random_state=fold_seed).split(X_np[train_val_idx])
        remaining_idx, early_stopping_idx = next(early_stopping_split)
        early_stopping_idx = train_val_idx[early_stopping_idx]
        remaining_data = train_val_idx[remaining_idx]
        
        # For hyperparameter optimization, split remaining data into train and validation
        train_val_split = KFold(n_splits=int(1/INNER_FOLD_EARLY_STOPPING_SPLIT), shuffle=True, random_state=fold_seed).split(X_np[remaining_data])
        train_idx, val_idx = next(train_val_split)
        
        # Convert indices to integer and map to remaining_data indices
        train_idx = np.asarray(remaining_data[train_idx], dtype=np.int64)
        val_idx = np.asarray(remaining_data[val_idx], dtype=np.int64)
        early_stopping_idx = np.asarray(early_stopping_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        
        logger.info(f"Initial split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Early stopping: {len(early_stopping_idx)}, Test: {len(test_idx)}")
        
        # default trial config iun case not provided
        if trial_config is None:
            trial_config = {
                'batch_size': BATCH_SIZE,
                'learning_rate_base': 0.0002,
                'learning_rate_offset': 1.0,
                'beta1': 0.5,
                'beta2': 0.999,
                'z_dim': 100,
                'hidden_dim': 256,
                'hidden_layers': 3,
                'activation': 'ReLU',
                'dropout_rate': 0.3,
                'disc_dropout_rate': 0.3,
                'use_self_attention': True,
                'loss_function': 'wgan',
                'conditioning_strategy': 'concat'
            }
            logger.info("Using default trial configuration")
        
        # dataloaders
        if len(train_idx) < trial_config['batch_size']:
            logger.warning(f"Training set size ({len(train_idx)}) is smaller than batch size ({trial_config['batch_size']}). Reducing batch size.")
            batch_size = max(1, len(train_idx) // 2)  # Ensure at least 1 sample per batch
        else:
            batch_size = trial_config['batch_size']

        # Ensure we have enough samples for validation...(e.g., if debug can be issue esp.)
        if len(val_idx) == 0:
            logger.warning("No validation samples. Using a small portion of training data for validation.")
            val_size = max(1, len(train_idx) // 5)  # Use 20% of training data for validation
            val_idx = train_idx[-val_size:]
            train_idx = train_idx[:-val_size]

        # Extract training data for standardization
        X_train = X_np[train_idx]
        y_train = y_np[train_idx]
        c_train = c_np[train_idx] if c_np is not None else None

        # dataloaders
        train_loader = create_dataloader(
            X_train, y_train,
            c_train if c_np is not None else None,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = create_dataloader(
            X_np[val_idx], y_np[val_idx],
            c_np[val_idx] if c_np is not None else None,
            batch_size=batch_size,
            shuffle=False
        )

        early_stopping_loader = create_dataloader(
            X_np[early_stopping_idx], y_np[early_stopping_idx],
            c_np[early_stopping_idx] if c_np is not None else None,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Run hyperparameter optimization
        if trial_config is None:
            logger.info("Starting hyperparameter optimization...")
            trial_config = optimize_hyperparameters(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                fold_seed=fold_seed,
                is_test=DEBUG_MODE
            )
            logger.info(f"Best hyperparameters found: {trial_config}")
        
        # Create test loader with standardization based on training data statistics
        X_test = X_np[test_idx]
        y_test = y_np[test_idx]
        _, [X_test_std] = standardize_data(X_train, [X_test], is_behavioral=mode=="image_to_behavior")
        _, [y_test_std] = standardize_data(y_train, [y_test], is_behavioral=mode=="behavior_to_image")
        
        # Handle conditional data for test set...if present
        c_test_std = None
        if c_np is not None:
            c_test = c_np[test_idx]
            _, [c_test_std] = standardize_data(c_train, [c_test], is_behavioral=True)
        
        test_loader = DataLoader(
            GANDataset(X_test_std, y_test_std, c_test_std if c_np is not None else None),
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True
        )
        
        # Calculate input and output dimensions based on mode
        if MODE == "behavior_to_image":
            x_dim = len(TARGET_VARIABLES)  # Behavioral features (input)
            
            # Get actual image shape from data
            if len(y_np.shape) > 2:  # If multi-dimensional
                output_shape = y_np.shape[1:]  # Get shape excluding batch dimension
                y_dim = np.prod(output_shape)  # Flattened size
            else:  # If already flattened
                y_dim = y_np.shape[1]
            
            logger.info(f"Mode: behavior_to_image - Input (behavior) dim: {x_dim}, Output (image) dim: {y_dim}")
            logger.info(f"Image shape: {output_shape if len(y_np.shape) > 2 else 'flattened'}, flattened size: {y_dim}")
            
            # get progressive hidden dimensions based on input/output sizes
            input_size = trial_config['z_dim'] + x_dim
            output_size = y_dim
            
            hidden_dims = []
            num_layers = trial_config['hidden_layers']
            for i in range(num_layers):
                # Calculate size using log progression
                alpha = (i + 1) / num_layers
                size = int(np.exp(
                    (1 - alpha) * np.log(input_size) + alpha * np.log(output_size)
                ))
                hidden_dims.append(size)
            
            logger.info(f"Calculated hidden dimensions: {hidden_dims}")
            
            # Initialize generator and discriminator
            generator = Generator(
                z_dim=trial_config['z_dim'],
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dims=hidden_dims,  # Dynamic hidden dimensions
                activation=trial_config['activation'],
                dropout_rate=trial_config['dropout_rate'],
                use_self_attention=trial_config['use_self_attention'],
                loss_function=trial_config['loss_function'],
                conditioning_strategy=trial_config['conditioning_strategy'],
                c_dim=0  # Set to 0 since we're not using conditional data...
            )
            
            discriminator = Discriminator(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dims=hidden_dims[::-1],  # Reverse the dimensions for discriminator
                activation=trial_config['activation'],
                dropout_rate=trial_config['disc_dropout_rate'],
                use_self_attention=trial_config['use_self_attention'],
                loss_function=trial_config['loss_function'],
                conditioning_strategy=trial_config['conditioning_strategy'],
                c_dim=0  # Set to 0 since we're not using conditional data
            )
        else:  # image_to_behavior
            x_dim = np.prod(X_np.shape[1:])  # Flattened image
            y_dim = y_np.shape[1] if len(y_np.shape) > 1 else 1  # Behavioral features
            logger.info(f"Mode: image_to_behavior - Using image data as input and behavioral data as target")
            
            # Use standard hidden dimensions for behavior generation
            hidden_dims = [trial_config['hidden_dim']] * trial_config['hidden_layers']
            
        logger.info(f"Hidden dimensions: {hidden_dims}")
        
        # Initialize models with optimized hyperparameters and move to device
        generator = Generator(
            z_dim=trial_config['z_dim'],
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            activation=trial_config['activation'],
            dropout_rate=trial_config['dropout_rate'],
            use_self_attention=trial_config['use_self_attention'],
            loss_function=trial_config['loss_function'],
            conditioning_strategy=trial_config['conditioning_strategy'],
            c_dim=0  # Set to 0 since we're not using conditional data
        )
        
        discriminator = Discriminator(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims[::-1],  # Reverse dimensions for discriminator
            activation=trial_config['activation'],
            dropout_rate=trial_config['disc_dropout_rate'],
            use_self_attention=trial_config['use_self_attention'],
            loss_function=trial_config['loss_function'],
            conditioning_strategy=trial_config['conditioning_strategy'],
            c_dim=0  # Set to 0 since we're not using conditional data
        )
        
        # Initialize weights before moving to device
        generator.apply(lambda m: init_weights(m, seed=fold_seed))
        discriminator.apply(lambda m: init_weights(m, seed=fold_seed))
        
        # Move models to device and turn on gradient computation
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        generator.train()
        discriminator.train()
        
        print_model_info(generator, "Generator")
        print_model_info(discriminator, "Discriminator")
        
        # Create trainer with optimized hyperparameters
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=early_stopping_loader,
            learning_rate=trial_config['learning_rate_base'],
            disc_learning_rate=trial_config['learning_rate_base'] * trial_config['learning_rate_offset'],
            beta1=trial_config['beta1'],
            beta2=trial_config['beta2'],
            n_critic=N_CRITIC_WGAN if trial_config['loss_function'] == 'wgan' else N_CRITIC_OTHER,
            lambda_gp=LAMBDA_GP
        )
            
        # Training loop with early stopping and learning rate scheduling
        best_val_loss = float('inf')
        val_metrics_history = {
            'val_real_validity': [],
            'val_fake_validity': [],
            'sample_variance': []
        }
        
        mode_collapse_counter = 0  # Track consecutive mode collapses
        
        for epoch in range(MIN_EPOCHS):
            logger.info(f"Starting epoch {epoch}")
            train_metrics = trainer.train_epoch(epoch)
            
            if early_stopping_loader:
                val_metrics = validate_models(generator, discriminator, early_stopping_loader, device, mode=MODE)
                val_metrics_history['val_real_validity'].append(val_metrics['val_real_validity'])
                val_metrics_history['val_fake_validity'].append(val_metrics['val_fake_validity'])
                val_metrics_history['sample_variance'].append(val_metrics.get('sample_variance', 0.0))
                if 'val_mse' not in val_metrics_history:
                    val_metrics_history['val_mse'] = []
                val_metrics_history['val_mse'].append(val_metrics.get('val_mse', 0.0))
                
                # Use different validation metrics based on mode
                if MODE == "image_to_behavior":
                    val_loss = val_metrics['val_mse']  # Use MSE as validation loss
                else:
                    val_loss = -val_metrics['val_real_validity']  # Original validation loss
                
                logger.info(f"Epoch {epoch} - val_loss: {val_loss:.4f}, variance: {val_metrics.get('sample_variance', 0.0):.6f}")
                if MODE == "image_to_behavior":
                    logger.info(f"MSE: {val_metrics['val_mse']:.4f}")
                else:  # behavior_to_image mode
                    logger.info(f"Real Validity: {val_metrics['val_real_validity']:.4f}, Fake Validity: {val_metrics['val_fake_validity']:.4f}")
                
                # Plot training progress
                plot_training_progress(
                    epoch=epoch,
                    d_losses=train_metrics['d_losses'],
                    g_losses=train_metrics['g_losses'],
                    val_metrics_history=val_metrics_history,
                    fold=outer_fold,
                    output_dir=os.path.join(OUTPUT_DIR, 'plots'),
                    mode=MODE,
                    is_final_epoch=(epoch == MIN_EPOCHS - 1)  # True on last epoch
                )
                
                # Check for mode collapse
                if val_metrics.get('mode_collapse', False):
                    mode_collapse_counter += 1
                    logger.warning(f"Mode collapse detected in epoch {epoch} (count: {mode_collapse_counter})")
                    
                    # If mode collapse persists for too long, stop training
                    if mode_collapse_counter >= 5:  # Allow up to 5 consecutive mode collapses
                        logger.error("Persistent mode collapse detected. Stopping training.")
                        raise Exception("Training stopped due to persistent mode collapse")
                else:
                    mode_collapse_counter = 0  # Reset counter if no mode collapse
                
                # Save checkpoint if validation improves and no mode collapse
                if val_loss < best_val_loss and not val_metrics.get('mode_collapse', False):
                    best_val_loss = val_loss
                    save_checkpoint(
                        generator,
                        trainer.g_optimizer,
                        epoch,
                        val_loss,
                        os.path.join('checkpoints', f'generator_fold{outer_fold}_best.pt'),
                        OUTPUT_DIR
                    )
                    save_checkpoint(
                        discriminator,
                        trainer.d_optimizer,
                        epoch,
                        val_loss,
                        os.path.join('checkpoints', f'discriminator_fold{outer_fold}_best.pt'),
                        OUTPUT_DIR
                    )
        
        # Plot and save final training progress at the end of the fold
        plot_training_progress(
            epoch,
            train_metrics['d_losses'],
            train_metrics['g_losses'],
            val_metrics_history,
            outer_fold,
            os.path.join(OUTPUT_DIR, 'plots'),
            save_plot=True,  # Save the final plot
            mode=MODE
        )
        
        # If in behavior_to_image mode, generate and save test images
        if MODE == "behavior_to_image":
            logger.info("Starting test image generation in behavior_to_image mode")
            generator.eval()
            all_generated_images = []
            all_true_images = []
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    logger.info(f"Processing test batch {batch_idx}")
                    real_samples, conditions = batch_data
                    batch_size = conditions.size(0)
                    z = torch.randn(batch_size, trial_config['z_dim']).to(device)
                    
                    # Generate fake images
                    fake_samples = generator(z, conditions.to(device))
                    logger.info(f"Generated fake samples for batch {batch_idx}, shape: {fake_samples.shape}")
                    
                    # Store results
                    all_generated_images.append(fake_samples.cpu().numpy())
                    all_true_images.append(real_samples.cpu().numpy())
                    
                    # Save individual test samples
                    results_dir = os.path.join(OUTPUT_DIR, 'results', f'fold_{outer_fold}', f'batch_{batch_idx}')
                    logger.info(f"Creating results directory: {results_dir}")
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Save batch data
                    logger.info(f"Saving test samples to {results_dir}")
                    np.save(os.path.join(results_dir, 'true_samples.npy'), real_samples.cpu().numpy())
                    np.save(os.path.join(results_dir, 'generated_samples.npy'), fake_samples.cpu().numpy())
                    np.save(os.path.join(results_dir, 'conditions.npy'), conditions.cpu().numpy())
                    
                    logger.info(f"Saved test samples for fold {outer_fold}, batch {batch_idx}")
                
                # Concatenate all batches
                logger.info("Concatenating batches...")
                generated_images = np.concatenate(all_generated_images, axis=0)
                true_images = np.concatenate(all_true_images, axis=0)
                logger.info(f"Concatenated shapes - generated: {generated_images.shape}, true: {true_images.shape}")
                
                # Initialize all_folds_data if not already initialized
                if 'true_data' not in all_folds_data:
                    all_folds_data['true_data'] = [None] * OUTER_FOLDS
                if 'generated_data' not in all_folds_data:
                    all_folds_data['generated_data'] = [None] * OUTER_FOLDS
                
                # Store data for this fold in the shared list
                logger.info(f"Storing data for fold {outer_fold} in shared list")
                all_folds_data['true_data'][outer_fold] = true_images
                all_folds_data['generated_data'][outer_fold] = generated_images

                # Save the matrix of generated data
                results_dir = os.path.join(OUTPUT_DIR, 'results')
                logger.info(f"Saving generated data matrix to {results_dir}")
                
                # Check if this is the final fold
                is_final_fold = outer_fold == OUTER_FOLDS - 1
                
                # Save matrices
                save_generated_data_matrix(
                    true_data=true_images,
                    generated_data=generated_images,
                    fold=outer_fold,
                    output_dir=results_dir,
                    mode=MODE,
                    is_final_fold=is_final_fold,
                    all_folds_data=all_folds_data if is_final_fold else None
                )

                # Perform image reconstruction for brain image generation mode
                logger.info("Starting image reconstruction...")
                # Get original image shape
                if len(y_np.shape) > 2:
                    original_shape = y_np.shape[1:]  # Get shape excluding batch dimension
                else:
                    # If data is flattened, get shape from transform info
                    transform_info = load_transform_info(OUTPUT_DIR)
                    if transform_info is None or not isinstance(transform_info, dict):
                        raise ValueError("Could not find transform info")
                    
                    # First try to get info from modality key
                    modality_info = transform_info.get(modality)
                    if modality_info is None:
                        # If not found, try using the first available modality
                        available_modalities = [k for k in transform_info.keys() if k != 'default']
                        if available_modalities:
                            modality = available_modalities[0]
                            modality_info = transform_info[modality]
                            logger.warning(f"No transform info found for modality {modality}, using first available: {modality}")
                        else:
                            # Try using default transform info
                            modality_info = transform_info.get('default')
                            if modality_info is None:
                                raise ValueError(f"Could not find transform info for modality: {modality}")
                    
                    # Get the original shape from either nested or flat structure
                    if isinstance(modality_info, dict):
                        original_shape = modality_info.get('orig_shape')
                        brain_mask = modality_info.get('brain_mask')
                    else:
                        raise ValueError(f"Invalid transform info structure for modality: {modality}")
                    
                    if original_shape is None:
                        raise ValueError(f"Could not find original shape for modality: {modality}")
                    
                    # Convert to tuple if it's not already
                    if not isinstance(original_shape, tuple):
                        try:
                            original_shape = tuple(original_shape)  # Convert to tuple if not already
                        except Exception as e:
                            logger.error(f"Error converting original shape to tuple: {str(e)}")
                            raise ValueError(f"Could not convert original shape to tuple: {original_shape}")
                    
                    if brain_mask is None:
                        raise ValueError(f"Could not find brain mask for modality: {modality}")

                    logger.info(f"Using transform info for modality: {modality}")
                    logger.info(f"Original shape: {original_shape}")
                    logger.info(f"Brain mask shape: {brain_mask.shape if brain_mask is not None else None}")
                    
                    # Verify brain mask matches the data
                    if brain_mask.sum() != true_images.shape[1]:
                        logger.error(f"Brain mask features ({brain_mask.sum()}) don't match data features ({true_images.shape[1]})")
                        raise ValueError("Brain mask and data features mismatch")
                
                if len(original_shape) != 3:
                    raise ValueError(f"Expected 3D shape but got shape with {len(original_shape)} dimensions: {original_shape}")
                
                logger.info(f"Using 3D shape {original_shape} for visualization")
                logger.info(f"Generated images shape: {generated_images.shape}")
                logger.info(f"True images shape: {true_images.shape}")
                
                # Calculate average of generated images for comparison
                average_generated = np.mean(generated_images, axis=0)
                
                # Save individual comparisons with average generated image
                for subject_idx in range(len(true_images)):
                    try:
                        logger.info(f"Processing subject {subject_idx}")
                        logger.info(f"Input shapes - true_images: {true_images.shape}, generated_images: {generated_images.shape}")
                        
                        # Reconstruct 3D volumes for visualization
                        true_3d = reconstruct_volume(true_images[subject_idx:subject_idx+1], brain_mask, original_shape)
                        generated_3d = reconstruct_volume(generated_images[subject_idx:subject_idx+1], brain_mask, original_shape)
                        avg_generated_3d = reconstruct_volume(average_generated.reshape(1, -1), brain_mask, original_shape)
                        
                        # Save transform info for future use
                        save_transform_info(brain_mask, original_shape, modality_info={'modality': modality.split('.')[0], 'feature': modality.split('.')[1]})

                        logger.info(f"Reconstructed shapes:")
                        logger.info(f"  true_3d: {true_3d.shape}")
                        logger.info(f"  generated_3d: {generated_3d.shape}")
                        logger.info(f"  avg_generated_3d: {avg_generated_3d.shape}")
                        
                        # Add channel dimension and transpose to match expected shape (N, C, H, W, D)
                        true_3d = np.expand_dims(true_3d, axis=1)  # Add channel dimension
                        generated_3d = np.expand_dims(generated_3d, axis=1)
                        avg_generated_3d = np.expand_dims(avg_generated_3d, axis=1)
                        
                        logger.info(f"Final shapes after adding channel dimension:")
                        logger.info(f"  true_3d: {true_3d.shape}")
                        logger.info(f"  generated_3d: {generated_3d.shape}")
                        logger.info(f"  avg_generated_3d: {avg_generated_3d.shape}")
                        
                        # Save comparison
                        save_comparison_images(
                            original_data=true_3d,
                            generated_data=generated_3d,
                            subject_idx=subject_idx,  # Changed from 0 to subject_idx
                            output_dir=results_dir,
                            average_generated=avg_generated_3d
                        )
                        logger.info(f"Successfully saved comparison for subject {subject_idx}")
                        
                    except Exception as e:
                        logger.error(f"Error processing subject {subject_idx}: {str(e)}")
                        logger.error(f"Stack trace:", exc_info=True)
                        continue
                
                # Save average images
                try:
                    # Ensure average_images directory exists
                    average_images_dir = os.path.join(OUTPUT_DIR, 'average_images')
                    os.makedirs(average_images_dir, exist_ok=True)
                    
                    save_average_test_images(
                        true_images=true_images,
                        generated_images=generated_images,
                        fold=outer_fold,
                        output_dir=average_images_dir,
                        original_shape=original_shape,
                        brain_mask=brain_mask,
                        mode=mode
                    )
                    logger.info(f"Successfully saved average test images for fold {outer_fold}")
                except Exception as e:
                    logger.error(f"Error saving average test images for fold {outer_fold}: {str(e)}")
                    raise
        
        logger.info(f"Fold {outer_fold} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in fold {outer_fold}: {str(e)}", exc_info=True)
        if error_dict is not None:
            error_dict[outer_fold] = str(e)
        raise
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def clean_previous_files():
    """Clean up previous output files (not cached data)."""
    # List of directories to clean
    dirs_to_clean = [
        os.path.join(OUTPUT_DIR, 'checkpoints'),
        os.path.join(OUTPUT_DIR, 'results'),
        os.path.join(OUTPUT_DIR, 'average_images'),
        os.path.join(OUTPUT_DIR, 'plots'),
        os.path.join(OUTPUT_DIR, 'logs')
    ]
    
    # Clean each directory
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            print(f"Cleaning directory: {dir_path}")
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

def clean_cache():
    """Clean up cached data files."""
    if os.path.exists(CACHE_DIR):
        print(f"Cleaning cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    if os.path.exists(os.path.dirname(TRANSFORM_CACHE_PATH)):
        print(f"Cleaning transforms directory: {os.path.dirname(TRANSFORM_CACHE_PATH)}")
        shutil.rmtree(os.path.dirname(TRANSFORM_CACHE_PATH))

def load_and_preprocess_data():
    """Load and preprocess data from CSV and MAT files"""
    # Load behavioral data
    beh_df = pd.read_csv(CSV_PATH)
    if DEBUG_MODE:
        beh_df = beh_df.head(DEBUG_SAMPLES)
    
    # Extract modalities
    raw_X, raw_y, included_subjects = extract_modalities(
        beh_df=beh_df,
        mat_dir=MAT_DIR,
        modalities=MODALITIES,
        target=TARGET_VARIABLES
    )
    
    # Set c_np to None since we don't have conditional data
    c_np = None
    
    return raw_X, raw_y, c_np

def main() -> None:
    manager = None
    try:
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'results'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'average_images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'transforms'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'cache'), exist_ok=True)

        # Set up logging
        logger = setup_logging(OUTPUT_DIR)
        logger.info("Starting GAN training")
        logger.info(f"Running in {MODE} mode")
        
        # Clean previous files if requested
        if CLEAN_PREVIOUS_FILES:
            clean_previous_files()
        
        # Clean cache if requested
        if CLEAN_CACHE:
            clean_cache()
        
        # Save settings
        save_settings()
        
        # Set global seed
        global_seed = set_global_seed()
        logger.info(f"Using global seed: {global_seed}")
        
        # Load and preprocess data
        raw_X, raw_y, c_np = load_and_preprocess_data()
        
        logger.info(f"Sample values - raw_X[0,0]: {raw_X[0,0]}, raw_y[0,0]: {raw_y[0,0]}")

        # Swap X and y if needed based on MODE
        if MODE == "behavior_to_image":
            X_np = raw_y  # behavioral data becomes input
            y_np = raw_X  # image data becomes target
            logger.info("Mode: behavior_to_image - Using behavioral data as input and image data as target")
        else:  # image_to_behavior
            X_np = raw_X  # image data becomes input
            y_np = raw_y  # behavioral data becomes target
            logger.info("Mode: image_to_behavior - Using image data as input and behavioral data as target")
            if c_np is not None:
                logger.warning("Conditional data is not supported in general, but especially not for image_to_behavior mode. Setting to None.")
                c_np = None

        # Debug logging
        logger.info(f"Data types after swap - X_np: {X_np.dtype}, y_np: {y_np.dtype}")
        logger.info(f"Sample values - X_np[0,0]: {X_np[0,0]}, y_np[0,0]: {y_np[0,0]}")

        if not validate_input_data(X_np, y_np, c_np):
            raise ValueError("Invalid input data")
        
        X_np = reshape_data(X_np)
        debug_data(X_np, y_np, c_np, "Initial data: ")
        
        # Get available GPUs
        available_gpus = get_available_gpus()
        logger.info(f"Available GPUs: {available_gpus}")
        
        # Initialize manager for shared resources
        manager = Manager()
        error_dict = manager.dict()
        all_test_predictions = manager.list([None] * OUTER_FOLDS)
        all_test_targets = manager.list([None] * OUTER_FOLDS)
        
        # Add tracking for generated and true data across folds
        all_folds_data = {
            'true_data': manager.list([None] * OUTER_FOLDS),
            'generated_data': manager.list([None] * OUTER_FOLDS)
        }

        # Create cross-validation splits with global seed
        kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=global_seed)

        # Run folds sequentially
        logger.info("Starting sequential fold processing")
        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(X_np)):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Processing fold {fold_idx} on device {device}")
            
            try:
                run_fold(
                    fold_idx,
                    train_val_idx,
                    test_idx,
                    device,
                    X_np,
                    y_np,
                    c_np,
                    all_test_predictions,
                    all_test_targets,
                    f"{MODALITIES[0]['modality']}.{MODALITIES[0]['feature']}",  # Get modality from settings
                    MODE,  # Use MODE from settings instead of hardcoded 'train'
                    global_seed,
                    error_dict,
                    all_folds_data
                )
                logger.info(f"Fold {fold_idx} completed successfully")
            except Exception as e:
                logger.error(f"Error in fold {fold_idx}: {str(e)}", exc_info=True)
                error_dict[fold_idx] = str(e)
                continue
            
            # Clear GPU memory between folds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Check for failed folds
        failed_folds = [i for i, msg in error_dict.items()]
        if failed_folds:
            logger.error(f"Folds {failed_folds} failed")
            if len(failed_folds) == OUTER_FOLDS:
                raise RuntimeError("All folds failed")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise

    finally:
        if manager:
            manager.shutdown()
        clear_gpu_memory()

if __name__ == "__main__":
    print("Starting main execution...")  # Debug print
    try:
        print("Setting up multiprocessing...")  # Debug print
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing setup complete...")  # Debug print
        
        main()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
