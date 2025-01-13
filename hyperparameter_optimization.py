import optuna
import numpy as np
import torch
import logging
from typing import Dict, Optional, Union, Tuple, List
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from gan_arch import Generator, Discriminator
from model_utils import init_weights, validate_models
from data_utils import create_dataloaders, GANDataset
from gan_trainer import GANTrainer
from gan_settings import (
    # Hyperparameter search settings
    HYPERPARAMETER_SEARCH_SPACE, N_TRIALS, INNER_FOLDS,
    # Model architecture settings
    USE_BATCH_NORM, USE_SELF_ATTENTION,
    # Training settings
    BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, MIN_EPOCHS,
    # Early stopping settings
    G_PATIENCE, D_PATIENCE, G_MIN_DELTA, D_MIN_DELTA,
    # Learning rate settings
    G_WARMUP_EPOCHS, D_WARMUP_EPOCHS, WARMUP_LR_INIT,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, MIN_LR,
    # Attention settings
    USE_ATTENTION_WARMUP, ATTENTION_WARMUP_EPOCHS,
    ATTENTION_WARMUP_FACTOR, ATTENTION_FINAL_FACTOR,
    # Loss function settings
    LAMBDA_GP, LAMBDA_MSE, N_CRITIC_WGAN, N_CRITIC_OTHER,
    # Gradient settings
    GRADIENT_CLIP_NORM, CLIP_VALUE,
    # Mode settings
    MODE
)

logger = logging.getLogger(__name__)

def get_default_hyperparameters() -> Dict[str, Union[int, float, str]]:
    """Return default hyperparameters if optimization fails"""
    return {
        'hidden_layers': HYPERPARAMETER_SEARCH_SPACE['hidden_layers'][1],  # middle value
        'hidden_dim': HYPERPARAMETER_SEARCH_SPACE['hidden_dim'][1],  # middle value
        'activation': HYPERPARAMETER_SEARCH_SPACE['activation'][0],  # first value
        'z_dim': HYPERPARAMETER_SEARCH_SPACE['z_dim'][0],  # first value
        'dropout_rate': sum(HYPERPARAMETER_SEARCH_SPACE['dropout_rate'])/2,  # middle value
        'disc_dropout_rate': sum(HYPERPARAMETER_SEARCH_SPACE['disc_dropout_rate'])/2,  # middle value
        'use_self_attention': HYPERPARAMETER_SEARCH_SPACE['use_self_attention'][0],  # only value
        'loss_function': HYPERPARAMETER_SEARCH_SPACE['loss_function'][0],  # only value
        'learning_rate_base': sum(HYPERPARAMETER_SEARCH_SPACE['learning_rate_base'])/2,  # middle value
        'learning_rate_offset': sum(HYPERPARAMETER_SEARCH_SPACE['learning_rate_offset'])/2,  # middle value
        'beta1': sum(HYPERPARAMETER_SEARCH_SPACE['beta1'])/2,  # middle value
        'beta2': HYPERPARAMETER_SEARCH_SPACE['beta2'][0],  # first value
        'conditioning_strategy': HYPERPARAMETER_SEARCH_SPACE['conditioning_strategy'][0]  # only value
    }

def optimize_hyperparameters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fold_seed: int,
    is_test: bool = False
) -> Dict[str, Union[int, float, str]]:
    """Run hyperparameter optimization using validation data"""
    
    # Use reduced settings if testing
    n_trials = 2 if is_test else N_TRIALS
    
    # Create and configure the study with a more lenient pruner
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=fold_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2 if is_test else 5,
            n_warmup_steps=5 if is_test else 20,
            interval_steps=5 if is_test else 10
        )
    )
    
    try:
        # Run optimization using the validation set for evaluation
        study.optimize(
            lambda trial: objective(
                trial,
                train_loader,
                val_loader,
                device,
                study,
                is_test
            ),
            n_trials=n_trials,
            timeout=None,
            catch=(Exception,)
        )
        
        # Return best parameters if any trial completed successfully
        if study.best_trial:
            return study.best_params
        else:
            logger.warning("No successful trials, using default hyperparameters")
            return get_default_hyperparameters()
            
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        return get_default_hyperparameters()

def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    study: optuna.Study,
    is_test: bool = False
) -> float:
    """Objective function for Optuna hyperparameter optimization"""
    try:
        # Sample hyperparameters from search space
        hidden_layers = trial.suggest_int('hidden_layers', 
            min(HYPERPARAMETER_SEARCH_SPACE['hidden_layers']), 
            max(HYPERPARAMETER_SEARCH_SPACE['hidden_layers']))
        hidden_dim = trial.suggest_categorical('hidden_dim', 
            HYPERPARAMETER_SEARCH_SPACE['hidden_dim'])
        activation = trial.suggest_categorical('activation', 
            HYPERPARAMETER_SEARCH_SPACE['activation'])
        z_dim = trial.suggest_categorical('z_dim', 
            HYPERPARAMETER_SEARCH_SPACE['z_dim'])
        dropout_rate = trial.suggest_float('dropout_rate', 
            min(HYPERPARAMETER_SEARCH_SPACE['dropout_rate']), 
            max(HYPERPARAMETER_SEARCH_SPACE['dropout_rate']))
        disc_dropout_rate = trial.suggest_float('disc_dropout_rate', 
            min(HYPERPARAMETER_SEARCH_SPACE['disc_dropout_rate']), 
            max(HYPERPARAMETER_SEARCH_SPACE['disc_dropout_rate']))
        use_self_attention = trial.suggest_categorical('use_self_attention', 
            HYPERPARAMETER_SEARCH_SPACE['use_self_attention'])
        loss_function = trial.suggest_categorical('loss_function', 
            HYPERPARAMETER_SEARCH_SPACE['loss_function'])
        learning_rate_base = trial.suggest_float('learning_rate_base', 
            min(HYPERPARAMETER_SEARCH_SPACE['learning_rate_base']), 
            max(HYPERPARAMETER_SEARCH_SPACE['learning_rate_base']))
        learning_rate_offset = trial.suggest_float('learning_rate_offset', 
            min(HYPERPARAMETER_SEARCH_SPACE['learning_rate_offset']), 
            max(HYPERPARAMETER_SEARCH_SPACE['learning_rate_offset']))
        beta1 = trial.suggest_float('beta1', 
            min(HYPERPARAMETER_SEARCH_SPACE['beta1']), 
            max(HYPERPARAMETER_SEARCH_SPACE['beta1']))
        beta2 = trial.suggest_categorical('beta2', 
            HYPERPARAMETER_SEARCH_SPACE['beta2'])
        conditioning_strategy = trial.suggest_categorical('conditioning_strategy',
            HYPERPARAMETER_SEARCH_SPACE['conditioning_strategy'])

        # Create hidden dimensions list
        hidden_dims = [hidden_dim] * hidden_layers

        # Get sample batch to determine dimensions
        X_sample, y_sample = next(iter(train_loader))
        
        # Calculate dimensions based on mode
        if MODE == "behavior_to_image":
            x_dim = y_sample.shape[1]  # Behavioral features
            y_dim = X_sample.shape[1]  # Flattened image
            logger.info(f"Mode: behavior_to_image - x_dim: {x_dim}, y_dim: {y_dim}")
        else:  # image_to_behavior
            x_dim = X_sample.shape[1]  # Flattened image
            y_dim = y_sample.shape[1]  # Behavioral features
            logger.info(f"Mode: image_to_behavior - x_dim: {x_dim}, y_dim: {y_dim}")

        # Initialize models with proper settings
        generator = Generator(
            z_dim=z_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            use_self_attention=use_self_attention,
            loss_function=loss_function,
            conditioning_strategy=conditioning_strategy
        ).to(device)

        discriminator = Discriminator(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims[::-1],
            activation=activation,
            dropout_rate=disc_dropout_rate,
            use_self_attention=use_self_attention,
            loss_function=loss_function,
            conditioning_strategy=conditioning_strategy
        ).to(device)

        generator.apply(init_weights)
        discriminator.apply(init_weights)

        # Create trainer with all settings
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate_base,
            beta1=beta1,
            beta2=beta2,
            n_critic=N_CRITIC_WGAN if loss_function == 'wgan' else N_CRITIC_OTHER,
            lambda_gp=LAMBDA_GP
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        min_epochs_completed = False
        
        # Use fewer epochs in test mode
        n_epochs = 10 if is_test else NUM_EPOCHS
        
        # Learning rate scheduling setup
        def get_warmup_lr(epoch, base_lr, warmup_epochs, warmup_lr_init):
            if epoch >= warmup_epochs:
                return base_lr
            return warmup_lr_init + (base_lr - warmup_lr_init) * epoch / warmup_epochs
        
        # Attention warmup setup
        if USE_ATTENTION_WARMUP:
            def get_attention_lr_factor(epoch):
                if epoch >= ATTENTION_WARMUP_EPOCHS:
                    return ATTENTION_FINAL_FACTOR
                return ATTENTION_WARMUP_FACTOR + (ATTENTION_FINAL_FACTOR - ATTENTION_WARMUP_FACTOR) * epoch / ATTENTION_WARMUP_EPOCHS
        
        for epoch in range(n_epochs):
            # Apply learning rate warmup
            current_g_lr = get_warmup_lr(epoch, learning_rate_base, G_WARMUP_EPOCHS, WARMUP_LR_INIT)
            current_d_lr = get_warmup_lr(epoch, learning_rate_base * learning_rate_offset, D_WARMUP_EPOCHS, WARMUP_LR_INIT)
            
            # Update learning rates
            for param_group in trainer.g_optimizer.param_groups:
                param_group['lr'] = current_g_lr
            for param_group in trainer.d_optimizer.param_groups:
                param_group['lr'] = current_d_lr
            
            # Apply attention warmup if enabled
            if USE_ATTENTION_WARMUP:
                attention_factor = get_attention_lr_factor(epoch)
                for name, param in generator.named_parameters():
                    if 'attention' in name:
                        for param_group in trainer.g_optimizer.param_groups:
                            if param in param_group['params']:
                                param_group['lr'] = current_g_lr * attention_factor
                for name, param in discriminator.named_parameters():
                    if 'attention' in name:
                        for param_group in trainer.d_optimizer.param_groups:
                            if param in param_group['params']:
                                param_group['lr'] = current_d_lr * attention_factor
            
            # Apply gradient clipping during training
            for param in generator.parameters():
                torch.nn.utils.clip_grad_norm_(param, GRADIENT_CLIP_NORM)
            for param in discriminator.parameters():
                torch.nn.utils.clip_grad_norm_(param, GRADIENT_CLIP_NORM)
                
            trainer.train_epoch(epoch)
            
            if val_loader:
                val_metrics = validate_models(generator, discriminator, val_loader, device)
                val_loss = -val_metrics['val_real_validity']
                
                # Report value to Optuna
                trial.report(val_loss, epoch)
                
                # Only consider pruning after minimum epochs and if loss is significantly worse
                if epoch >= (2 if is_test else MIN_EPOCHS):
                    min_epochs_completed = True
                    if trial.should_prune() and val_loss > 2 * study.best_value:
                        raise optuna.TrialPruned()
                
                # Early stopping logic with settings
                if val_loss < best_val_loss - G_MIN_DELTA:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Only stop if we've completed minimum epochs
                if min_epochs_completed and patience_counter >= (2 if is_test else G_PATIENCE):
                    break

        return best_val_loss

    except Exception as e:
        logger.error(f"Trial failed: {str(e)}")
        raise optuna.TrialPruned() 