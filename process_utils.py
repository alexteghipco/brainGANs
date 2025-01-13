import torch
import gc
from typing import Dict, List, Optional, Tuple
import logging
from multiprocessing import Process
import os
import numpy as np

logger = logging.getLogger(__name__)

def get_device_for_fold(fold_idx: int, num_gpus: int) -> torch.device:
    """Get the appropriate device for a given fold"""
    if torch.cuda.is_available() and num_gpus > 0:
        gpu_idx = fold_idx % num_gpus
        return torch.device(f'cuda:{gpu_idx}')
    return torch.device('cpu')

def clear_gpu_memory() -> None:
    """Clear GPU memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_fold_wrapper(args: Tuple) -> None:
    """Wrapper function for running a single fold in a separate process"""
    try:
        (
            fold_idx, train_val_idx, test_idx, device,
            X_np, y_np, c_np, all_test_predictions,
            all_test_targets, modality, mode, global_seed,
            error_dict, all_folds_data
        ) = args
        
        logger.info(f"Starting fold {fold_idx} on device {device}")
        logger.info(f"Train/val indices shape: {train_val_idx.shape}")
        logger.info(f"Test indices shape: {test_idx.shape}")
        logger.info(f"X shape: {X_np.shape}")
        logger.info(f"y shape: {y_np.shape}")
        if c_np is not None:
            logger.info(f"c shape: {np.array(c_np).shape}")
        
        # Import here to avoid circular imports
        from gan_train import run_fold
        
        try:
            run_fold(
                fold_idx, train_val_idx, test_idx, device,
                X_np, y_np, c_np, all_test_predictions,
                all_test_targets, modality, mode, global_seed,
                error_dict, all_folds_data
            )
        except Exception as e:
            logger.error(f"Error in fold {fold_idx}:", exc_info=True)
            if error_dict is not None:
                error_dict[fold_idx] = str(e)
            raise
        
    except Exception as e:
        logger.error(f"Error in fold wrapper {fold_idx}:", exc_info=True)
        if error_dict is not None:
            error_dict[fold_idx] = str(e)
        raise

def run_fold_with_recovery(fold_idx: int, *args: Tuple) -> Optional[None]:
    """Run a fold with error recovery"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Clear GPU memory before each attempt
            clear_gpu_memory()
            
            # Run the fold
            run_fold_wrapper((fold_idx, *args))
            return None
            
        except RuntimeError as e:
            # Handle CUDA out of memory errors specially
            if "out of memory" in str(e):
                logger.error(f"CUDA out of memory in fold {fold_idx}. Attempting recovery...")
                clear_gpu_memory()  # Clear GPU memory immediately
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Force CUDA memory cleanup
            
            retry_count += 1
            logger.warning(
                f"Fold {fold_idx} failed (attempt {retry_count}/{max_retries}): {str(e)}"
            )
            
            if retry_count == max_retries:
                logger.error(f"Fold {fold_idx} failed after {max_retries} attempts")
                raise
            
            # Exponential backoff between retries
            import time
            wait_time = 10 * (2 ** (retry_count - 1))  # 10s, 20s, 40s
            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        
        except Exception as e:
            # For non-CUDA errors, increment retry counter and log
            retry_count += 1
            logger.warning(
                f"Fold {fold_idx} failed with error (attempt {retry_count}/{max_retries}): {str(e)}"
            )
            
            if retry_count == max_retries:
                logger.error(f"Fold {fold_idx} failed after {max_retries} attempts")
                raise
            
            # Standard wait time for non-CUDA errors
            import time
            time.sleep(10)
    
    return None

def cleanup_processes(processes: List[Process]) -> None:
    """Clean up multiprocessing resources"""
    if processes:
        for p in processes:
            if p.is_alive():
                logger.warning(f"Terminating process {p.pid}")
                p.terminate()
                p.join()

def setup_process_logging(output_dir: str, fold_idx: int) -> None:
    """Set up logging for a process"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(output_dir, f'fold_{fold_idx}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger = logging.getLogger()
    logger.addHandler(file_handler)

def get_available_gpus() -> List[int]:
    """Get list of available GPU devices"""
    if not torch.cuda.is_available():
        return []
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(i):
                torch.cuda.memory_allocated()
            available_gpus.append(i)
        except Exception:
            continue
    
    return available_gpus

def distribute_folds_to_gpus(
    n_folds: int,
    available_gpus: List[int]
) -> Dict[int, int]:
    """Distribute folds across available GPUs"""
    if not available_gpus:
        return {i: -1 for i in range(n_folds)}  # -1 indicates CPU
    
    fold_to_gpu = {}
    for fold_idx in range(n_folds):
        gpu_idx = available_gpus[fold_idx % len(available_gpus)]
        fold_to_gpu[fold_idx] = gpu_idx
    
    return fold_to_gpu 