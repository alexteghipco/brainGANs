import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import pandas as pd
from collections import defaultdict
from gan_settings import (
    BEHAVIORAL_STANDARDIZATION, IMAGING_STANDARDIZATION,
    MINMAX_RANGE, ROBUST_QUANTILE_RANGE, STANDARDIZE_FEATURES_JOINTLY,
    MODE, OUTPUT_DIR
)
import os
import pickle

logger = logging.getLogger(__name__)

# Global variable to store transform info
_transform_info = None

def reshape_data(X: np.ndarray) -> np.ndarray:
    """Reshape input data to the correct format"""
    logger.info(f"Reshaping data from shape: {X.shape}")
    
    # Only reshape if it's image data (more than 2 dimensions)
    if len(X.shape) > 2:
        # Handle different image formats
        if len(X.shape) == 3:  # (batch, height, width)
            logger.info("Reshaping 3D data")
            X = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 4:  # (batch, channels, height, width) or (batch, height, width, channels)
            logger.info("Reshaping 4D data")
            if X.shape[1] > X.shape[3]:  # Channels are in wrong dimension
                logger.info("Transposing 4D data")
                X = np.transpose(X, (0, 3, 1, 2))
            X = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 5:  # (batch, channels, depth, height, width)
            logger.info("Reshaping 5D data")
            X = X.reshape(X.shape[0], -1)
    elif len(X.shape) == 1:
        # Handle 1D data
        logger.info("Reshaping 1D data")
        X = X.reshape(-1, 1)
    
    logger.info(f"Final data shape: {X.shape}")
    return X

def validate_data_shapes(X: np.ndarray, y: np.ndarray) -> bool:
    """Validate input data shapes"""
    logger.info(f"Validating data shapes - X: {X.shape}, y: {y.shape}")
    
    # Allow higher dimensional data (images) in either X or y
    if len(X.shape) > 2 and len(y.shape) > 2:
        logger.error("Both X and y cannot be image data")
        return False
    
    if len(X.shape) > 5 or len(y.shape) > 5:
        logger.error(f"Data dimensions too high - X: {len(X.shape)}, y: {len(y.shape)}")
        return False
    
    if X.shape[0] != y.shape[0]:
        logger.error(f"Mismatched sample counts: X={X.shape[0]}, y={y.shape[0]}")
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.error("X contains NaN or infinite values")
        return False
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        logger.error("y contains NaN or infinite values")
        return False
    
    logger.info("Data shapes validation passed")
    return True

def preprocess_y_data(y: np.ndarray) -> np.ndarray:
    """Preprocess y data by handling empty strings and converting to float32"""
    logger.info("Preprocessing y data")
    
    # If y is already float type and has no NaN/inf values, return as is
    if np.issubdtype(y.dtype, np.floating):
        if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
            logger.info("y data is already properly formatted")
            return y.astype(np.float32)
    
    try:
        # Convert y data to pandas DataFrame for easier handling
        y_df = pd.DataFrame(y)
        
        # Replace empty strings, whitespace, and 'nan' strings with NaN
        y_df = y_df.replace(r'^\s*$', np.nan, regex=True)
        y_df = y_df.replace('nan', np.nan)
        
        # Convert all columns to numeric, coercing errors to NaN
        y_df = y_df.apply(pd.to_numeric, errors='coerce')
        
        # Calculate column means excluding NaN values
        column_means = y_df.mean(skipna=True)
        
        # Fill NaN with column means
        y_df = y_df.fillna(column_means)
        
        # Check for any remaining NaN values
        if y_df.isna().any().any():
            logger.error("Failed to fill all NaN values in y data")
            raise ValueError("Failed to fill all NaN values")
        
        logger.info("Successfully preprocessed y data")
        return y_df.values.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error preprocessing y data: {str(e)}")
        raise ValueError(f"Failed to preprocess y data: {str(e)}")

def validate_input_data(
    X: np.ndarray,
    y: np.ndarray,
    c: Optional[np.ndarray] = None
) -> bool:
    """Validate input data for shape, type, and values"""
    try:
        # Check for None
        if X is None or y is None:
            logger.error("Input data cannot be None")
            return False
        
        # Convert to float32 if needed
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        # Check shapes
        if len(X.shape) < 1 or len(y.shape) < 1:
            logger.error("Input data must have at least one dimension")
            return False
        
        if X.shape[0] != y.shape[0]:
            logger.error(f"Number of samples mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
            return False
        
        # Check for NaN and inf values
        if np.isnan(X).any() or np.isinf(X).any():
            logger.error("X contains NaN or inf values")
            return False
        
        if np.isnan(y).any() or np.isinf(y).any():
            logger.error("y contains NaN or inf values")
            return False
        
        # Validate conditional data if present
        if c is not None:
            if c.dtype != np.float32:
                c = c.astype(np.float32)
            
            if c.shape[0] != X.shape[0]:
                logger.error(f"Number of samples mismatch: X has {X.shape[0]}, c has {c.shape[0]}")
                return False
            
            if np.isnan(c).any() or np.isinf(c).any():
                logger.error("c contains NaN or inf values")
                return False
        
        logger.info("Input data validation passed")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}, c shape: {getattr(c, 'shape', None)}")
        return True
    
    except Exception as e:
        logger.error(f"Error during data validation: {str(e)}")
        return False

def debug_data(
    X: np.ndarray,
    y: np.ndarray,
    c: Optional[Union[np.ndarray, List]] = None,
    prefix: str = ""
) -> None:
    """Print debug information about data arrays"""
    logger = logging.getLogger(__name__)
    
    try:
        if X.dtype.kind in ['i', 'f']:  # Only show range for numeric data
            logger.debug(f"{prefix}X range: [{X.min():.3f}, {X.max():.3f}]")
        logger.debug(f"{prefix}X shape: {X.shape}")
        logger.debug(f"{prefix}X dtype: {X.dtype}")
        
        if y.dtype.kind in ['i', 'f']:  # Only show range for numeric data
            logger.debug(f"{prefix}y range: [{y.min():.3f}, {y.max():.3f}]")
        logger.debug(f"{prefix}y shape: {y.shape}")
        logger.debug(f"{prefix}y dtype: {y.dtype}")
        
        if c is not None:
            if isinstance(c, np.ndarray) and c.dtype.kind in ['i', 'f']:
                logger.debug(f"{prefix}c range: [{c.min():.3f}, {c.max():.3f}]")
            logger.debug(f"{prefix}c shape: {c.shape if isinstance(c, np.ndarray) else len(c)}")
            logger.debug(f"{prefix}c dtype: {c.dtype if isinstance(c, np.ndarray) else type(c)}")
    except Exception as e:
        logger.warning(f"Error in debug_data: {str(e)}")

class GANDataset(Dataset):
    """Dataset class for GAN training"""
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray] = None,
        transform=None
    ):
        """Initialize dataset with input data, targets, and optional conditions"""
        logger.info(f"Creating dataset with shapes - X: {X.shape}, y: {y.shape}, c: {getattr(c, 'shape', None)}")
        
        # Convert input data to float32 tensors
        try:
            # Ensure X is 2D
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            self.X = torch.FloatTensor(X)
            
            # Ensure y is 2D
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            self.y = torch.FloatTensor(y)
            
            logger.info(f"Reshaped tensors - X: {self.X.shape}, y: {self.y.shape}")
        except Exception as e:
            logger.error(f"Error converting data to tensors: {str(e)}")
            raise ValueError("Failed to convert data to tensors")
        
        # Handle conditional data
        self.c = None
        if c is not None:
            try:
                if isinstance(c, (np.ndarray, list)):
                    c = np.array(c, dtype=np.float32)
                    if len(c.shape) == 1:
                        c = c.reshape(-1, 1)
                    self.c = torch.FloatTensor(c)
                    logger.info(f"Reshaped conditions: {self.c.shape}")
                else:
                    logger.error(f"Unexpected condition type: {type(c)}")
                    raise ValueError("Invalid condition type")
            except Exception as e:
                logger.error(f"Error converting conditions to tensor: {str(e)}")
                raise ValueError("Failed to convert conditions to tensor")
        
        self.transform = transform
        logger.info("Dataset initialization completed")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get data
        X = self.X[idx]
        y = self.y[idx]
        
        # Apply transform if available
        if self.transform:
            X = self.transform(X)
        
        # Return with or without conditions
        if self.c is not None:
            return y, X, self.c[idx]  # Note: order is (real_samples, conditions, c)
        return y, X  # Note: order is (real_samples, conditions)

def standardize_data(
    train_data: np.ndarray,
    other_data: List[np.ndarray],
    is_behavioral: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Standardize data using only training data statistics to avoid leakage.
    
    Args:
        train_data: Training data to compute statistics from
        other_data: List of other data arrays to standardize using training statistics
        is_behavioral: Whether this is behavioral data (True) or imaging data (False)
        
    Returns:
        Tuple of (standardized_train_data, list_of_standardized_other_data)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\nStandardizing {'behavioral' if is_behavioral else 'imaging'} data:")
    logger.info(f"Initial train_data type: {train_data.dtype}, shape: {train_data.shape}")
    for i, data in enumerate(other_data):
        logger.info(f"Initial other_data[{i}] type: {data.dtype}, shape: {data.shape}")
    
    # Convert data to float32 and handle any string data
    try:
        train_data = train_data.astype(np.float32)
        other_data = [data.astype(np.float32) for data in other_data]
        logger.info("Successfully converted data to float32 directly")
    except ValueError as e:
        logger.warning(f"Direct float32 conversion failed: {str(e)}")
        # If conversion fails, try to handle string data
        if isinstance(train_data, np.ndarray) and train_data.dtype.kind in ['U', 'S']:
            logger.info("Detected string data, attempting conversion through pandas")
            # Convert string data to numeric, replacing empty strings with NaN
            train_data = pd.DataFrame(train_data).apply(pd.to_numeric, errors='coerce').values
            other_data = [pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').values for data in other_data]
            
            # Fill NaN values with 0
            train_data = np.nan_to_num(train_data, nan=0.0)
            other_data = [np.nan_to_num(data, nan=0.0) for data in other_data]
            
            # Convert to float32
            train_data = train_data.astype(np.float32)
            other_data = [data.astype(np.float32) for data in other_data]
            logger.info("Successfully converted string data to float32 through pandas")
        else:
            raise ValueError(f"Failed to convert data to numeric format: {str(e)}")
    
    logger.info(f"Final train_data type: {train_data.dtype}, shape: {train_data.shape}")
    for i, data in enumerate(other_data):
        logger.info(f"Final other_data[{i}] type: {data.dtype}, shape: {data.shape}")

    standardization_method = BEHAVIORAL_STANDARDIZATION if is_behavioral else IMAGING_STANDARDIZATION
    
    if standardization_method is None:
        return train_data, other_data
    
    if standardization_method == 'per_sample' and not is_behavioral:
        # Standardize each sample independently
        train_standardized = (train_data - train_data.mean(axis=1, keepdims=True)) / (train_data.std(axis=1, keepdims=True) + 1e-8)
        other_standardized = [(data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8) for data in other_data]
        return train_standardized, other_standardized
    
    # For joint standardization of imaging data
    if not is_behavioral and STANDARDIZE_FEATURES_JOINTLY:
        train_data = train_data.reshape(train_data.shape[0], -1)
        other_data = [data.reshape(data.shape[0], -1) for data in other_data]
    
    if standardization_method == 'standard':
        # Standard scaling (zero mean, unit variance)
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        # Handle constant features
        train_std = np.where(train_std == 0, 1, train_std)
        
        train_standardized = (train_data - train_mean) / train_std
        other_standardized = [(data - train_mean) / train_std for data in other_data]
    
    elif standardization_method == 'minmax':
        # MinMax scaling
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        train_range = train_max - train_min
        # Handle constant features
        train_range = np.where(train_range == 0, 1, train_range)
        
        scale = MINMAX_RANGE[1] - MINMAX_RANGE[0]
        train_standardized = MINMAX_RANGE[0] + scale * (train_data - train_min) / train_range
        other_standardized = [MINMAX_RANGE[0] + scale * (data - train_min) / train_range for data in other_data]
    
    elif standardization_method == 'robust':
        # Robust scaling using quantiles
        q_low, q_high = np.percentile(train_data, ROBUST_QUANTILE_RANGE, axis=0)
        train_iqr = q_high - q_low
        # Handle constant features
        train_iqr = np.where(train_iqr == 0, 1, train_iqr)
        
        train_standardized = (train_data - q_low) / train_iqr
        other_standardized = [(data - q_low) / train_iqr for data in other_data]
    
    else:
        raise ValueError(f"Unknown standardization method: {standardization_method}")
    
    # Reshape back for imaging data if needed
    if not is_behavioral and STANDARDIZE_FEATURES_JOINTLY:
        original_shape = list(train_data.shape)
        original_shape[1:] = [-1]  # Flatten all dimensions except the first
        train_standardized = train_standardized.reshape(original_shape)
        other_standardized = [data.reshape(original_shape) for data in other_standardized]
    
    return train_standardized, other_standardized

def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    c: Optional[np.ndarray] = None,
    batch_size: int = 32,
    train_val_split: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders with proper standardization"""
    try:
        # Set seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Clean data first
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split indices if validation is requested
        if train_val_split < 1.0:
            n_samples = len(X_clean)
            n_train = int(n_samples * train_val_split)
            indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
        else:
            train_indices = np.arange(len(X_clean))
            val_indices = np.array([])
            
        logger.info(f"Split sizes - train: {len(train_indices)}, val: {len(val_indices)}")
        
        # Standardize X data using only training data statistics
        X_train = X_clean[train_indices]
        other_X = [X_clean[val_indices]] if len(val_indices) > 0 else []
        X_train_std, X_other_std = standardize_data(X_train, other_X)
        
        # Standardize y data using only training data statistics
        y_train = y_clean[train_indices]
        other_y = [y_clean[val_indices]] if len(val_indices) > 0 else []
        y_train_std, y_other_std = standardize_data(y_train, other_y)
        
        # Handle conditional data if present
        c_train_std = None
        c_val_std = None
        if c is not None:
            c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            c_train = c_clean[train_indices]
            other_c = [c_clean[val_indices]] if len(val_indices) > 0 else []
            c_train_std, c_other_std = standardize_data(c_train, other_c)
            if len(val_indices) > 0:
                c_val_std = c_other_std[0]
        
        # Create datasets
        train_dataset = GANDataset(
            X_train_std,
            y_train_std,
            c_train_std,
            transform=None
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_indices)),
            shuffle=shuffle,
            pin_memory=True
        )
        
        if len(val_indices) > 0:
            val_dataset = GANDataset(
                X_other_std[0],
                y_other_std[0],
                c_val_std,
                transform=None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(batch_size, len(val_indices)),
                shuffle=False,
                pin_memory=True
            )
        else:
            val_loader = None
        
        logger.info("Dataloaders created successfully with standardization")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        raise ValueError(f"Failed to create dataloaders: {str(e)}")

def load_transform_info(output_dir: str, modality: str = None) -> Dict[str, Any]:
    """Load transform info from pickle file.
    
    Args:
        output_dir: Output directory containing transforms/transform_info.pkl
        modality: Optional modality key to load specific transform info
        
    Returns:
        Dictionary containing transform info
    """
    transform_path = os.path.join(output_dir, 'transforms', 'transform_info.pkl')
    if os.path.exists(transform_path):
        with open(transform_path, 'rb') as f:
            transform_info = pickle.load(f)
            logger.info(f"Loaded transform info from {transform_path}")
            
            # If a specific modality is requested, try to get it
            if modality is not None:
                if modality in transform_info:
                    return {modality: transform_info[modality]}
                else:
                    # Try to find first available modality
                    available_modalities = [k for k in transform_info.keys() if k != 'default']
                    if available_modalities:
                        first_modality = available_modalities[0]
                        logger.warning(f"No transform info found for modality {modality}, using first available: {first_modality}")
                        return {first_modality: transform_info[first_modality]}
                    elif 'default' in transform_info:
                        logger.warning(f"No transform info found for modality {modality}, using default")
                        return {'default': transform_info['default']}
            
            return transform_info
            
    return {}

def get_brain_mask(transform_info: Dict[str, Any], modality: str = None) -> Optional[np.ndarray]:
    """Get brain mask from transform info.
    
    Args:
        transform_info: Dictionary containing transform info
        modality: Optional modality to get brain mask for
        
    Returns:
        Brain mask as boolean array or None if not found
    """
    # If transform_info is a nested dictionary with modalities
    if modality is not None and modality in transform_info:
        brain_mask = transform_info[modality].get('brain_mask')
    else:
        # Try to get brain mask directly from transform_info
        brain_mask = transform_info.get('brain_mask')
    
    if brain_mask is not None:
        brain_mask = brain_mask.astype(bool)
        logger.info(f"Loaded brain mask with {brain_mask.sum()} valid features")
        return brain_mask
    return None

def apply_brain_mask(data: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply brain mask to data.
    
    Args:
        data: Data array of shape (n_samples, depth, height, width) or (n_samples, n_features)
        brain_mask: Optional boolean mask to apply
        
    Returns:
        Masked data array of shape (n_samples, n_features)
    """
    if brain_mask is None:
        # If no mask, flatten data if needed
        if len(data.shape) > 2:
            return data.reshape(data.shape[0], -1)
        return data
    
    # If data is not flattened, flatten it first
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    # Verify mask matches data
    if brain_mask.shape[0] != data.shape[1]:
        raise ValueError(f"Brain mask shape {brain_mask.shape} doesn't match data shape {data.shape}")
    
    # Apply mask
    return data[:, brain_mask]

def reconstruct_volume(data: np.ndarray, brain_mask: Optional[np.ndarray], original_shape: Tuple[int, int, int]) -> np.ndarray:
    """Reconstruct 3D volume from masked data.
    
    Args:
        data: Data array of shape (n_samples, n_features)
        brain_mask: Boolean mask used to mask the data
        original_shape: Original 3D shape (depth, height, width)
        
    Returns:
        Reconstructed volume of shape (n_samples, channels=1, height, width, depth)
    """
    try:
        n_samples = data.shape[0]
        total_voxels = np.prod(original_shape)
        
        logger.info(f"Reconstructing volume - Input shape: {data.shape}, Target shape: {original_shape}")
        
        if brain_mask is None:
            raise ValueError("Brain mask is required for reconstruction")
        
        # Verify mask matches expected size
        if brain_mask.size != total_voxels:
            raise ValueError(f"Brain mask size ({brain_mask.size}) doesn't match expected size ({total_voxels})")
        
        # Verify data features match valid voxels in mask
        n_valid_voxels = np.sum(brain_mask)
        if data.shape[1] != n_valid_voxels:
            raise ValueError(f"Data features ({data.shape[1]}) don't match number of valid voxels in mask ({n_valid_voxels})")
        
        # Initialize full volumes with zeros
        full_data = np.zeros((n_samples, total_voxels))
        
        # Fill in the masked values
        full_data[:, brain_mask] = data
        
        # Reshape to 4D (n_samples, depth, height, width)
        reconstructed = full_data.reshape(n_samples, *original_shape)
        
        # Add channel dimension and transpose to match expected shape (n_samples, channels=1, height, width, depth)
        reconstructed = np.expand_dims(reconstructed, axis=1)  # Add channel dimension after batch
        reconstructed = np.transpose(reconstructed, (0, 1, 3, 4, 2))  # Move depth to last dimension
        
        logger.info(f"Successfully reconstructed volume with shape: {reconstructed.shape}")
        return reconstructed
        
    except Exception as e:
        logger.error(f"Error in reconstruct_volume: {str(e)}")
        logger.error(f"Input shapes - data: {data.shape}, mask: {brain_mask.shape if brain_mask is not None else None}")
        logger.error(f"Target shape: {original_shape}")
        logger.error("Stack trace:", exc_info=True)
        raise 

def save_transform_info(brain_mask, orig_shape, downsample_factor=None, value_range=None, modality_info=None):
    """Save brain mask and original shape for later reconstruction.
    
    Args:
        brain_mask (np.ndarray): Boolean array indicating valid voxels
        orig_shape (tuple): Original shape of data (x, y, z)
        downsample_factor (int): Optional downsampling factor
        value_range (tuple): Optional (min, max) values of original data
        modality_info (dict): Optional dictionary containing modality and feature information
    """
    # Create transform info dictionary
    transform_info = {
        'brain_mask': brain_mask,
        'orig_shape': orig_shape,
        'downsample_factor': downsample_factor,
        'value_range': value_range
    }
    
    logger.info("Saving transform info:")
    logger.info(f"  Brain mask shape: {brain_mask.shape}")
    logger.info(f"  Original shape: {orig_shape}")
    logger.info(f"  Downsample factor: {downsample_factor}")
    if value_range is not None:
        logger.info(f"  Value range: [{value_range[0]:.3f}, {value_range[1]:.3f}]")
    if modality_info is not None:
        logger.info(f"  Modality: {modality_info['modality']}.{modality_info['feature']}")
    
    # Cache the transform info both in memory and on disk
    global _transform_info
    
    # Initialize _transform_info as a dict of modalities if it doesn't exist
    if _transform_info is None:
        _transform_info = {}
    
    # Store transform info by modality
    if modality_info is not None:
        # Use the full modality string as the key (e.g., 'fmri.dat')
        modality_key = f"{modality_info['modality']}.{modality_info['feature']}"
        _transform_info[modality_key] = transform_info
    else:
        # Store as default if no modality info provided
        _transform_info['default'] = transform_info
    
    # Create transforms directory if it doesn't exist
    transforms_dir = os.path.join(OUTPUT_DIR, 'transforms')
    os.makedirs(transforms_dir, exist_ok=True)
    
    # Save to output directory
    transform_path = os.path.join(transforms_dir, 'transform_info.pkl')
    try:
        with open(transform_path, 'wb') as f:
            pickle.dump(_transform_info, f)
        logger.info(f"Successfully saved transform info to {transform_path}")
    except Exception as e:
        logger.error(f"Error saving transform info: {str(e)}")
        raise 