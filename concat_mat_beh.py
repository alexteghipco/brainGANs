import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os
import glob
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from gan_settings import *
from scipy.ndimage import zoom
from scipy.io import loadmat
import pickle
import logging
from gan_settings import TARGET_VARIABLES

logger = logging.getLogger(__name__)

def get_voxel_size_from_header(hdr):
    """Extract voxel dimensions from fMRI header matrix.
    
    Args:
        hdr: Header information containing transformation matrix
        
    Returns:
        list: Voxel dimensions [x, y, z] in mm
    """
    if not hasattr(hdr, 'mat'):
        return FALLBACK_VOXEL_SIZE
        
    # Extract scaling factors from the tsf matrix
    mat = hdr.mat
    voxel_size = [
        abs(float(mat[0,0])),  # x 
        abs(float(mat[1,1])),  # y 
        abs(float(mat[2,2]))   # z 
    ]
    
    return voxel_size

def downsample_brain_image(image, voxel_size=None, downsample_factor=DOWNSAMPLE_FACTOR,
                          interpolation=DOWNSAMPLE_INTERPOLATION):
    """Downsample a brain image.
    
    Args:
        image (np.ndarray): Input image
        voxel_size (list, optional): Original voxel dimensions [x,y,z] in mm
        downsample_factor (float): Factor by which to downsample. Set to 1 to disable downsampling.
        interpolation (str): Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        np.ndarray: Downsampled image
    """
    if downsample_factor == 1:
        return image
    
    # Calculate target shape for downsampling using floor division
    current_shape = np.array(image.shape)
    target_shape = (current_shape // downsample_factor).astype(int)
    
    # Ensure minimum size of 1 in each dimension
    target_shape = np.maximum(target_shape, 1)
    
    # Convert interpolation method string to order parameter
    order_dict = {'nearest': 0, 'linear': 1, 'cubic': 3}
    order = order_dict.get(interpolation, 1)
    
    # Calculate exact zoom factors to match target shape
    zoom_factors = target_shape.astype(float) / current_shape
    
    # Perform downsampling
    downsampled = zoom(image, zoom_factors, order=order)
    
    return downsampled

def is_voxelwise(data):
    """Check if data contains voxelwise images (not regional data)"""
    if isinstance(data, dict) and 'hdr' in data:
        # Check if data matches voxelwise dimensions
        img_data = data['data']
        return len(img_data.shape) == 4  # (n_samples, x, y, z)
    if isinstance(data, np.ndarray):
        # For arrays, check if it's 4D with spatial dimensions
        return len(data.shape) == 4  # (n_samples, x, y, z)
    return False

def get_image_dims(data):
    """Get image dimensions from the data.
    
    Args:
        data: Input data (dict with header or numpy array)
    
    Returns:
        tuple: Image dimensions (x, y, z)
    """
    if isinstance(data, dict) and 'data' in data:
        # Get dims from the actual image data
        img_data = data['data']
        if len(img_data.shape) == 4:  # (n_samples, x, y, z)
            return img_data.shape[1:]
        elif len(img_data.shape) == 3:  # Single image (x, y, z)
            return img_data.shape
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 4:  # (n_samples, x, y, z)
            return data.shape[1:]
        elif len(data.shape) == 3:  # Single image (x, y, z)
            return data.shape
    
    raise ValueError("Could not determine image dimensions from data")

def create_brain_mask(data):
    """Create a mask based on empty voxel/ROI threshold.
    Removes any voxel/ROI where more than EMPTY_VOXEL_THRESHOLD proportion of subjects have a 0 value.
    
    Args:
        data (np.ndarray): Array of shape (n_samples, features) where features can be voxels or ROIs
    
    Returns:
        np.ndarray: Boolean mask of shape (features,) indicating valid voxels/ROIs
        tuple: Original data shape
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Get original shape before any reshaping
    orig_shape = data.shape[1:] if len(data.shape) > 2 else (data.shape[1],)
    
    # Reshape to 2D if needed (n_samples, features)
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    # Count proportion of zeros for each feature
    zero_proportion = np.mean(data == 0, axis=0)
    
    # Create mask - keep features that are non-zero in enough subjects
    mask = zero_proportion < EMPTY_VOXEL_THRESHOLD
    
    # Print mask statistics
    n_valid = np.sum(mask)
    if n_valid > 0:
        logger.info(f"Mask created with {n_valid} valid features "
                   f"({n_valid/len(mask)*100:.1f}% of total)")
        
        # Calculate statistics for valid features
        valid_data = data[:, mask]
        logger.info(f"Value range in mask: [{np.min(valid_data):.3f}, "
                   f"{np.max(valid_data):.3f}]")
        logger.info(f"Mean value in mask: {np.mean(valid_data):.3f}")
    else:
        logger.warning("No valid features in mask")
        # Use a simple fallback - keep features that are non-zero in at least one subject
        mask = ~np.all(data == 0, axis=0)
        n_valid = np.sum(mask)
        logger.info(f"Using fallback mask with {n_valid} features "
                   f"({n_valid/len(mask)*100:.1f}% of total)")
    
    return mask, orig_shape

def apply_brain_mask(data, brain_mask):
    """Apply brain mask to data.
    
    Args:
        data (np.ndarray): 4D array of shape (n_samples, x, y, z)
        brain_mask (np.ndarray): 1D boolean array of shape (x*y*z,)
    
    Returns:
        np.ndarray: 2D array of shape (n_samples, n_valid_voxels)
    """
    if len(data.shape) != 4:
        raise ValueError("Expected 4D array (n_samples, x, y, z)")
    
    n_samples = data.shape[0]
    n_voxels = brain_mask.size
    expected_voxels = np.prod(data.shape[1:])
    
    if n_voxels != expected_voxels:
        raise ValueError(f"Brain mask size ({n_voxels}) does not match "
                        f"data shape ({expected_voxels})")
    
    # If mask is empty, return a small subset of voxels
    if not np.any(brain_mask):
        print("Warning: Empty brain mask, using random subset of voxels")
        n_subset = min(1000, expected_voxels)  # Take at most 1000 voxels
        brain_mask = np.zeros(n_voxels, dtype=bool)
        subset_indices = np.random.choice(n_voxels, n_subset, replace=False)
        brain_mask[subset_indices] = True
    
    # Reshape data to 2D (n_samples, n_voxels)
    flat_data = data.reshape(n_samples, -1)
    
    # Apply mask
    masked_data = flat_data[:, brain_mask]
    
    # Store the original value range for reconstruction
    save_transform_info(brain_mask, data.shape[1:], downsample_factor=DOWNSAMPLE_FACTOR,
                       value_range=(np.min(data), np.max(data)))
    
    print(f"Applied brain mask: {masked_data.shape}")
    print(f"Value range: [{masked_data.min():.3f}, {masked_data.max():.3f}]")
    return masked_data

_transform_info = None

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
    
    print("Saving transform info:")
    print(f"  Brain mask shape: {brain_mask.shape}")
    print(f"  Original shape: {orig_shape}")
    print(f"  Downsample factor: {downsample_factor}")
    if value_range is not None:
        print(f"  Value range: [{value_range[0]:.3f}, {value_range[1]:.3f}]")
    if modality_info is not None:
        print(f"  Modality: {modality_info['modality']}.{modality_info['feature']}")
    
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
        print(f"Successfully saved transform info to {transform_path}")
    except Exception as e:
        print(f"Error saving transform info: {str(e)}")
        raise
    
    return transform_info

def apply_brain_mask(data, mask=None, orig_shape=None):
    """Apply brain mask to data, handling both masking and unmasking.
    
    Args:
        data (np.ndarray): Input data
        mask (np.ndarray, optional): Boolean mask for valid voxels
        orig_shape (tuple, optional): Original 3D shape for unmasking
    
    Returns:
        np.ndarray: Masked/unmasked data
    """
    if mask is None:
        return data
    
    data_shape = np.array(data.shape)
    
    if len(data_shape) == 4:  # 4D data: (n_samples, x, y, z)
        # Flatten each sample independently and apply mask
        n_samples = data_shape[0]
        flat_data = data.reshape(n_samples, -1)  # (n_samples, n_voxels)
        masked_data = flat_data[:, mask]  # (n_samples, n_valid_voxels)
        return masked_data
    
    elif len(data_shape) == 3:  # Single 3D image
        # Flatten to 1D
        flat_data = data.reshape(-1)
        # Apply mask
        return flat_data[mask]
    
    elif len(data_shape) == 2:  # 2D data (n_samples, n_valid_voxels) to be unmasked
        if orig_shape is None:
            raise ValueError("orig_shape required for unmasking 2D data")
        # Initialize full volume with zeros for each sample
        n_samples = data_shape[0]
        n_voxels = np.prod(orig_shape)
        # Ensure mask is boolean and correct size
        mask = mask.astype(bool)
        if mask.size != n_voxels:
            # Create a new mask of correct size
            new_mask = np.zeros(n_voxels, dtype=bool)
            min_size = min(mask.size, n_voxels)
            new_mask[:min_size] = mask[:min_size]
            mask = new_mask
        
        # Count valid voxels
        n_valid = np.sum(mask)
        if data.shape[1] != n_valid:
            print(f"Warning: Data has {data.shape[1]} features but mask has {n_valid} valid voxels")
            # Adjust data size to match mask
            if data.shape[1] > n_valid:
                data = data[:, :n_valid]
            else:
                padded_data = np.zeros((n_samples, n_valid))
                padded_data[:, :data.shape[1]] = data
                data = padded_data
        
        # Create output array and reshape for assignment
        full_data = np.zeros((n_samples,) + orig_shape)
        flat_data = full_data.reshape(n_samples, -1)
        
        # Place data back in masked positions
        flat_data[:, mask] = data
        return full_data
    
    elif len(data_shape) == 1:  # 1D data to be unmasked
        if orig_shape is None:
            raise ValueError("orig_shape required for unmasking 1D data")
        # Initialize full volume with zeros
        n_voxels = np.prod(orig_shape)
        # Ensure mask is boolean and correct size
        mask = mask.astype(bool)
        if mask.size != n_voxels:
            # Create a new mask of correct size
            new_mask = np.zeros(n_voxels, dtype=bool)
            min_size = min(mask.size, n_voxels)
            new_mask[:min_size] = mask[:min_size]
            mask = new_mask
        
        # Count valid voxels
        n_valid = np.sum(mask)
        if data.size != n_valid:
            print(f"Warning: Data has {data.size} features but mask has {n_valid} valid voxels")
            # Adjust data size to match mask
            if data.size > n_valid:
                data = data[:n_valid]
            else:
                padded_data = np.zeros(n_valid)
                padded_data[:data.size] = data
                data = padded_data
        
        # Create output array and reshape for assignment
        full_data = np.zeros(orig_shape)
        flat_data = full_data.reshape(-1)
        
        # Place data back in masked positions
        flat_data[mask] = data
        return full_data
    
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

class MyDataset(Dataset):
    """Dataset class for GAN training"""
    def __init__(self, X, y, c=None):
        """Initialize dataset with optional conditional vector
        
        Args:
            X: Input data (voxelwise images or other features)
            y: Target data (behavior)
            c (optional): Conditional vector
        """
        if DEBUG_VERBOSE:
            print("\nInitializing Dataset:")
            print(f"Input data shape: {X.shape if isinstance(X, (np.ndarray, torch.Tensor)) else 'dict with header'}")
            print(f"Target data shape: {y.shape}")
            print(f"Conditional data shape: {c.shape if c is not None else 'None'}")
        
        # Process the input data
        modality_info = {'modality': 'unknown', 'feature': 'unknown'}  # Default info if not provided
        X_data = process_single_modality(X, modality_info)
        
        # Convert everything to tensors, ensuring we handle both numpy arrays and tensors
        self.X = torch.FloatTensor(X_data) if isinstance(X_data, np.ndarray) else X_data.float()
        self.y = torch.FloatTensor(y) if isinstance(y, np.ndarray) else y.float()
        self.c = torch.FloatTensor(c) if c is not None and isinstance(c, np.ndarray) else c
        
        if DEBUG_VERBOSE:
            print(f"Final tensor shapes - X: {self.X.shape}, y: {self.y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.c is not None:
            return self.X[idx], self.y[idx], self.c[idx]
        return self.X[idx], self.y[idx]

def remove_nan_inf(X, y=None, c=None):
    """Handle NaN and Inf values in data arrays.
    
    Args:
        X: Input data array (numpy array or torch tensor)
        y: Target data array (optional, numpy array or torch tensor)
        c: Conditional data array (optional, numpy array or torch tensor)
        
    Returns:
        Tuple of cleaned arrays (X, y, c) with NaN/Inf values handled appropriately
    """
    if DEBUG_VERBOSE:
        print("\nHandling NaN/Inf values:")
        print(f"Initial shapes - X: {X.shape}, y: {y.shape if y is not None else 'None'}, c: {c.shape if c is not None else 'None'}")
    
    # Convert torch tensors to numpy arrays for consistent handling
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.numpy() if y is not None and isinstance(y, torch.Tensor) else y
    c_np = c.numpy() if c is not None and isinstance(c, torch.Tensor) else c
    
    # Create mask for samples to keep
    keep_mask = np.ones(len(X_np), dtype=bool)
    
    # For 4D data, check each sample's spatial dimensions
    if len(X_np.shape) == 4:
        # Replace NaN/Inf with 0 in neuroimaging data
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove samples where ALL values are 0, NaN, or Inf
        all_invalid_mask = np.all(X_np == 0, axis=(1,2,3))
        keep_mask &= ~all_invalid_mask
        
        if DEBUG_VERBOSE:
            print(f"Found {np.sum(all_invalid_mask)} samples with all invalid neuroimaging data")
    else:
        # For other data types, replace NaN/Inf with 0
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove samples where ALL values are 0, NaN, or Inf
        all_invalid_mask = np.all(X_np == 0, axis=tuple(range(1, len(X_np.shape))))
        keep_mask &= ~all_invalid_mask
    
    # Check behavioral data
    if y_np is not None:
        # Remove samples where ANY behavioral data is NaN/Inf
        any_invalid_y = np.any(np.isnan(y_np) | np.isinf(y_np), axis=tuple(range(1, len(y_np.shape))))
        keep_mask &= ~any_invalid_y
        
        if DEBUG_VERBOSE:
            print(f"Found {np.sum(any_invalid_y)} samples with any invalid behavioral data")
    
    # Check conditional data
    if c_np is not None:
        # Replace NaN/Inf with 0 in conditional data
        c_np = np.nan_to_num(c_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    if DEBUG_VERBOSE:
        n_invalid = len(keep_mask) - np.sum(keep_mask)
        print(f"Total samples removed: {n_invalid}")
        print(f"Number of valid samples remaining: {np.sum(keep_mask)}")
    
    # Apply mask and convert back to original type
    X_clean = torch.FloatTensor(X_np[keep_mask]) if isinstance(X, torch.Tensor) else X_np[keep_mask]
    y_clean = torch.FloatTensor(y_np[keep_mask]) if y is not None and isinstance(y, torch.Tensor) else y_np[keep_mask] if y is not None else None
    c_clean = torch.FloatTensor(c_np[keep_mask]) if c is not None and isinstance(c, torch.Tensor) else c_np[keep_mask] if c is not None else None
    
    if DEBUG_VERBOSE:
        print(f"Final shapes - X: {X_clean.shape}, y: {y_clean.shape if y_clean is not None else 'None'}, c: {c_clean.shape if c_clean is not None else 'None'}")
    
    return X_clean, y_clean, c_clean

def save_subject_info(output_dir, subjects_info):
    """Save subject information to a log file.
    
    Args:
        output_dir (str): Directory to save the log file
        subjects_info (dict): Dictionary containing subject information
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'subjects_info.txt')
    
    with open(log_path, 'w') as f:
        f.write("Subject Information Log\n")
        f.write("=====================\n\n")
        
        # Write summary
        f.write("Summary:\n")
        f.write(f"Total subjects processed: {subjects_info['total_processed']}\n")
        f.write(f"Subjects included: {len(subjects_info['included_subjects'])}\n")
        f.write(f"Subjects excluded: {len(subjects_info['excluded_subjects'])}\n\n")
        
        # Write shape information if available
        if 'shape_counts' in subjects_info:
            f.write("Dataset Shapes:\n")
            for shape, count in subjects_info['shape_counts'].items():
                f.write(f"{shape}: {count} subjects\n")
            f.write("\n")
        
        # Write included subjects
        f.write("Included Subjects:\n")
        f.write("Index\tSID\tShape\n")
        for idx, sid, shape in subjects_info['included_subjects']:
            f.write(f"{idx}\t{sid}\t{shape}\n")
        f.write("\n")
        
        # Write excluded subjects
        f.write("Excluded Subjects:\n")
        f.write("SID\tReason\n")
        for sid, reason in subjects_info['excluded_subjects']:
            f.write(f"{sid}\t{reason}\n")

def load_mat_data(subject_id: str, mat_dir: str, modality: str, feature: str) -> Optional[np.ndarray]:
    """Load data from a .mat file, handling missing files and structures gracefully.
    
    Args:
        subject_id: Subject identifier
        mat_dir: Directory containing .mat files
        modality: Modality to extract (e.g., 'fmri', 'fa')
        feature: Feature to extract (e.g., 'dat', 'mean')
    
    Returns:
        Optional[np.ndarray]: Loaded data or None if file/structure is missing
    """
    try:
        # Construct mat file path
        mat_path = os.path.join(mat_dir, f"{subject_id}.mat")
        
        # Check if file exists
        if not os.path.exists(mat_path):
            logger.warning(f"Missing .mat file for subject {subject_id}")
            return None
        
        # Load mat file
        try:
            mat_data = loadmat(mat_path)
        except Exception as e:
            logger.warning(f"Error loading .mat file for subject {subject_id}: {str(e)}")
            return None
        
        # Check if modality exists
        if modality not in mat_data:
            logger.warning(f"Missing modality {modality} for subject {subject_id}")
            return None
        
        # Get modality data (handle structured array)
        modality_data = mat_data[modality]
        if isinstance(modality_data, np.ndarray) and modality_data.dtype.fields is not None:
            # For structured arrays, get the first element
            modality_data = modality_data[0, 0]
        
        # Check if feature exists in modality data
        if not hasattr(modality_data, 'dtype') or feature not in modality_data.dtype.names:
            logger.warning(f"Missing feature {feature} in {modality} for subject {subject_id}")
            return None
        
        # Get feature data (handle structured array)
        feature_data = modality_data[feature]
        if isinstance(feature_data, np.ndarray) and feature_data.dtype.fields is not None:
            # For structured arrays, get the first element
            feature_data = feature_data[0, 0]
        
        # Check if data is empty or all zeros
        if feature_data is None or (isinstance(feature_data, np.ndarray) and feature_data.size == 0):
            logger.warning(f"Empty data for subject {subject_id} in {modality}.{feature}")
            return None
        
        # Check expected shape for raw 3D modalities
        if modality in RAW_3D_MODALITIES:
            expected_shape = RAW_3D_MODALITIES[modality]
            if feature_data.shape != expected_shape:
                logger.warning(f"Unexpected shape for subject {subject_id} in {modality}.{feature}: "
                             f"got {feature_data.shape}, expected {expected_shape}")
                return None
        
        return feature_data
        
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {str(e)}")
        return None

def extract_modality(beh_df: pd.DataFrame, mat_dir: str, modality: str = 'fmri', 
                    feature: str = 'dat', target: str = 'WABTotal',
                    output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract a single modality from .mat files.
    
    Args:
        beh_df: DataFrame with behavioral data (first column should be subject IDs)
        mat_dir: Directory containing .mat files
        modality: Modality to extract (e.g., 'fmri', 'fa')
        feature: Feature to extract (e.g., 'dat', 'mean')
        target: Target variable from behavioral data
        output_dir: Optional directory to save outputs
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X data, y data, and list of included subject IDs
    """
    logger.info(f"\nExtracting modality: {modality} (feature: {feature})")
    
    # Get initial sample size
    initial_size = len(beh_df)
    logger.info(f"Initial sample size: {initial_size}")
    
    if DEBUG_MODE:
        logger.info(f"Debug mode: limiting to {DEBUG_SAMPLES} samples")
        beh_df = beh_df.head(DEBUG_SAMPLES)
    
    # Get subject ID column (first column)
    subject_col = beh_df.columns[0]
    logger.info(f"Using '{subject_col}' as subject ID column")
    
    # Track subjects and their data
    X_list = []
    y_list = []
    included_subjects = []
    excluded_subjects = []
    
    # Process each subject
    for i, (idx, row) in enumerate(beh_df.iterrows(), 1):
        subject_id = row[subject_col]
        logger.info(f"Processing subject {i}/{len(beh_df)} (ID: {subject_id})")
        
        # Load mat data
        feature_data = load_mat_data(subject_id, mat_dir, modality, feature)
        
        if feature_data is None:
            excluded_subjects.append((subject_id, "Missing or invalid imaging data"))
            continue
            
        # Replace NaN/Inf in imaging data with 0
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if all values are 0 (completely invalid data)
        if np.all(feature_data == 0):
            excluded_subjects.append((subject_id, "All imaging values are 0"))
            continue
        
        # Check behavioral data - exclude if ANY target variable is NaN/Inf
        if isinstance(target, str):
            target_vars = [target]
        else:
            target_vars = target
            
        try:
            # Convert target values to float32, replacing empty strings with NaN
            target_values = pd.to_numeric(row[target_vars], errors='coerce').astype(np.float32).values
            
            # Check for NaN/Inf values
            if np.any(np.isnan(target_values)) or np.any(np.isinf(target_values)):
                excluded_subjects.append((subject_id, f"Invalid behavioral measures found"))
                continue
                
        except Exception as e:
            logger.warning(f"Error processing behavioral data for subject {subject_id}: {str(e)}")
            excluded_subjects.append((subject_id, f"Error in behavioral data: {str(e)}"))
            continue
            
        # Check conditional variables if they exist
        if CONDITIONAL_VARIABLES:
            try:
                # Convert conditional values to float, replacing empty strings with NaN
                conditional_values = pd.to_numeric(row[CONDITIONAL_VARIABLES], errors='coerce').values
                
                # Check for NaN/Inf values
                if np.any(np.isnan(conditional_values)) or np.any(np.isinf(conditional_values)):
                    excluded_subjects.append((subject_id, f"Invalid conditional variables found"))
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing conditional data for subject {subject_id}: {str(e)}")
                excluded_subjects.append((subject_id, f"Error in conditional data: {str(e)}"))
                continue
        
        # Store valid data
        X_list.append(feature_data)
        y_list.append(target_values)
        included_subjects.append(subject_id)
    
    # Convert lists to arrays
    if not X_list:
        raise ValueError("No valid subjects found")
    
    X = np.stack(X_list)
    y = np.array(y_list)
    
    # Log subject inclusion/exclusion
    logger.info(f"\nSuccessfully extracted data for {len(included_subjects)} subjects")
    logger.info(f"Excluded {len(excluded_subjects)} subjects:")
    for subject_id, reason in excluded_subjects:
        logger.info(f"  - {subject_id}: {reason}")
    
    # Save subject info if output directory provided
    if output_dir:
        subject_info = {
            'included': included_subjects,
            'excluded': excluded_subjects
        }
        save_subject_info(output_dir, subject_info)
    
    # Log final data shape
    logger.info(f"Final data shape: {X.shape}")
    
    return X, y, included_subjects

def save_to_cache(X: np.ndarray, y: np.ndarray, subjects: List[str]) -> None:
    """Save processed data to cache.
    
    Args:
        X (np.ndarray): Feature data
        y (np.ndarray): Target data
        subjects (List[str]): List of subject IDs
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    if os.path.exists(CACHE_FILENAME):
        logger.warning(f"⚠️ Overwriting existing cached data at: {CACHE_FILENAME}")
        
    cache_data = {
        'X': X,
        'y': y,
        'subjects': subjects,
        'modalities': MODALITIES,
        'target_variables': TARGET_VARIABLES,
        'timestamp': pd.Timestamp.now()
    }
    
    logger.info(f"Saving data to cache: {CACHE_FILENAME}")
    logger.info(f"Cache data shapes - X: {X.shape}, y: {y.shape}")
    
    with open(CACHE_FILENAME, 'wb') as f:
        pickle.dump(cache_data, f)

def load_from_cache() -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Load data from cache if it exists and is valid.
    
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, List[str]]]: Cached data (X, y, subjects) or None if cache is invalid
    """
    if not os.path.exists(CACHE_FILENAME):
        logger.info("No cache file found")
        return None
        
    try:
        with open(CACHE_FILENAME, 'rb') as f:
            cache_data = pickle.load(f)
            
        # Validate cache data
        if not all(k in cache_data for k in ['X', 'y', 'subjects', 'modalities', 'target_variables']):
            logger.warning("Cache file is missing required fields")
            return None
            
        # Check if modalities and target variables match current settings
        if cache_data['modalities'] != MODALITIES or cache_data['target_variables'] != TARGET_VARIABLES:
            logger.warning("Cache settings do not match current settings")
            return None
            
        logger.warning(f"⚠️ Using cached data from: {CACHE_FILENAME} (timestamp: {cache_data['timestamp']})")
        logger.info(f"Cache data shapes - X: {cache_data['X'].shape}, y: {cache_data['y'].shape}")
        
        return cache_data['X'], cache_data['y'], cache_data['subjects']
        
    except Exception as e:
        logger.error(f"Error loading cache: {str(e)}")
        return None

def process_single_modality(X, modality_info):
    """Process a single modality's data, handling both voxelwise and ROI data.
    
    Args:
        X: Input data array or dict with header
        modality_info: Dict containing modality and feature information
        
    Returns:
        np.ndarray: Processed data (2D array: n_samples x n_features)
    """
    # Handle data with header information first
    if isinstance(X, dict) and 'hdr' in X:
        voxel_size = get_voxel_size_from_header(X['hdr'])
        header_info = X['hdr']
        X = X['data']  # Extract actual image data
    else:
        voxel_size = None
    
    # Convert torch tensor to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    
    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check if this is voxelwise data
    is_voxel = is_voxelwise(X)
    logger.info(f"Processing {modality_info['modality']}.{modality_info['feature']}: "
               f"{'voxelwise' if is_voxel else 'ROI'} data")
    
    # Store original shape before any processing
    orig_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
    
    if is_voxel and DOWNSAMPLE_FACTOR > 1:
        # Downsample only if it's voxelwise data and factor > 1
        logger.info(f"Downsampling {modality_info['modality']}.{modality_info['feature']} "
                   f"from shape {X.shape}")
        X = np.array([downsample_brain_image(img, voxel_size) for img in X])
        logger.info(f"New shape after downsampling: {X.shape}")
        orig_shape = X.shape[1:]  # Update orig_shape after downsampling
    else:
        # For ROI data, just ensure it's 2D (n_samples, features)
        if len(X.shape) > 2:
            orig_shape = X.shape[1:]  # Store original shape before reshaping
            X = X.reshape(X.shape[0], -1)
            logger.info(f"Reshaped ROI data from {X.shape[0]}, {orig_shape} to {X.shape}")
    
    # Create and apply mask
    mask, _ = create_brain_mask(X)
    if mask is not None:
        # For ROI data or already 2D data, we need to handle masking differently
        if len(X.shape) == 2:
            # Directly apply mask to 2D data
            X = X[:, mask]
        else:
            # For 3D/4D data, use the standard apply_brain_mask function
            X = apply_brain_mask(X, mask)
        
        # Save transform info for later use
        save_transform_info(
            brain_mask=mask,
            orig_shape=orig_shape,
            downsample_factor=DOWNSAMPLE_FACTOR if is_voxel and DOWNSAMPLE_FACTOR > 1 else None,
            value_range=(np.min(X), np.max(X)),
            modality_info=modality_info  # Save which modality this transform belongs to
        )
    
    return X

def extract_modalities(beh_df: pd.DataFrame, mat_dir: str, modalities: List[Dict[str, str]], 
                    target: str = 'WABTotal', output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract multiple modalities from .mat files.
    
    Args:
        beh_df: DataFrame with behavioral data (first column should be subject IDs)
        mat_dir: Directory containing .mat files
        modalities: List of dicts with modality and feature names
        target: Target variable(s) from behavioral data. If a string, will use that single target.
               If None, will use all TARGET_VARIABLES from settings.
        output_dir: Optional directory to save outputs
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X data, y data, and list of included subject IDs
    """
    # Try to load from cache if enabled
    if USE_CACHED_DATA:
        logger.info("Checking for cached data...")
        logger.info(f"Cache file path: {CACHE_FILENAME}")
        logger.info(f"Cache directory exists: {os.path.exists(os.path.dirname(CACHE_FILENAME))}")
        cached_data = load_from_cache()
        if cached_data is not None:
            logger.info("Using cached data")
            return cached_data
        logger.info("No valid cache found, processing data...")
    
    # Use all target variables if none specified
    if target is None:
        target = TARGET_VARIABLES
    
    # Make a copy of the DataFrame to avoid modifying the original
    beh_df = beh_df.copy()
    
    # Get subject ID column (first column)
    subject_col = beh_df.columns[0]
    logger.info(f"Using '{subject_col}' as subject ID column")
    
    # Extract first modality
    X1, y, included_subjects = extract_modality(
        beh_df=beh_df,
        mat_dir=mat_dir,
        modality=modalities[0]['modality'],
        feature=modalities[0]['feature'],
        target=target,
        output_dir=output_dir
    )
    
    # Process first modality
    X1_processed = process_single_modality(X1, modalities[0])
    
    # Set index on behavioral data for faster lookups
    beh_df.set_index(subject_col, inplace=True)
    
    # Get behavioral data for included subjects
    if isinstance(target, str):
        y = beh_df.loc[included_subjects, target].values
    else:
        y = beh_df.loc[included_subjects, target].values
    
    # Extract and process additional modalities if specified
    if len(modalities) > 1:
        processed_Xs = [X1_processed]  # Start with first processed modality
        
        # Create a new DataFrame for subsequent modalities with subject ID as a column
        subsequent_beh_df = beh_df.reset_index()
        
        for mod_dict in modalities[1:]:
            # Extract modality using the reset DataFrame
            mod_X, _, mod_subjects = extract_modality(
                beh_df=subsequent_beh_df,
                mat_dir=mat_dir,
                modality=mod_dict['modality'],
                feature=mod_dict['feature'],
                target=target
            )
            
            # Verify subjects match
            if mod_subjects != included_subjects:
                raise ValueError(f"Subject mismatch between modalities: "
                              f"{mod_dict['modality']} has different subjects")
            
            # Process modality
            mod_X_processed = process_single_modality(mod_X, mod_dict)
            processed_Xs.append(mod_X_processed)
        
        # Combine all processed modalities (all should be 2D at this point)
        X = np.concatenate(processed_Xs, axis=1)
        logger.info(f"Combined {len(modalities)} modalities:")
        for i, mod_dict in enumerate(modalities):
            logger.info(f"  {mod_dict['modality']}.{mod_dict['feature']}: {processed_Xs[i].shape[1]} features")
    else:
        X = X1_processed
    
    # Always save to cache
    save_to_cache(X, y, included_subjects)
    
    # Return data and subject list
    return X, y, included_subjects
