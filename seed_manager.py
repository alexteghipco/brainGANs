# seed_manager.py
"""
Seed Management Module

This module provides random seed management for ensuring reproducibility across multiple 
libraries/frameworks. It generates, saves, and loads random seeds consistently across all randomization sources.

Key Components:
--------------
1. Seed Generation (generate_random_seed)
   - Creates time-based random seeds
   - Ensures 32-bit compatibility using bitwise operations

2. Seed Persistence (save_seed, load_seed)
   - Save Function:
     * Stores seeds in JSON format
     * Default file: 'random_seed.json'
     * Enables experiment reproduction across sessions
   
   - Load Function:
     * Retrieves previously saved seeds if you have one...

3. Global Seed Management (set_global_seed)
   - Comprehensive seeding across libraries:
     * Python's random module
     * NumPy's random number generator
     * PyTorch CPU operations
     * PyTorch CUDA operations (when available)
     * CUDNN backend settings
   
   - Features:
     * Automatic seed generation if none provided
     * Persistent storage of generated seeds
     * Complete PyTorch determinism configuration
     * Multi-GPU support through cuda.manual_seed_all

Implementation Details:
---------------------
1. Seed Generation:
   - Uses system time (milliseconds) as entropy source
   - Applies 32-bit mask (0xFFFFFFFF) for compatibility
   - Ensures consistent seed format across systems

2. Determinism Controls:
   - Sets cudnn.deterministic = True for reproducible convolutions
   - Disables cudnn.benchmark to ensure consistent algorithms
   - Applies seeds to all available CUDA devices

Usage Examples:
-------------
1. Basic Usage (Auto-generation):
   ```python
   from seed_manager import set_global_seed
   
   # Automatically generates and saves a seed
   seed = set_global_seed()
   ```

2. Manual Seed Setting:
   ```python
   # Set a specific seed
   seed = set_global_seed(42)
   ```

3. Persistence Operations:
   ```python
   from seed_manager import save_seed, load_seed
   
   # Save current seed
   save_seed(42, 'experiment_seed.json')
   
   # Load saved seed
   saved_seed = load_seed('experiment_seed.json')
   ```

Note: While this ensures reproducibility of random operations, other factors 
like hardware, software versions, and parallel processing can still affect 
exact reproducibility.
"""

import random
import numpy as np
import torch
import time
import json
import os
from gan_settings import LOGS_DIR  # Import the correct LOGS_DIR

def generate_random_seed():
    """Generate a random seed based on current time"""
    return int(time.time() * 1000) & 0xFFFFFFFF  # Ensure it fits in 32 bits

def save_seed(seed, filename='random_seed.json'):
    """Save the seed to a file"""
    os.makedirs(LOGS_DIR, exist_ok=True)  # Create logs directory if it doesn't exist
    filepath = os.path.join(LOGS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump({'seed': seed}, f)

def load_seed(filename='random_seed.json'):
    """Load the seed from a file"""
    filepath = os.path.join(LOGS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)['seed']
    return None

def set_global_seed(seed=None):
    """Set random seed for reproducibility across all libraries"""
    if seed is None:
        # Try to load existing seed, if not generate new one
        seed = load_seed()
        if seed is None:
            seed = generate_random_seed()
            save_seed(seed)
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed
