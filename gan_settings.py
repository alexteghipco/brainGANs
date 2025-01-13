# gan_settings.py
"""
brainGAN Package Overview

This package is designed to train and evaluate Generative Adversarial Networks (GANs) that either generate neuroimaging data from behavioral data or 
behavioral data from neuroimaging data. This is *sort of* the entry point for the package. It defines all the variables we will use even though it
does not initiate the training/evaluation process (see `gan_train.py` for that).

### Assumptions
  - Check dependencies...requirements.txt
  - We assume our lab's organizational structure for data. You can actually download a ton of data here to use with this package: https://openneuro.org/datasets/ds004414/versions/1.0.0
    + WAB_forAlex.csv: A CSV file containing behavioral data. You just need a .csv where the first column is the subject ID and the second through whichever column is behavioral data.
      * Each column must have a name. The name of the first column does not matter, but you should know the other column names as we will ask for them below.
    + MASTER_FILES_08_2019: A directory containing .mat files with brain data.
      - Each mat file has a modality-based structure. For example, fa_jhu means fa values for each region of the JHU atlas. This field should contain one with FA and one with the region 
      labels (again, see our openneuro repo for example files). For voxelwsie data we need a header and an image field (i.e., dat and hdr).
      - You can use explore_mat.py to explore the contents of the mat files and their structure if you are unfamiliar with our organizational structure.
      - mat files are always named as {subject_id}.mat where subject_id is row of the first column of the csv file.
    + If your data is in a different format, you can modify the data loading code in:
      * `concat_mat_beh.py`      
      
### Code Features
  - Should adapt to whether you have CPU, GPU, or multiple GPUs.
  - Warn before overwriting previous results.

### Mode-Specific Behavior
The package operates in two modes controlled by the MODE setting:
1. "behavior_to_image":
   - Input: Behavioral scores from CSV file
   - Output: Generated brain images matching the input scores
   - Loss curves: Shows D_loss and G_loss for image generation, real/fake validity scores, and sample variance
   - Evaluation: Focuses on image quality and anatomical accuracy
   - Visualization: Multiple slice views (Sagittal, Coronal, Axial) of generated images

2. "image_to_behavior":
   - Input: Brain images from .mat files
   - Output: Generated behavioral scores
   - Loss curves: Shows D_loss and G_loss for behavior prediction, MSE loss instead of validity scores
   - Evaluation: Focuses on prediction accuracy using MSE
   - Visualization: Plots of predicted vs actual behavioral scores

### Understanding Loss Curves
The training progress plots show three key aspects of training:

1. Training Losses (Left Plot):
   - D_loss: Discriminator loss, should stabilize around zero for good training
   - G_loss: Generator loss, should also stabilize but may show more variation
   - Convergence is indicated by both losses stabilizing, not necessarily reaching zero

2. Validation Metrics (Middle Plot):
   - For behavior_to_image:
     * Real Validity: How well discriminator identifies real images (higher is better)
     * Fake Validity: How well discriminator spots generated images (lower means generator is improving)
   - For image_to_behavior:
     * MSE Loss: Mean squared error between predicted and actual behavioral scores (lower is better)

3. Sample Variance (Right Plot):
   - Shows diversity in generated samples
   - Too low variance suggests mode collapse
   - Too high variance might indicate unstable training
   - Should stabilize as training progresses

### GAN Implementation Notes
  - The GAN uses self-attention mechanism (both Generator and Discriminator) and several alternative activation/loss functions.
  - Technically, there is support for conditional GANs, but we don't use them here and the code has undergone extensive refactoring/edits so there is no longer any guarantee that this will work out of the box.
  - We use nested CV for training and evaluation. You should be too if you are tuning hyperparameters, as otherwise your pipeline is leaking data. And you should be tuning hyperparameters if you want better performance.
  - Standardization is performed within the nested CV loop, not before it (again, avoiding a common pain point of data leakage in ML).
  - Evaluation data for early stopping is independent of training, validation and test data--*always* (yet again, a source of data leakage and ultimately invisible overfitting since your generalization 
    will look "good" on your test set). The split is 80/20 within each validation dataset, be it in the inner or outer loop.
  - For early stopping we track patience, min change in loss, a min number of epochs passed, and save checkpoints.
  - Separate early stopping for Generator and Discriminator so that we don't stop the training process early depending on who is not improving.
  - Checkpoints save model states, optimizer states, loss, epoch and "best score" (the best loss seen so far).
  - We use adam optimizer for both Generator and Discriminator. Feel free to update in gan_train.py.
  - We use gradient penalty for WGAN training.
  - We use a warmup period for learning rate scheduling that increases the learning rate linearly, then switch to a ReduceLROnPlateau scheduler 
    (warmup settings are configurable, but ReduceLROnPlateau uses default values of:
      + 0.5 for the factor reducing lr and 10 for patience)
  - Training and evaluation plots generated as we go for inner and outer folds (for both the Generator and Discriminator).
  - Early stopping is based on MSE (validation loss) and so is the WarmupScheduler. MSE is also used for hyperparameter tuning. 
    Note, we have the GAN generate many predictions and get a point-estimate for an input it hasn't seen.
  - Seeds for folds etc are generated using seed_manager.py

### Code Structure
  - Core Files:
    - `gan_settings.py`: Contains global settings and hyperparameters for training GANs. You are here.
    - `gan_train.py`: Main script for training GAN models. Handles data loading, model initialization, training loops, and nested cross-validation.
    - `gan_eval.py`: Script for evaluating trained models.
    - `gan_arch.py`: Defines the GAN architectures (Generator and Discriminator) dynamically.
    - `gan_trainer.py`: Contains the GAN trainer class that manages the training process.

  - Data Processing:
    - `data_utils.py`: Data loading, preprocessing, and augmentation utilities.
    - `process_utils.py`: General data processing utilities.
    - `concat_mat_beh.py`: Utilities for handling and processing .mat files and behavioral data.

  - Training Support:
    - `training_utils.py`: Training helpers, including early stopping, LR scheduling, and loss functions.
    - `model_utils.py`: Model-related utility functions.
    - `metrics.py`: Evaluation metrics and performance calculations.
    - `visualization.py`: Plotting and visualization utilities.
    - `seed_manager.py`: Manages random seeds for reproducibility.

  - Hyperparameter Optimization:
    - `hyperparameter_optimization.py`: Optuna-based hyperparameter optimization implementation.
    - `test_hyperparameter_optimization.py`: Tests for hyperparameter optimization.

  - Testing:
    - `test_gan.py`: Unit tests for GAN functionality.
    - `test_data_processing.py`: Tests for data processing pipeline.
    - `test_gan_dimensions.py`: Tests for GAN input/output dimensions.

  - Utility:
    - `cleanup_utils.py`: Utilities for cleaning up old files and outputs.
    - `requirements.txt`: Python package dependencies.

### Output Structure
outputs/
├── cache/                  # Cached intermediate results
│   ├── data_cache.pkl     # Cached preprocessed X and y data
│   └── transform_info.pkl  # Cached transform information (masks, 3D shapes, etc.)
├── checkpoints/           # Model checkpoints
│   ├── outer_fold_{n}/    # Separate directory for each outer fold
│   │   ├── inner_fold_{m}/  # Separate directory for each inner fold
│   │   │   ├── generator_checkpoint.pt
│   │   │   └── discriminator_checkpoint.pt
│   │   └── best_model.pt   # Best model from inner fold validation
│   └── best_model.pt      # Best overall model
├── logs/                  # Training logs
│   ├── random_seed.json   # Random seeds used for reproducibility
│   ├── settings.json      # Copy of all settings used for the run
│   └── training_log.txt   # Detailed training progress log
├── plots/                 # Generated plots and visualizations
│   ├── outer_fold_{n}/    # Plots for each outer fold
│   │   ├── loss_curves/   # Training and validation loss plots
│   │   ├── predictions/   # Generated vs. real data plots
│   │   └── attention/     # Attention weight visualizations
│   └── final_results/     # Aggregate results plots
└── results/              # Evaluation results
    ├── outer_fold_{n}/    # Results for each outer fold
    │   ├── metrics.json   # Performance metrics
    │   ├── predictions/   # Model predictions
    │   │   ├── behavior_to_image/  # When MODE="behavior_to_image"
    │   │   │   ├── batch_{k}/     # Generated images for each batch
    │   │   │   │   ├── comparison_subject{i}.png  # Side-by-side comparisons
    │   │   │   │   └── generated_subject{i}.npy   # Raw generated data
    │   │   │   └── average_images/  # Average results across subjects
    │   │   │       ├── average_comparison_sagittal.png
    │   │   │       ├── average_comparison_coronal.png
    │   │   │       └── average_comparison_axial.png
    │   │   └── image_to_behavior/  # When MODE="image_to_behavior"
    │   │       ├── predicted_scores.csv  # Generated behavioral scores
    │   │       └── score_comparisons/    # Visualization of predictions
    │   │           ├── scatter_plots.png # Predicted vs actual scores
    │   │           └── error_dist.png    # Distribution of prediction errors
    │   ├── true_data_fold{n}.npy    # Ground truth data for fold
    │   └── generated_data_fold{n}.npy # Generated data for fold
    └── aggregate_results/  # Combined results across all folds
        ├── combined_true_data.npy
        └── combined_generated_data.npy

### Running the Package
  - To train a GAN model, execute the following command in your terminal:
    ```
    python gan_train.py
    ```
  - To evaluate a trained model, run:
    ```
    python gan_eval.py
    ```

The settings below will document more features of the package, take a close look at them and the documentation. Note, in departing from the older version of this code, we use Was distance
for early stopped with WGAN (i.e., specifically now in the inner folds as well). Note, conditional GANs are not debugged and missing features since we made many changes to the code and
they did not work well in our early experiments (e.g., they are not scaled or transformed in thhe current code...).
"""
import matplotlib.pyplot as plt
import os
import numpy as np

##################
''' Input paths '''
##################
# Here we specify paths to files containing the data we will use for training
CSV_PATH = 'WAB_forAlex.csv' # Path to the CSV file containing behavioral data (i.e., each row is a patient, each column is a feature). First column is subject ID and its name does not matter (other columns must have names you know/can specify below so you need to name the columns in the file).
MAT_DIR = r'C:\Users\alext\Downloads\master_newest\MASTER_FILES_08_2019' # Directory containing .mat files with neuroimaging data (i.e., each file in MAT_DIR is a .mat file which has a name like {subject_id}.mat if 'subject_id' is the first column of the CSV file).

###########################################
''' Input data extraction settings '''
###########################################
# Define which modalities and features to extract from .mat files for training
MODALITIES = [
    {
        'modality': 'fmri',  # This should match what you're actually using
        'feature': 'dat'     # This should match your data structure
    }
]

# Note, you can have multiple modalities and features like so, in which case we concatenate them (everything gets vectorized/flattened, which is how we can make this work):
#MODALITIES = [
#    {
#        'modality': 'fmri',  # Functional MRI data
#        'feature': 'dat'     # Raw fMRI data
#     },
#    {
#        'modality': 'fa',    # Fractional Anisotropy from DTI
#        'feature': 'jhu'     # Using JHU atlas regions
#    },
#    {
#        'modality': 'cbf',   # Cerebral Blood Flow
#        'feature': 'dat'     # Raw CBF data
#    }
#]

# Here we predefine 3D image modalities and their expected shapes for your project--this is to make sure minority of subjects that may have data in a different shape are dropped and don't throw errors during training. 
RAW_3D_MODALITIES = {
    'T1': (175, 209, 151),    # T1 image dims
    'fmri': (79, 95, 69),     # Functional MRI
    'fmrib': (79, 95, 69),    # Another fMRI variant? Idk it's a structure in the mat files...on that note, you can add more modalities here if you need--there are more in the mat files (*this is only for voxelwise data*)
    'fa': (175, 209, 151),    # Fractional Anisotropy
    'md': (175, 209, 151),    # Mean Diffusivity
    'cbf': (79, 95, 69),      # Cerebral Blood Flow
    'lesion': (157, 189, 156) # Binary lesion map 
}

# Which variables to extract from the CSV file? (if this is your y you may want to just use one variable but you can use multiple for a multi-task learning-like problem)
TARGET_VARIABLES = ['WABTotal','PNT_Baseline', 'WABCompreh', 'WABNaming', 'WABRepetition', 'WABSponSpeech'] # these values will be specific to your data...

''' Network task setup '''
MODE = "image_to_behavior"  # Are the gans generating images from behavior (behavior_to_image) or generating behavior from images (image_to_behavior)?

#####################################
''' Model and training parameters '''
#####################################
# This is the main group of settings that you will need to hone in on for your project. There are lots of knobs to turn, but you can keep the defaults for most settings if you are not sure what to do.

# Default Model Architecture Settings (in case of debugging, or if tuning fails for whatever reason--e.g., if you set up tuning to fail because you don't actually want to do tuning)
HIDDEN_DIMS = [512, 256, 256]  # Hidden layer dimensions for both generator and discriminator
Z_DIM = 100  # Dimension of latent space
DROPOUT_RATE = 0.2  # Dropout rate for generator
DISC_DROPOUT_RATE = 0.2  # Dropout rate for discriminator
ACTIVATION = 'ReLU'  # Activation function: 'ReLU', 'SiLU', or 'Mish'
USE_SELF_ATTENTION = True  # Whether to use self-attention in the models or not
LOSS_FUNCTION = 'wgan'  # Loss function: 'gan' (vanilla GAN), 'wgan' (Wasserstein GAN), or 'lsgan' (Least Squares GAN)

# General Training Settings
BATCH_SIZE = 16  # Reduced for debugging, you should probably set this to 16 or 32. Batch size controls how many samples are processed at a time (i.e., before we update the weights). Smaller values = less memory usage but slower training (but less likely to overfit).
NUM_EPOCHS = 100 # Max number of epochs for training (i.e., pass through the entire dataset). This is lower for debugging, should be 1000 or more. More passes help with stability but can also lead to overfitting (neuroimaging data is noisy and high-dimensional, we need stability: small learning rates, more passes through the data; more computational resources will be needed for this).

# Early Stopping Settings: patience for early stopping for generator and discriminator, respectively (G_ and D_). That is, number of epochs to wait before terminating training based on other criteria below. You may have to experiment with these parameters to find a good balance.
# These are very much problem and project-specific so there is no one-size-fits-all solution. Monitor loss and look for signs that the model is stopping before it might be able to decrease validation loss.
G_PATIENCE = 100 # This is lower for debugging, should be 100 or more.
D_PATIENCE = 100 # Same as above.

# Minimum change to qualify as an improvement for avoiding early stopping. Same as patience above--monitor loss curves for optimal project-specific values.
G_MIN_DELTA = 0.001 
D_MIN_DELTA = 0.001 

 # Minimum number of epochs before early stopping can be triggered at all. Same as patience above--monitor loss. This determines number of epochs we will train for no matter what.
MIN_EPOCHS = 200 # Should be ~200 or more.

# Learning Rate Scheduling
G_WARMUP_EPOCHS = 50  # Number of epochs for warmup for generator. Warmup is a technique to avoid instability in the early stages of training, where weights are initialized essentially at random so chance can cause idiosyncratic updates that destabilize training. This is lower for debugging, should be 50 or more.
D_WARMUP_EPOCHS = 25  # Number of epochs for warmup for discriminator. Same as above but for discriminator. This is lower for debugging, should be 25 or more.
WARMUP_LR_INIT = 1e-6 # Initial learning rate during warmup.
LR_SCHEDULER_FACTOR = 1  # Factor to reduce learning rate by. This is lower for debugging, should be 1.0 or more.
LR_SCHEDULER_PATIENCE = 50  # Number of epochs to wait before reducing learning rate. This is lower for debugging, should be 50 or more.
MIN_LR = 1e-7  # Minimum learning rate. That is, during warmup we will reduce the learning rate by a factor of 0.5 after 10 epochs based on the settings above but this is the minimum learning rate we will ever use.
# *NOTE* post-warm up we actually switch to ReduceLROnPlateau. This scheduler will reduce the learning rate when the validation loss plateaus. Again, we want to avoid overshooting the optimal solution so we need a smart way of reducing the learning rate after that initial learning phase where it can be beneficial to learn faster.

# Attention warmup settings--helps stabilize attention mechanisms during early training. This is independent of the learning rate warmup above, controlling the learning rate for attention layers.
USE_ATTENTION_WARMUP = True  # Okay, well this will determine whether to use separate learning rate warmup for attention layers.
ATTENTION_WARMUP_EPOCHS = 100  # Number of epochs for attention warmup (longer than G_WARMUP_EPOCHS)
ATTENTION_WARMUP_FACTOR = 0.3  # Initial learning rate factor for attention (30% of base lr or so)
ATTENTION_FINAL_FACTOR = 1.2  # Final learning rate factor for attention (120% of base lr or so)

# Coefficient for gradient penalty (only in WGAN): a higher value can enforce stronger penalty on gradients but may lead to unstable training. This forces the discriminator to be 
# 1-Lipschitz continuous (gradient norms close to 1), which helps prevent gradient explosions/vanishing gradients. So, higher values = less likelihood of mode collapse.
LAMBDA_GP = 10.0  # Weight for gradient penalty in WGAN-GP
LAMBDA_MSE = 10.0  # Weight for MSE loss in generator (i.e., how much does MSE loss contribute to the generator's total loss function? Generator loss = MSE + GAN loss or adversial loss PLUS outputs are numerically close to real data via MSE)

# Maximum gradient norm for clipping. This helps prevent exploding gradients/mode collapse
GRADIENT_CLIP_NORM = 1.0  # How? It limits the maximum norm (magnitude) of gradients during training--affects both models

# Number of discriminator updates per generator update--a good discriminator is key to good overla performance. You probably want this to be higher than you think you do.
N_CRITIC_WGAN = 5  # Default ratio for WGAN as per original paper
N_CRITIC_OTHER = 1  # Default 1:1 ratio for standard GAN/LSGAN, can be increased to help with mode collapse but probably not necessary as we don't need to maintain the Lipschitz constraint (again this only applies to the other loss functions). Could be helpful if you are getting some severe mode collapse.

# WGAN specific parameters
CLIP_VALUE = 0.1  # Weight clipping value for WGAN. Using a less restrictive value (0.1) since we also have gradient penalty for stability. Enforces the Lipschitz constraint, limiting the maximum absolute value that weights in the discriminator can take

# Number of predictions to generate per test input (we use this to evaluate the model's performance as a point estimate of the generative distribution that it has learned)
N_SAMPLES = 50  # Controls how many different predictions are generated for each test input using different noise vectors (z). Higher values give better estimates of the model's output distribution but increase evaluation time
                
# Diversity threshold for generated samples
VARIANCE_THRESHOLD = 5e-4  # Minimum variance required across generated samples to ensure diversity, otherwise we assume mode collapse occurred and stop training

# Choose between batch normalization and layer normalization for the models.
# BatchNorm can help with training stability but may be incompatible with some loss functions (e.g., WGAN)
# LayerNorm is more stable with different batch sizes and loss functions
USE_BATCH_NORM = False  # Set to True to use BatchNorm, False to use LayerNorm

###################################
''' Image Preprocessing Settings '''
###################################
# Downsample brain images
DOWNSAMPLE_FACTOR = 6  # Increased from 2 to 4 to reduce memory usage more aggressively. Set to 1 to disable downsampling.
FALLBACK_VOXEL_SIZE = [2, 2, 2] # Fallback voxel size if header information is not available -- this is pretty standard...but feel free to adjust based on your own project. I ran into some mat files with missing structures. 
DOWNSAMPLE_INTERPOLATION = 'linear' # Interpolation method for downsampling. Options: 'linear', 'nearest', 'cubic'

# Brain masking settings
EMPTY_VOXEL_THRESHOLD = 0.2  # Remove voxels that are empty in <20% of subjects. Set to 0 to disable masking.
EMPTY_VOXEL_VALUES = [0, np.nan]  # Values considered as "empty" voxels

''' Cross-validation parameters '''
# Determines nested CV and some Optuna (hyper)parameters -- Optuna performs bayesian optimization. Bayesian optimization learns a function that maps hyperparameters to performance so you do not need to search the whole hyperparameter space (and gets a better optimization usually than grid search)
OUTER_FOLDS = 4 # More outer folds can decrease error around point estimate of performance but depends on your dataset size and compute
INNER_FOLD_EARLY_STOPPING_SPLIT = 0.2  # 20% of data reserved for early stopping in inner folds
OUTER_FOLD_EARLY_STOPPING_SPLIT = 0.2  # 20% of data reserved for early stopping in outer folds
INNER_FOLDS = 4 # Increase inner folds before outer folds, it will stabalize hyperparameter selection which has a bigger impact on performance
N_TRIALS = 10  # Number of trials or evaluations for hyperparameter optimization--you should not aim for any lower than this, even if going behavior --> image

''' Hyperparameter search space '''
# Defines the search space for hyperparameter optimization. Note, this is what we use, but there are more options available for some settings than displayed here
HYPERPARAMETER_SEARCH_SPACE = {
    'hidden_layers': [3, 4, 5],  # More layers needed for complex image generation
    'hidden_dim': [512, 1024, 2048],  # Much larger dimensions needed for image data
    'activation': ['Mish', 'SiLU', 'ReLU'],  # Include ReLU as it's a proven activation function
    'z_dim': [100, 200, 300],  # Larger latent space for more variation in generated images
    'dropout_rate': [0.2, 0.3, 0.4],  # Dropout for generator
    'disc_dropout_rate': [0.2, 0.3, 0.4],  # Separate dropout for discriminator
    'use_self_attention': [True],  # Always use attention for image generation
    'loss_function': ['wgan','gan','lsgan'],  # WGAN tends to work better for high-dimensional outputs
    'learning_rate_base': [1e-4, 2e-4, 5e-4],  # Base learning rate
    'learning_rate_offset': [1.0, 2.0, 5.0],  # Discriminator learning rate multiplier
    'beta1': [0.0, 0.1, 0.5],  # Lower beta1 for WGAN, maybe even 0 for stability. In general, if 0, adam only uses the first moment (mean) of the gradient. If 0.5, it uses both moments (mean and variance) of the gradient. 
    'beta2': [0.9, 0.999], # Second moment tracks how much gradients vary. Larger vlaues = smaller effective learning rate.
    'conditioning_strategy': ['concat']
}

''' Previous files '''
CLEAN_PREVIOUS_FILES = True # If True, cleans up old files (not cached data, but outputs like random seeds, hyperparameter jsons, results, etc) before starting a new run.
USE_CACHED_DATA = False # If True, uses cached data to speed up data loading. Cached data will be generated automatically if it doesn't exist.
CLEAN_CACHE = True # If True, cleans up cached data before starting a new run.

''' Other settings '''
# Number of subprocesses for data loading (DataLoader).
NUM_WORKERS = 0 # A higher value can speed up data loading but require more memory.

# Error handling settings -- this is really for multi-GPU training but gets applied across the board for simplicity.
CONTINUE_ON_FOLD_FAILURE = False  # If True, continues training with partial results when some folds fail. If False, stops training if any fold fails.
MAX_FAILED_FOLDS_RATIO = 0.5  # Maximum ratio of folds that can fail before stopping training (only used if CONTINUE_ON_FOLD_FAILURE is True)

''' Debug Settings '''
DEBUG_MODE = False  # Set to True to run with limited samples
DEBUG_SAMPLES = 230  # Number of samples to use in debug mode
DEBUG_VERBOSE = True  # Set to True for additional print statements
N_SUBJECTS_PLOT_DEBUG = 3 # Number of subjects to *plot* in debug mode for some functions (e.g., test_data_processing.py; note we take first n_subjects from the list)

''' Data Standardization Settings '''
# Standardization method for behavioral data
# Options: 'standard' (zero mean, unit variance), 'minmax' (scale to [0,1]), 'robust' (using median and IQR), None (no standardization)
BEHAVIORAL_STANDARDIZATION = 'standard'

# Standardization method for imaging data
# Options: 'standard' (zero mean, unit variance), 'minmax' (scale to [0,1]), 'robust' (using median and IQR), 
#         'per_sample' (standardize each sample independently), None (no standardization)
IMAGING_STANDARDIZATION = 'standard'

# Settings for minmax scaling
MINMAX_RANGE = (0, 1)  # Range for minmax scaling

# Settings for robust scaling
ROBUST_QUANTILE_RANGE = (25.0, 75.0)  # Quantile range for robust scaling (in percentiles)

# Whether to standardize features independently or jointly for imaging data
# Only applies when IMAGING_STANDARDIZATION is not None or 'per_sample'
STANDARDIZE_FEATURES_JOINTLY = False  # If True, compute statistics across all features; if False, standardize each feature independently

''' Conditional GAN settings '''
# Settings for conditional GAN operation. I gave up on this at some point because we were getting poor results so it's not well tested...
CONDITIONAL_VARIABLES = []  # List of variables to use as conditions from behavioral csv file (e.g., ['age', 'sex']). Set to empty list [] to disable conditional GAN
CONDITIONAL_EMBEDDING_DIM = 0  # Dimension of embeddings for categorical conditional variables, set > 0 if using categorical conditions
CONDITIONAL_CONTINUOUS_SCALING = 'standard'  # Scaling method for continuous conditional variables: 'standard', 'minmax', or None 

######################################
''' Plotting and logging intervals '''
######################################
PLOT_INTERVAL = 10  # Plot training progress every n epochs
LOG_INTERVAL = 100  # Log training metrics every n batches

###################
''' Output paths '''
###################
# Here we specify output path structures. You can ignore this...we place outputs in a directory called 'output' in the same directory as this file.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')  # Base directory for all outputs called 'output'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  # model checkpoints go here
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')  # evaluation results go here
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')  # plots...
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')  # logs...
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')  # Directory for caching intermediate results
CACHE_FILENAME = os.path.join(CACHE_DIR, 'data_cache.pkl')  # Path to cached data file
TRANSFORM_CACHE_PATH = os.path.join(OUTPUT_DIR, 'transforms', 'transform_info.pkl')  # Path to cached transform info (just meta-data about downsampling, mask, etc)

# Create all output directories above
for directory in [OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR, CACHE_DIR, os.path.dirname(TRANSFORM_CACHE_PATH)]:
    os.makedirs(directory, exist_ok=True)
    

# save settings to JSON
def save_settings():
    """Save all settings to a JSON file in the output directory"""
    import json
    
    # Get all uppercase variables (which are our settings)
    settings = {
        name: value for name, value in globals().items()
        if name.isupper() and not name.startswith('__')
    }
    
    # Convert non-serializable values to strings
    for key, value in settings.items():
        if isinstance(value, (np.ndarray, np.generic)):
            settings[key] = value.tolist()
        elif not isinstance(value, (bool, int, float, str, list, dict, type(None))):
            settings[key] = str(value)
    
    # Save to file
    settings_path = os.path.join(OUTPUT_DIR, 'settings.json')
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)

# Import visualization functions after settings are defined
from visualization import create_managed_subplots
