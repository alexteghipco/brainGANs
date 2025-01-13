# 🧠 brainGAN

> A PyTorch-based GAN framework for bidirectional generation between neuroimaging data (fMRI, DTI) and behavioral measures, with a focus on robust training and reproducibility.

## ✨ Key Features

### 🔄 Bidirectional Generation
- **Image → Behavior**: Generate behavioral predictions from neuroimaging data (single or multiple modalities, voxelwise or ROI-wise)
- **Behavior → Image**: Generate synthetic neuroimaging data from behavioral scores or batteries of scores

### 🏗️ Architecture
- Optional self-attention mechanisms for capturing long-range dependencies
- GAN, lsGAN, and/or WGAN loss functions
- Flexible network architecture with optimizable hidden layers and other components
- Some support for conditional generation (cGAN is a work in progress)

### 🚂 Robust Training Pipeline
- Nested cross-validation with Optuna (Bayesian)hyperparameter optimization
- Smart learning rate scheduling with warmup periods and plateau reduction
- Early stopping with customizable patience independently for generator and discriminator
- Gradient penalty and/or clipping for stability
- Multi-GPU support with parallel fold processing (but can default to CPU if no GPU is available)

## 🛠️ Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### ⚠️ Important
Edit `gan_settings.py` before running!

### Usage
```python
python gan_train.py
```

## 📁 Project Structure

```
├── 🎯 Core Files
│   ├── gan_settings.py     # Global settings and hyperparameters (start here!)
│   ├── gan_train.py        # Main training script with nested CV
│   ├── gan_arch.py         # GAN architecture definitions used by gan_train.py
│   ├── gan_eval.py         # Standalone evaluation script (not used during training)
│   └── gan_trainer.py      # Training orchestration and management
│
├── 🧮 Data Processing
│   ├── data_utils.py       # Data loading and preprocessing
│   ├── process_utils.py    # GPU/device management and process orchestration
│   └── concat_mat_beh.py   # MAT and behavioral data handling
│
├── 🔧 Training Support
│   ├── training_utils.py   # Early stopping, LR scheduling
│   ├── model_utils.py      # Model initialization, checkpointing, and validation
│   ├── metrics.py          # Loss functions used by gan_trainer.py
│   ├── visualization.py    # Plotting and visualization tools
│   └── seed_manager.py     # Random seed management
│
├── 🎛️ Hyperparameter Optimization
│   ├── hyperparameter_optimization.py  # Optuna optimization used by gan_train.py
│   └── test_hyperparameter_optimization.py  # Optimization tests
│
├── 🧪 Testing
│   ├── test_gan.py              # Core GAN functionality tests
│   ├── test_data_processing.py  # Data pipeline tests
│   └── test_gan_dimensions.py   # Input/output dimension tests
│
├── 🧹 Utility
│   └── requirements.txt    # Package dependencies
│
└── 📂 outputs/             # Generated during training
    ├── cache/             # Intermediate results
    │   ├── data_cache.pkl     # Preprocessed data
    │   └── transform_info.pkl # Transform metadata
    │
    ├── checkpoints/      # Model states
    │   ├── outer_fold_{n}/
    │   │   ├── inner_fold_{m}/
    │   │   │   ├── generator_checkpoint.pt
    │   │   │   └── discriminator_checkpoint.pt
    │   │   └── best_model.pt
    │   └── best_model.pt
    │
    ├── logs/            # Training records
    │   ├── random_seed.json
    │   ├── settings.json
    │   └── training_log.txt
    │
    ├── plots/          # Visualizations
    │   ├── outer_fold_{n}/
    │   │   ├── loss_curves/
    │   │   └── predictions/
    │   └── final_results/
    │
    └── results/        # Evaluation outputs
        ├── outer_fold_{n}/
        │   ├── predictions/
        │   │   ├── behavior_to_image/    # When MODE="behavior_to_image"
        │   │   │   ├── batch_{k}/
        │   │   │   │   ├── comparison_subject{i}.png
        │   │   │   │   └── generated_subject{i}.npy
        │   │   │   └── average_images/
        │   │   │       ├── average_comparison_sagittal.png
        │   │   │       ├── average_comparison_coronal.png
        │   │   │       └── average_comparison_axial.png
        │   │   └── image_to_behavior/    # When MODE="image_to_behavior"
        │   │       ├── predicted_scores.csv
        │   │       └── score_comparisons/
        │   │           ├── scatter_plots.png
        │   │           └── error_dist.png
        │   ├── true_data_fold{n}.npy
        │   └── generated_data_fold{n}.npy
        └── aggregate_results/
            ├── combined_true_data.npy
            └── combined_generated_data.npy
```

#### 1️⃣ GAN Architecture (`gan_arch.py`)

##### Base Architecture
- 🏗️ **Generator & Discriminator Foundation**
  - Shared `BaseGANModule` with configurable dimensions
  - Smart normalization selection (batch/layer) and dynamicweight initialization strategiesdepending on loss function
  - Dropout regularization (configurable rate)
  - Other configurable components (e.g., depth, activation functions, GAN type)

##### Attention Mechanism
- 🔍 **Multi-Head Self-Attention**
  - 4-head scaled dot-product attention
  - Learnable attention strength (γ parameter)
  - Strategic placement at network intervals
  - Dimension-scaled transformations (d⁻⁰·⁵)

#### 2️⃣ Training Pipeline (`training_utils.py`)

##### Training Management
- 🛑 **Early Stopping**
  ```python
  Generator:     Discriminator Validity ↑ (behavior_to_image)
                 MSE ↓ (image_to_behavior)
  Discriminator: W-distance/accuracy ↑
  ```
  - Auto-checkpoint at peak performance
  - Configurable minimum epochs and change thresholds
  - Different validation metrics used based on mode:
    - behavior_to_image: Uses discriminator's real/fake validity scores during training, MS-SSIM for final evaluation
    - image_to_behavior: Uses MSE between predicted and actual behavioral scores

##### Learning Rate Control
- 📈 **Advanced Scheduling**
  - Warmup → Plateau reduction
  - Independent attention-specific optimization

##### Stability Measures
- 🔒 **Training Safeguards**
  - WGAN gradient penalty
  - Norm-based gradient clipping
  - Hybrid metric scoring (MSE + MS-SSIM)
  - Multiple generations per input for point-estimates of performance
  - Nested cross-validation for hyperparameter optimization 

#### 3️⃣ Evaluation System (`gan_eval.py`)

##### Metrics Suite
- 📊 **Performance Tracking**
  ```
  MSE, RMSE, MAE    → Lower is better
  MS-SSIM           → Higher is better
  ```
  - Multi-prediction correlation handling

##### Results Processing
- 📈 **Analysis Pipeline**
  - Cross-validation aggregation
  - Statistical analysis (correlation, MSE, MAE)
  - Modality-specific evaluations

### 🔍 Data Processing
- automatic downsampling of brain images
- masking out empty voxels across the group of subjects analyzed
- reconstruction of outputs in original space

#### 1️⃣ Data Integration
- 📥 **Input Processing**
  - Structured .mat parsing
  - Automated normalization with robust scaling for behavioral scores
  - Modality-based feature extraction

#### 2️⃣ Reproducibility
- 🎲 **Randomization Control**
  - Global seed management
  - Framework-wide determinism
  - Consistent cross-validation

## 📫 Contact
alex.teghipco@uci.edu