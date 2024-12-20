# 🧠 brainGAN

> A PyTorch-based GAN framework for bidirectional generation between neuroimaging data (fMRI, DTI) and behavioral measures, with a focus on stable training and reproducibility.

## ✨ Key Features

### 🔄 Bidirectional Generation
- **Image → Behavior**: Generate behavioral predictions from neuroimaging data
- **Behavior → Image**: Generate synthetic neuroimaging data from behavioral scores
- Support for multiple modalities (fMRI, DTI)

### 🏗️ Advanced Architecture
- Self-attention mechanisms for capturing long-range dependencies
- Conditional GAN support for demographic or other features (WIP)
- Flexible network architecture with configurable hidden layers, other architectural elements
- Multiple loss function options (WGAN, LSGAN, GAN)

### 🚂 Robust Training Pipeline
- Nested cross-validation with Optuna hyperparameter optimization
- Smart learning rate scheduling with warmup and plateau periods
- Early stopping with customizable patience, min epochs, min change
- Gradient penalty and clipping for stability
- Multi-GPU support with parallel fold processing
- Data caching, automatic checkpointing

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
├── 🎯 Core GAN Implementation
│   ├── gan_arch.py         # GAN architecture definitions
│   ├── gan_config.py       # Configuration constants
│   ├── gan_settings.py     # Global settings and parameters
│   ├── gan_train.py        # Training implementation
│   ├── gan_eval.py         # Evaluation functions
│   └── training_utils.py   # Training helper functions
│
├── 🧪 Testing 
│   └── gan_test.py         # Unit tests and integration tests
│
├── 📊 Data Processing
│   ├── concat_mat_beh.py   # Data concatenation utilities
│   └── seed_manager.py     # Random seed management
│
├── 🔍 Data Exploration (search ARC on openneuro for examples)
│   ├── explore_data.py     # CSV data exploration
│   ├── explore_mat.py      # MAT file exploration
│   └── explore_modalities.py # Modality analysis
│
├── ⚙️ Configuration
│   ├── requirements.txt    # Project dependencies
│
└── 📂 outputs/
    ├── checkpoints/       # Model states
    ├── results/          # Evaluation results
    ├── plots/           # Generated figures
    ├── logs/            # Training logs
    └── cache/           # Cached data
```

## 🔧 Implementation Details

### 🎯 Core Components

#### 1️⃣ GAN Architecture (`gan_arch.py`)

##### Base Architecture
- 🏗️ **Generator & Discriminator Foundation**
  - Shared `BaseGANModule` with configurable dimensions
  - Normalization selection (batch/layer)
  - Dropout regularization (configurable rate)
  - Depth, dropout, latent space, and other configurable parameters

##### Attention Mechanism for Interactions
- 🔍 **Multi-Head Self-Attention**
  - 4-head scaled dot-product attention
  - Learnable attention strength (γ parameter)
  - Strategic placement at network intervals
  - Dimension-scaled transformations (d⁻⁰·⁵)

#### 2️⃣ Training Pipeline (`training_utils.py`)

##### Training Management
- 🛑 **Early Stopping**
  ```python
  Generator:     MSE/MS-SSIM ↓
  Discriminator: W-distance/accuracy ↑
  ```
  - Auto-checkpoint at peak performance
  - Configurable minimum epochs

##### Learning Rate Control
- 📈 **Advanced Scheduling**
  - Warmup → Plateau reduction
  - Independent attention warmup
  - Independent discriminator/generator warmup

##### Stability Measures
- 🔒 **Training Safeguards**
  - WGAN gradient penalty
  - Norm-based gradient clipping
  - Hybrid metric scoring (MSE + MS-SSIM)

#### 3️⃣ Evaluation System (`gan_eval.py`)

##### Metrics Suite
- 📊 **Performance Tracking**
  ```
  MSE, RMSE, MAE    → Lower is better
  MS-SSIM           → Higher is better
  ```
  - Multi-prediction correlation handling
  - Robust to numerical instabilities

##### Results Processing
- 📈 **Analysis Pipeline**
  - Cross-validation aggregation
  - KS statistical testing
  - Modality-specific evaluations (e.g., rois vs voxelwise)

### 🔍 Data Processing

#### 1️⃣ Data Integration
- 📥 **Input Processing**
  - Structured MAT parsing
  - Automated normalization with robust and normal normalization methods
  - Noramlization *inside* inner folds to ensure no leakage
  - Automatic downsampling of images based on header
  - Automatic masking to remove empty voxels across most individuals
  - Outputs for facilitating reconstruction
  - Modality-based feature extraction from our lab's file convention system (see ARC on openneuro for examples)

#### 2️⃣ Reproducibility
- 🎲 **Randomization Control**
  - Global seed management for framework-wide determinism

## 📫 Contact
alex.teghipco@uci.edu
