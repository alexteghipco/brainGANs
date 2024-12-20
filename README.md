# 🧠 brainGAN

> A PyTorch-based GAN framework for bidirectional generation between neuroimaging data (fMRI, DTI) and behavioral measures, with a focus on robust training and reproducibility.

## ✨ Key Features

### 🔄 Bidirectional Generation
- **Image → Behavior**: Generate behavioral predictions from neuroimaging data
- **Behavior → Image**: Generate synthetic neuroimaging data from behavioral scores
- Support for multiple modalities (fMRI, DTI)

### 🏗️ Advanced Architecture
- Self-attention mechanisms for capturing long-range dependencies
- Conditional GAN support for demographic features
- Flexible network architecture with configurable hidden layers
- Multiple loss function options (WGAN, LSGAN, GAN)

### 🚂 Robust Training Pipeline
- Nested cross-validation with Optuna hyperparameter optimization
- Smart learning rate scheduling with warmup periods
- Early stopping with customizable patience
- Gradient penalty and clipping for stability
- Multi-GPU support with parallel fold processing

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
  - Smart normalization selection (batch/layer)
  - Dropout regularization (configurable rate)

##### Attention Mechanism
- 🔍 **Multi-Head Self-Attention**
  - 4-head scaled dot-product attention
  - Learnable attention strength (γ parameter)
  - Strategic placement at network intervals
  - Dimension-scaled transformations (d⁻⁰·⁵)

##### Conditional Generation
- 🔄 **Input Processing**
  ```
  [noise_vector | condition_vector | demographics]
  ```
- Flexible conditioning strategies
- Modality-matched output dimensions

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
  - Attention-specific optimization
  - Reduction factor: 0.5
  - Plateau patience: 10 epochs

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
  - Modality-specific evaluations

### 🔍 Data Processing

#### 1️⃣ Data Integration
- 📥 **Input Processing**
  - Structured MAT parsing
  - Automated normalization
  - Modality-based feature extraction

#### 2️⃣ Reproducibility
- 🎲 **Randomization Control**
  - Global seed management
  - Framework-wide determinism
  - Consistent cross-validation

#### 3️⃣ Data Exploration
- 🔬 **Analysis Tools**
  - Recursive structure analysis
  - Distribution statistics
  - Automated error detection
  - Content validation

## 📫 Contact
alex.teghipco@uci.edu