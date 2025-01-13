import torch
import numpy as np
import pytest
from gan_arch import Generator, Discriminator
from gan_trainer import GANTrainer
from torch.utils.data import DataLoader, TensorDataset
from data_utils import (
    reshape_data, validate_input_data, standardize_data,
    GANDataset, create_dataloaders
)
from gan_settings import (
    Z_DIM, HIDDEN_DIMS, ACTIVATION, DROPOUT_RATE, DISC_DROPOUT_RATE,
    USE_SELF_ATTENTION, LOSS_FUNCTION, BATCH_SIZE, LAMBDA_GP,
    N_CRITIC_WGAN, N_CRITIC_OTHER, MODE, TARGET_VARIABLES
)

def test_data_dimensions():
    """Test data reshaping and dimension handling"""
    # Test 3D data
    batch_size = 4
    depth, height, width = 10, 12, 9
    X_3d = np.random.randn(batch_size, depth, height, width)
    y_3d = np.random.randn(batch_size, len(TARGET_VARIABLES))
    
    # Test reshaping
    X_reshaped = reshape_data(X_3d)
    assert len(X_reshaped.shape) == 2, f"Expected 2D array, got shape {X_reshaped.shape}"
    assert X_reshaped.shape[0] == batch_size, f"Expected batch size {batch_size}, got {X_reshaped.shape[0]}"
    assert X_reshaped.shape[1] == depth * height * width, f"Expected flattened size {depth * height * width}, got {X_reshaped.shape[1]}"

def test_gan_dataset():
    """Test GAN dataset creation and data loading"""
    batch_size = 4
    x_dim = 100
    y_dim = 6  # Number of target variables
    
    # Create dummy data
    X = np.random.randn(batch_size, x_dim)
    y = np.random.randn(batch_size, y_dim)
    
    # Test dataset creation
    dataset = GANDataset(X, y)
    assert len(dataset) == batch_size, f"Expected dataset length {batch_size}, got {len(dataset)}"
    
    # Test data loading
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    assert len(batch) == 2, "Expected tuple of (X, y)"
    assert batch[0].shape == (2, x_dim), f"Expected X shape (2, {x_dim}), got {batch[0].shape}"
    assert batch[1].shape == (2, y_dim), f"Expected y shape (2, {y_dim}), got {batch[1].shape}"

def test_standardization():
    """Test data standardization"""
    batch_size = 4
    x_dim = 100
    
    # Create dummy data
    X_train = np.random.randn(batch_size, x_dim)
    X_test = np.random.randn(batch_size//2, x_dim)
    
    # Test standardization
    X_train_std, [X_test_std] = standardize_data(X_train, [X_test], is_behavioral=False)
    assert X_train_std.shape == X_train.shape, f"Expected shape {X_train.shape}, got {X_train_std.shape}"
    assert X_test_std.shape == X_test.shape, f"Expected shape {X_test.shape}, got {X_test_std.shape}"

def test_behavior_to_image_dimensions():
    """Test GAN dimensions for behavior to image generation"""
    batch_size = 4
    z_dim = Z_DIM
    x_dim = len(TARGET_VARIABLES)  # Behavioral features
    depth, height, width = 10, 12, 9
    y_dim = depth * height * width  # Flattened image dimension
    
    # Initialize models
    generator = Generator(
        z_dim=z_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=HIDDEN_DIMS,
        activation=ACTIVATION,
        dropout_rate=DROPOUT_RATE,
        use_self_attention=USE_SELF_ATTENTION,
        loss_function=LOSS_FUNCTION
    )
    
    discriminator = Discriminator(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=HIDDEN_DIMS[::-1],
        activation=ACTIVATION,
        dropout_rate=DISC_DROPOUT_RATE,
        use_self_attention=USE_SELF_ATTENTION,
        loss_function=LOSS_FUNCTION
    )
    
    # Test forward pass dimensions
    z = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, x_dim)
    fake_y = generator(z, x)
    
    assert fake_y.shape == (batch_size, y_dim), f"Generator output shape mismatch: expected {(batch_size, y_dim)}, got {fake_y.shape}"
    
    # Test discriminator dimensions
    d_out_fake = discriminator(x, fake_y.detach())
    assert d_out_fake.shape == (batch_size, 1), f"Discriminator output shape mismatch: expected {(batch_size, 1)}, got {d_out_fake.shape}"

def test_image_to_behavior_dimensions():
    """Test GAN dimensions for image to behavior generation"""
    batch_size = 4
    z_dim = Z_DIM
    depth, height, width = 10, 12, 9
    x_dim = depth * height * width  # Flattened image dimension
    y_dim = len(TARGET_VARIABLES)  # Behavioral features
    
    # Initialize models
    generator = Generator(
        z_dim=z_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=HIDDEN_DIMS,
        activation=ACTIVATION,
        dropout_rate=DROPOUT_RATE,
        use_self_attention=USE_SELF_ATTENTION,
        loss_function=LOSS_FUNCTION
    )
    
    discriminator = Discriminator(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=HIDDEN_DIMS[::-1],
        activation=ACTIVATION,
        dropout_rate=DISC_DROPOUT_RATE,
        use_self_attention=USE_SELF_ATTENTION,
        loss_function=LOSS_FUNCTION
    )
    
    # Test forward pass dimensions
    z = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, x_dim)
    fake_y = generator(z, x)
    
    assert fake_y.shape == (batch_size, y_dim), f"Generator output shape mismatch: expected {(batch_size, y_dim)}, got {fake_y.shape}"
    
    # Test discriminator dimensions
    d_out_fake = discriminator(x, fake_y.detach())
    assert d_out_fake.shape == (batch_size, 1), f"Discriminator output shape mismatch: expected {(batch_size, 1)}, got {d_out_fake.shape}"

def test_end_to_end_training():
    """Test end-to-end training pipeline with small dummy data"""
    batch_size = 4
    z_dim = Z_DIM
    x_dim = len(TARGET_VARIABLES)  # Behavioral features
    depth, height, width = 10, 12, 9
    y_dim = depth * height * width  # Flattened image dimension
    
    # Create dummy data for behavior to image generation
    X = torch.randn(batch_size * 2, x_dim)  # Behavioral features
    y = torch.randn(batch_size * 2, y_dim)  # Flattened image data
    
    # Create dataloaders using TensorDataset to ensure correct order
    dataset = TensorDataset(y, X)  # Note: order is (real_samples, conditions)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models with progressive hidden dimensions
    hidden_dims = [
        min(d, max(z_dim + x_dim, y_dim)) for d in [
            (z_dim + x_dim) * 2,  # First layer expands
            (z_dim + x_dim) * 4,  # Second layer expands more
            y_dim,  # Final hidden layer matches output
        ]
    ]
    
    generator = Generator(
        z_dim=z_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        activation=ACTIVATION,
        dropout_rate=DROPOUT_RATE,
        use_self_attention=USE_SELF_ATTENTION,
        loss_function=LOSS_FUNCTION
    )
    
    discriminator = Discriminator(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims[::-1],
        activation=ACTIVATION,
        dropout_rate=DISC_DROPOUT_RATE,
        use_self_attention=USE_SELF_ATTENTION,
        loss_function=LOSS_FUNCTION
    )
    
    # Initialize trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        n_critic=N_CRITIC_WGAN if LOSS_FUNCTION == 'wgan' else N_CRITIC_OTHER,
        lambda_gp=LAMBDA_GP
    )
    
    # Test one training epoch
    train_metrics = trainer.train_epoch(epoch=0)
    assert isinstance(train_metrics, dict), "Expected dictionary of metrics"
    assert len(train_metrics) > 0, "Expected non-empty metrics dictionary"

if __name__ == "__main__":
    pytest.main([__file__]) 