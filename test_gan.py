import torch
import numpy as np
import pytest
from gan_arch import Generator, Discriminator
from gan_trainer import GANTrainer
from torch.utils.data import TensorDataset, DataLoader
from gan_settings import (
    Z_DIM, HIDDEN_DIMS, ACTIVATION, DROPOUT_RATE, DISC_DROPOUT_RATE,
    USE_SELF_ATTENTION, LOSS_FUNCTION, BATCH_SIZE, LAMBDA_GP,
    N_CRITIC_WGAN, N_CRITIC_OTHER, DEBUG_SAMPLES, MODE,
    TARGET_VARIABLES, RAW_3D_MODALITIES, USE_BATCH_NORM,
    GRADIENT_CLIP_NORM, LAMBDA_MSE, CLIP_VALUE,
    NUM_WORKERS, DEBUG_MODE
)

def test_vector_to_vector_gan():
    """Test GAN with vector inputs and vector outputs"""
    # Test parameters
    batch_size = BATCH_SIZE if not DEBUG_MODE else min(BATCH_SIZE, DEBUG_SAMPLES)
    z_dim = Z_DIM
    x_dim = len(TARGET_VARIABLES)  # Use actual number of behavioral features
    y_dim = 3  # 3 behavioral features as output
    hidden_dims = HIDDEN_DIMS
    n_critic = N_CRITIC_WGAN if LOSS_FUNCTION == 'wgan' else N_CRITIC_OTHER
    
    # Create dummy data
    n_samples = DEBUG_SAMPLES if DEBUG_MODE else 20
    X = torch.randn(n_samples, x_dim)  # Input behavioral features
    y = torch.randn(n_samples, y_dim)  # Output behavioral features
    
    # Create dataloaders - order: real_samples (y), conditions (X)
    dataset = TensorDataset(y, X)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # Initialize models
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
    
    # Test forward pass
    z = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, x_dim)
    fake_y = generator(z, x)
    
    # For vector outputs, the shape should be [batch_size, y_dim]
    assert fake_y.shape == (batch_size, y_dim), f"Generator output shape mismatch: expected {(batch_size, y_dim)}, got {fake_y.shape}"
    
    # Test discriminator with batch_size samples
    x_batch = X[:batch_size]  # Take only batch_size samples
    y_batch = y[:batch_size]  # Take only batch_size samples
    d_out_real = discriminator(x_batch, y_batch)
    d_out_fake = discriminator(x_batch, fake_y.detach())
    assert d_out_real.shape == (batch_size, 1), f"Discriminator output shape mismatch for real data"
    assert d_out_fake.shape == (batch_size, 1), f"Discriminator output shape mismatch for fake data"
    
    # Test trainer initialization
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,  # Using a fixed learning rate for testing
        beta1=0.0,
        beta2=0.9,
        n_critic=n_critic,
        lambda_gp=LAMBDA_GP
    )
    
    # Test one training epoch
    train_metrics = trainer.train_epoch(epoch=0)
    assert 'train_d_loss' in train_metrics
    assert 'train_g_loss' in train_metrics

def test_vector_to_image_gan():
    """Test GAN with vector inputs and image outputs"""
    # Test parameters
    batch_size = BATCH_SIZE if not DEBUG_MODE else min(BATCH_SIZE, DEBUG_SAMPLES)
    z_dim = Z_DIM
    x_dim = len(TARGET_VARIABLES)  # Use actual number of behavioral features
    
    # Use actual image shape from settings if available, otherwise use test shape
    modality = list(RAW_3D_MODALITIES.keys())[0]  # Take first modality as example
    image_shape = RAW_3D_MODALITIES[modality]
    y_dim = np.prod(image_shape)  # Flattened image dimension
    
    hidden_dims = HIDDEN_DIMS
    n_critic = N_CRITIC_WGAN if LOSS_FUNCTION == 'wgan' else N_CRITIC_OTHER
    
    # Create dummy data
    n_samples = DEBUG_SAMPLES if DEBUG_MODE else 20
    X = torch.randn(n_samples, x_dim)  # Input behavioral features
    y = torch.randn(n_samples, y_dim)  # Flattened image data
    
    # Create dataloaders - order: real_samples (y), conditions (X)
    dataset = TensorDataset(y, X)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # Initialize models
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
    
    # Test forward pass
    z = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, x_dim)
    fake_y = generator(z, x)
    
    # The generator will try to reshape the output for image data
    # We need to flatten it for the discriminator
    fake_y_flat = fake_y.view(batch_size, -1)
    assert fake_y_flat.shape == (batch_size, y_dim), f"Generator output shape mismatch after flattening: expected {(batch_size, y_dim)}, got {fake_y_flat.shape}"
    
    # Test discriminator with batch_size samples
    x_batch = X[:batch_size]  # Take only batch_size samples
    y_batch = y[:batch_size]  # Take only batch_size samples
    d_out_real = discriminator(x_batch, y_batch)
    d_out_fake = discriminator(x_batch, fake_y_flat.detach())
    assert d_out_real.shape == (batch_size, 1), f"Discriminator output shape mismatch for real data"
    assert d_out_fake.shape == (batch_size, 1), f"Discriminator output shape mismatch for fake data"
    
    # Test trainer initialization
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,  # Using a fixed learning rate for testing
        beta1=0.0,
        beta2=0.9,
        n_critic=n_critic,
        lambda_gp=LAMBDA_GP
    )
    
    # Test one training epoch
    train_metrics = trainer.train_epoch(epoch=0)
    assert 'train_d_loss' in train_metrics
    assert 'train_g_loss' in train_metrics

if __name__ == "__main__":
    pytest.main([__file__]) 