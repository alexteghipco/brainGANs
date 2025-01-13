import pytest
import torch
import numpy as np
from model_utils import validate_models
from torch.utils.data import DataLoader, TensorDataset
from gan_settings import MODE, VARIANCE_THRESHOLD

def test_image_to_behavior_validation():
    """Test validation metrics for image_to_behavior mode"""
    batch_size = 4
    image_dim = 100  # Flattened image dimension
    behavior_dim = 6  # Number of behavioral features
    
    # Create mock generator that outputs behavioral predictions
    class MockGenerator:
        def __init__(self):
            self.z_dim = 10
            self.eval_called = False
        
        def eval(self):
            self.eval_called = True
        
        def __call__(self, z, conditions):
            # Return predictable outputs for testing
            batch_size = conditions.size(0)
            return torch.ones(batch_size, behavior_dim)

    # Create mock discriminator
    class MockDiscriminator:
        def __init__(self):
            self.eval_called = False
        
        def eval(self):
            self.eval_called = True
        
        def __call__(self, x, conditions):
            batch_size = x.size(0)
            return torch.zeros(batch_size, 1)

    # Create test data
    real_samples = torch.randn(batch_size * 2, behavior_dim)  # Behavioral data
    conditions = torch.randn(batch_size * 2, image_dim)  # Image data
    dataset = TensorDataset(real_samples, conditions)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Initialize mock models
    generator = MockGenerator()
    discriminator = MockDiscriminator()

    # Test validation in image_to_behavior mode
    metrics = validate_models(
        generator=generator,
        discriminator=discriminator,
        val_loader=val_loader,
        device=torch.device('cpu'),
        mode="image_to_behavior"
    )

    # Verify metrics
    assert 'val_mse' in metrics, "MSE should be present in metrics"
    assert 'mode_collapse' in metrics, "Mode collapse flag should be present"
    assert 'sample_variance' in metrics, "Sample variance should be present"
    assert metrics['val_real_validity'] == 0.0, "Real validity should be 0 in image_to_behavior mode"
    assert metrics['val_fake_validity'] == 0.0, "Fake validity should be 0 in image_to_behavior mode"
    assert generator.eval_called, "Generator eval() should be called"
    assert discriminator.eval_called, "Discriminator eval() should be called"

def test_behavior_to_image_validation():
    """Test validation metrics for behavior_to_image mode"""
    batch_size = 4
    image_dim = 100  # Flattened image dimension
    behavior_dim = 6  # Number of behavioral features
    
    # Create mock generator that outputs images
    class MockGenerator:
        def __init__(self):
            self.z_dim = 10
            self.eval_called = False
        
        def eval(self):
            self.eval_called = True
        
        def __call__(self, z, conditions):
            # Return predictable outputs for testing
            batch_size = conditions.size(0)
            return torch.randn(batch_size, image_dim)  # Random to ensure variance

    # Create mock discriminator
    class MockDiscriminator:
        def __init__(self):
            self.eval_called = False
        
        def eval(self):
            self.eval_called = True
        
        def __call__(self, x, conditions):
            batch_size = x.size(0)
            return torch.ones(batch_size, 1)  # Always predict real

    # Create test data
    real_samples = torch.randn(batch_size * 2, image_dim)  # Image data
    conditions = torch.randn(batch_size * 2, behavior_dim)  # Behavioral data
    dataset = TensorDataset(real_samples, conditions)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Initialize mock models
    generator = MockGenerator()
    discriminator = MockDiscriminator()

    # Test validation in behavior_to_image mode
    metrics = validate_models(
        generator=generator,
        discriminator=discriminator,
        val_loader=val_loader,
        device=torch.device('cpu'),
        mode="behavior_to_image"
    )

    # Verify metrics
    assert 'val_real_validity' in metrics, "Real validity should be present in metrics"
    assert 'val_fake_validity' in metrics, "Fake validity should be present in metrics"
    assert 'mode_collapse' in metrics, "Mode collapse flag should be present"
    assert 'sample_variance' in metrics, "Sample variance should be present"
    assert metrics['val_mse'] == 0.0, "MSE should be 0 in behavior_to_image mode"
    assert metrics['val_real_validity'] > 0, "Real validity should be positive"
    assert metrics['val_fake_validity'] > 0, "Fake validity should be positive"
    assert generator.eval_called, "Generator eval() should be called"
    assert discriminator.eval_called, "Discriminator eval() should be called"

def test_mode_collapse_detection():
    """Test mode collapse detection in both modes"""
    batch_size = 4
    image_dim = 100
    behavior_dim = 6
    
    # Create mock generator that outputs constant values (simulating mode collapse)
    class MockCollapsingGenerator:
        def __init__(self):
            self.z_dim = 10
        
        def eval(self):
            pass
        
        def __call__(self, z, conditions):
            batch_size = conditions.size(0)
            return torch.ones(batch_size, behavior_dim)  # Constant output

    class MockDiscriminator:
        def eval(self):
            pass
        
        def __call__(self, x, conditions):
            batch_size = x.size(0)
            return torch.zeros(batch_size, 1)

    # Create test data
    real_samples = torch.randn(batch_size * 2, behavior_dim)
    conditions = torch.randn(batch_size * 2, image_dim)
    dataset = TensorDataset(real_samples, conditions)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Initialize mock models
    generator = MockCollapsingGenerator()
    discriminator = MockDiscriminator()

    # Test validation with mode collapse
    metrics = validate_models(
        generator=generator,
        discriminator=discriminator,
        val_loader=val_loader,
        device=torch.device('cpu'),
        mode="image_to_behavior"
    )

    # Verify mode collapse detection
    assert metrics['mode_collapse'], "Should detect mode collapse with constant generator output"
    assert metrics['sample_variance'] < VARIANCE_THRESHOLD, "Sample variance should be below threshold"

def test_empty_validation_data():
    """Test validation with empty data loader"""
    batch_size = 4
    image_dim = 100
    behavior_dim = 6
    
    # Create empty dataset
    dataset = TensorDataset(
        torch.randn(0, behavior_dim),  # Empty tensors
        torch.randn(0, image_dim)
    )
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Create mock models
    class MockGenerator:
        def __init__(self):
            self.z_dim = 10
        def eval(self):
            pass
        def __call__(self, z, conditions):
            return torch.randn(conditions.size(0), behavior_dim)

    class MockDiscriminator:
        def eval(self):
            pass
        def __call__(self, x, conditions):
            return torch.zeros(x.size(0), 1)

    generator = MockGenerator()
    discriminator = MockDiscriminator()

    # Test validation with empty data
    metrics = validate_models(
        generator=generator,
        discriminator=discriminator,
        val_loader=val_loader,
        device=torch.device('cpu'),
        mode="image_to_behavior"
    )

    # Verify default metrics for empty data
    assert metrics['val_real_validity'] == 0.0, "Should return default real validity"
    assert metrics['val_fake_validity'] == 0.0, "Should return default fake validity"
    assert metrics['val_mse'] == float('inf'), "Should return inf MSE for empty data"
    assert metrics['mode_collapse'], "Should indicate mode collapse for empty data"
    assert metrics['sample_variance'] == 0.0, "Should return zero variance for empty data" 