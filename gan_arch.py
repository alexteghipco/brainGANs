# gan_arch.py
"""
GAN Architecture Module

Implements Generative Adversarial Network (GAN) architecture specifically designed 
for bidirectional generation between neuroimaging data and behavioral scores.

Architecture Components:
----------------------
1. Self-Attention Mechanism (SelfAttention)
   - Purpose: Enables the network to capture long-range dependencies in the data
   - Implementation:
     * Query, Key, Value transformations using linear projections
     * Scaled dot-product attention with softmax normalization
     * Residual connection with learnable scaling (gamma parameter)
     * Dimension-specific scaling factor (dim ** -0.5) for numerical stability
   - Usage: Both Generator and Discriminator networks

2. Generator Network
   - Input Processing:
     * Latent vector (z_dim): Random noise for generation diversity
     * Condition vector (x_dim): Input domain features
     * Optional demographic features (c_dim): Additional conditioning
   
   - Architecture Design:
     * Dynamic hidden layer configuration with variable width and depth
     * Layer Composition (for each hidden layer):
       1. Linear transformation
       2. Layer normalization for training stability
       3. Configurable activation function (ReLU, SiLU, Mish)
       4. Optional dropout for regularization
       5. Self-attention module after initial layers
     
   - Special Features:
     * Conditional generation support through concatenation
     * Weight initialization based on activation function:
       - Kaiming initialization for ReLU
       - Modified Kaiming for SiLU/Mish
       - Xavier/Glorot for other activations
     * Zero-initialization for biases to start from a neutral state

3. Discriminator Network
   - Input Processing:
     * Real/Generated samples (y_dim)
     * Condition vector (x_dim)
     * Optional demographic features (c_dim)
   
   - Architecture Design:
     * Symmetric hidden layer configuration with Generator
     * Layer Composition:
       1. Linear transformation
       2. Layer normalization
       3. Configurable activation
       4. Optional dropout
       5. Strategic self-attention at network middle
     * Single output neuron for real/fake classification

Key Features:
------------
1. Tunable Network Architecture
   - Configurable network depth and width
   - Multiple activation function options
   - Adjustable dropout rates
   - Optional self-attention mechanisms

2. Conditioning Strategies
   - Concat: Direct feature concatenation
   - Supports demographic feature integration

3. Loss Function Compatibility
   - GAN: Traditional adversarial loss
   - WGAN: Wasserstein GAN loss
   - LSGAN: Least Squares GAN loss <-- worked best for generating behavioral scores from images

4. Training Stability Features
   - Layer normalization for gradient stability <-- you could try batch norm but will run into compatibility issues with some loss functions
   - Dynamic weight initializations
   - Residual connections in attention mechanisms
   - Dropout for regularization <-- USE this. You need it going from behavior to image or the other way around.
   - *CONSIDER: adding l2 regularization in layers

Implementation Details:
---------------------
1. Weight Initialization:
   - Generator output layer: Activation-specific initialization
   - Hidden layers: Standard initialization with zero bias
   - Attention layers: Default PyTorch initialization

2. Forward Pass Mechanics:
   - Generator: z ⊕ X [⊕ c] → hidden layers → output
   - Discriminator: Y ⊕ X [⊕ c] → hidden layers → scalar
   Where ⊕ represents concatenation, and [⊕ c] is optional demographic features

Usage:
------
```python
# Generator instantiation
generator = Generator(
    x_dim=256,           # Input feature dimension
    y_dim=1,             # Output dimension
    z_dim=100,           # Latent space dimension
    hidden_dims=[512, 256, 128],  # Hidden layer dimensions
    activation='ReLU',
    dropout_rate=0.3,
    loss_function='wgan',
    conditioning_strategy='concat',
    use_self_attention=True,
    c_dim=0
)

# Discriminator instantiation
discriminator = Discriminator(
    x_dim=256,           # Input feature dimension
    y_dim=1,             # Generated/real sample dimension
    hidden_dims=[128, 256, 512],  # Hidden layer dimensions
    activation='ReLU',
    dropout_rate=0.3,
    loss_function='wgan'
)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gan_settings import USE_BATCH_NORM
import numpy as np
import logging

# Add at the start of the file after imports
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 1)
        
        # Multi-head projections
        self.query = nn.Linear(dim, self.head_dim * num_heads)
        self.key = nn.Linear(dim, self.head_dim * num_heads)
        self.value = nn.Linear(dim, self.head_dim * num_heads)
        
        # Output projection
        self.proj = nn.Linear(self.head_dim * num_heads, dim)
        
        self.scale = self.head_dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Register all parameters
        self.add_module('query', self.query)
        self.add_module('key', self.key)
        self.add_module('value', self.value)
        self.add_module('proj', self.proj)
        self.register_parameter('gamma', self.gamma)
    
    def forward(self, x):
        B, N = x.shape
        
        # Project queries, keys, and values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, -1, self.head_dim)
        k = k.view(B, self.num_heads, -1, self.head_dim)
        v = v.view(B, self.num_heads, -1, self.head_dim)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.view(B, -1)
        
        # Output projection with residual connection
        out = self.proj(out)
        out = self.gamma * out + x
        
        return out

class BaseGANModule(nn.Module):
    """Base class for GAN modules with shared functionality/architecture"""
    
    def __init__(self, input_dim, hidden_dims, activation, dropout_rate,
                 use_self_attention=True, attention_interval=2):
        super(BaseGANModule, self).__init__()
        self.use_self_attention = use_self_attention
        self.attention_interval = attention_interval
        
        # Convert activation string to function and store for weight init
        if isinstance(activation, str):
            if activation == 'ReLU':
                self.activation_type = 'relu'
                activation = nn.ReLU()
            elif activation == 'SiLU':
                self.activation_type = 'silu'
                activation = nn.SiLU()
            elif activation == 'Mish':
                self.activation_type = 'mish'
                activation = nn.Mish()
            else:
                raise ValueError(f"Unknown activation function: {activation}")
        else:
            self.activation_type = 'linear'
        
        layers = []
        current_dim = input_dim
        
        for idx, h_dim in enumerate(hidden_dims):
            # Main layers block
            layer_block = []
            
            # Linear layer
            layer_block.append(nn.Linear(current_dim, h_dim))
            
            # Normalization - using LayerNorm by default
            layer_block.append(nn.LayerNorm(h_dim))
            
            # Activation
            layer_block.append(activation)
            
            # Dropout
            if dropout_rate > 0:
                layer_block.append(nn.Dropout(dropout_rate))
            
            # Self-attention
            if use_self_attention and idx % attention_interval == 0:
                layer_block.append(SelfAttention(h_dim))
            
            # Add layer block
            layers.extend(layer_block)
            current_dim = h_dim
        
        self.model = nn.Sequential(*layers)
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            if self.activation_type == 'relu':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif self.activation_type in ['silu', 'mish']:
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        try:
            # Ensure input is properly batched (has batch dimension)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
            
            # Ensure input is flattened
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            # Pass through model
            x = self.model(x)
            
            # Validate output
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError("Output contains NaN or inf values")
            
            return x
        
        except Exception as e:
            logger.error(f"Error in BaseGANModule forward pass: {str(e)}")
            raise

class Generator(BaseGANModule):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dims, activation, dropout_rate,
                 loss_function, conditioning_strategy='concat', use_self_attention=True, c_dim=0):
        # Calculate input dimension based on z_dim and x_dim (behavioral features)
        input_dim = z_dim + x_dim
        if c_dim > 0 and conditioning_strategy == 'concat':
            input_dim += c_dim
        
        # Store configuration
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.conditioning_strategy = conditioning_strategy
        self.loss_function = loss_function
        
        # Initialize base network
        super().__init__(input_dim, hidden_dims, activation, dropout_rate, use_self_attention)
        
        # Add output layer
        self.output_layer = nn.Linear(hidden_dims[-1], y_dim)
        
        # Log architecture details
        logger.info("Generator architecture:")
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Hidden dimensions: {hidden_dims}")
        logger.info(f"Output dimension: {y_dim}")
    
    def forward(self, z, conditions, c=None):
        try:
            # Ensure inputs are properly batched (have batch dimension)
            if z.dim() == 1:
                z = z.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
            if conditions.dim() == 1:
                conditions = conditions.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
            
            # Ensure conditions is flattened and has correct dimensions
            if conditions.dim() > 2:
                conditions = conditions.view(conditions.size(0), -1)
            
            # Validate input dimensions
            if conditions.size(1) != self.x_dim:
                raise ValueError(f"Input conditions dimension mismatch: expected {self.x_dim}, got {conditions.size(1)}")
            if z.size(1) != self.z_dim:
                raise ValueError(f"Input z dimension mismatch: expected {self.z_dim}, got {z.size(1)}")
            
            # Concatenate inputs based on conditioning strategy
            if self.conditioning_strategy == 'concat':
                if c is not None and self.c_dim > 0:
                    if c.dim() == 1:
                        c = c.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
                    x = torch.cat([z, conditions, c], dim=1)
                else:
                    x = torch.cat([z, conditions], dim=1)
            else:
                x = torch.cat([z, conditions], dim=1)
            
            # Pass through base network
            x = self.model(x)
            
            # Generate output
            output = self.output_layer(x)
            
            # Validate output shape
            if output.shape[-1] != self.y_dim:
                raise ValueError(f"Output shape mismatch: expected {self.y_dim} features, got {output.shape[-1]}")
            
            # Validate output values
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("Output contains NaN or inf values")
            
            return output
        
        except Exception as e:
            logger.error(f"Error in Generator forward pass: {str(e)}")
            raise

class Discriminator(BaseGANModule):
    def __init__(self, x_dim, y_dim, hidden_dims, activation, dropout_rate,
                 loss_function, conditioning_strategy='concat', use_self_attention=True, c_dim=0):
        # Calculate input dimension
        input_dim = x_dim + y_dim
        if c_dim > 0 and conditioning_strategy == 'concat':
            input_dim += c_dim
        
        # Store configuration
        self.conditioning_strategy = conditioning_strategy
        self.loss_function = loss_function
        self.c_dim = c_dim
        
        # Initialize base network
        super().__init__(input_dim, hidden_dims, activation, dropout_rate, use_self_attention)
        
        # Add output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, y, X, c=None):
        try:
            # Ensure inputs are properly batched (have batch dimension)
            if y.dim() == 1:
                y = y.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
            if X.dim() == 1:
                X = X.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
            
            # Ensure X is flattened
            if X.dim() > 2:
                X = X.view(X.size(0), -1)
            
            # Ensure y is flattened
            if y.dim() > 2:
                y = y.view(y.size(0), -1)
            
            # Combine inputs based on conditioning strategy
            if self.conditioning_strategy == 'concat':
                if c is not None and self.c_dim > 0:
                    if c.dim() == 1:
                        c = c.unsqueeze(0)  # Add batch dimension [D] -> [1, D]
                    input_tensor = torch.cat([y, X, c], dim=1)
                else:
                    input_tensor = torch.cat([y, X], dim=1)
            else:
                input_tensor = torch.cat([y, X], dim=1)
            
            # Pass through base network
            hidden = super().forward(input_tensor)
            
            # Generate validity score
            return self.output_layer(hidden)
        
        except Exception as e:
            logger.error(f"Error in Discriminator forward pass: {str(e)}")
            raise
