"""
Pure Decision Transformer Configuration

This module provides configuration management for the Pure Decision Transformer,
including configurable loss functions and training parameters.
"""

import os
from typing import Dict, Any


# Default configuration for Pure Decision Transformer
DEFAULT_PURE_DT_CONFIG = {
    # Model Architecture
    'embed_dim': 256,           # Transformer embedding dimension
    'num_layers': 4,            # Number of transformer layers
    'num_heads': 8,             # Number of attention heads
    'max_context_len': 20,      # Maximum context window for sequences
    
    # Training Parameters
    'learning_rate': 1e-4,      # Adam learning rate
    'weight_decay': 1e-5,       # L2 regularization
    'train_frequency': 5,       # Train every N actions (same as bandit)
    'min_buffer_size': 5,      # Minimum experience buffer size to start training
    
    # Loss Function Configuration
    'loss_type': 'bandit',  # Options: 'cross_entropy', 'selective', 'hybrid', 'bandit'
    'selective_confidence_threshold': 0.8,  # For hybrid loss approach
    'action_entropy_coeff': 0.0001,  # Entropy coefficient for discrete actions (bandit loss)
    'coord_entropy_coeff': 0.00001,  # Entropy coefficient for coordinates (bandit loss)
    
    # Action Sampling  
    'temperature': 1.0,         # Temperature for action sampling (1.0 = normal, 0.0 = greedy)
    'temperature_decay': 0.99,  # Optional temperature decay over time
    'min_temperature': 0.1,     # Minimum temperature floor
    
    # Experience Management
    'max_buffer_size': 50000,   # Maximum experience buffer size
    'experience_sample_rate': 1.0,  # Fraction of experiences to use for training
    
    # Sequence Processing
    'max_training_experiences': 64,  # Maximum sequence length for training
    'context_length': 5,       # Context window for training/inference
    'sequence_stride': 1,       # Stride for creating training sequences
    'padding_action_idx': 4101, # Action index used for padding (PAD token)
    
    # Model Initialization
    'freeze_cnn_backbone': False,  # Whether to freeze CNN backbone during training
    'use_pretrained_cnn': False,    # Whether to initialize from bandit CNN weights
    'cnn_transfer_path': None,     # Path to bandit model for CNN transfer (auto-detect if None)
    
    # Training Schedule
    'epochs_per_training':5,   # Number of epochs per training session
    'gradient_clip_norm': 1.0,  # Gradient clipping norm
    'early_stopping_patience': None,  # Optional early stopping (None = disabled)
    
    # Logging and Visualization  
    'log_training_metrics': True,     # Log training metrics to tensorboard
    'log_action_distributions': True, # Log action probability distributions
    'save_model_checkpoints': False,  # Save model checkpoints during training
    'checkpoint_frequency': 1000,     # Save checkpoint every N training steps
}

# Device-specific configurations
CPU_PURE_DT_CONFIG = {
    **DEFAULT_PURE_DT_CONFIG,
    'embed_dim': 128,           # Smaller model for CPU
    'num_layers': 2,            # Fewer layers for CPU
    'context_length': 10,       # Shorter context for CPU
    'max_training_experiences': 30,
}

GPU_PURE_DT_CONFIG = {
    **DEFAULT_PURE_DT_CONFIG, 
    'embed_dim': 512,           # Larger model for GPU
    'num_layers': 6,            # More layers for GPU
    'context_length': 20,       # Longer context for GPU
    'max_training_experiences': 100,
}

# Loss function specific configurations
CROSS_ENTROPY_CONFIG = {
    **DEFAULT_PURE_DT_CONFIG,
    'loss_type': 'cross_entropy',
    'learning_rate': 1e-4,      # Standard learning rate for dense gradients
    'epochs_per_training': 1,   # Single epoch for fast dense updates
}

SELECTIVE_CONFIG = {
    **DEFAULT_PURE_DT_CONFIG,
    'loss_type': 'selective', 
    'learning_rate': 5e-4,      # Higher learning rate for sparse gradients
    'epochs_per_training': 3,   # More epochs for sparse selective updates
}

HYBRID_CONFIG = {
    **DEFAULT_PURE_DT_CONFIG,
    'loss_type': 'hybrid',
    'selective_confidence_threshold': 0.8,
    'learning_rate': 2e-4,      # Balanced learning rate
    'epochs_per_training': 2,   # Balanced epoch count
}

BANDIT_CONFIG = {
    **DEFAULT_PURE_DT_CONFIG,
    'loss_type': 'bandit',
    'learning_rate': 1e-4,      # Same as original bandit
    'epochs_per_training': 1,   # Single epoch like original bandit
    'action_entropy_coeff': 0.0001,  # Action exploration bonus
    'coord_entropy_coeff': 0.00001,  # Coordinate exploration bonus
    'weight_decay': 0,          # No weight decay (bandit doesn't use it)
}


def load_pure_dt_config(device: str = None, loss_type: str = None) -> Dict[str, Any]:
    """Load Pure DT configuration based on device and loss type preferences.
    
    Args:
        device: 'cpu', 'cuda', or None for auto-detection
        loss_type: 'cross_entropy', 'selective', 'hybrid', or None for default
        
    Returns:
        Configuration dictionary
    """
    # Start with default config
    config = DEFAULT_PURE_DT_CONFIG.copy()
    
    # Apply device-specific config
    if device == 'cpu':
        config.update(CPU_PURE_DT_CONFIG)
    elif device == 'cuda' or device == 'gpu':
        config.update(GPU_PURE_DT_CONFIG)
    elif device is None:
        # Auto-detect device
        import torch
        if torch.cuda.is_available():
            config.update(GPU_PURE_DT_CONFIG)
        else:
            config.update(CPU_PURE_DT_CONFIG)
    
    # Apply loss-specific config  
    if loss_type == 'cross_entropy':
        config.update(CROSS_ENTROPY_CONFIG)
    elif loss_type == 'selective':
        config.update(SELECTIVE_CONFIG)
    elif loss_type == 'hybrid':
        config.update(HYBRID_CONFIG)
    elif loss_type == 'bandit':
        config.update(BANDIT_CONFIG)
    
    # Override with environment variables if present
    config = _apply_environment_overrides(config)
    
    return config


def _apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    env_mapping = {
        'PURE_DT_LEARNING_RATE': ('learning_rate', float),
        'PURE_DT_CONTEXT_LENGTH': ('context_length', int),
        'PURE_DT_LOSS_TYPE': ('loss_type', str),
        'PURE_DT_TEMPERATURE': ('temperature', float),
        'PURE_DT_EMBED_DIM': ('embed_dim', int),
        'PURE_DT_NUM_LAYERS': ('num_layers', int),
        'PURE_DT_EPOCHS': ('epochs_per_training', int),
    }
    
    for env_var, (config_key, type_func) in env_mapping.items():
        if env_var in os.environ:
            try:
                config[config_key] = type_func(os.environ[env_var])
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid value for {env_var}: {e}")
    
    return config


def get_loss_config_summary(config: Dict[str, Any]) -> str:
    """Get a summary of the loss configuration for logging."""
    loss_type = config.get('loss_type', 'unknown')
    lr = config.get('learning_rate', 'unknown')
    epochs = config.get('epochs_per_training', 'unknown')
    context = config.get('context_length', 'unknown')
    
    summary = f"Pure DT Config: loss={loss_type}, lr={lr}, epochs={epochs}, context={context}"
    
    if loss_type == 'hybrid':
        threshold = config.get('selective_confidence_threshold', 'unknown')
        summary += f", threshold={threshold}"
    
    return summary


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_keys = [
        'embed_dim', 'num_layers', 'num_heads', 'learning_rate',
        'context_length', 'loss_type', 'temperature'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate ranges
    if config['context_length'] < 1:
        raise ValueError("context_length must be >= 1")
    
    if config['temperature'] < 0:
        raise ValueError("temperature must be >= 0")
    
    if config['loss_type'] not in ['cross_entropy', 'selective', 'hybrid', 'bandit']:
        raise ValueError(f"Invalid loss_type: {config['loss_type']}")
    
    if config['embed_dim'] % config['num_heads'] != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    
    return True


# Export main functions
__all__ = [
    'DEFAULT_PURE_DT_CONFIG',
    'CPU_PURE_DT_CONFIG', 
    'GPU_PURE_DT_CONFIG',
    'CROSS_ENTROPY_CONFIG',
    'SELECTIVE_CONFIG',
    'HYBRID_CONFIG',
    'BANDIT_CONFIG',
    'load_pure_dt_config',
    'get_loss_config_summary',
    'validate_config'
]