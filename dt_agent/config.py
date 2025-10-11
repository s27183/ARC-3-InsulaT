"""
Pure Decision Transformer Configuration

This module provides configuration management for the Pure Decision Transformer,
including configurable loss functions and training parameters.
"""

import os
from typing import Dict, Any


# Default configuration for Pure Decision Transformer
DEFAULT_DT_CONFIG = {
    # Model Architecture
    "embed_dim": 256,  # Transformer embedding dimension
    "num_layers": 4,  # Number of transformer layers
    "num_heads": 8,  # Number of attention heads
    "max_context_len": 300,  # Maximum positional embedding capacity (must be >= longest head-specific context)

    # Training Parameters
    "learning_rate": 1e-4,  # Adam learning rate
    "weight_decay": 1e-5,  # L2 regularization
    "train_frequency": 10,  # Train every N actions (reduced to save compute with longer context)
    "min_buffer_size": 10,  # Minimum experience buffer size to start training

    # Loss Function Configuration
    "action_entropy_coeff": 0.0001,  # Entropy coefficient for discrete actions
    "coord_entropy_coeff": 0.00001,  # Entropy coefficient for coordinates

    # Temporal Credit Assignment
    # TODO: experiment with/without temporal credit and with different decay rates to do A/B test
    "temporal_credit": True, # Backward temporal credit assignment in a state/action sequence
    "eligibility_decay": 0.8,  # Exponential decay for temporal credit (0.8 = moderate decay, 1.0 = no temporal credit)

    # Hierarchical Context Windows (Head-Specific)
    "change_context_len": 5,  # Context length for change head (immediate effects)
    "completion_context_len": 50,  # Context length for completion head (goal sequences)
    "gameover_context_len": 200,  # Context length for GAME_OVER head (failure causal chains)

    # Head-Specific Eligibility Decay (for temporal credit assignment)
    "change_eligibility_decay": 0.7,  # Fast decay for immediate effects
    "completion_eligibility_decay": 0.8,  # Medium decay for goal-oriented actions
    "gameover_eligibility_decay": 0.9,  # Slow decay for long causal chains

    # Importance-Weighted Replay (Head-Specific Replay Sizes)
    "change_replay_size": 16,  # Number of change sequences per training round
    "completion_replay_size": 320,  # Number of completion sequences per training round
    "gameover_replay_size": 320,  # Number of GAME_OVER sequences per training round

    # Replay Variation (for completion and GAME_OVER sequences)
    "replay_variation_min": 0.8,  # Minimum variation factor (80% of target length)
    "replay_variation_max": 1.0,  # Maximum variation factor (100% of target length)

    # Experience Management
    "max_buffer_size": 200000,  # Maximum experience buffer size
    "experience_sample_rate": 1.0,  # Fraction of experiences to use for training

    # Training Schedule
    "batch_size": 16,  # Batch size for training
    "epochs_per_training": 1,  # Number of epochs per training session
    "gradient_clip_norm": 1.0,  # Gradient clipping norm

    # ViT State Encoder Configuration
    "vit_patch_size": 8,  # Patch size (8×8 = 64 patches for 64×64 grid)
    "vit_num_layers": 4,  # ViT transformer layers
    "vit_num_heads": 8,  # ViT attention heads
    "vit_dropout": 0.1,  # ViT dropout rate
    "vit_use_cls_token": True,  # Use CLS token (True) vs global pooling (False)
    "vit_cell_embed_dim": 64,  # Dimension for learned cell embeddings (0-15)
}

# Device-specific configurations
CPU_PURE_DT_CONFIG = {
    **DEFAULT_DT_CONFIG,
    "embed_dim": 128,  # Smaller model for CPU
    "num_layers": 2,  # Fewer layers for CPU
    "max_context_len": 150,  # Must accommodate gameover_context_len
    "batch_size": 16,  # Smaller batch for CPU
    # ViT configuration for CPU
    "vit_num_layers": 2,  # Fewer ViT layers for CPU
    "vit_num_heads": 4,  # Fewer attention heads for CPU (128/4 = 32)
    "vit_patch_size": 8,  # Keep patch size same
    "vit_cell_embed_dim": 32,  # Smaller cell embeddings for CPU
    # Hierarchical context for CPU (shorter windows)
    "change_context_len": 10,
    "completion_context_len": 50,
    "gameover_context_len": 150,
    # Smaller replay sizes for CPU
    "change_replay_size": 8,
    "completion_replay_size": 40,
    "gameover_replay_size": 80,
}

GPU_PURE_DT_CONFIG = {
    **DEFAULT_DT_CONFIG,
    "embed_dim": 512,  # Larger model for GPU
    "num_layers": 6,  # More layers for GPU
    "max_context_len": 400,  # Must accommodate gameover_context_len
    "batch_size": 16,  # Smaller batch to fit longer context in memory
    # ViT configuration for GPU
    "vit_num_layers": 6,  # More ViT layers for GPU
    "vit_num_heads": 16,  # More attention heads for GPU (512/16 = 32)
    "vit_patch_size": 8,  # Keep patch size same
    "vit_cell_embed_dim": 128,  # Larger cell embeddings for GPU
    # Hierarchical context for GPU (longer windows for better performance)
    "change_context_len": 20,
    "completion_context_len": 150,
    "gameover_context_len": 400,
    # Larger replay sizes for GPU
    "change_replay_size": 32,
    "completion_replay_size": 160,
    "gameover_replay_size": 320,
}


def load_dt_config(device: str = None) -> Dict[str, Any]:
    """Load Pure DT configuration based on device.

    Args:
        device: 'cpu', 'cuda', or None for auto-detection

    Returns:
        Configuration dictionary
    """
    # Start with default config
    config = DEFAULT_DT_CONFIG.copy()

    # Apply device-specific config
    if device == "cpu":
        config.update(CPU_PURE_DT_CONFIG)
    elif device == "cuda" or device == "gpu":
        config.update(GPU_PURE_DT_CONFIG)
    elif device is None:
        # Auto-detect device
        import torch

        if torch.cuda.is_available():
            config.update(GPU_PURE_DT_CONFIG)
        else:
            config.update(CPU_PURE_DT_CONFIG)

    # Override with environment variables if present
    config = _apply_environment_overrides(config)

    return config


def _apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    env_mapping = {
        "PURE_DT_LEARNING_RATE": ("learning_rate", float),
        "PURE_DT_MAX_CONTEXT_LEN": ("max_context_len", int),
        "PURE_DT_EMBED_DIM": ("embed_dim", int),
        "PURE_DT_NUM_LAYERS": ("num_layers", int),
        "PURE_DT_EPOCHS": ("epochs_per_training", int),
    }

    for env_var, (config_key, type_func) in env_mapping.items():
        if env_var in os.environ:
            try:
                config[config_key] = type_func(os.environ[env_var])
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid value for {env_var}: {e}")

    return config


def get_loss_config_summary(config: Dict[str, Any]) -> str:
    """Get a summary of the configuration for logging."""
    lr = config.get("learning_rate", "unknown")
    epochs = config.get("epochs_per_training", "unknown")
    context = config.get("max_context_len", "unknown")

    summary = (
        f"Pure DT Config: lr={lr}, epochs={epochs}, context={context}"
    )

    return summary


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_keys = [
        "embed_dim",
        "num_layers",
        "num_heads",
        "learning_rate",
        "max_context_len",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate ranges
    if config["max_context_len"] < 1:
        raise ValueError("max_context_len must be >= 1")

    if config["embed_dim"] % config["num_heads"] != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    # Validate hierarchical context lengths respect max_context_len constraint
    max_context = config["max_context_len"]
    change_context = config.get("change_context_len", 0)
    completion_context = config.get("completion_context_len", 0)
    gameover_context = config.get("gameover_context_len", 0)

    if change_context > max_context:
        raise ValueError(
            f"change_context_len ({change_context}) exceeds max_context_len ({max_context})"
        )
    if completion_context > max_context:
        raise ValueError(
            f"completion_context_len ({completion_context}) exceeds max_context_len ({max_context})"
        )
    if gameover_context > max_context:
        raise ValueError(
            f"gameover_context_len ({gameover_context}) exceeds max_context_len ({max_context})"
        )

    return True


# Export main functions
__all__ = [
    "DEFAULT_DT_CONFIG",
    "CPU_PURE_DT_CONFIG",
    "GPU_PURE_DT_CONFIG",
    "load_dt_config",
    "get_loss_config_summary",
    "validate_config",
]
