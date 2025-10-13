"""
InsulaAgent Configuration

This module provides configuration management for InsulaAgent, an online supervised
learning architecture with insular cortex-inspired multi-level integration.
Includes configurable model architecture and training parameters.
"""

import os
from typing import Dict, Any


# Default configuration for InsulaAgent
DEFAULT_DT_CONFIG = {
    # Model Architecture
    "embed_dim": 256,  # Transformer embedding dimension
    "num_layers": 4,  # Number of transformer layers
    "num_heads": 8,  # Number of attention heads
    "max_context_len": 40,  # Maximum positional embedding capacity (must be >= longest head-specific context)

    # Training Parameters
    "learning_rate": 1e-4,  # Adam learning rate
    "weight_decay": 1e-5,  # L2 regularization
    "train_frequency": 5,  # Train every N actions (reduced to save compute with longer context)
    "min_buffer_size": 5,  # Minimum experience buffer size to start training

    # Loss Function Configuration
    "action_entropy_coeff": 0.0001,  # Entropy coefficient for discrete actions
    "coord_entropy_coeff": 0.00001,  # Entropy coefficient for coordinates

    # Temporal Credit Assignment
    # TODO: experiment with/without temporal credit and with different decay rates to do A/B test
    "temporal_credit": False, # Backward temporal credit assignment in a state/action sequence
    "eligibility_decay": 0.8,  # Exponential decay for temporal credit (0.8 = moderate decay, 1.0 = no temporal credit)

    # Hierarchical Context Windows (Head-Specific)
    # Note: Context length determines temporal credit assignment window (how far back to look)
    # Effective context ≈ -ln(0.01)/ln(γ) where γ is eligibility_decay
    # Design principle: Match context_len to effective context (95-100% coverage for efficiency)
    "change_context_len": 5,   # Immediate effects (γ=0.7 → ~13-step effective, 38% coverage - tight for speed)
    "completion_context_len": 20,  # Goal sequences (γ=0.8 → ~21-step effective, 95% coverage)
    "gameover_context_len": 40,    # Failure chains (γ=0.9 → ~44-step effective, 91% coverage)

    # Head-Specific Eligibility Decay (for temporal credit assignment)
    "use_learned_decay": False,  # Toggle learned vs fixed decay rates
    "change_eligibility_decay": 0.7,  # Fast decay for immediate effects (init value if learned)
    "completion_eligibility_decay": 0.8,  # Medium decay for goal-oriented actions (init value if learned)
    "gameover_eligibility_decay": 0.9,  # Slow decay for long causal chains (init value if learned)

    # Importance-Weighted Replay (Head-Specific Replay Sizes)
    # NOTE: Replay size = batch size (how many sequences to sample per training round)
    # Rationale based on event frequency AND buffer persistence:
    "change_replay_size": 16,  # Change is frequent (50-70% of actions) → small batch
    "completion_replay_size": 160,  # Completion is rare (1 per level) AND buffer clears immediately → large batch to maximize learning before buffer clears
    "gameover_replay_size": 16,  # GAME_OVER is moderately frequent (10-30%) AND persists in buffer across multiple training rounds → small batch to avoid oversampling same failures

    # Replay Variation (for completion and GAME_OVER sequences)
    "replay_variation_min": 0.5,  # Minimum variation factor (80% of target length)
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
    "vit_pos_dim_ratio": 0.5,  # Position encoding dimension as ratio of cell_embed_dim (pos_dim = cell_embed_dim * ratio)
    "vit_use_patch_pos_encoding": False,  # Whether to use patch-level positional encoding (redundant with cell-level)

    # Head Configuration (Triple-Head Architecture)
    "use_change_head": True,        # Always True (change head is required baseline)
    "use_completion_head": True,    # Optional: predict level completion (goal-oriented)
    "use_gameover_head": True,      # Optional: predict GAME_OVER avoidance (safety-aware)
}

# Device-specific configurations
CPU_PURE_DT_CONFIG = {
    **DEFAULT_DT_CONFIG,
    "embed_dim": 128,  # Smaller model for CPU
    "num_layers": 2,  # Fewer layers for CPU
    "max_context_len": 150,  # Must accommodate gameover_context_len
    # ViT configuration for CPU
    "vit_num_layers": 2,  # Fewer ViT layers for CPU
    "vit_num_heads": 4,  # Fewer attention heads for CPU (128/4 = 32)
    "vit_patch_size": 8,  # Keep patch size same
    "vit_cell_embed_dim": 32,  # Smaller cell embeddings for CPU
    # Hierarchical context for CPU (same as DEFAULT)
    "change_context_len": 5,
    "completion_context_len": 20,
    "gameover_context_len": 40,
    # Smaller replay sizes for CPU
    "change_replay_size": 16,
    "completion_replay_size": 160,
    "gameover_replay_size": 16,  # Match change_replay_size (see rationale in DEFAULT_DT_CONFIG)
}

GPU_PURE_DT_CONFIG = {
    **DEFAULT_DT_CONFIG,
    # Use DEFAULT_DT_CONFIG values for balanced GPU config
    # (Previous aggressive config: embed_dim=512, num_layers=6 → 50M params)
    # (New conservative config matches DEFAULT → ~10M params, reasonable scaling from 2.3M CPU)
    "embed_dim": 256,  # Same as DEFAULT (2× CPU)
    "num_layers": 4,  # Same as DEFAULT (2× CPU)
    "num_heads": 8,  # Same as DEFAULT
    "max_context_len": 300,  # Same as DEFAULT
    # ViT configuration for GPU
    "vit_num_layers": 4,  # Same as DEFAULT (2× CPU)
    "vit_num_heads": 8,  # Same as DEFAULT (2× CPU, 256/8 = 32)
    "vit_patch_size": 8,  # Keep patch size same
    "vit_cell_embed_dim": 64,  # Same as DEFAULT (2× CPU)
    # Hierarchical context for GPU (same as DEFAULT)
    "change_context_len": 5,
    "completion_context_len": 20,
    "gameover_context_len": 40,
    # Replay sizes for GPU (modest increase from CPU)
    "change_replay_size": 32,  # 2× CPU
    "completion_replay_size": 320,  # 2× CPU
    "gameover_replay_size": 32,  # Match change_replay_size scaling (2× CPU, see rationale in DEFAULT_DT_CONFIG)
}


def load_dt_config(device: str = None) -> Dict[str, Any]:
    """Load InsulaAgent configuration based on device.

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
        f"InsulaAgent Config: lr={lr}, epochs={epochs}, context={context}"
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

    # Validate head configuration
    use_change = config.get("use_change_head", True)
    use_completion = config.get("use_completion_head", True)
    use_gameover = config.get("use_gameover_head", True)

    # Change head must always be enabled (it's the baseline)
    if not use_change:
        raise ValueError("use_change_head must be True (change head is required)")

    # At least one head must be enabled
    if not (use_change or use_completion or use_gameover):
        raise ValueError("At least one prediction head must be enabled")

    # Validate hierarchical context lengths only for enabled heads
    max_context = config["max_context_len"]

    # Always validate change context (required head)
    change_context = config.get("change_context_len", 0)
    if change_context > max_context:
        raise ValueError(
            f"change_context_len ({change_context}) exceeds max_context_len ({max_context})"
        )

    # Only validate completion context if completion head is enabled
    if use_completion:
        completion_context = config.get("completion_context_len", 0)
        if completion_context > max_context:
            raise ValueError(
                f"completion_context_len ({completion_context}) exceeds max_context_len ({max_context})"
            )

    # Only validate gameover context if gameover head is enabled
    if use_gameover:
        gameover_context = config.get("gameover_context_len", 0)
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
