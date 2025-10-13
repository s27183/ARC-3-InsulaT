"""
InsulaAgent Configuration

This module provides configuration management for InsulaAgent, an online supervised
learning architecture with insular cortex-inspired multi-level integration.
Includes configurable model architecture and training parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InsulaConfig:
    """InsulaAgent Configuration.

    Organized into logical sections:
    - CORE ARCHITECTURE: Model structure
    - TRAINING CONFIGURATION: Optimization and training schedule
    - TEMPORAL CREDIT ASSIGNMENT: Credit assignment settings
    - EXPERIENCE REPLAY: Replay buffer and sampling
    """

    # ============================================================================
    # CORE ARCHITECTURE
    # ============================================================================

    # Decision Transformer (Temporal Processing)
    embed_dim: int = 256  # Transformer embedding dimension
    num_layers: int = 4  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    max_context_len: int = 40  # Maximum positional embedding capacity

    # Hierarchical Context Windows (Head-Specific)
    change_context_len: int = 5  # Immediate effects (γ=0.7 → ~13-step effective)
    completion_context_len: int = 20  # Goal sequences (γ=0.8 → ~21-step effective)
    gameover_context_len: int = 40  # Failure chains (γ=0.9 → ~44-step effective)

    # ViT State Encoder (Spatial Processing)
    vit_patch_size: int = 8  # Patch size (8×8 = 64 patches for 64×64 grid)
    vit_num_layers: int = 4  # ViT transformer layers
    vit_num_heads: int = 8  # ViT attention heads
    vit_dropout: float = 0.1  # ViT dropout rate
    vit_use_cls_token: bool = True  # Use CLS token vs global pooling
    vit_cell_embed_dim: int = 64  # Dimension for learned cell embeddings (0-15)
    vit_pos_dim_ratio: float = 0.5  # Position encoding dimension ratio
    vit_use_patch_pos_encoding: bool = False  # Patch-level positional encoding

    # Multi-Head Prediction Architecture
    use_change_head: bool = True  # Always True (change head is required)
    use_completion_head: bool = True  # Optional: predict level completion
    use_gameover_head: bool = True  # Optional: predict GAME_OVER avoidance

    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================

    # Optimizer Settings
    learning_rate: float = 1e-4  # Adam learning rate
    weight_decay: float = 1e-5  # L2 regularization
    gradient_clip_norm: float = 1.0  # Gradient clipping norm

    # Training Schedule
    train_frequency: int = 5  # Train every N actions
    batch_size: int = 16  # Batch size for training
    epochs_per_training: int = 1  # Number of epochs per training session
    min_buffer_size: int = 5  # Minimum experience buffer size to start training

    # Loss Function
    action_entropy_coeff: float = 0.0001  # Entropy coefficient for discrete actions
    coord_entropy_coeff: float = 0.00001  # Entropy coefficient for coordinates

    # ============================================================================
    # TEMPORAL CREDIT ASSIGNMENT
    # ============================================================================

    temporal_credit: bool = False  # Train on all timesteps or final only

    # Eligibility Decay Rates (ONLY used when temporal_credit=True)
    use_learned_decay: bool = False  # Learn decay rates during training
    change_eligibility_decay: float = 0.7  # Fast decay for immediate effects
    completion_eligibility_decay: float = 0.8  # Medium decay for goal sequences
    gameover_eligibility_decay: float = 0.9  # Slow decay for failure chains

    # ============================================================================
    # EXPERIENCE REPLAY
    # ============================================================================

    # Buffer Management
    max_buffer_size: int = 200000  # Maximum experience buffer size
    experience_sample_rate: float = 1.0  # Fraction of experiences to use

    # Head-Specific Replay Sizes (Importance-Weighted Sampling)
    change_replay_size: int = 16  # Change is frequent → small batch
    completion_replay_size: int = 160  # Completion is rare → large batch
    gameover_replay_size: int = 16  # GAME_OVER persists → small batch

    # Sequence Length Variation
    replay_variation_min: float = 0.5  # Minimum variation factor
    replay_variation_max: float = 1.0  # Maximum variation factor

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate ranges
        if self.max_context_len < 1:
            raise ValueError("max_context_len must be >= 1")

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Change head must always be enabled
        if not self.use_change_head:
            raise ValueError("use_change_head must be True (change head is required)")

        # At least one head must be enabled
        if not (self.use_change_head or self.use_completion_head or self.use_gameover_head):
            raise ValueError("At least one prediction head must be enabled")

        # Validate hierarchical context lengths
        if self.change_context_len > self.max_context_len:
            raise ValueError(
                f"change_context_len ({self.change_context_len}) exceeds "
                f"max_context_len ({self.max_context_len})"
            )

        if self.use_completion_head and self.completion_context_len > self.max_context_len:
            raise ValueError(
                f"completion_context_len ({self.completion_context_len}) exceeds "
                f"max_context_len ({self.max_context_len})"
            )

        if self.use_gameover_head and self.gameover_context_len > self.max_context_len:
            raise ValueError(
                f"gameover_context_len ({self.gameover_context_len}) exceeds "
                f"max_context_len ({self.max_context_len})"
            )

    def summary(self) -> str:
        """Get a summary of the configuration for logging."""
        return (
            f"InsulaConfig: lr={self.learning_rate}, "
            f"epochs={self.epochs_per_training}, "
            f"context={self.max_context_len}"
        )


# ============================================================================
# Factory Functions
# ============================================================================

def default_config() -> InsulaConfig:
    """Create default InsulaAgent configuration.

    Returns:
        InsulaConfig with default values
    """
    return InsulaConfig()


def cpu_config() -> InsulaConfig:
    """Create CPU-optimized InsulaAgent configuration.

    Returns:
        InsulaConfig optimized for CPU training
    """
    return InsulaConfig(
        # Smaller model for CPU
        embed_dim=128,
        num_layers=2,
        num_heads=4,  # 128/4 = 32
        max_context_len=150,
        # ViT configuration for CPU
        vit_num_layers=2,
        vit_num_heads=4,
        vit_cell_embed_dim=32,
        # Context lengths (same as default)
        change_context_len=5,
        completion_context_len=20,
        gameover_context_len=40,
        # Replay sizes (same as default)
        change_replay_size=16,
        completion_replay_size=160,
        gameover_replay_size=16,
    )


def gpu_config() -> InsulaConfig:
    """Create GPU-optimized InsulaAgent configuration.

    Returns:
        InsulaConfig optimized for GPU training
    """
    return InsulaConfig(
        # Balanced GPU config (same as default for most params)
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_context_len=300,
        # ViT configuration for GPU
        vit_num_layers=4,
        vit_num_heads=8,
        vit_cell_embed_dim=64,
        # Context lengths (same as default)
        change_context_len=5,
        completion_context_len=20,
        gameover_context_len=40,
        # Larger replay sizes for GPU
        change_replay_size=32,
        completion_replay_size=320,
        gameover_replay_size=32,
    )


def load_config(device: Optional[str] = None) -> InsulaConfig:
    """Load InsulaAgent configuration based on device.

    Args:
        device: 'cpu', 'cuda', 'gpu', or None for auto-detection

    Returns:
        InsulaConfig instance
    """
    if device == "cpu":
        config = cpu_config()
    elif device in ("cuda", "gpu"):
        config = gpu_config()
    elif device is None:
        # Auto-detect device
        import torch
        if torch.cuda.is_available():
            config = gpu_config()
        else:
            config = cpu_config()
    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu', 'cuda', 'gpu', or None")

    # Apply environment variable overrides if present
    config = _apply_environment_overrides(config)

    return config


def _apply_environment_overrides(config: InsulaConfig) -> InsulaConfig:
    """Apply environment variable overrides to configuration.

    Args:
        config: InsulaConfig instance to modify

    Returns:
        Modified InsulaConfig instance
    """
    env_mapping = {
        "PURE_DT_LEARNING_RATE": ("learning_rate", float),
        "PURE_DT_MAX_CONTEXT_LEN": ("max_context_len", int),
        "PURE_DT_EMBED_DIM": ("embed_dim", int),
        "PURE_DT_NUM_LAYERS": ("num_layers", int),
        "PURE_DT_EPOCHS": ("epochs_per_training", int),
    }

    for env_var, (attr_name, type_func) in env_mapping.items():
        if env_var in os.environ:
            try:
                value = type_func(os.environ[env_var])
                setattr(config, attr_name, value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid value for {env_var}: {e}")

    # Re-validate after overrides
    config.validate()

    return config


# Export main functions
__all__ = [
    "InsulaConfig",
    "default_config",
    "cpu_config",
    "gpu_config",
    "load_config",
]
