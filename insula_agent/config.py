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

    # Unified Context Length (All Heads)
    # Number of past actions (k) for sequence construction
    # Creates sequences: k+1 states + k actions = (2k+1) total elements
    # With context_len=50: 51 states + 50 actions = 101 elements
    # Sequence structure: [s₀, a₀, s₁, a₁, ..., s₄₉, a₄₉, s₅₀]
    # Positional embeddings sized accordingly: (2*50 + 1) = 101 positions
    #
    # Temporal hierarchy achieved through head-specific decay rates (γ):
    # - Change head (γ=0.725): Strong recency bias → focus on recent 5 actions (80% weight)
    # - Completion head (γ=0.851): Moderate recency bias → focus on recent 10 actions (80% weight)
    # - Gameover head (γ=0.926): Mild recency bias → focus on recent 20 actions (80% weight)
    #
    # Rationale for 50:
    # - Extended temporal context for complex multi-step reasoning
    # - Captures longer causal chains (up to 40+ action sequences)
    # - Hierarchical focus windows (5/10/20 actions) maintained via decay rates
    # - Efficient with modern transformers (101 tokens = 2×50+1)
    # - Allows gameover head to see full failure causality across extended episodes
    context_len: int = 5

    # ViT State Encoder (Spatial Processing)
    vit_patch_size: int = 8  # Default Patch size (8×8 = 64 patches for 64×64 grid) - will be replaced by dynamic patch size per game
    vit_num_layers: int = 4  # ViT transformer layers
    vit_num_heads: int = 8  # ViT attention heads
    vit_dropout: float = 0.1  # ViT dropout rate
    vit_use_cls_token: bool = True  # Use CLS token vs global pooling
    vit_cell_embed_dim: int = 64  # Dimension for learned cell embeddings (0-15)
    vit_pos_dim_ratio: float = 0.5  # Position encoding dimension ratio
    vit_use_patch_pos_encoding: bool = False  # Patch-level positional encoding

    # Multi-Head Prediction Architecture
    # TODO: enable/disable learned heads for ablation studies
    use_change_head: bool = True  # Always True (change head is required)
    use_change_momentum_head: bool = True  # Predict change momentum (optional, for ablation studies)
    use_completion_head: bool = False  # Predict level completion (trajectory-level rewards)
    use_gameover_head: bool = False  # Predict GAME_OVER avoidance (trajectory-level rewards)

    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================

    # Optimizer Settings
    # TODO: should we use a different learning rates for the ViT and decision models?
    learning_rate: float = 1e-4  # Adam learning rate
    weight_decay: float = 1e-5  # L2 regularization
    gradient_clip_norm: float = 1.0  # Gradient clipping norm

    # Training Schedule
    train_frequency: int = 5  # Train every N actions
    epochs_per_training: int = 1  # Number of epochs per training session

    # Level Completion Behavior
    reset_on_level_completion: bool = False  # Reset model/optimizer on level completion (ablation study)
                                              # False (default): Transfer learning across levels
                                              # True: Fresh start each level (independent learning)

    #

    # ============================================================================
    # MULTI-TIMESTEP FORWARD PREDICTION
    # ============================================================================

    # Enable training on all timesteps vs final timestep only
    # When True: Model predicts at ALL states in sequence (PAST + PRESENT forward predictions)
    # When False: Model predicts only at final state (PRESENT forward prediction only)
    # Recommendation: True (provides k+1 training signals per sequence, improves representations)
    temporal_update: bool = True

    # Temporal Weighting (ONLY used when temporal_update=True)
    # Controls relative weighting of predictions at different timesteps in sequence
    # γ = 1.0: Equal weighting (all predictions contribute equally)
    # γ < 1.0: Recency bias (recent predictions weighted more heavily)
    #
    # Head-Specific Rationale:
    # - Change head (γ=1.0): Action-level rewards with immediate causality
    #   → Each (state, action, change_reward) tuple independently valid
    #   → No temporal structure needed for direct cause-effect
    #
    # - Completion head (γ=0.8): Trajectory-level rewards (all actions get credit)
    #   → Early actions may be exploratory/incidental, not goal-directed
    #   → Moderate recency bias denoises: recent actions more likely critical
    #
    # - Gameover head (γ=0.9): Trajectory-level rewards (all actions penalized)
    #   → Failure causality can be immediate OR delayed
    #   → Mild recency bias balances immediate vs cascading failures
    #
    # Memory Reconsolidation: Buffer stores action-level rewards, replay assigns
    # trajectory-level rewards (matches hippocampal replay + dopamine modulation)
    use_learned_decay: bool = False  # Learn decay rates during training (experimental)

    # Hierarchical Temporal Decay Rates (Optimized for Multi-Timescale Learning)
    # All heads see same 50-step context, but weight timesteps differently:
    # - γ closer to 0: Strong recency bias (focus on recent actions)
    # - γ closer to 1: Weak recency bias (nearly uniform weighting)
    # Computed to concentrate 80% of weight on target window for context_len=50
    change_temporal_update_decay: float = 1.0  # Focus on recent 3 actions (immediate effects)
    change_momentum_temporal_update_decay: float = 1.0  # Focus on recent 3 actions (same as change)
    completion_temporal_update_decay: float = 0.95  # Focus on recent 20 actions (medium-term planning)
    gameover_temporal_update_decay: float = 0.85  # Focus on recent 10 actions (long-term causality)

    # ============================================================================
    # EXPERIENCE REPLAY
    # ============================================================================

    # Buffer Management
    max_buffer_size: int = 200_000  # Maximum experience buffer size

    # Trajectory Reward Assignment (Memory Reconsolidation)
    # When enabled: Assign trajectory-level rewards during replay, not in buffer
    # - Completion: All actions in successful sequence get reward=1.0
    # - Gameover: All actions in failed sequence get reward=0.0
    # - Change: Keeps action-level rewards (immediate causality)
    # Matches biological mechanism: Dopamine modulates replayed sequences (Gomperts et al., 2015)
    use_trajectory_rewards: bool = True  # Enable reward revaluation during replay

    # Head-Specific Replay Sizes (Importance-Weighted Sampling)
    change_replay_size: int = 8  # Change is frequent → small batch (shared with momentum head)
    # Note: change_momentum_replay_size removed - momentum head shares change_replay_size
    completion_replay_size: int = 8  # Completion is rare → large batch
    gameover_replay_size: int = 8  # GAME_OVER persists → small batch

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate context length
        if self.context_len < 1:
            raise ValueError("context_len must be >= 1")

        # Validate embed_dim divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Change head must always be enabled
        if not self.use_change_head:
            raise ValueError("use_change_head must be True (change head is required)")

        # At least one head must be enabled
        if not (self.use_change_head or self.use_change_momentum_head or self.use_completion_head or self.use_gameover_head):
            raise ValueError("At least one prediction head must be enabled")

        # Validate temporal decay rates (0, 1]
        if not (0 < self.change_temporal_update_decay <= 1.0):
            raise ValueError(
                f"change_temporal_decay ({self.change_temporal_update_decay}) must be in (0, 1]"
            )

        if not (0 < self.change_momentum_temporal_update_decay <= 1.0):
            raise ValueError(
                f"change_momentum_temporal_update_decay ({self.change_momentum_temporal_update_decay}) must be in (0, 1]"
            )

        if not (0 < self.completion_temporal_update_decay <= 1.0):
            raise ValueError(
                f"completion_temporal_decay ({self.completion_temporal_update_decay}) must be in (0, 1]"
            )

        if not (0 < self.gameover_temporal_update_decay <= 1.0):
            raise ValueError(
                f"gameover_temporal_decay ({self.gameover_temporal_update_decay}) must be in (0, 1]"
            )

    def summary(self) -> str:
        """Get a summary of the configuration for logging."""
        return (
            f"InsulaConfig: lr={self.learning_rate}, "
            f"epochs={self.epochs_per_training}, "
            f"context={self.context_len}"
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
        InsulaConfig optimized for CPU training (smaller for fast iteration)
    """
    return InsulaConfig(
        # Smaller transformer for CPU
        embed_dim=128,
        num_layers=2,
        num_heads=4,  # 128/4 = 32
        context_len=5,  # Shorter context for CPU (faster development/testing)
        # Smaller ViT for CPU
        vit_num_layers=2,
        vit_num_heads=4,
        vit_cell_embed_dim=32,
        # Smaller replay sizes for faster training
        change_replay_size=8,       # Shared by change and momentum heads
        completion_replay_size=8,
        gameover_replay_size=8,
        # CPU-specific decay rates (for context_len=25)
        change_temporal_update_decay=0.725,
        completion_temporal_update_decay=0.859,
        gameover_temporal_update_decay=0.990,
    )


def gpu_config() -> InsulaConfig:
    """Create GPU-optimized InsulaAgent configuration for H100.

    Balanced scaling for context_len=50:
    - 2x longer context (50 vs 25 steps)
    - 2.3x model capacity (24M vs 11M parameters)
    - Reduced batch sizes to avoid OOM (32-64 vs 128)
    - Memory usage: ~40-60 GB during training (with gradients + activations)

    Returns:
        InsulaConfig optimized for H100 GPU training
    """
    return InsulaConfig(
        # Scaled Transformer architecture (H100-optimized)
        embed_dim=256,              # 256 → 384 (1.5x capacity)
        num_layers=4,               # 4 → 6 (deeper reasoning)
        num_heads=8,               # 8 → 12 (more attention patterns)
        context_len=5,             # 25 → 50 (2x longer context)
        # ViT architecture (keep same - spatial is already good)
        vit_num_layers=4,
        vit_num_heads=8,
        vit_cell_embed_dim=64,
        # Balanced replay sizes for H100 memory constraints
        # With context_len=50 (101 tokens), batch_size=128 causes OOM
        change_replay_size=64,      # Adaptive batch rarely hits this anyway
        completion_replay_size=64,  # Key: 32x oversampling still strong signal
        gameover_replay_size=64,    # Adaptive batch rarely hits this anyway
        # Decay rates optimized for context_len=50
        change_temporal_update_decay=0.5,          # Focus on recent 5 actions
        change_momentum_temporal_update_decay=0.5, # Focus on recent 5 actions
        completion_temporal_update_decay=0.95,      # Focus on recent 20 actions
        gameover_temporal_update_decay=0.85,        # Focus on recent 10 actions
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
        "PURE_DT_CONTEXT_LEN": ("context_len", int),
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
