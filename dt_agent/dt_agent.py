"""
Pure Decision Transformer for ARC-AGI-3

This module implements a Pure Decision Transformer that replaces the hybrid bandit-DT
architecture with a unified transformer-based approach for direct action prediction.

Architecture:
- StateEncoder: CNN backbone (reuses proven bandit architecture)
- ActionEmbedding: Handles 4101 actions (ACTION1-5 + 64x64 coordinates)
- PureDecisionTransformer: Interleaved state-action sequences → direct action classification
- ActionSampler: Temperature sampling with action masking

Key Features:
- State-action sequences with local context (k=15-20 steps)
- Direct action classification over 4101 action space
- Configurable loss functions (cross-entropy, selective, hybrid)
- Temperature-based exploration
- Reuses bandit CNN for spatial reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import numpy as np
import random
import time
import logging
import os
import sys
import hashlib
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState

from dt_agent.utils import (
    setup_experiment_directory,
    setup_logging,
    get_environment_directory,
)
from dt_agent.config import load_dt_config, get_loss_config_summary, validate_config


# ============================================================================
# DEPRECATED: CNN StateEncoder (replaced by ViTStateEncoder)
# ============================================================================
# This CNN-based StateEncoder has been replaced by ViTStateEncoder for better
# handling of non-local causality in ARC-AGI-3. The Vision Transformer provides
# global attention from layer 1, which is critical for dynamic grid games.
#
# Keeping this class for reference and potential future comparisons.
# ============================================================================


class StateEncoder(nn.Module):
    """
    [DEPRECATED] CNN State Encoder - replaced by ViTStateEncoder.

    This CNN-based encoder uses 4 convolutional layers for spatial reasoning.
    It has been replaced by ViTStateEncoder which provides:
    - Global attention from layer 1 (better for non-local causality)
    - Hierarchical transformer architecture (ViT spatial + Transformer temporal)
    - Learned spatial relationships via self-attention

    Kept for reference and potential CNN vs ViT comparisons.
    """

    def __init__(self, input_channels=16, embed_dim=256, freeze_weights=False):
        super().__init__()

        # Shared convolutional backbone (from bandit ActionModel)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Spatial pooling and projection to transformer dimension
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)  # 256×64×64 → 256×1×1
        self.state_projection = nn.Linear(256, embed_dim)

        if freeze_weights:
            # Option: freeze pre-trained bandit weights for transfer learning
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, grid_states):
        """
        Args:
            grid_states: [batch, 16, 64, 64] - One-hot encoded grids
        Returns:
            state_repr: [batch, embed_dim] - State representations
        """
        # Shared convolutional backbone
        x = F.relu(self.conv1(grid_states))  # [batch, 32, 64, 64]
        x = F.relu(self.conv2(x))  # [batch, 64, 64, 64]
        x = F.relu(self.conv3(x))  # [batch, 128, 64, 64]
        x = F.relu(self.conv4(x))  # [batch, 256, 64, 64]

        # Global spatial representation
        x = self.spatial_pool(x).flatten(1)  # [batch, 256]
        state_repr = self.state_projection(x)  # [batch, embed_dim]

        return state_repr


# ============================================================================
# End of deprecated CNN StateEncoder
# ============================================================================


class ViTStateEncoder(nn.Module):
    """Vision Transformer State Encoder with Learned Cell Embeddings.

    Encodes 64×64 grids into vector representations using patch-based
    self-attention with learned embeddings for each cell value (0-15).

    This approach is more efficient than one-hot encoding:
    - 16x fewer values per patch (64 cells vs 1024 one-hot values)
    - Learned color representations (like word embeddings in NLP)
    - Standard transformer architecture philosophy

    Args:
        num_colors: Number of possible cell values (default: 16 for colors 0-15)
        embed_dim: Transformer embedding dimension
        cell_embed_dim: Dimension for each cell embedding (0-15)
        patch_size: Size of each square patch (default: 8 for 8×8 patches)
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_cls_token: Whether to use CLS token (True) or global pooling (False)
    """

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 256,
        cell_embed_dim: int = 64,
        patch_size: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.num_colors = num_colors
        self.cell_embed_dim = cell_embed_dim
        self.patch_size = patch_size
        self.grid_size = 64
        self.num_patches = (self.grid_size // patch_size) ** 2  # 64 patches for 8×8
        self.use_cls_token = use_cls_token

        # Learned cell embedding: each color (0-15) → vector
        self.cell_embedding = nn.Embedding(num_colors, cell_embed_dim)

        # Patch projection: aggregated cell embeddings → transformer dimension
        self.patch_projection = nn.Linear(cell_embed_dim, embed_dim)

        # 2D learnable positional embeddings for patch grid
        num_patches_per_dim = self.grid_size // patch_size  # 8
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches_per_dim, num_patches_per_dim, embed_dim) * 0.02
        )

        # CLS token for global representation
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # Post-norm (standard ViT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def _extract_patches(self, grid_states: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from integer grid.

        Args:
            grid_states: [batch, 64, 64] - Integer grid with values 0-15

        Returns:
            patches: [batch, num_patches_h, num_patches_w, patch_size*patch_size]
                     [batch, 8, 8, 64] - 64 cells per patch
        """
        batch_size = grid_states.shape[0]

        # Unfold to extract patches: [batch, 8, 8, 8, 8]
        patches = grid_states.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )

        # Flatten each patch: [batch, 8, 8, 64]
        num_patches_per_dim = self.grid_size // self.patch_size
        patches = patches.reshape(
            batch_size,
            num_patches_per_dim,
            num_patches_per_dim,
            self.patch_size * self.patch_size,  # 64 cells per patch
        )

        return patches

    def _embed_and_aggregate_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed each cell and aggregate within patches.

        Args:
            patches: [batch, 8, 8, 64] - Integer cell values 0-15

        Returns:
            patch_embeddings: [batch, 8, 8, cell_embed_dim]
        """
        # Embed each cell: [batch, 8, 8, 64, cell_embed_dim]
        # Ensure integer type for embedding lookup
        cell_embeddings = self.cell_embedding(patches.long())

        # Aggregate within each patch (mean pooling)
        # [batch, 8, 8, cell_embed_dim]
        patch_embeddings = cell_embeddings.mean(dim=3)

        return patch_embeddings

    def forward(self, grid_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_states: [batch, 64, 64] - Integer grid with values 0-15

        Returns:
            state_repr: [batch, embed_dim] - State representations
        """
        batch_size = grid_states.shape[0]

        # Extract patches: [batch, 8, 8, 64]
        patches = self._extract_patches(grid_states)

        # Embed cells and aggregate: [batch, 8, 8, cell_embed_dim]
        patch_repr = self._embed_and_aggregate_patches(patches)

        # Project to transformer dimension: [batch, 8, 8, embed_dim]
        x = self.patch_projection(patch_repr)

        # Add 2D positional embeddings
        x = x + self.pos_embed

        # Flatten to sequence: [batch, 64, embed_dim]
        x = x.reshape(batch_size, self.num_patches, -1)

        # Prepend CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, 65, embed_dim]

        # Transformer encoding
        x = self.transformer(x)

        # Extract global representation
        if self.use_cls_token:
            state_repr = x[:, 0]  # [batch, embed_dim] - CLS token
        else:
            state_repr = x.mean(dim=1)  # [batch, embed_dim] - Average pooling

        # Final normalization
        state_repr = self.norm(state_repr)

        return state_repr


class ActionEmbedding(nn.Module):
    """Action embedding for 4101 action vocabulary: ACTION1-5 + coordinates."""

    def __init__(self, embed_dim=256):
        super().__init__()
        # 4101 actions: ACTION1-5 (indices 0-4) + coordinates (indices 5-4100)
        self.action_embedding = nn.Embedding(4101, embed_dim)

    def forward(self, action_indices):
        """
        Args:
            action_indices: [batch, seq_len] with values in [0, 4100]
        Returns:
            action_embeddings: [batch, seq_len, embed_dim]
        """
        return self.action_embedding(action_indices)


class DecisionTransformer(nn.Module):
    """End-to-end transformer for ARC-AGI action prediction using state-action sequences."""

    def __init__(
        self,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_context_len=20,
        # ViT encoder parameters
        vit_cell_embed_dim=64,
        vit_patch_size=8,
        vit_num_layers=4,
        vit_num_heads=8,
        vit_dropout=0.1,
        vit_use_cls_token=True,
    ):
        super().__init__()

        # Component modules - Use ViT State Encoder with learned cell embeddings
        self.state_encoder = ViTStateEncoder(
            num_colors=16,
            embed_dim=embed_dim,
            cell_embed_dim=vit_cell_embed_dim,
            patch_size=vit_patch_size,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            dropout=vit_dropout,
            use_cls_token=vit_use_cls_token,
        )
        self.action_embedding = ActionEmbedding(embed_dim=embed_dim)

        # Positional encoding for temporal context
        self.pos_embedding = nn.Parameter(
            torch.randn(max_context_len * 2, embed_dim) * 0.02
        )

        # Decoder-only transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Action head for predicting changes caused by discrete actions (ACTION1-5)
        self.change_action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 5),  # ACTION1-5
        )

        # Coordinate head for predicting changes caused by spatial actions (64x64 coordinates)
        self.change_coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4096),  # 64x64 coordinates
        )

        # Action head for predicting level completion caused by discrete actions (ACTION1-5)
        self.completion_action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 5),
        )

        # Action head for predicting level completion caused by spatial actions (64x64 coordinates)
        self.completion_coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4096),
        )

    def build_state_action_sequence(self, states, actions) -> torch.Tensor:
        """Build interleaved state-action sequence: [s₀, a₀, s₁, a₁, ..., s_{t-1}, a_{t-1}, s_t]

        Args:
            states: [batch, seq_len+1, 64, 64] - k+1 integer grids with cell values 0-15
            actions: [batch, seq_len] - k past actions (excluding current to predict)

        Returns:
            sequence: [batch, 2*seq_len+1, embed_dim] - Interleaved state-action sequence
        """
        batch_size = states.shape[0]
        seq_len = actions.shape[1]  # k past actions
        sequence_tokens = []

        # Build interleaved sequence: s₀, a₀, s₁, a₁, ..., s_{k-1}, a_{k-1}, s_k
        for t in range(seq_len):
            # State at time t
            state_repr = self.state_encoder(states[:, t])  # [batch, embed_dim]
            sequence_tokens.append(state_repr)

            # Action at time t
            action_embed = self.action_embedding(actions[:, t])  # [batch, embed_dim]
            sequence_tokens.append(action_embed)

        # Add final state (current state) - no action after this (we predict it)
        final_state_repr = self.state_encoder(states[:, -1])  # [batch, embed_dim]
        sequence_tokens.append(final_state_repr)

        # Stack sequence tokens: [batch, 2*seq_len+1, embed_dim]
        sequence = torch.stack(sequence_tokens, dim=1)

        # Add positional encoding
        seq_positions = min(sequence.shape[1], self.pos_embedding.shape[0])
        sequence = sequence + self.pos_embedding[:seq_positions].unsqueeze(0)

        return sequence

    def forward(self, states, actions) -> dict[str, torch.Tensor]:
        """
        Args:
            states: [batch, seq_len+1, 64, 64] - k+1 integer grids with cell values 0-15 (past + current)
            actions: [batch, seq_len] - k past actions (0-4100)

        Returns:
            action_logits: [batch, 4101] - Logits over full action space for next action
        """
        # Build interleaved state-action sequence
        sequence = self.build_state_action_sequence(states, actions)

        # Create causal attention mask for autoregressive modeling
        seq_len = sequence.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(sequence.device)

        # Transformer forward pass (autoregressive)
        transformer_output = self.transformer(
            sequence,
            tgt_mask=causal_mask,
            memory=sequence,  # Self-attention over the sequence
        )

        # Extract final representation (current state)
        final_repr = transformer_output[:, -1]  # [batch, embed_dim]

        # Multi-head prediction
        change_action_logits = self.change_action_head(
            final_repr
        )  # [batch, 5] - ACTION1-5
        completion_action_logits = self.completion_action_head(
            final_repr
        )  # [batch, 5] - ACTION1-5
        change_coord_logits = self.change_coord_head(
            final_repr
        )  # [batch, 4096] - coordinates
        completion_coord_logits = self.completion_coord_head(
            final_repr
        )  # [batch, 4096] - coordinates

        # Concatenate for compatibility with existing interface
        change_logits = torch.cat(
            [change_action_logits, change_coord_logits], dim=1
        )  # [batch, 4101]
        completion_logits = torch.cat(
            [completion_action_logits, completion_coord_logits], dim=1
        )  # [batch, 4101]

        return {
            "change_logits": change_logits,
            "completion_logits": completion_logits,
        }


class DTAgent(Agent):
    """Self-contained Decision Transformer Agent for ARC-AGI-3.

    This agent uses a unified transformer architecture for direct action prediction.

    Features:
    - State-action sequence modeling with local context
    - Temperature-based exploration with action masking
    - Experience buffer with uniqueness tracking
    - Integrated logging
    - Full Agent interface compatibility
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Initialize random seeds for reproducibility
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()

        # Override MAX_ACTIONS - no limit for DT
        self.MAX_ACTIONS = float("inf")

        # Setup experiment directory and logging
        self.base_dir, log_file = setup_experiment_directory()
        setup_logging(log_file)
        self.logger = logging.getLogger(f"DTAgent_{self.game_id}")

        env_dir = get_environment_directory(self.base_dir, self.game_id)
        tensorboard_dir = os.path.join(env_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"DT Agent using device: {self.device}")

        # Model initialization
        self.pure_dt_config = load_dt_config(device=str(self.device))
        validate_config(self.pure_dt_config)

        config_summary = get_loss_config_summary(self.pure_dt_config)

        self.pure_dt_model = DecisionTransformer(
            embed_dim=self.pure_dt_config["embed_dim"],
            num_layers=self.pure_dt_config["num_layers"],
            num_heads=self.pure_dt_config["num_heads"],
            max_context_len=self.pure_dt_config["max_context_len"],
            # ViT encoder parameters
            vit_cell_embed_dim=self.pure_dt_config.get("vit_cell_embed_dim", 64),
            vit_patch_size=self.pure_dt_config.get("vit_patch_size", 8),
            vit_num_layers=self.pure_dt_config.get("vit_num_layers", 4),
            vit_num_heads=self.pure_dt_config.get("vit_num_heads", 8),
            vit_dropout=self.pure_dt_config.get("vit_dropout", 0.1),
            vit_use_cls_token=self.pure_dt_config.get("vit_use_cls_token", True),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.pure_dt_model.parameters(),
            lr=self.pure_dt_config["learning_rate"],
            weight_decay=self.pure_dt_config["weight_decay"],
        )

        # Grid info
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16

        # Scores as level indicators
        self.current_score = None

        # Experience buffer for training with uniqueness tracking
        self.experience_buffer = deque(maxlen=self.pure_dt_config["max_buffer_size"])
        self.experience_hashes = set()
        self.train_frequency = self.pure_dt_config["train_frequency"]

        # Game state and action mapping
        self.prev_frame = None
        self.prev_action_idx = None

        self.action_list = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]
        self.logger.info(f"DT Agent initialized for game_id: {self.game_id}")

    def _frame_to_tensor(self, latest_frame: FrameData) -> None | torch.Tensor:
        """Convert frame data to integer tensor for ViT with learned embeddings.

        Returns integer grid [64, 64] with values 0-15 instead of one-hot encoding.
        This is 16x more memory efficient and aligns with transformer best practices.
        """
        # Convert the frame to a numpy array
        frame = np.array(latest_frame.frame[-1], dtype=np.int64)

        assert frame.shape == (self.grid_size, self.grid_size)

        # Keep as integer tensor: (64, 64) with values 0-15
        tensor = torch.from_numpy(frame).long()

        return tensor

    def _hash_experience(self, frame: np.array, action_idx: int) -> str:
        """Compute hash for frame+action combination to ensure uniqueness."""
        assert frame.shape == (self.grid_size, self.grid_size)
        frame_bytes = frame.tobytes()

        # Create hash from frame + action combination
        hash_input = frame_bytes + str(action_idx).encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()

    def _train_dt_model(self):
        """Train the Pure Decision Transformer on collected experiences using configurable loss."""
        if len(self.experience_buffer) < self.pure_dt_config["min_buffer_size"]:
            return

        # Sample experiences for sequence creation
        sample_size = min(
            int(
                len(self.experience_buffer)
                * self.pure_dt_config["experience_sample_rate"]
            ),
            self.pure_dt_config["max_training_experiences"],
        )

        if sample_size < self.pure_dt_config["max_context_len"] + 1:
            return

        # Create training sequences from experience buffer
        sequences = self._create_training_sequences(sample_size)
        if not sequences:
            return

        # Convert sequences to tensors
        states_batch = torch.stack([seq["states"] for seq in sequences]).to(self.device)
        actions_batch = torch.stack([seq["actions"] for seq in sequences]).to(
            self.device
        )
        target_actions_batch = torch.stack([seq["target_action"] for seq in sequences]).to(
            self.device
        )
        change_rewards_batch = torch.stack(
            [seq["change_reward"] for seq in sequences]
        ).to(self.device)
        completion_rewards_batch = torch.stack(
            [seq["completion_reward"] for seq in sequences]
        ).to(self.device)

        # Log training start info
        num_sequences = len(sequences)
        self.logger.info(f"🎯 Starting DT training: {num_sequences} sequences")

        # Training loop with configurable epochs
        for epoch in range(self.pure_dt_config["epochs_per_training"]):
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.pure_dt_model(states_batch, actions_batch)  # [batch, 4101]

            # Compute configurable loss
            loss, metrics = self._compute_loss(
                change_logits=logits["change_logits"],
                completion_logits=logits["completion_logits"],
                target_actions=target_actions_batch,
                change_rewards=change_rewards_batch,
                completion_rewards=completion_rewards_batch,
            )

            # Gradient clipping and backward pass
            loss.backward()
            if self.pure_dt_config["gradient_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.pure_dt_model.parameters(),
                    self.pure_dt_config["gradient_clip_norm"],
                )
            self.optimizer.step()

            # Log training metrics
            self._log_pure_dt_metrics(loss, metrics, epoch)

        # Log training completion
        final_loss = loss.item() if "loss" in locals() else 0.0
        final_accuracy = metrics.get("accuracy", 0.0) if "metrics" in locals() else 0.0
        self.logger.info(
            f"✅ DT training completed: final_loss={final_loss:.4f}, accuracy={final_accuracy:.3f}"
        )

        # Clean up GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _create_training_sequences(self, sample_size):
        """Create state-action sequences for DT training."""
        sequences = []
        context_len = self.pure_dt_config["max_context_len"]

        # Ensure sample_size is an integer
        sample_size = int(sample_size)

        # Sample the starting indices for the sequences
        replace = (
            False if sample_size <= len(self.experience_buffer) - context_len else True
        )
        exp_start_idx = np.random.choice(
            len(self.experience_buffer) - context_len, size=sample_size, replace=replace
        )

        experience_buffer = list(self.experience_buffer)
        for i in exp_start_idx:
            sequence_experiences = experience_buffer[i : i + context_len]

            # Build state sequence [s_0, s_1, ..., s_k] (k+1 states)
            states = []
            for exp in sequence_experiences:
                state_tensor = torch.from_numpy(exp["state"]).long()
                states.append(state_tensor)
            states = torch.stack(states)  # [k+1, 64, 64]

            # Build action sequence [a_0, a_1, ..., a_{k-1}] (k past actions)
            actions = []
            for exp in sequence_experiences[
                :-1
            ]:  # Exclude last (we predict its action)
                actions.append(exp["action_idx"])
            actions = torch.tensor(actions, dtype=torch.long)  # [k]

            # Target action and reward (what we want to predict)
            target_action = sequence_experiences[-1]["action_idx"]
            change_target_reward = sequence_experiences[-1]["change_reward"]
            completion_target_reward = sequence_experiences[-1]["completion_reward"]

            sequences.append(
                {
                    "states": states,
                    "actions": actions,
                    "target_action": torch.tensor(target_action, dtype=torch.long),
                    "change_reward": torch.tensor(
                        change_target_reward, dtype=torch.float32
                    ),
                    "completion_reward": torch.tensor(
                        completion_target_reward, dtype=torch.float32
                    ),
                }
            )

        return sequences

    def _compute_loss(
        self,
        change_logits: torch.Tensor,
        completion_logits: torch.Tensor,
        target_actions: torch.Tensor,
        change_rewards: torch.Tensor,
        completion_rewards: torch.Tensor,
    ):
        """
        Compute configurable loss for DT training.

        Args:
            change_logits: logits for predicting actions that cause frame changes
            completion_logits: logits for predicting actions that complete a level
            target_actions: target action index for the current state
            change_rewards: binary reward for frame change (0 or 1)
            completion_rewards: binary reward for level completion (0 or 1)
        Returns:
            loss: scalar loss value
            metrics: dictionary with metrics for logging
        """

        # Gather only the logits for selected actions
        selected_change_logits = change_logits.gather(
            dim=1, index=target_actions.unsqueeze(1)
        ).squeeze(1)
        selected_completion_logits = completion_logits.gather(
                dim=1, index= target_actions.unsqueeze(1)
        ).squeeze(1)

        # Binary cross-entropy with rewards as binary labels
        change_loss = F.binary_cross_entropy_with_logits(
            selected_change_logits, change_rewards
        )
        completion_loss = F.binary_cross_entropy_with_logits(
                selected_completion_logits, completion_rewards
        )

        # Add action diversity regularization (encourage exploring more actions)
        # Compute probabilities separately for each head
        change_probs = torch.sigmoid(change_logits)
        completion_probs = torch.sigmoid(completion_logits)

        # Split into action and coordinate spaces for each head
        change_action_probs = change_probs[:, :5]      # [batch, 5]
        change_coord_probs = change_probs[:, 5:]       # [batch, 4096]
        completion_action_probs = completion_probs[:, :5]    # [batch, 5]
        completion_coord_probs = completion_probs[:, 5:]     # [batch, 4096]

        # Calculate diversity bonus (mean probability) separately for each head
        # Higher mean = model considers more actions viable = more diversity
        change_action_diversity = change_action_probs.mean(dim=1).mean(dim=0)
        change_coord_diversity = change_coord_probs.mean(dim=1).mean(dim=0)
        completion_action_diversity = completion_action_probs.mean(dim=1).mean(dim=0)
        completion_coord_diversity = completion_coord_probs.mean(dim=1).mean(dim=0)

        # Average diversity across both heads
        action_diversity = (change_action_diversity + completion_action_diversity) / 2
        coord_diversity = (change_coord_diversity + completion_coord_diversity) / 2

        # Diversity coefficients (configurable)
        action_coeff = self.pure_dt_config.get("action_entropy_coeff", 0.0001)
        coord_coeff = self.pure_dt_config.get("coord_entropy_coeff", 0.00001)

        # Total loss with diversity regularization
        # Subtracting encourages higher mean probability = more action diversity
        loss = (
            change_loss + completion_loss
            - action_coeff * action_diversity
            - coord_coeff * coord_diversity
        )

        # Calculate accuracy: did we correctly predict whether action causes frame change?
        # This matches what the loss is optimizing for (bandit-style frame change prediction)
        change_accuracy = (torch.sigmoid(selected_change_logits) > 0.5) == change_rewards
        completion_accuracy = (torch.sigmoid(selected_completion_logits) > 0.5) == completion_rewards
        accuracy = (change_accuracy & completion_accuracy).float().mean()


        metrics = {
            "accuracy": accuracy.item(),
            "change_loss": change_loss.item(),
            "completion_loss": completion_loss.item(),
            "action_diversity": action_diversity.item(),
            "coord_diversity": coord_diversity.item(),
            "total_loss": loss.item(),
        }


        return loss, metrics

    def _log_pure_dt_metrics(self, loss, metrics, epoch):
        """Log DT training metrics to tensorboard."""
        step = self.action_counter

        self.writer.add_scalar("DTAgent/loss", loss.item(), step)
        self.writer.add_scalar("DTAgent/accuracy", metrics.get("accuracy", 0), step)

        # Log bandit-specific metrics if available
        if "change_loss" in metrics:
            self.writer.add_scalar("DTAgent/change_loss", metrics["change_loss"], step)
        if "completion_loss" in metrics:
            self.writer.add_scalar(
                "DTAgent/completion_loss", metrics["completion_loss"], step
            )
        if "action_diversity" in metrics:
            self.writer.add_scalar(
                "DTAgent/action_diversity", metrics["action_diversity"], step
            )
        if "coord_diversity" in metrics:
            self.writer.add_scalar(
                "DTAgent/coord_diversity", metrics["coord_diversity"], step
            )

        # Loss-type specific metrics
        if "positive_samples" in metrics:
            self.writer.add_scalar(
                "DTAgent/positive_samples", metrics["positive_samples"], step
            )
        if "high_confidence_frac" in metrics:
            self.writer.add_scalar(
                "DTAgent/high_confidence_frac", metrics["high_confidence_frac"], step
            )
        if "mean_confidence" in metrics:
            self.writer.add_scalar(
                "DTAgent/mean_confidence", metrics["mean_confidence"], step
            )

    def _has_time_elapsed(self) -> bool:
        """Check if 8 hours have elapsed since start."""
        elapsed_hours = time.time() - self.start_time
        return elapsed_hours >= 8 * 3600 - 5 * 60  # 8 hours with 5 minute buffer

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                self._has_time_elapsed(),
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """
        Choose action using Decision Transformer predictions.

        Args:
            frames: List of previous frames
            latest_frame: Latest frame data

        Returns:
            action: The action to take
        """

        # Check level completion
        self._check_level_completion(latest_frame=latest_frame)

        # Reset when the game state is either NOT_PLAYED or GAME_OVER
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.prev_frame = None
            self.prev_action_idx = None
            action = GameAction.RESET
            action.reasoning = "Game needs reset."
            return action

        # Convert current frame to torch tensor
        current_frame = self._frame_to_tensor(latest_frame)

        # Store unique experience
        if current_frame is not None:
            self._store_experience(current_frame, current_score=latest_frame.score)

        # If frame processing failed, reset tracking and return random action
        if current_frame is None:
            print("Error detected!")
            self.prev_frame = None
            self.prev_action_idx = None
            action = random.choice(self.action_list[:5])  # Random ACTION1-ACTION5
            action.reasoning = (
                f"Encountered a no-op frame, use a random action - {action.value}"
            )
            return action

        # Get action predictions from DT model (following bandit pattern exactly)
        action_idx, coord_idx, selected_action = self.select_action(
            latest_frame_torch=current_frame, latest_frame=latest_frame
        )

        # Store current frame and action for next experience creation
        # Keep as integer grid [64, 64] with values 0-15
        self.prev_frame = current_frame.cpu().numpy().astype(np.int64)

        # Store unified action index: 0-4 for ACTION1-5, 5+ for coordinates
        if action_idx < 5:
            self.prev_action_idx = action_idx
        else:
            self.prev_action_idx = 5 + coord_idx  # Unified action space

        # Increment action counter
        self.action_counter += 1

        # Train DT model periodically
        if self.action_counter % self.train_frequency == 0:
            buffer_size = len(self.experience_buffer)
            if buffer_size >= self.pure_dt_config["min_buffer_size"]:
                self.logger.info(
                    f"🤖 Training DT model... (buffer size: {buffer_size})"
                )
                self._train_dt_model()
            else:
                self.logger.debug(
                    f"Skipping training: buffer size {buffer_size} < min_buffer_size {self.pure_dt_config['min_buffer_size']}"
                )

        return selected_action

    def _check_level_completion(self, latest_frame: FrameData) -> None | GameAction:
        # Check if score has changed and log score at action count

        if latest_frame.score != self.current_score:
            self.logger.info(
                f"Score changed from {self.current_score} to {latest_frame.score} at action {self.action_counter}"
            )

            # Clear experience buffer when reaching new level
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            self.logger.info("Cleared experience buffer - new level reached")

            # Reset previous tracking
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = latest_frame.score


    def _store_experience(self, current_frame: torch.Tensor, current_score: int):
        if self.prev_frame is not None:
            # Compute hash for uniqueness check
            experience_hash = self._hash_experience(
                self.prev_frame, self.prev_action_idx
            )

            # Only store if unique
            if experience_hash not in self.experience_hashes:
                # Convert current frame to numpy int64 for comparison
                latest_frame_np = current_frame.cpu().numpy().astype(np.int64)
                frame_changed = not np.array_equal(self.prev_frame, latest_frame_np)
                level_completion = current_score != self.current_score

                experience = {
                    "state": self.prev_frame,  # Integer grid [64, 64]
                    "action_idx": self.prev_action_idx,  # Unified action index
                    "change_reward": 1.0 if frame_changed else 0.0,
                    "completion_reward": 1.0 if level_completion else 0.0,
                }
                self.experience_buffer.append(experience)
                self.experience_hashes.add(experience_hash)

                # Log replay buffer size periodically
                self.writer.add_scalar(
                    "DTAgent/replay_buffer_size",
                    len(self.experience_buffer),
                    self.action_counter,
                )
                self.writer.add_scalar(
                    "DTAgent/replay_unique_hashes",
                    len(self.experience_hashes),
                    self.action_counter,
                )

    def select_action(
        self, latest_frame_torch: torch.Tensor, latest_frame: FrameData
    ) -> tuple[int, int, GameAction]:
        with torch.no_grad():
            # Build state-action sequence and get logits
            max_context_len = self.pure_dt_config["max_context_len"]

            # Cold start - random valid action if no experience
            if len(self.experience_buffer) < 1:
                selected_action = self._random_valid_action(
                    latest_frame.available_actions
                )
                # Set default values for tracking
                if selected_action.value <= 5:
                    action_idx = selected_action.value - 1
                    coords = None
                    coord_idx = None
                else:
                    action_idx = 5
                    coords = (0, 0)
                    coord_idx = 0
            else:
                # Build state-action sequence for inference
                states, actions = self._build_inference_sequence(
                    latest_frame_torch, max_context_len
                )

                # Get action logits from DT
                logits = self.pure_dt_model(
                    states, actions
                )  # [1, 4101]
                change_logits = logits["change_logits"].squeeze(0)  # (4101,)
                completion_logits = logits["completion_logits"].squeeze(0)

                # Sample from combined action space (following bandit pattern exactly)
                action_idx, coords, coord_idx = self._sample_from_combined_output(
                    change_logits, completion_logits, latest_frame.available_actions
                )

                # Create GameAction directly (following bandit pattern exactly)
                if action_idx < 5:
                    # Selected ACTION1-ACTION5
                    selected_action = self.action_list[action_idx]
                    selected_action.reasoning = "DT prediction"
                else:
                    # Selected a coordinate - treat as ACTION6 (following bandit pattern exactly)
                    selected_action = GameAction.ACTION6
                    y, x = coords
                    selected_action.set_data({"x": x, "y": y})
                    selected_action.reasoning = "DT coordinate prediction"
                    self.logger.info(
                        f"📍 ACTION6 selected: coordinates ({x}, {y}) -> coord_idx={coord_idx}"
                    )

        return action_idx, coord_idx, selected_action

    def _sample_from_combined_output(
        self,
        change_logits: torch.Tensor,
        completion_logits: torch.Tensor,
        available_actions=None,
    ) -> tuple[int, tuple | None, int | None]:
        """Sample from combined 5 + 64x64 action space with masking for invalid actions.

        Args:
            change_logits (torch.Tensor): logits for predicting changes caused by both discrete and coordinate actions
            completion_logits (torch.Tensor): logits for predicting level completion caused by both discrete and coordinate actions
            available_actions (list, optional): list of available actions, default to None

        Returns:
            action_idx (int): index of selected action
            coord_idx (tuple): index of selected coordinate
            selected_action (GameAction): selected action
        """
        # Split logits
        change_action_logits = change_logits[:5]  # First 5
        change_coord_logits = change_logits[5:]  # Remaining 4096
        completion_action_logits = completion_logits[:5] # First 5
        completion_coord_logits = completion_logits[5:] # Remaining 4096

        # Apply masking based on available_actions if provided
        if available_actions is not None and len(available_actions) > 0:
            # Create mask for action logits (ACTION1-ACTION5 = indices 0-4)
            action_mask = torch.full_like(change_action_logits, float("-inf"))
            action6_available = False

            for action in available_actions:
                # Extract action value if it's a GameAction enum
                action_id = action.value

                if 1 <= action_id <= 5:  # ACTION1-ACTION5
                    action_mask[action_id - 1] = 0.0  # Unmask valid actions
                elif action_id == 6:  # ACTION6
                    action6_available = True

            # Apply mask to action logits
            change_action_logits = change_action_logits + action_mask
            completion_action_logits = completion_action_logits + action_mask

            # If ACTION6 (coordinate action) is not available, mask all coordinate logits
            if not action6_available:
                coord_mask = torch.full_like(change_coord_logits, float("-inf"))
                change_coord_logits = change_coord_logits + coord_mask
                completion_coord_logits = completion_coord_logits + coord_mask

        # Apply sigmoid
        change_action_probs = torch.sigmoid(change_action_logits)
        completion_action_probs = torch.sigmoid(completion_action_logits)
        change_coord_probs = torch.sigmoid(change_coord_logits)
        completion_coord_probs = torch.sigmoid(completion_coord_logits)

        # For fair sampling: treat coordinates as one action type with total prob divided by 4096
        change_coord_probs_scaled = change_coord_probs / self.num_coordinates
        completion_coord_probs_scaled = completion_coord_probs / self.num_coordinates

        # Combine for sampling (normalize)
        change_probs_sampling = torch.cat(
            [change_action_probs, change_coord_probs_scaled]
        )
        change_probs_sampling = change_probs_sampling / change_probs_sampling.sum()
        completion_probs_sampling = torch.cat(
            [completion_action_probs, completion_coord_probs_scaled]
        )
        completion_probs_sampling = (
            completion_probs_sampling / completion_probs_sampling.sum()
        )
        probs_sampling = change_probs_sampling * completion_probs_sampling

        # Renormalize after multiplication (to ensure sum=1.0 for np.random.choice)
        probs_sampling = probs_sampling / probs_sampling.sum()

        # Sample from normalized space
        selected_idx = np.random.choice(
            len(probs_sampling), p=probs_sampling.cpu().numpy()
        )

        if selected_idx < 5:
            # Selected one of ACTION1-ACTION5
            return selected_idx, None, None
        else:
            # Selected a coordinate (index 5-4100)
            coord_idx = selected_idx - 5
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx

    def _random_valid_action(self, available_actions):
        """Generate random valid action for cold start or error fallback."""
        if available_actions and len(available_actions) > 0:
            selected_action = random.choice(available_actions)
            selected_action.reasoning = "Random valid action (fallback)"
            return selected_action
        else:
            # Default fallback
            action = random.choice(self.action_list[:5])  # Random ACTION1-5
            action.reasoning = "Random action (no constraints available)"
            return action

    def _build_inference_sequence(self, current_frame, max_context_len):
        """Build state-action sequence for Decision Transformer inference."""
        # Get recent experiences up to max context length
        available_context = min(int(max_context_len), len(self.experience_buffer))
        recent_experiences = (
            list(self.experience_buffer)[-available_context:]
            if available_context > 0
            else []
        )

        # Build states: recent states + current state
        states_list = []
        for exp in recent_experiences:
            states_list.append(torch.from_numpy(exp["state"]).long())
        states_list.append(current_frame)  # Add current state

        # Build actions: recent actions (no current action - we're predicting it)
        actions_list = []
        for exp in recent_experiences:
            actions_list.append(exp["action_idx"])

        # Handle cold start: if no actions yet, create minimal sequence with one dummy action
        # This is only needed to satisfy model input requirements, not for actual padding
        if len(actions_list) == 0:
            # Use ACTION1 (index 0) as dummy action for cold start
            # The model won't rely on this since there's no history anyway
            actions_list.append(0)
            # Add a zero state to maintain sequence structure [s0, a0, s1]
            states_list = [torch.zeros_like(current_frame)] + states_list

        # Convert to tensors
        states = (
            torch.stack(states_list).unsqueeze(0).to(self.device)
        )  # [1, seq_len+1, 16, 64, 64]
        actions = (
            torch.tensor(actions_list).unsqueeze(0).to(self.device)
        )  # [1, seq_len]

        return states, actions
