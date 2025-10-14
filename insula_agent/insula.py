"""
Insula: Insular Cortex-Inspired Online Supervised Learning agent for ARC-AGI-3

This module implements Insula, an online supervised learning architecture with
multi-level insular cortex-inspired integration. NOT a reinforcement learning model—
supervision signals are generated on-the-fly from game API outcomes.

Architecture (Five-Level Hierarchy):
1. Cell Integration: Color + position → unified cell representation (insular-inspired)
2. Spatial Integration (Posterior Insula): ViT processes 64×64 grid → spatial state
3. Temporal Integration (Anterior Insula): Decision Transformer adds action history context
4. Decision Projection (Insula→Striatum): Triple heads predict action probabilities
5. Learning Systems (Hippocampus+Striatum): Replay with temporal hindsight traces

Components:
- ViT State Encoder: Patch-based attention (8×8 patches) with learnable per-patch alpha
- Action Embedding: Learned embeddings for 4101 actions (ACTION1-5 + 64x64 coordinates)
- Decision Transformer: Causal attention over state-action sequences
- Triple-Head Prediction: Change (exploration) + Completion (goal) + GAME_OVER (safety)
- Multiplicative Action Sampling: Combines all three heads for balanced decisions

Key Features:
- Online supervised learning: Labels from game API outcomes (change/completion/gameover)
- Hierarchical context windows: 15/100/300 steps for change/completion/gameover heads
- Head-specific temporal decay: 0.7/0.8/0.9 for multi-timescale replay weighting
- Importance-weighted replay: 1:5:10 ratio (critical events replayed more)
- Temporal hindsight traces: Individual action evaluation after seeing full trajectory
- Joint optimization: Single optimizer step on accumulated gradients from all heads

Biological Inspiration:
- Insular cortex: Multi-level integration hub (cell→spatial→temporal→decision)
- VTA/SNc dopamine: Change/Completion heads (reward prediction)
- Habenula/RMTg: GAME_OVER head (aversive prediction)
- Hippocampal replay: Episodic memory with importance-weighted prioritization
- Basal ganglia: Multiplicative integration of Go/NoGo pathways
"""

from typing import Any
import random
import time
import logging
from collections import deque
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState

from insula_agent.config import load_config
from insula_agent.models import DecisionModel
from insula_agent.trainer import train_model
from insula_agent.utils import setup_logging


class Insula(Agent):
    """Insula agent for ARC-AGI-3.

    Online supervised learning agent with insular cortex-inspired multi-level integration.
    Generates supervision signals on-the-fly from game outcomes (NOT reinforcement learning).

    Features:
    - State-action sequence modeling with hierarchical temporal context
    - Triple-head prediction: Change/Completion/GAME_OVER
    - Temperature-based exploration with action masking
    - Experience buffer for episodic replay
    - Temporal hindsight traces for retrospective action evaluation
    - Importance-weighted replay (1:10:10 ratio)
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

        # Override MAX_ACTIONS - no limit for Insula
        self.MAX_ACTIONS = float("inf")

        # Setup experiment directory and logging
        base_dir = Path.cwd() / "insula_agent/logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        log_file = base_dir / "run.log"
        setup_logging(log_file)
        self.logger = logging.getLogger(f"InsulaAgent_{self.game_id}")

        # Log card id
        self.logger.info(f"Card ID: {self.card_id}")

        tensorboard_dir = base_dir / f"{self.game_id}/tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tensorboard_dir))

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Insula Agent using device: {self.device}")

        # Load configuration
        self.config = load_config(device=str(self.device))

        # Lazy model initialization - deferred until first frame to detect grid size
        self.decision_model = None  # Will be initialized on first valid frame
        self.optimizer = None  # Will be initialized with model

        # Grid info - will be detected from first frame
        self.grid_size = None  # Detected dynamically from first frame
        self.num_coordinates = None  # Calculated after grid_size detection
        self.num_colours = 16  # Fixed for ARC (0-15 colors)

        # Scores as level indicators
        self.current_score = 0

        # Experience buffer for training
        self.experience_buffer = deque(maxlen=self.config.max_buffer_size)
        self.train_frequency = self.config.train_frequency

        # Game state and action mapping
        self.prev_frame = None
        self.prev_action_idx = None
        self.win_counter = 0

        self.action_list = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]
        self.logger.info(f"Insula Agent initialized for game_id: {self.game_id}")

    def _frame_to_tensor(self, latest_frame: FrameData) -> None | torch.Tensor:
        """Convert frame data to integer tensor for ViT with learned embeddings.

        Args:
            latest_frame: FrameData object with frame data and metadata

        Returns:
            [H, W] tensor with values 0-15, or None if invalid
        """
        # Convert the frame to a numpy array
        frame = np.array(latest_frame.frame[-1], dtype=np.int64)

        # Validation: First frame (grid_size not yet detected)
        if self.grid_size is None:
            # Basic validation: ensure it's 2D
            if frame.ndim != 2:
                self.logger.error(f"Invalid frame: expected 2D, got shape {frame.shape}")
                return None

            # Validate color values are in range 0-15
            if frame.max() > 15 or frame.min() < 0:
                self.logger.error(
                    f"Invalid frame values: expected 0-15, got range [{frame.min()}, {frame.max()}]"
                )
                return None
        else:
            # Validation: Subsequent frames (grid_size already detected)
            if frame.shape != (self.grid_size, self.grid_size):
                self.logger.error(
                    f"Frame size mismatch: expected {self.grid_size}×{self.grid_size}, "
                    f"got {frame.shape[0]}×{frame.shape[1]}"
                )
                return None

        # Keep as integer tensor with detected shape
        tensor = torch.from_numpy(frame).long()

        return tensor

    def _calculate_patch_size(self, grid_size: int) -> int:
        """Calculate optimal patch size that divides grid_size evenly.

        Prioritizes patch sizes close to config default (8) for efficiency.
        Falls back to smaller sizes if needed for divisibility.

        Args:
            grid_size: Detected grid size (H or W, assumes square grid)

        Returns:
            patch_size: Patch size that divides grid_size evenly
        """
        config_patch_size = self.config.vit_patch_size

        # Try config patch size first
        if grid_size % config_patch_size == 0:
            return config_patch_size

        # Try patch sizes in order of preference (decreasing from 32 to 1)
        candidates = sorted([1, 2, 4, 8, 16, 32], reverse=True)

        for patch_size in candidates:
            if patch_size <= grid_size and grid_size % patch_size == 0:
                self.logger.info(
                    f"Config patch_size={config_patch_size} doesn't divide grid_size={grid_size}. "
                    f"Using patch_size={patch_size} instead."
                )
                return patch_size

        # Fallback: patch_size=1 (each cell is a patch)
        self.logger.warning(
            f"No efficient patch size found for grid_size={grid_size}. "
            f"Using patch_size=1 (slow!)."
        )
        return 1

    def _initialize_model_from_frame(self, frame_tensor: torch.Tensor) -> None:
        """Initialize model dynamically based on detected grid size from first frame.

        Args:
            frame_tensor: First frame tensor with shape [H, W]
        """
        from insula_agent.models import DecisionModel

        # Detect grid size from first frame
        h, w = frame_tensor.shape[0], frame_tensor.shape[1]

        # Handle non-square grids - use larger dimension for safety
        if h != w:
            self.logger.warning(
                f"Non-square grid detected: {h}×{w}. "
                f"Using larger dimension: {max(h, w)}"
            )
            self.grid_size = max(h, w)
        else:
            self.grid_size = h

        self.num_coordinates = self.grid_size * self.grid_size

        # Calculate optimal patch size
        patch_size = self._calculate_patch_size(self.grid_size)

        self.logger.info(
            f"🎯 Detected grid size: {h}×{w}. "
            f"Using grid_size={self.grid_size}, patch_size={patch_size}, "
            f"num_patches={self.grid_size//patch_size}×{self.grid_size//patch_size}"
        )

        # Initialize model with detected parameters
        self.decision_model = DecisionModel(
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_context_len=self.config.max_context_len,
            # ViT encoder parameters with dynamic patch size
            vit_cell_embed_dim=self.config.vit_cell_embed_dim,
            vit_patch_size=patch_size,  # 🔥 Dynamic based on grid size!
            vit_num_layers=self.config.vit_num_layers,
            vit_num_heads=self.config.vit_num_heads,
            vit_dropout=self.config.vit_dropout,
            vit_use_cls_token=self.config.vit_use_cls_token,
            vit_pos_dim_ratio=self.config.vit_pos_dim_ratio,
            vit_use_patch_pos_encoding=self.config.vit_use_patch_pos_encoding,
            # Head configuration
            use_completion_head=self.config.use_completion_head,
            use_gameover_head=self.config.use_gameover_head,
            # Learned decay configuration
            use_learned_decay=self.config.use_learned_decay,
            change_decay_init=self.config.change_temporal_decay,
            completion_decay_init=self.config.completion_temporal_decay,
            gameover_decay_init=self.config.gameover_temporal_decay,
        ).to(self.device)

        # Set to eval mode for inference (returns 2D logits [batch, 4101])
        # Training mode is set in trainer.py (returns 3D logits [batch, seq_len+1, 4101])
        self.decision_model.eval()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.decision_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.logger.info(
            f"✅ Model initialized successfully: {self.grid_size}×{self.grid_size} grid, "
            f"patch_size={patch_size}, embed_dim={self.config.embed_dim}, "
            f"total_params={sum(p.numel() for p in self.decision_model.parameters()):,}"
        )

    def _has_time_elapsed(self) -> bool:
        """Check if 8 hours have elapsed since start."""
        elapsed_hours = time.time() - self.start_time
        return elapsed_hours >= 8 * 3600 - 5 * 60  # 8 hours with 5 minute buffer

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""

        # Keep track of the number of wins
        if latest_frame.state is GameState.WIN:
            self.win_counter += 1

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
        # Convert current frame to torch tensor
        current_frame = None
        if latest_frame.state is not GameState.NOT_PLAYED:
            current_frame = self._frame_to_tensor(latest_frame)

            # Lazy initialization: Initialize model on first valid frame
            if self.decision_model is None and current_frame is not None:
                self._initialize_model_from_frame(current_frame)

            # Store unique experience
            if current_frame is not None:
                self._store_experience(current_frame, latest_frame)

        # Train the model
        if self._should_train_model(latest_frame):
            buffer_size = len(self.experience_buffer)
            if buffer_size >= self.config.min_buffer_size:
                self.logger.info(
                    f"🤖 Training Insula model. Game: {self.game_id} with (buffer size: {buffer_size})"
                )
                train_model(
                    model=self.decision_model,
                    optimizer=self.optimizer,
                    experience_buffer=self.experience_buffer,
                    config=self.config,
                    device=self.device,
                    writer=self.writer,
                    logger=self.logger,
                    game_id=self.game_id,
                    action_counter=self.action_counter,
                )
            else:
                self.logger.debug(
                    f"Skipping training: buffer size {buffer_size} < min_buffer_size {self.config.min_buffer_size}"
                )

        # Check level completion
        self._check_level_completion(latest_frame=latest_frame)

        # Reset when the game when it's initialized or when GAME_OVER is encountered
        if latest_frame.state in  [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.prev_frame = None
            self.prev_action_idx = None
            action = GameAction.RESET
            if latest_frame.state is GameState.NOT_PLAYED:
                action.reasoning = "Game needs reset due to NOT_PLAYED"
                self.logger.info(f"Game {self.game_id} is initialized")
            else:
                action.reasoning = "Game needs reset due to GAME_OVER"
                self.logger.info(f"🤖 GAME OVER. Game {self.game_id} is reset. Current score: {self.current_score}")
            return action


        # If frame processing failed, reset tracking and return a random action
        if current_frame is None:
            self.logger.info(f"Error in frame processing for Game {self.game_id}. Use randome action selection")
            self.prev_frame = None
            self.prev_action_idx = None
            action = random.choice(self.action_list[:5])  # Random ACTION1-ACTION5
            action.reasoning = (
                f"Encountered a no-op frame, use a random action - {action.value}"
            )
            return action

        # If no reset and no error, get action predictions from Insula model (following bandit pattern exactly)
        action_idx, coord_idx, selected_action = self._select_action(
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


        return selected_action

    def _should_train_model(self, latest_frame: FrameData) -> bool:
        should_train_model = (
              self.action_counter % self.train_frequency == 0 or
              latest_frame.state == GameState.GAME_OVER or
              latest_frame.score > self.current_score
        )
        return should_train_model

    def _check_level_completion(self, latest_frame: FrameData) -> None | GameAction:
        # Check if score has changed and log score at action count

        if latest_frame.score != self.current_score:
            self.logger.info(f"Score changed from {self.current_score} to {latest_frame.score} for game {self.game_id} at action {self.action_counter}")
            self.logger.info(f"Game {self.game_id} reached level {latest_frame.score} at action {self.action_counter}")

            # Clear experience buffer when reaching new level (always do this)
            self.experience_buffer.clear()
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = latest_frame.score

            # Transfer Learning Strategy: Keep trained model by default
            # Rationale: Game rules remain constant, ViT learns abstract spatial patterns
            # that transfer across levels, Insula learns action semantics
            #
            # If observing overfitting or poor transfer, consider these options:

            # OPTION 1: Reset only ViT encoder (relearn spatial patterns, keep Insula action knowledge)
            # Pros: Adapts to new grid structures while preserving action semantics
            # Cons: Loses learned spatial abstractions (symmetries, transformations)
            # from insula_agent.models import ViTStateEncoder
            # self.pure_dt_model.state_encoder = ViTStateEncoder(
            #     cell_embed_dim=self.config.get("vit_cell_embed_dim", 64),
            #     patch_size=self.config.get("vit_patch_size", 8),
            #     num_layers=self.config.get("vit_num_layers", 4),
            #     num_heads=self.config.get("vit_num_heads", 8),
            #     dropout=self.config.get("vit_dropout", 0.1),
            #     use_cls_token=self.config.get("vit_use_cls_token", True),
            #     embed_dim=self.config["embed_dim"],
            # ).to(self.device)
            # # Re-register encoder parameters in optimizer
            # self.optimizer = torch.optim.Adam(
            #     self.pure_dt_model.parameters(),
            #     lr=self.config["learning_rate"],
            #     weight_decay=self.config["weight_decay"],
            # )
            # self.logger.info(f"🔄 Reset ViT encoder for level {latest_frame.score}")

            # OPTION 2: Reduce learning rate (fine-tuning mode)
            # Pros: Gentler adaptation to new level, preserves most learned knowledge
            # Cons: May learn new patterns too slowly
            # decay_factor = 0.5
            # for param_group in self.optimizer.param_groups:
            #     old_lr = param_group['lr']
            #     param_group['lr'] *= decay_factor
            #     self.logger.info(f"📉 Reduced learning rate: {old_lr:.6f} → {param_group['lr']:.6f}")

            # OPTION 3: Reset optimizer state (clear momentum/Adam statistics)
            # Pros: Removes optimization momentum from previous level's gradients
            # Cons: Loses adaptive learning rate benefits
            # self.optimizer = torch.optim.Adam(
            #     self.pure_dt_model.parameters(),
            #     lr=self.config["learning_rate"],
            #     weight_decay=self.config["weight_decay"],
            # )
            # self.logger.info(f"🔄 Reset optimizer state for level {latest_frame.score}")

            # OPTION 4: Full model reset (nuclear option - NOT RECOMMENDED)
            # Only use if levels are completely independent tasks
            # self.pure_dt_model = DecisionTransformer(...).to(self.device)
            # self.optimizer = torch.optim.Adam(...)
            # self.logger.info(f"🔄 Full model reset for level {latest_frame.score}")


    def _store_experience(self, current_frame: torch.Tensor, latest_frame: FrameData):
        if self.prev_frame is not None:
            # Convert current frame to numpy int64 for comparison
            latest_frame_np = current_frame.cpu().numpy().astype(np.int64)
            frame_changed = not np.array_equal(self.prev_frame, latest_frame_np)
            level_completion = latest_frame.score != self.current_score
            # Inverted GAME_OVER: 1.0 = survived (good), 0.0 = GAME_OVER (bad)
            game_over_occurred = latest_frame.state == GameState.GAME_OVER
            gameover_reward = 0.0 if game_over_occurred else 1.0

            experience = {
                "state": self.prev_frame,  # Integer grid [64, 64]
                "action_idx": self.prev_action_idx,  # Unified action index
                "change_reward": 1.0 if frame_changed else 0.0,
                "completion_reward": 1.0 if level_completion else 0.0,
                "gameover_reward": gameover_reward,  # 1.0 = survived, 0.0 = died
            }
            self.experience_buffer.append(experience)

            # Log replay buffer size periodically
            self.writer.add_scalar(
                "InsulaAgent/replay_buffer_size",
                len(self.experience_buffer),
                self.action_counter,
            )

    def _select_action(
        self, latest_frame_torch: torch.Tensor, latest_frame: FrameData
    ) -> tuple[int, int, GameAction]:

        self.decision_model.eval()
        with torch.no_grad():
            # Dynamic inference context length based on enabled heads
            # Rationale: Each head needs ≥ its training context length for correct predictions
            # Use longest context among enabled heads (attention handles longer context gracefully)
            inference_context_len = self.config.change_context_len  # Start with change (always enabled)

            if self.config.use_completion_head:
                inference_context_len = max(inference_context_len, self.config.completion_context_len)

            if self.config.use_gameover_head:
                inference_context_len = max(inference_context_len, self.config.gameover_context_len)

            # Log inference context (debug level, only once on first inference)
            if self.action_counter == 1:
                enabled_heads = ["change"]
                if self.config.use_completion_head:
                    enabled_heads.append("completion")
                if self.config.use_gameover_head:
                    enabled_heads.append("gameover")
                self.logger.debug(
                    f"Inference context length: {inference_context_len} steps "
                    f"(enabled heads: {', '.join(enabled_heads)})"
                )

            # Cold start - random valid action if no experience
            if len(self.experience_buffer) < 1:
                selected_action = self._random_valid_action(
                    latest_frame.available_actions
                )
                # Set default values for tracking
                if selected_action.value <= 5:
                    action_idx = selected_action.value - 1
                    coord_idx = None
                else:
                    action_idx = 5
                    coord_idx = 0
            else:
                # Build state-action sequence for inference
                states, actions = self._build_inference_sequence(
                    latest_frame_torch, inference_context_len
                )

                # Get action logits from Insula (variable number of heads)
                logits = self.decision_model(states, actions)

                # Always get change logits (required)
                change_logits = logits["change_logits"].squeeze(0)  # [4101]

                # Conditionally get completion logits (optional)
                completion_logits = None
                if "completion_logits" in logits:
                    completion_logits = logits["completion_logits"].squeeze(0)  # [4101]

                # Conditionally get gameover logits (optional)
                gameover_logits = None
                if "gameover_logits" in logits:
                    gameover_logits = logits["gameover_logits"].squeeze(0)  # [4101]

                # Sample from combined action space (multiplicative combination with variable heads)
                action_idx, coords, coord_idx = self._sample_action(
                    change_logits, completion_logits, gameover_logits, latest_frame.available_actions
                )

                # Create GameAction directly (following bandit pattern exactly)
                if action_idx < 5:
                    # Selected ACTION1-ACTION5
                    selected_action = self.action_list[action_idx]
                    selected_action.reasoning = "discrete action prediction"
                else:
                    # Selected a coordinate - treat as ACTION6 (following bandit pattern exactly)
                    selected_action = GameAction.ACTION6
                    y, x = coords
                    selected_action.set_data({"x": x, "y": y})
                    selected_action.reasoning = "coordinate action prediction"
                    self.logger.info(
                        f"{self.game_id} - ACTION6: coordinates ({x}, {y}) -> coord_idx={coord_idx}"
                    )

        return action_idx, coord_idx, selected_action

    def _sample_action(
        self,
        change_logits: torch.Tensor,
        completion_logits: torch.Tensor | None,  # Can be None if head disabled
        gameover_logits: torch.Tensor | None,    # Can be None if head disabled
        available_actions=None,
    ) -> tuple[int, tuple | None, int | None]:
        """Sample from combined action space using multiplicative combination of variable heads.

        Args:
            change_logits: [4101] - Logits for predicting frame changes (always present)
            completion_logits: [4101] or None - Logits for predicting level completion (optional)
            gameover_logits: [4101] or None - Logits for predicting GAME_OVER (optional, inverted for avoidance)
            available_actions: List of available actions for masking

        Returns:
            action_idx: Index of selected action
            coords: Coordinates if ACTION6 selected
            coord_idx: Flattened coordinate index
        """
        # Split change logits (always present)
        change_action_logits = change_logits[:5]  # [5]
        change_coord_logits = change_logits[5:]  # [4096]

        # Split completion logits if present
        if completion_logits is not None:
            completion_action_logits = completion_logits[:5]  # [5]
            completion_coord_logits = completion_logits[5:]  # [4096]
        else:
            completion_action_logits = None
            completion_coord_logits = None

        # Split gameover logits if present
        if gameover_logits is not None:
            gameover_action_logits = gameover_logits[:5]  # [5]
            gameover_coord_logits = gameover_logits[5:]  # [4096]
        else:
            gameover_action_logits = None
            gameover_coord_logits = None

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

            # Apply mask to change head (always present)
            change_action_logits = change_action_logits + action_mask

            # Apply mask to completion head if present
            if completion_action_logits is not None:
                completion_action_logits = completion_action_logits + action_mask

            # Apply mask to gameover head if present
            if gameover_action_logits is not None:
                gameover_action_logits = gameover_action_logits + action_mask

            # If ACTION6 (coordinate action) is not available, mask all coordinate logits
            if not action6_available:
                coord_mask = torch.full_like(change_coord_logits, float("-inf"))
                change_coord_logits = change_coord_logits + coord_mask

                if completion_coord_logits is not None:
                    completion_coord_logits = completion_coord_logits + coord_mask

                if gameover_coord_logits is not None:
                    gameover_coord_logits = gameover_coord_logits + coord_mask

        # Apply sigmoid to convert logits to probabilities (change head always present)
        change_action_probs = torch.sigmoid(change_action_logits)  # [5]
        change_coord_probs = torch.sigmoid(change_coord_logits)  # [4096]

        # For fair sampling: treat coordinates as one action type with total prob divided by 4096
        change_coord_probs_scaled = change_coord_probs / self.num_coordinates

        # Combine action and coordinate probabilities for change head
        change_probs_sampling = torch.cat([change_action_probs, change_coord_probs_scaled])  # [4101]
        change_probs_sampling = change_probs_sampling / change_probs_sampling.sum()

        # Start with change head probabilities (always present)
        probs_sampling = change_probs_sampling  # [4101]

        # Multiply by completion head probabilities if available
        if completion_action_logits is not None:
            completion_action_probs = torch.sigmoid(completion_action_logits)  # [5]
            completion_coord_probs = torch.sigmoid(completion_coord_logits)  # [4096]
            completion_coord_probs_scaled = completion_coord_probs / self.num_coordinates

            completion_probs_sampling = torch.cat([completion_action_probs, completion_coord_probs_scaled])  # [4101]
            completion_probs_sampling = completion_probs_sampling / completion_probs_sampling.sum()

            # Multiplicative combination
            probs_sampling = probs_sampling * completion_probs_sampling  # [4101]

        # Multiply by inverted gameover head probabilities if available
        if gameover_action_logits is not None:
            gameover_action_probs = torch.sigmoid(gameover_action_logits)  # [5]
            gameover_coord_probs = torch.sigmoid(gameover_coord_logits)  # [4096]
            gameover_coord_probs_scaled = gameover_coord_probs / self.num_coordinates

            gameover_probs_sampling = torch.cat([gameover_action_probs, gameover_coord_probs_scaled])  # [4101]
            gameover_probs_sampling = gameover_probs_sampling / gameover_probs_sampling.sum()

            # CRITICAL: Invert gameover probabilities for avoidance (1.0 - p)
            # gameover_probs predicts "will cause GAME_OVER", we want to AVOID those actions
            gameover_probs_sampling_inverted = 1.0 - gameover_probs_sampling + 1e-10  # [4101]

            # Multiplicative combination
            probs_sampling = probs_sampling * gameover_probs_sampling_inverted  # [4101]

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

    def _build_inference_sequence(self, current_frame, context_len):
        """Build state-action sequence for Decision Transformer inference.

        Args:
            current_frame: Current frame tensor [64, 64]
            context_len: Number of past experiences to include. This is dynamically
                        selected based on enabled heads - each head requires at least
                        its training context length for correct predictions:
                        - Change-only: 5 steps (8× faster than fixed 40)
                        - Change+Completion: 20 steps (2× faster than fixed 40)
                        - All heads: 40 steps (same as fixed, but semantically correct)

        Returns:
            states: [1, seq_len+1, 64, 64] - State sequence including current
            actions: [1, seq_len] - Past actions (current action will be predicted)
        """
        # Get recent experiences up to max context length
        available_context = min(int(context_len), len(self.experience_buffer))
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
