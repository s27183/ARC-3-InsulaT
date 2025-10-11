"""
Vision Decision Transformer (ViDT) for ARC-AGI-3

This module implements a Decision Transformer with Vision Transformer state encoder,
triple-head prediction, and hierarchical context windows for multi-objective RL.

Architecture:
- ViT State Encoder: Patch-based attention (8×8 patches) with learnable per-patch alpha mixing
- Action Embedding: Learned embeddings for 4101 actions (ACTION1-5 + 64x64 coordinates)
- Decision Transformer: Causal attention over state-action sequences with hierarchical contexts
- Triple-Head Prediction: Change (exploration) + Completion (goal) + GAME_OVER (safety)
- Multiplicative Action Sampling: Combines all three heads for balanced decision-making

Key Features:
- Hierarchical context windows: 15/100/300 steps for change/completion/gameover heads
- Head-specific eligibility decay: 0.7/0.8/0.9 for multi-timescale credit assignment
- Importance-weighted replay: 1:5:10 ratio (16:80:160 sequences per training round)
- Outcome-anchored sampling: Completion/gameover sequences end at critical events
- Joint optimization: Single optimizer step on accumulated gradients from all heads
- Gradient accumulation: ~256 gradient contributions per optimizer step

Biological Inspiration:
- VTA/SNc dopamine systems: Change/Completion heads (approach/reward)
- Habenula/RMTg: GAME_OVER head (avoidance/punishment)
- Hippocampal reverse replay: Backward temporal credit assignment
- Basal ganglia: Multiplicative integration of Go/NoGo pathways
"""

from typing import Any
import random
import time
import logging
import hashlib
from collections import deque
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState

from dt_agent.config import load_dt_config, validate_config
from dt_agent.models import DecisionTransformer
from dt_agent.trainer import train_dt_model
from dt_agent.utils import setup_logging


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
        base_dir = Path.cwd() / "dt_agent/logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        log_file = base_dir / "run.log"
        setup_logging(log_file)
        self.logger = logging.getLogger(f"DTAgent_{self.game_id}")

        tensorboard_dir = base_dir / f"{self.game_id}/tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tensorboard_dir))

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"DT Agent using device: {self.device}")

        # Model initialization
        self.config = load_dt_config(device=str(self.device))
        validate_config(self.config)

        self.pure_dt_model = DecisionTransformer(
            embed_dim=self.config["embed_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            max_context_len=self.config["max_context_len"],
            # ViT encoder parameters
            vit_cell_embed_dim=self.config.get("vit_cell_embed_dim", 64),
            vit_patch_size=self.config.get("vit_patch_size", 8),
            vit_num_layers=self.config.get("vit_num_layers", 4),
            vit_num_heads=self.config.get("vit_num_heads", 8),
            vit_dropout=self.config.get("vit_dropout", 0.1),
            vit_use_cls_token=self.config.get("vit_use_cls_token", True),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.pure_dt_model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Grid info
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16

        # Scores as level indicators
        self.current_score = 0

        # Experience buffer for training with uniqueness tracking
        self.experience_buffer = deque(maxlen=self.config["max_buffer_size"])
        self.hashed_experiences = set()
        self.train_frequency = self.config["train_frequency"]

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

        Args:
            latest_frame: FrameData object with frame data and metadata

        Returns:
            [64, 64] tensor with values 0-15
        """
        # Convert the frame to a numpy array
        frame = np.array(latest_frame.frame[-1], dtype=np.int64)

        try:
            assert frame.shape == (self.grid_size, self.grid_size)
        except AssertionError:
            self.logger.error(
                f"Error in frame shape: {frame.shape} != {self.grid_size}x{self.grid_size}"
            )
            return None

        # Keep as integer tensor: (64, 64) with values 0-15
        tensor = torch.from_numpy(frame).long()

        return tensor

    def _hash_experience(self, frame: np.ndarray, action_idx: int) -> str:
        """Compute hash for frame+action combination using BLAKE2b.

        Input: frame shape [64, 64] + action_idx (int)
        Output: 64-character hex string (512-bit hash)
        """
        assert frame.shape == (self.grid_size, self.grid_size)

        # Combine frame bytes and action
        hash_input = frame.tobytes() + str(action_idx).encode("utf-8")
        return hashlib.blake2b(hash_input).hexdigest()

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
        # Reset when the game when it's initilized
        if latest_frame.state == GameState.NOT_PLAYED:
            self.experience_buffer.clear()
            self.hashed_experiences.clear()
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = 0
            action = GameAction.RESET
            action.reasoning = "Game needs reset due to NOT_PLAYED or GAME_OVER"
            return action

        # Convert current frame to torch tensor
        current_frame = self._frame_to_tensor(latest_frame)

        # Store unique experience
        if current_frame is not None:
            self._store_experience(current_frame, latest_frame)

        # Train the model
        if self._should_train_model(latest_frame):
            buffer_size = len(self.experience_buffer)
            if buffer_size >= self.config["min_buffer_size"]:
                self.logger.info(
                    f"🤖 Training DT model. Game: {self.game_id} with (buffer size: {buffer_size})"
                )
                train_dt_model(
                    model=self.pure_dt_model,
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
                    f"Skipping training: buffer size {buffer_size} < min_buffer_size {self.config['min_buffer_size']}"
                )

        # Check level completion and perform reset
        self._check_level_completion(latest_frame=latest_frame)

        # Reset when the game state is GAME_OVER
        if latest_frame.state == GameState.GAME_OVER:
            self.experience_buffer.clear()
            self.hashed_experiences.clear()
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = 0
            action = GameAction.RESET
            action.reasoning = "Game needs reset due to NOT_PLAYED or GAME_OVER"
            return action

        # If frame processing failed, reset tracking and return random action
        if current_frame is None:
            print("Error detected in converting latest frame to tensor!")
            self.prev_frame = None
            self.prev_action_idx = None
            action = random.choice(self.action_list[:5])  # Random ACTION1-ACTION5
            action.reasoning = (
                f"Encountered a no-op frame, use a random action - {action.value}"
            )
            return action

        # If no reset and error, get action predictions from DT model (following bandit pattern exactly)
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


        return selected_action

    def _should_train_model(self, latest_frame: FrameData) -> bool:
        should_train_model = (
              self.action_counter % self.train_frequency == 0 or
              latest_frame.state in [GameState.WIN, GameState.GAME_OVER] or
              latest_frame.score != self.current_score
        )
        return should_train_model

    def _check_level_completion(self, latest_frame: FrameData) -> None | GameAction:
        # Check if score has changed and log score at action count

        if latest_frame.score != self.current_score:
            self.logger.info(
                f"Score changed from {self.current_score} to {latest_frame.score} at action {self.action_counter}"
            )
            self.logger.info("Cleared experience buffer - new level reached")

            # Clear experience buffer when reaching new level
            self.experience_buffer.clear()
            self.hashed_experiences.clear()
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = latest_frame.score

            #TODO: Should we reset the model as well? But retaining the model across levels may lead to better generalization


    def _store_experience(self, current_frame: torch.Tensor, latest_frame: FrameData):
        if self.prev_frame is not None:
            # Compute hash for uniqueness check
            hashed_experience = self._hash_experience(
                self.prev_frame, self.prev_action_idx
            )

            # Only store if unique
            if hashed_experience not in self.hashed_experiences:
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
                self.hashed_experiences.add(hashed_experience)

                # Log replay buffer size periodically
                self.writer.add_scalar(
                    "DTAgent/replay_buffer_size",
                    len(self.experience_buffer),
                    self.action_counter,
                )
                self.writer.add_scalar(
                    "DTAgent/replay_unique_hashes",
                    len(self.hashed_experiences),
                    self.action_counter,
                )

    def select_action(
        self, latest_frame_torch: torch.Tensor, latest_frame: FrameData
    ) -> tuple[int, int, GameAction]:

        self.pure_dt_model.eval()
        with torch.no_grad():
            # Build state-action sequence and get logits
            max_context_len = self.config["max_context_len"]

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

                # Get action logits from DT (three heads)
                logits = self.pure_dt_model(states, actions)
                change_logits = logits["change_logits"].squeeze(0)  # [4101]
                completion_logits = logits["completion_logits"].squeeze(0)  # [4101]
                gameover_logits = logits["gameover_logits"].squeeze(0)  # [4101]

                # Sample from combined action space (multiplicative combination with three heads)
                action_idx, coords, coord_idx = self._sample_from_combined_output(
                    change_logits, completion_logits, gameover_logits, latest_frame.available_actions
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
        gameover_logits: torch.Tensor,
        available_actions=None,
    ) -> tuple[int, tuple | None, int | None]:
        """Sample from combined action space using multiplicative combination of three heads.

        Args:
            change_logits: [4101] - Logits for predicting frame changes
            completion_logits: [4101] - Logits for predicting level completion
            gameover_logits: [4101] - Logits for predicting GAME_OVER (inverted for avoidance)
            available_actions: List of available actions for masking

        Returns:
            action_idx: Index of selected action
            coords: Coordinates if ACTION6 selected
            coord_idx: Flattened coordinate index
        """
        # Split logits into action (first 5) and coordinate (remaining 4096) spaces
        change_action_logits = change_logits[:5]  # [5]
        change_coord_logits = change_logits[5:]  # [4096]
        completion_action_logits = completion_logits[:5]  # [5]
        completion_coord_logits = completion_logits[5:]  # [4096]
        gameover_action_logits = gameover_logits[:5]  # [5]
        gameover_coord_logits = gameover_logits[5:]  # [4096]

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

            # Apply mask to all three heads' action logits
            change_action_logits = change_action_logits + action_mask
            completion_action_logits = completion_action_logits + action_mask
            gameover_action_logits = gameover_action_logits + action_mask

            # If ACTION6 (coordinate action) is not available, mask all coordinate logits
            if not action6_available:
                coord_mask = torch.full_like(change_coord_logits, float("-inf"))
                change_coord_logits = change_coord_logits + coord_mask
                completion_coord_logits = completion_coord_logits + coord_mask
                gameover_coord_logits = gameover_coord_logits + coord_mask

        # Apply sigmoid to convert logits to probabilities
        change_action_probs = torch.sigmoid(change_action_logits)  # [5]
        completion_action_probs = torch.sigmoid(completion_action_logits)  # [5]
        gameover_action_probs = torch.sigmoid(gameover_action_logits)  # [5]
        change_coord_probs = torch.sigmoid(change_coord_logits)  # [4096]
        completion_coord_probs = torch.sigmoid(completion_coord_logits)  # [4096]
        gameover_coord_probs = torch.sigmoid(gameover_coord_logits)  # [4096]

        # For fair sampling: treat coordinates as one action type with total prob divided by 4096
        change_coord_probs_scaled = change_coord_probs / self.num_coordinates
        completion_coord_probs_scaled = completion_coord_probs / self.num_coordinates
        gameover_coord_probs_scaled = gameover_coord_probs / self.num_coordinates

        # Combine action and coordinate probabilities for each head
        change_probs_sampling = torch.cat([change_action_probs, change_coord_probs_scaled])  # [4101]
        change_probs_sampling = change_probs_sampling / change_probs_sampling.sum()

        completion_probs_sampling = torch.cat([completion_action_probs, completion_coord_probs_scaled])  # [4101]
        completion_probs_sampling = completion_probs_sampling / completion_probs_sampling.sum()

        gameover_probs_sampling = torch.cat([gameover_action_probs, gameover_coord_probs_scaled])  # [4101]
        gameover_probs_sampling = gameover_probs_sampling / gameover_probs_sampling.sum()

        # CRITICAL: Invert gameover probabilities for avoidance (1.0 - p)
        # gameover_probs predicts "will cause GAME_OVER", we want to AVOID those actions
        gameover_probs_sampling_inverted = 1.0 - gameover_probs_sampling + 1e-10  # [4101]

        # Multiplicative combination: change * completion * (1 - gameover)
        probs_sampling = (
            change_probs_sampling *
            completion_probs_sampling *
            gameover_probs_sampling_inverted
        )  # [4101]

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
