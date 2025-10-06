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

from dt_agent.utils import setup_experiment_directory, setup_logging, get_environment_directory
from dt_agent.config import load_dt_config, get_loss_config_summary, validate_config


class StateEncoder(nn.Module):
    """CNN State Encoder - reuses proven bandit architecture for spatial reasoning."""
    
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
        x = F.relu(self.conv2(x))            # [batch, 64, 64, 64]  
        x = F.relu(self.conv3(x))            # [batch, 128, 64, 64]
        x = F.relu(self.conv4(x))            # [batch, 256, 64, 64]
        
        # Global spatial representation
        x = self.spatial_pool(x).flatten(1)  # [batch, 256]
        state_repr = self.state_projection(x)  # [batch, embed_dim]
        
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
    
    def __init__(self, embed_dim=256, num_layers=4, num_heads=8, max_context_len=20):
        super().__init__()
        
        # Component modules
        self.state_encoder = StateEncoder(embed_dim=embed_dim, freeze_weights=False)
        self.action_embedding = ActionEmbedding(embed_dim=embed_dim)
        
        # Positional encoding for temporal context
        self.pos_embedding = nn.Parameter(torch.randn(max_context_len * 2, embed_dim) * 0.02)
        
        # Decoder-only transformer (GPT-style)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Dual-head architecture like bandit model
        # Action head for discrete actions (ACTION1-5)
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 5)  # ACTION1-5
        )
        
        # Coordinate head for spatial actions (64x64 coordinates)
        self.coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4096)  # 64x64 coordinates
        )
        
    def build_state_action_sequence(self, states, actions):
        """Build interleaved state-action sequence: [s₀, a₀, s₁, a₁, ..., s_{t-1}, a_{t-1}, s_t]
        
        Args:
            states: [batch, seq_len+1, 16, 64, 64] - k+1 states (including current)
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
        
    def forward(self, states, actions):
        """
        Args:
            states: [batch, seq_len+1, 16, 64, 64] - k+1 states (past + current)
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
            memory=sequence  # Self-attention over the sequence
        )
        
        # Extract final representation (current state)
        final_repr = transformer_output[:, -1]  # [batch, embed_dim]
        
        # Dual-head prediction
        action_logits = self.action_head(final_repr)  # [batch, 5] - ACTION1-5
        coord_logits = self.coord_head(final_repr)    # [batch, 4096] - coordinates
        
        # Concatenate for compatibility with existing interface
        combined_logits = torch.cat([action_logits, coord_logits], dim=1)  # [batch, 4101]
        
        return combined_logits



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
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DT Agent using device: {self.device}")
        
        # Setup experiment directory and logging
        self.base_dir, log_file = setup_experiment_directory()
        setup_logging(log_file)
        
        # Get environment-specific directory
        env_dir = get_environment_directory(self.base_dir, self.game_id)
        tensorboard_dir = os.path.join(env_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        self.writer = SummaryWriter(tensorboard_dir)
        self.current_score = -1
        
        # Setup logger
        self.logger = logging.getLogger(f"DTAgent_{self.game_id}")
        
        # Configuration for visualization and logging
        self.save_action_visualizations = False  # Set to True to enable image generation
        self.vis_save_frequency = 100  # Save images every N steps
        self.vis_samples_per_save = 1  # Number of visualization samples to save each time
        self.log_dir = env_dir
        
        # Grid and action space configuration
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16
        
        # Load DT configuration
        self.pure_dt_config = load_dt_config(device=str(self.device))
        validate_config(self.pure_dt_config)
        
        # Log configuration
        config_summary = get_loss_config_summary(self.pure_dt_config)
        self.logger.info(f"DT initialized: {config_summary}")
        print(f"DT initialized: {config_summary}")
        
        # Initialize DT model
        self.pure_dt_model = DecisionTransformer(
            embed_dim=self.pure_dt_config['embed_dim'],
            num_layers=self.pure_dt_config['num_layers'],
            num_heads=self.pure_dt_config['num_heads'],
            max_context_len=self.pure_dt_config['max_context_len']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.pure_dt_model.parameters(),
            lr=self.pure_dt_config['learning_rate'],
            weight_decay=self.pure_dt_config['weight_decay']
        )
        
        # Action sampling now done directly in choose_action() like bandit
        
        # Experience buffer for training with uniqueness tracking
        self.experience_buffer = deque(maxlen=self.pure_dt_config['max_buffer_size'])
        self.experience_hashes = set()
        self.train_frequency = self.pure_dt_config['train_frequency']
        
        # Track previous state/action for experience creation
        self.prev_frame = None
        self.prev_action_idx = None
        
        # Action mapping: ACTION1-ACTION5
        self.action_list = [
            GameAction.ACTION1,
            GameAction.ACTION2, 
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]
        
        print(f"DT Agent logging to: {tensorboard_dir}")
        self.logger.info(f"DT Agent initialized for game_id: {self.game_id}")
        if self.save_action_visualizations:
            self.logger.info(
                f"Action visualizations enabled: saving {self.vis_samples_per_save} samples every {self.vis_save_frequency} steps"
            )
    
    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        """Convert frame data to tensor format for the model."""
        # Convert frame to numpy array with color indices 0-15
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        # Take the last frame (in case of an animation of frames)
        frame = frame[-1]
        
        assert frame.shape == (self.grid_size, self.grid_size)
        
        # One-hot encode: (64, 64) -> (16, 64, 64)
        tensor = torch.zeros(
            self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32
        )
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        
        return tensor.to(self.device)
    
    def _compute_experience_hash(self, frame: np.array, action_idx: int) -> str:
        """Compute hash for frame+action combination to ensure uniqueness."""
        assert frame.shape == (self.num_colours, self.grid_size, self.grid_size)
        frame_bytes = frame.tobytes()
        
        # Create hash from frame + action combination
        hash_input = frame_bytes + str(action_idx).encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()
    
    def _train_dt_model(self):
        """Train the Pure Decision Transformer on collected experiences using configurable loss."""
        if len(self.experience_buffer) < self.pure_dt_config['min_buffer_size']:
            return
        
        # Sample experiences for sequence creation
        sample_size = min(
            int(len(self.experience_buffer) * self.pure_dt_config['experience_sample_rate']),
            self.pure_dt_config['max_training_experiences']
        )
        
        if sample_size < self.pure_dt_config['context_length'] + 1:
            return
        
        # Create training sequences from experience buffer
        sequences = self._create_training_sequences(sample_size)
        if not sequences:
            return
        
        # Convert sequences to tensors
        states_batch = torch.stack([seq['states'] for seq in sequences]).to(self.device)
        actions_batch = torch.stack([seq['actions'] for seq in sequences]).to(self.device)
        targets_batch = torch.stack([seq['targets'] for seq in sequences]).to(self.device)
        rewards_batch = torch.stack([seq['rewards'] for seq in sequences]).to(self.device)
        
        # Log training start info
        num_sequences = len(sequences)
        num_positive_rewards = (rewards_batch > 0).sum().item()
        loss_type = self.pure_dt_config['loss_type']
        self.logger.info(f"🎯 Starting DT training: {num_sequences} sequences, {num_positive_rewards} positive rewards, loss_type={loss_type}")
        
        # Training loop with configurable epochs
        for epoch in range(self.pure_dt_config['epochs_per_training']):
            self.optimizer.zero_grad()
            
            # Forward pass
            action_logits = self.pure_dt_model(states_batch, actions_batch)  # [batch, 4101]
            
            # Compute configurable loss
            loss, metrics = self._compute_pure_dt_loss(action_logits, targets_batch, rewards_batch)
            
            # Gradient clipping and backward pass
            loss.backward()
            if self.pure_dt_config['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.pure_dt_model.parameters(), 
                    self.pure_dt_config['gradient_clip_norm']
                )
            self.optimizer.step()
            
            # Log training metrics
            if self.save_action_visualizations:
                self._log_pure_dt_metrics(loss, metrics, epoch)
        
        # Log training completion
        final_loss = loss.item() if 'loss' in locals() else 0.0
        final_accuracy = metrics.get('accuracy', 0.0) if 'metrics' in locals() else 0.0
        self.logger.info(f"✅ DT training completed: final_loss={final_loss:.4f}, accuracy={final_accuracy:.3f}")
        
        # Clean up GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _create_training_sequences(self, sample_size):
        """Create state-action sequences for DT training."""
        sequences = []
        context_len = self.pure_dt_config['context_length']
        
        # Ensure sample_size is an integer
        sample_size = int(sample_size)
        
        # Sample the starting indices for the sequences
        replace = False if sample_size <= len(self.experience_buffer)-context_len else True
        exp_start_idx = np.random.choice(
                len(self.experience_buffer) - context_len,
                size=sample_size,
                replace=replace
        )

        experience_buffer = list(self.experience_buffer)
        for i in exp_start_idx:
            sequence_experiences = experience_buffer[i:i + context_len]
            
            # Build state sequence [s_0, s_1, ..., s_k] (k+1 states)
            states = []
            for exp in sequence_experiences:
                state_tensor = torch.from_numpy(exp['state']).float()
                states.append(state_tensor)
            states = torch.stack(states)  # [k+1, 16, 64, 64]
            
            # Build action sequence [a_0, a_1, ..., a_{k-1}] (k past actions)
            actions = []
            for exp in sequence_experiences[:-1]:  # Exclude last (we predict its action)
                actions.append(exp['action_idx'])
            actions = torch.tensor(actions, dtype=torch.long)  # [k]
            
            # Target action and reward (what we want to predict)
            target_action = sequence_experiences[-1]['action_idx']
            target_reward = sequence_experiences[-1]['reward']
            
            sequences.append({
                'states': states,
                'actions': actions,
                'targets': torch.tensor(target_action, dtype=torch.long),
                'rewards': torch.tensor(target_reward, dtype=torch.float32)
            })
        
        return sequences
    
    def _compute_pure_dt_loss(self, action_logits, targets, rewards):
        """Compute configurable loss for DT training."""
        loss_type = self.pure_dt_config['loss_type']
        
        if loss_type == 'cross_entropy':
            # Standard cross-entropy loss (dense updates)
            loss = F.cross_entropy(action_logits, targets)
            metrics = {'accuracy': (action_logits.argmax(dim=1) == targets).float().mean()}
            
        elif loss_type == 'bandit':
            # Bandit-style: Binary cross-entropy on selected action logits only
            # This matches the original bandit optimization process exactly
            
            # Gather only the logits for selected actions
            selected_logits = action_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Binary cross-entropy with rewards as binary labels
            main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)
            
            # Add entropy regularization like bandit (encourage exploration)
            all_probs = torch.sigmoid(action_logits)
            
            # Split into action and coordinate spaces for separate entropy
            action_probs = all_probs[:, :5]
            coord_probs = all_probs[:, 5:]
            
            # Calculate entropy bonus (mean sigmoid activation)
            action_entropy = action_probs.mean()
            coord_entropy = coord_probs.mean()
            
            # Dynamic entropy coefficients (can be made configurable)
            action_coeff = self.pure_dt_config.get('action_entropy_coeff', 0.0001)
            coord_coeff = self.pure_dt_config.get('coord_entropy_coeff', 0.00001)
            
            # Total loss with entropy regularization
            loss = main_loss - action_coeff * action_entropy - coord_coeff * coord_entropy
            
            # Calculate accuracy: did we correctly predict whether action causes frame change?
            # This matches what the loss is optimizing for (bandit-style frame change prediction)
            accuracy = (
                ((torch.sigmoid(selected_logits) > 0.5) == rewards).float().mean()
            )

            metrics = {
                'accuracy': accuracy.item(),
                'main_loss': main_loss.item(),
                'action_entropy': action_entropy.item(),
                'coord_entropy': coord_entropy.item(),
                'total_loss': loss.item()
            }
            
        elif loss_type == 'selective':
            # Selective loss - only update on positive rewards (sparse updates)
            positive_mask = rewards > 0
            if positive_mask.sum() > 0:
                loss = F.cross_entropy(action_logits[positive_mask], targets[positive_mask])
                metrics = {
                    'accuracy': ((action_logits.argmax(dim=1)>0.5) == targets).float().mean(),
                    'positive_samples': positive_mask.sum().float()
                }
            else:
                loss = torch.tensor(0.0, device=action_logits.device, requires_grad=True)
                metrics = {'accuracy': 0.0, 'positive_samples': 0.0}

        
        return loss, metrics
    
    def _log_pure_dt_metrics(self, loss, metrics, epoch):
        """Log DT training metrics to tensorboard."""
        step = self.action_counter
        
        self.writer.add_scalar("DT/loss", loss.item(), step)
        self.writer.add_scalar("DT/accuracy", metrics.get('accuracy', 0), step)
        
        # Log bandit-specific metrics if available
        if 'main_loss' in metrics:
            self.writer.add_scalar("DT/main_loss", metrics['main_loss'], step)
        if 'action_entropy' in metrics:
            self.writer.add_scalar("DT/action_entropy", metrics['action_entropy'], step)
        if 'coord_entropy' in metrics:
            self.writer.add_scalar("DT/coord_entropy", metrics['coord_entropy'], step)
        
        # Loss-type specific metrics
        if 'positive_samples' in metrics:
            self.writer.add_scalar("DT/positive_samples", metrics['positive_samples'], step)
        if 'high_confidence_frac' in metrics:
            self.writer.add_scalar("DT/high_confidence_frac", metrics['high_confidence_frac'], step)
        if 'mean_confidence' in metrics:
            self.writer.add_scalar("DT/mean_confidence", metrics['mean_confidence'], step)
    
    def _has_time_elapsed(self) -> bool:
        """Check if 8 hours have elapsed since start."""
        elapsed_hours = time.time() - self.start_time
        return elapsed_hours >= 8 * 3600 - 5 * 60  # 8 hours with 5 minute buffer
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any([
            latest_frame.state is GameState.WIN,
            self._has_time_elapsed(),
        ])
    
    def choose_action(self, frames: list[FrameData], latest_frame_data: FrameData) -> GameAction:
        """Choose action using Decision Transformer predictions."""

        # Reset the game when certain conditions are met
        self._reset_if_required(latest_frame_data=latest_frame_data)

        # Convert current frame to tensor
        current_frame = self._frame_to_tensor(latest_frame_data)
        
        # If frame processing failed, reset tracking and return random action
        if current_frame is None:
            print("Error detected!")
            self.prev_frame = None
            self.prev_action_idx = None
            action = random.choice(self.action_list[:5])  # Random ACTION1-ACTION5
            action.reasoning = f"Skipped error frame, use a random action - {action.value}"
            return action
        
        # Store unique experience
        self._store_experience(current_frame)
        
        # Get action predictions from DT model (following bandit pattern exactly)
        action_idx, coord_idx, selected_action = self.select_action(
                latest_frame_torch=current_frame,
                latest_frame_data= latest_frame_data
        )
        
        # Store current frame and action for next experience creation
        self.prev_frame = current_frame.cpu().numpy().astype(bool)
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
            if buffer_size >= self.pure_dt_config['min_buffer_size']:
                self.logger.info(f"🤖 Training DT model... (buffer size: {buffer_size})")
                self._train_dt_model()
            else:
                self.logger.debug(f"Skipping training: buffer size {buffer_size} < min_buffer_size {self.pure_dt_config['min_buffer_size']}")

        return selected_action

    def _reset_if_required(self, latest_frame_data: FrameData) -> None | GameAction:
        # Check if score has changed and log score at action count

        if latest_frame_data.score != self.current_score:
            if self.save_action_visualizations:
                self.writer.add_scalar("Agent/score", latest_frame_data.score, self.action_counter)

            self.logger.info(
                f"Score changed from {self.current_score} to {latest_frame_data.score} at action {self.action_counter}"
            )
            print(f"Score changed from {self.current_score} to {latest_frame_data.score} at action {self.action_counter}")

            # Clear experience buffer when reaching new level
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            self.logger.info("Cleared experience buffer - new level reached")
            print("Cleared experience buffer - new level reached")

            # Reset DT model and optimizer for new level
            self.pure_dt_model = DecisionTransformer(
                embed_dim=self.pure_dt_config['embed_dim'],
                num_layers=self.pure_dt_config['num_layers'],
                num_heads=self.pure_dt_config['num_heads'],
                max_context_len=self.pure_dt_config['max_context_len']
            ).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.pure_dt_model.parameters(),
                lr=self.pure_dt_config['learning_rate'],
                weight_decay=self.pure_dt_config['weight_decay']
            )

            self.logger.info("Reset DT model and optimizer for new level")
            print("Reset DT model and optimizer for new level")

            # Reset previous tracking
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = latest_frame_data.score

        if latest_frame_data.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Reset previous tracking on game reset
            self.prev_frame = None
            self.prev_action_idx = None
            action = GameAction.RESET
            action.reasoning = "Game needs reset."
            return action

    def _store_experience(self, latest_frame: torch.Tensor):
        if self.prev_frame is not None:
            # Compute hash for uniqueness check
            experience_hash = self._compute_experience_hash(self.prev_frame, self.prev_action_idx)

            # Only store if unique
            if experience_hash not in self.experience_hashes:
                # Convert current frame to numpy bool for comparison
                latest_frame_np = latest_frame.cpu().numpy().astype(bool)
                frame_changed = not np.array_equal(self.prev_frame, latest_frame_np)

                experience = {
                    "state": self.prev_frame,  # Already numpy bool
                    "action_idx": self.prev_action_idx,  # Unified action index
                    "reward": 1.0 if frame_changed else 0.0,
                }
                self.experience_buffer.append(experience)
                self.experience_hashes.add(experience_hash)

                # Log replay buffer size periodically
                if self.save_action_visualizations:
                    self.writer.add_scalar(
                        "Agent/replay_buffer_size", len(self.experience_buffer), self.action_counter
                    )
                    self.writer.add_scalar(
                        "Agent/replay_unique_hashes", len(self.experience_hashes), self.action_counter
                    )

    def select_action(self, latest_frame_torch: torch.Tensor, latest_frame_data: FrameData) -> tuple[int, int, GameAction]:
        with torch.no_grad():
            # Build state-action sequence and get logits
            context_length = self.pure_dt_config['context_length']

            # Cold start - random valid action if no experience
            if len(self.experience_buffer) < 1:
                selected_action = self._random_valid_action(latest_frame_data.available_actions)
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
                        latest_frame_torch, context_length
                )

                # Get action logits from DT
                combined_logits = self.pure_dt_model(states, actions)  # [1, 4101]
                combined_logits = combined_logits.squeeze(0)  # (4101,)

                # Sample from combined action space (following bandit pattern exactly)
                action_idx, coords, coord_idx, all_probs = (
                    self._sample_from_combined_output(
                        combined_logits, latest_frame_data.available_actions
                    )
                )

                # Store probabilities for visualization
                self._last_all_probs = all_probs

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
                    self.logger.info(f"📍 ACTION6 selected: coordinates ({x}, {y}) -> coord_idx={coord_idx}")

        return action_idx, coord_idx, selected_action

    def _sample_from_combined_output(
        self, combined_logits: torch.Tensor, available_actions=None
    ) -> tuple[int, tuple, int, np.ndarray]:
        """Sample from combined 5 + 64x64 action space with masking for invalid actions.
        
        Copied exactly from bandit model to ensure identical behavior.
        """
        # Split logits
        action_logits = combined_logits[:5]  # First 5
        coord_logits = combined_logits[5:]  # Remaining 4096

        # Apply masking based on available_actions if provided
        if available_actions is not None and len(available_actions) > 0:
            # Create mask for action logits (ACTION1-ACTION5 = indices 0-4)
            action_mask = torch.full_like(action_logits, float("-inf"))
            action6_available = False

            for action in available_actions:
                # Extract action value if it's a GameAction enum
                action_id = action.value

                if 1 <= action_id <= 5:  # ACTION1-ACTION5
                    action_mask[action_id - 1] = 0.0  # Unmask valid actions
                elif action_id == 6:  # ACTION6
                    action6_available = True

            # Apply mask to action logits
            action_logits = action_logits + action_mask

            # If ACTION6 (coordinate action) is not available, mask all coordinate logits
            if not action6_available:
                coord_mask = torch.full_like(coord_logits, float("-inf"))
                coord_logits = coord_logits + coord_mask

        # Apply sigmoid
        action_probs = torch.sigmoid(action_logits)
        coord_probs_raw = torch.sigmoid(coord_logits)

        # For fair sampling: treat coordinates as one action type with total prob divided by 4096
        coord_probs_scaled = coord_probs_raw / self.num_coordinates

        # Combine for sampling (normalize)
        all_probs_sampling = torch.cat([action_probs, coord_probs_scaled])
        all_probs_sampling = all_probs_sampling / all_probs_sampling.sum()
        all_probs_sampling_np = all_probs_sampling.cpu().numpy()

        # Sample from normalized space
        selected_idx = np.random.choice(
            len(all_probs_sampling_np), p=all_probs_sampling_np
        )

        # Return unnormalized sigmoid values for visualization
        coord_probs_viz = torch.sigmoid(coord_logits)  # Raw sigmoid for visualization
        all_probs_viz = torch.cat([action_probs, coord_probs_viz])
        all_probs_viz_np = all_probs_viz.cpu().numpy()

        if selected_idx < 5:
            # Selected one of ACTION1-ACTION5
            return selected_idx, None, None, all_probs_viz_np
        else:
            # Selected a coordinate (index 5-4100)
            coord_idx = selected_idx - 5
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx, all_probs_viz_np
    
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
    
    def _build_inference_sequence(self, current_frame, context_length):
        """Build state-action sequence for Decision Transformer inference."""
        # Get recent experiences up to context length
        available_context = min(int(context_length), len(self.experience_buffer))
        recent_experiences = list(self.experience_buffer)[-available_context:] if available_context > 0 else []

        # Build states: recent states + current state
        states_list = []
        for exp in recent_experiences:
            states_list.append(torch.from_numpy(exp['state']).float())
        states_list.append(current_frame)  # Add current state

        # Build actions: recent actions (no current action - we're predicting it)
        actions_list = []
        for exp in recent_experiences:
            actions_list.append(exp['action_idx'])

        # Handle cold start: if no actions yet, create minimal sequence with one dummy action
        # This is only needed to satisfy model input requirements, not for actual padding
        if len(actions_list) == 0:
            # Use ACTION1 (index 0) as dummy action for cold start
            # The model won't rely on this since there's no history anyway
            actions_list.append(0)
            # Add a zero state to maintain sequence structure [s0, a0, s1]
            states_list = [torch.zeros_like(current_frame)] + states_list
        
        # Convert to tensors
        states = torch.stack(states_list).unsqueeze(0).to(self.device)  # [1, seq_len+1, 16, 64, 64]
        actions = torch.tensor(actions_list).unsqueeze(0).to(self.device)  # [1, seq_len]
        
        return states, actions