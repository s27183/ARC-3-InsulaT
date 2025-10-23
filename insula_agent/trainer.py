"""
InsulaAgent Training Module

This module contains all training-related functions for InsulaAgent,
including unified context length, temporal hindsight traces,
and importance-weighted replay.

Uses online supervised learning with self-generated labels from game outcomes.

Key features:
- Unified context length (25 steps for all heads)
- Temporal hierarchy via head-specific decay rates (γ=1.0, 0.8, 0.9), not sequence length
- Outcome-anchored sampling for completion/gameover heads
- Trajectory-level reward revaluation (memory reconsolidation)
- Direct batching (all sequences same length → no grouping needed)
"""

from typing import Any
import random
from collections import deque, defaultdict
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from insula_agent.config import InsulaConfig


# ============================================================================
# Sequence Creation Functions
# ============================================================================

def extract_sequence(
    experience_buffer: deque,
    start_idx: int,
    context_len: int,
) -> dict[str, torch.Tensor]:
    """Extract a state-action sequence from experience buffer.

    Args:
        experience_buffer: Deque of experience dictionaries
        start_idx: Starting index in experience buffer
        context_len: Number of past actions (k)

    Returns:
        Dictionary with sequence data:
            - "states": torch.Tensor [k+1, 64, 64] - Grid states
            - "actions": torch.Tensor [k] - Past action indices
            - "target_action": torch.Tensor [] - Final action (scalar)
            - "change_reward": torch.Tensor [] - Final change reward (scalar)
            - "completion_reward": torch.Tensor [] - Final completion reward (scalar)
            - "gameover_reward": torch.Tensor [] - Final gameover reward (scalar)
            - "all_action_indices": torch.Tensor [k+1] - All actions for temporal credit
            - "all_change_rewards": torch.Tensor [k+1] - All change rewards
            - "all_change_momentum_rewards": torch.Tensor [k+1] - All change momentum rewards
            - "all_completion_rewards": torch.Tensor [k+1] - All completion rewards
            - "all_gameover_rewards": torch.Tensor [k+1] - All gameover rewards
    """
    # Extract k+1 experiences
    experiences = list(experience_buffer)[start_idx : start_idx + context_len + 1]

    # Build state sequence [s_0, s_1, ..., s_k] (k+1 states)
    states = torch.stack([
        torch.from_numpy(exp["state"]).long() for exp in experiences
    ])  # [k+1, 64, 64]

    # Build action sequence [a_0, a_1, ..., a_{k-1}] (k past actions)
    actions = torch.tensor(
        [exp["action_idx"] for exp in experiences[:-1]], dtype=torch.long
    )  # [k]

    # Target action and rewards (final state)
    target_action = torch.tensor(experiences[-1]["action_idx"], dtype=torch.long)
    change_reward = torch.tensor(experiences[-1]["change_reward"], dtype=torch.float32)
    completion_reward = torch.tensor(experiences[-1]["completion_reward"], dtype=torch.float32)
    gameover_reward = torch.tensor(experiences[-1]["gameover_reward"], dtype=torch.float32)

    # Temporal credit data (ALL actions and rewards including current)
    all_action_indices = torch.tensor(
        [exp["action_idx"] for exp in experiences], dtype=torch.long
    )  # [k+1]
    all_change_rewards = torch.tensor(
        [exp["change_reward"] for exp in experiences], dtype=torch.float32
    )  # [k+1]
    all_change_momentum_rewards = torch.tensor(
        [exp["change_momentum_reward"] for exp in experiences], dtype=torch.float32
    )  # [k+1]
    all_completion_rewards = torch.tensor(
        [exp["completion_reward"] for exp in experiences], dtype=torch.float32
    )  # [k+1]
    all_gameover_rewards = torch.tensor(
        [exp["gameover_reward"] for exp in experiences], dtype=torch.float32
    )  # [k+1]

    return {
        "states": states,
        "actions": actions,
        "target_action": target_action,
        "change_reward": change_reward,
        "completion_reward": completion_reward,
        "gameover_reward": gameover_reward,
        "all_action_indices": all_action_indices,
        "all_change_rewards": all_change_rewards,
        "all_change_momentum_rewards": all_change_momentum_rewards,
        "all_completion_rewards": all_completion_rewards,
        "all_gameover_rewards": all_gameover_rewards,
    }


def apply_trajectory_rewards(
    sequence: dict[str, torch.Tensor],
    config: InsulaConfig,
) -> dict[str, torch.Tensor]:
    """Apply trajectory-level reward revaluation during replay (Memory Reconsolidation).

    This implements the biological mechanism of dopaminergic modulation during
    hippocampal replay (Gomperts et al., 2015). Experience buffer stores action-level
    rewards, but during replay we assign trajectory-level rewards to learn which
    action sequences lead to success/failure.

    IMPORTANT: This does NOT modify the experience buffer - only the replayed sequence!

    Args:
        sequence: Sequence dictionary from extract_sequence()
        config: InsulaConfig with use_trajectory_rewards flag

    Returns:
        Modified sequence dictionary with trajectory-level rewards:
            - "all_completion_rewards": Set to 1.0 for actions where change_reward==1.0 (if completion)
            - "all_gameover_rewards": Set to 0.0 for actions where change_reward==1.0 (if gameover)
            - "all_change_rewards": UNCHANGED (action-level rewards, immediate causality)
            - "all_change_momentum_rewards": UNCHANGED (action-level rewards, comparative causality)

    Rationale:
        - Only productive actions (those that changed the grid) receive trajectory credit/blame
        - Invalid/no-op actions (change_reward==0.0) remain at action-level rewards
        - Completion: If sequence ends in success, credit productive actions
        - Gameover: If sequence ends in failure, penalize productive actions
        - Change: Grid changes have immediate causality (action-level rewards valid)
    """
    if not config.use_trajectory_rewards:
        return sequence  # Return unchanged if disabled

    # Create a copy to avoid modifying original tensors
    sequence = sequence.copy()

    # Get final rewards (scalar tensors)
    final_completion = sequence["completion_reward"].item()  # 1.0 (completion) or 0.0 (no completion)
    final_gameover = sequence["gameover_reward"].item()      # 1.0 (alive) or 0.0 (dead)

    # Trajectory-level reward assignment for COMPLETION
    # If final action led to level completion, credit actions that changed the grid
    # Rationale: Only productive actions (change_reward=1.0) should receive credit
    if final_completion == 1.0:
        mask = sequence["all_change_rewards"][:-1] == 1.0  # [k] boolean mask (exclude final state)
        sequence["all_completion_rewards"][:-1][mask] = 1.0

    # Trajectory-level reward assignment for GAMEOVER
    # If final action led to GAME_OVER (death), penalize actions that changed the grid
    # Rationale: Only productive actions contributed to failure (invalid actions are irrelevant)
    if final_gameover == 0.0:
        sequence["all_gameover_rewards"][:] = 0.0
        mask = sequence["all_completion_rewards"][:-1] == 1.0  # [k] boolean mask (exclude final state)
        sequence["all_gameover_rewards"][:-1][mask] = 1.0

    # Change rewards stay UNCHANGED - action-level causality (immediate)
    # sequence["all_change_rewards"] is NOT modified

    return sequence


def create_change_seqs(
    experience_buffer: deque,
    config: InsulaConfig,
) -> list[dict[str, torch.Tensor]]:
    """Create change sequences: random sampling without replacement (when possible).

    Samples WITHOUT replacement within batch to maximize diversity and reduce
    gradient correlation. Falls back to WITH replacement if buffer is small.

    Supports adaptive context length: uses min(context_len, buffer_len - 1) for
    early training when buffer is small.

    Args:
        experience_buffer: Deque of experience dictionaries
        config: Configuration dictionary with:
            - "change_replay_size": int (default 16)
            - "context_len": int (default 50, maximum context length)

    Returns:
        List of sequence dictionaries with adaptive length.
        Returns empty list if buffer too small (< 2 experiences).
    """
    replay_size = config.change_replay_size
    context_len = config.context_len  # Maximum context length
    buffer_len = len(experience_buffer)

    # Need at least 2 experiences (1 transition: s0, a0, s1)
    if buffer_len < 2:
        return []

    # Adaptive context: use available buffer size if smaller than context_len
    # Early training: buffer_len=5 → actual_context_len=4 (4 actions + 5 states)
    # Late training: buffer_len=100 → actual_context_len=50 (capped at context_len)
    actual_context_len = min(context_len, buffer_len - 1)

    # Calculate possible start positions using actual_context_len
    max_start_idx = buffer_len - actual_context_len - 1
    num_possible_starts = max_start_idx + 1

    # Adaptive batch size: use min(replay_size, num_possible_starts)
    # Honest learning - no oversampling for change sequences
    # Early game: Small batches (1-10 sequences)
    # Late game: Full batches (128 sequences)
    effective_batch_size = min(replay_size, num_possible_starts)
    sampled_starts = np.random.choice(
        num_possible_starts,
        size=effective_batch_size,
        replace=False  # Always sample without replacement
    )

    sequences = []
    for start_idx in sampled_starts:
        sequence = extract_sequence(experience_buffer, int(start_idx), actual_context_len)
        # Apply trajectory reward revaluation (memory reconsolidation)
        sequence = apply_trajectory_rewards(sequence, config)
        sequences.append(sequence)

    return sequences


# Note: create_change_momentum_seqs() removed - momentum head uses same sequences as change head
# Both are exploration signals with action-level rewards, so they share the same sampling strategy


def create_completion_seqs(
    experience_buffer: deque,
    config: InsulaConfig,
) -> list[dict[str, torch.Tensor]]:
    """Create completion sequences: outcome-anchored, adaptive context length.

    Supports adaptive context length: sequences ending at completion events use
    min(context_len, event_idx) to handle early game completions.

    Args:
        experience_buffer: Deque of experience dictionaries
        config: Configuration dictionary with:
            - "completion_replay_size": int (default 128)
            - "context_len": int (default 50, maximum context length)

    Returns:
        List of sequence dictionaries with adaptive length.
        Returns empty list if no completion events found.
    """
    replay_size = config.completion_replay_size
    context_len = config.context_len  # Maximum context length

    # Find all completion events in buffer
    completion_idx = [
        i for i, exp in enumerate(experience_buffer)
        if exp["completion_reward"] == 1.0
    ]

    if not completion_idx:
        return []  # No completion events yet

    # KEEP WITH replacement for completion sequences (intentional oversampling)
    # Rationale: Buffer is CLEARED after level completion, so this is the ONLY
    # chance to learn from success. Oversampling ensures strong gradient signal
    # for this rare, critical event before it's lost forever.
    if len(completion_idx) < replay_size:
        # SPARSE: Sample WITH replacement (oversample the rare success)
        sampled_indices = np.random.choice(completion_idx, size=replay_size, replace=True)
    else:
        # ABUNDANT: Sample WITHOUT replacement (subsample)
        sampled_indices = np.random.choice(completion_idx, size=replay_size, replace=False)

    sequences = []
    for completion_idx in sampled_indices:
        # Adaptive context: use min(context_len, completion_idx) to handle early completions
        # Example: completion at index 3 → actual_context_len=3 (can't use full context_len=50)
        actual_context_len = min(context_len, completion_idx)  # Can't go before buffer start

        # Extract sequence ENDING at completion_idx
        start_idx = completion_idx - actual_context_len
        sequence = extract_sequence(experience_buffer, start_idx, actual_context_len)
        # Apply trajectory reward revaluation (memory reconsolidation)
        sequence = apply_trajectory_rewards(sequence, config)
        sequences.append(sequence)

    return sequences


def create_gameover_seqs(
    experience_buffer: deque,
    config: InsulaConfig,
) -> list[dict[str, torch.Tensor]]:
    """Create GAME_OVER sequences: outcome-anchored, adaptive context length.

    Supports adaptive context length: sequences ending at GAME_OVER events use
    min(context_len, event_idx) to handle early game failures.

    Args:
        experience_buffer: Deque of experience dictionaries
        config: Configuration dictionary with:
            - "gameover_replay_size": int (default 128)
            - "context_len": int (default 50, maximum context length)

    Returns:
        List of sequence dictionaries with adaptive length.
        Returns empty list if no GAME_OVER events found.
    """
    replay_size = config.gameover_replay_size
    context_len = config.context_len  # Maximum context length

    # Find all GAME_OVER events in buffer (gameover_reward == 0.0 means GAME_OVER occurred)
    gameover_idx = [
        i for i, exp in enumerate(experience_buffer)
        if exp["gameover_reward"] == 0.0
    ]

    if not gameover_idx:
        return []  # No GAME_OVER events yet

    # Adaptive batch size: use min(replay_size, num_gameover_events)
    # Honest learning - no oversampling for gameover sequences
    # Buffer persists within level, diversity grows naturally
    effective_batch_size = min(replay_size, len(gameover_idx))
    sampled_indices = np.random.choice(
        gameover_idx,
        size=effective_batch_size,
        replace=False  # Always sample without replacement
    )

    sequences = []
    for gameover_idx in sampled_indices:
        # Adaptive context: use min(context_len, gameover_idx) to handle early failures
        # Example: GAME_OVER at index 2 → actual_context_len=2 (can't use full context_len=50)
        actual_context_len = min(context_len, gameover_idx)  # Can't go before buffer start

        # Extract sequence ENDING at gameover_idx
        start_idx = gameover_idx - actual_context_len
        sequence = extract_sequence(experience_buffer, start_idx, actual_context_len)
        # Apply trajectory reward revaluation (memory reconsolidation)
        sequence = apply_trajectory_rewards(sequence, config)
        sequences.append(sequence)

    return sequences


# ============================================================================
# Loss Computation Functions
# ============================================================================

def compute_head_loss(head_logits: torch.Tensor, target_actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Compute loss for a single head WITHOUT temporal credit (only final action).

    Args:
        head_logits: [batch, 4102] - Predicted logits for this head
        target_actions: [batch] - Final action indices (scalar per sequence)
        rewards: [batch] - Final rewards for this head (1.0 or 0.0)


    Returns:
        loss: Scalar loss tensor (binary cross-entropy)
    """
    # Gather only the logits for final actions
    selected_logits = head_logits.gather(
        dim=1, index=target_actions.unsqueeze(1)
    ).squeeze(1)  # [batch]

    # Binary cross-entropy with rewards as binary labels
    loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

    return loss


def compute_head_loss_with_temporal_credit(logits: torch.Tensor, all_action_indices: torch.Tensor,
                                           all_rewards: torch.Tensor, temporal_update_decay: torch.Tensor) -> torch.Tensor:
    """Compute loss for a single head with per-timestep predictions and temporal credit assignment.

    Vectorized implementation for GPU efficiency.

    Implements continuous forward modeling: each state's prediction is evaluated against its action.
    This matches hippocampal replay where all moments in a sequence get reactivated and updated.

    Each head (change/completion/gameover) has its own temporal decay rate.

    Args:
        logits: [batch, seq_len, 4102] - Predicted logits at each state (training mode)
        all_action_indices: [batch, seq_len] - ALL actions in sequence
        all_rewards: [batch, seq_len] - Rewards for this head (1.0 or 0.0)
        temporal_update_decay: Head-specific decay rate (scalar: 1.0, 0.8, or 0.9)

    Returns:
        loss: Scalar loss tensor (temporally weighted binary cross-entropy)

    Temporal Weighting:
        - Weight formula: w_t = γ^(seq_len - 1 - t)
        - Most recent (t=seq_len-1): γ^0 = 1.0 (full weight)
        - Oldest (t=0): γ^(seq_len-1) (smallest weight)
        - Example with γ=0.8, seq_len=26:
          * t=25 (newest): 0.8^0 = 1.000
          * t=20: 0.8^5 = 0.328
          * t=10: 0.8^15 = 0.035
          * t=0 (oldest): 0.8^25 = 0.004
    """
    batch_size, seq_len, _ = logits.shape

    # Compute temporal weights for ALL timesteps at once: [seq_len]
    # torch.arange(seq_len - 1, -1, -1) creates countdown: [seq_len-1, seq_len-2, ..., 1, 0]
    # This represents "steps from end" for each timestep
    # Example with seq_len=26: [25, 24, 23, ..., 1, 0]
    steps_from_end = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32, device=logits.device)  # [seq_len]
    temporal_weights = temporal_update_decay ** steps_from_end  # [seq_len]

    # Select logits for actions taken at each timestep (vectorized gather)
    # logits: [batch, seq_len, 4102]
    # all_action_indices: [batch, seq_len]
    # Need to expand indices to [batch, seq_len, 1] to gather along dim=2
    selected_logits = logits.gather(
        dim=2,  # Gather along action dimension
        index=all_action_indices.unsqueeze(2)  # [batch, seq_len] → [batch, seq_len, 1]
    ).squeeze(2)  # [batch, seq_len, 1] → [batch, seq_len]

    # Compute BCE loss for ALL timesteps at once: [batch, seq_len]
    losses = F.binary_cross_entropy_with_logits(
        selected_logits,  # [batch, seq_len]
        all_rewards,      # [batch, seq_len]
        reduction='none'  # Keep per-example losses
    )  # [batch, seq_len]

    # Apply temporal weights via broadcasting: [batch, seq_len] * [seq_len]
    # Broadcasting expands temporal_weights to [1, seq_len], then element-wise multiply
    weighted_losses = losses * temporal_weights.unsqueeze(0)  # [batch, seq_len]

    # Sum across both batch and time dimensions
    total_loss = weighted_losses.sum()  # [batch, seq_len] → scalar

    # Total weight = sum of temporal weights across time × batch size
    # This normalizes by the "effective number of weighted examples"
    total_weight = temporal_weights.sum() * batch_size  # scalar

    # Normalize by cumulative weight to get weighted average loss
    # Makes loss independent of batch size and sequence length
    loss = total_loss / total_weight

    return loss


# ============================================================================
# Training Functions
# ============================================================================

def train_head_batch(
    model: torch.nn.Module,
    sequences: list[dict[str, torch.Tensor]],
    head_type: str,
    config: InsulaConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Train a single batch for a specific head.

    Performs forward + backward pass. Gradients accumulate.
    Does NOT call optimizer.step() - that's done once after all batches.

    Args:
        model: DecisionTransformer model
        sequences: List of sequence dictionaries (all same length)
        head_type: Which head to train ("change", "completion", "gameover")
        config: Configuration dictionary
        device: torch.device

    Returns:
        loss: Scalar loss tensor for this batch
        metrics: Dictionary with loss and other stats
            - "loss": float
            - "head": str
            - "batch_size": int
            - "seq_length": int
    """
    # Convert sequences to tensors
    states_batch = torch.stack([seq["states"] for seq in sequences]).to(device)  # [B, seq_len+1, 64, 64]
    actions_batch = torch.stack([seq["actions"] for seq in sequences]).to(device)  # [B, seq_len]

    # Get rewards and temporal credit data for this head
    all_action_indices = torch.stack([seq["all_action_indices"] for seq in sequences]).to(device)  # [B, seq_len+1]

    if head_type == "change":
        all_rewards = torch.stack([seq["all_change_rewards"] for seq in sequences]).to(device)  # [B, seq_len+1]
        temporal_decay = model.change_decay  # Use model property (learned or fixed)
    elif head_type == "change_momentum":
        all_rewards = torch.stack([seq["all_change_momentum_rewards"] for seq in sequences]).to(device)  # [B, seq_len+1]
        temporal_decay = model.change_momentum_decay  # Use model property (learned or fixed)
    elif head_type == "completion":
        all_rewards = torch.stack([seq["all_completion_rewards"] for seq in sequences]).to(device)  # [B, seq_len+1]
        temporal_decay = model.completion_decay  # Use model property (learned or fixed)
    elif head_type == "gameover":
        all_rewards = torch.stack([seq["all_gameover_rewards"] for seq in sequences]).to(device)  # [B, seq_len+1]
        temporal_decay = model.gameover_decay  # Use model property (learned or fixed)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    # Forward pass (model in training mode)
    # Pass temporal_update flag to control whether to compute all timestep predictions or just final
    temporal_update = config.temporal_update
    logits_dict = model(states_batch, actions_batch, temporal_credit=temporal_update)

    # Extract logits for this head
    # Shape depends on temporal_update: [B, seq_len+1, 4102] if True, [B, 4102] if False
    head_logits = logits_dict[f"{head_type}_logits"]

    # Compute loss based on temporal_update config
    if temporal_update:
        # Use temporal replay weighting (all actions with temporal decay)
        loss = compute_head_loss_with_temporal_credit(logits=head_logits, all_action_indices=all_action_indices,
                                                      all_rewards=all_rewards, temporal_update_decay=temporal_decay)
    else:
        # Use only final action (no temporal credit)
        target_actions = all_action_indices[:, -1]  # [B] - final action
        rewards = all_rewards[:, -1]  # [B] - final reward

        loss = compute_head_loss(head_logits=head_logits, target_actions=target_actions, rewards=rewards)

    # Backward pass (gradients accumulate in model.parameters())
    loss.backward()

    metrics = {
        "loss": loss.item(),
        "head": head_type,
        "batch_size": len(sequences),
        "seq_length": len(sequences[0]["actions"]),
    }

    return loss, metrics


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experience_buffer: deque,
    config: InsulaConfig,
    device: torch.device,
    writer: SummaryWriter,
    logger: logging.Logger,
    game_id: str,
    action_counter: int,
) -> None:
    """Train Insula with unified context length and gradient accumulation.

    Note: Buffer size check performed by caller (play.py).
    This function assumes buffer is large enough for sequence creation.

    Args:
        model: DecisionTransformer model to train
        optimizer: Adam optimizer
        experience_buffer: Deque containing experience dictionaries
        config: Configuration dictionary with training parameters
        device: torch.device (cuda or cpu)
        writer: TensorBoard SummaryWriter for logging
        logger: Python logger for text logging
        game_id: Unique game identifier for logging
        action_counter: Current action count for logging x-axis

    Returns:
        None (modifies model in-place)
    """
    # === STEP 1: Create head-specific sequences ===
    # Always create change sequences (required)
    # Note: change_momentum head reuses these same sequences (both are exploration signals)
    change_sequences = create_change_seqs(experience_buffer, config)

    # Conditionally create completion sequences (optional)
    if config.use_completion_head:
        completion_sequences = create_completion_seqs(experience_buffer, config)
    else:
        completion_sequences = []

    # Conditionally create gameover sequences (optional)
    if config.use_gameover_head:
        gameover_sequences = create_gameover_seqs(experience_buffer, config)
    else:
        gameover_sequences = []

    # Skip training if no sequences
    if not change_sequences and not completion_sequences and not gameover_sequences:
        return None

    # === STEP 2: Direct batching (all sequences same length) ===
    # With unified context_len, all sequences have same length → no grouping needed
    # Simply count how many batches we'll process (1 per head if sequences exist)
    # Note: change and momentum share sequences, so count 1 or 2 batches if change_sequences exist (depending on use_change_momentum_head)
    num_batches = (1 if change_sequences else 0) + \
                  (1 if config.use_change_momentum_head and change_sequences else 0) + \
                  (1 if config.use_completion_head and completion_sequences else 0) + \
                  (1 if config.use_gameover_head and gameover_sequences else 0)

    # Log training info
    logger.info(
        f"🎯 Starting Insula training. Game: {game_id}. "
        f"{len(change_sequences)} change/momentum sequences (shared), "
        f"{len(completion_sequences)} completion, {len(gameover_sequences)} gameover "
        f"→ {num_batches} batches (unified context_len={config.context_len})"
    )

    # === STEP 3: Training loop with gradient accumulation ===
    model.train()

    for epoch in range(config.epochs_per_training):
        optimizer.zero_grad()  # Zero gradients at start of epoch

        accumulated_metrics = {
            "change_loss": 0.0,
            "change_momentum_loss": 0.0,
            "completion_loss": 0.0,
            "gameover_loss": 0.0,
            "total_loss": 0.0,
            "num_batches": 0,
        }

        # === STEP 4: Process change head (direct batching) ===
        if change_sequences:
            loss, metrics = train_head_batch(
                model, change_sequences, head_type="change", config=config, device=device
            )
            accumulated_metrics["change_loss"] += metrics["loss"]
            accumulated_metrics["num_batches"] += 1

        # === STEP 5: Process change_momentum head (if enabled, reuses change sequences) ===
        if config.use_change_momentum_head and change_sequences:
            loss, metrics = train_head_batch(
                model, change_sequences, head_type="change_momentum", config=config, device=device
            )
            accumulated_metrics["change_momentum_loss"] += metrics["loss"]
            accumulated_metrics["num_batches"] += 1

        # === STEP 6: Process completion head (if enabled, direct batching) ===
        if config.use_completion_head and completion_sequences:
            loss, metrics = train_head_batch(
                model, completion_sequences, head_type="completion", config=config, device=device
            )
            accumulated_metrics["completion_loss"] += metrics["loss"]
            accumulated_metrics["num_batches"] += 1

        # === STEP 6: Process gameover head (if enabled, direct batching) ===
        if config.use_gameover_head and gameover_sequences:
            loss, metrics = train_head_batch(
                model, gameover_sequences, head_type="gameover", config=config, device=device
            )
            accumulated_metrics["gameover_loss"] += metrics["loss"]
            accumulated_metrics["num_batches"] += 1

        # === STEP 7: Single optimizer step (accumulated gradients) ===
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )

        optimizer.step()

        # === STEP 8: Log metrics ===
        # Average losses across batches
        accumulated_metrics["total_loss"] = (
            accumulated_metrics["change_loss"] +
            accumulated_metrics["change_momentum_loss"] +
            accumulated_metrics["completion_loss"] +
            accumulated_metrics["gameover_loss"]
        ) / max(accumulated_metrics["num_batches"], 1)

        log_hierarchical_dt_metrics(writer, accumulated_metrics, action_counter, config)

        # Log decay rates (if using learned decay)
        if config.use_learned_decay:
            # Access decay rates via model properties (works for both learned and fixed)
            # Build decay message with only enabled heads
            decay_parts = [f"change={model.change_decay:.4f}"]
            if config.use_change_momentum_head:
                decay_parts.append(f"change_momentum={model.change_momentum_decay:.4f}")
            if config.use_completion_head:
                decay_parts.append(f"completion={model.completion_decay:.4f}")
            if config.use_gameover_head:
                decay_parts.append(f"gameover={model.gameover_decay:.4f}")

            logger.info(f"  Decay rates (epoch {epoch}): {', '.join(decay_parts)}")

            # Log to TensorBoard (only enabled heads)
            writer.add_scalar("InsulaAgent/decay/change", model.change_decay, action_counter)
            if config.use_change_momentum_head:
                writer.add_scalar("InsulaAgent/decay/change_momentum", model.change_momentum_decay, action_counter)
            if config.use_completion_head:
                writer.add_scalar("InsulaAgent/decay/completion", model.completion_decay, action_counter)
            if config.use_gameover_head:
                writer.add_scalar("InsulaAgent/decay/gameover", model.gameover_decay, action_counter)

    # Log completion
    logger.info(
        f"✅ Insula training completed. Game: {game_id}. "
        f"loss={accumulated_metrics['total_loss']:.4f}"
    )

    # Clean up GPU memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# Logging Functions
# ============================================================================

def log_hierarchical_dt_metrics(
    writer: SummaryWriter,
    metrics: dict[str, float],
    action_counter: int,
    config: InsulaConfig = None,
) -> None:
    """Log hierarchical training metrics to tensorboard.

    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary with:
            - "total_loss": float
            - "num_batches": int
            - "change_loss": float (optional)
            - "change_momentum_loss": float (optional)
            - "completion_loss": float (optional)
            - "gameover_loss": float (optional)
        action_counter: Current action count for x-axis
        config: Configuration dictionary (for checking enabled heads)

    Returns:
        None
    """
    # Log overall metrics
    writer.add_scalar("InsulaAgent/total_loss", metrics["total_loss"], action_counter)
    writer.add_scalar("InsulaAgent/num_batches", metrics["num_batches"], action_counter)

    # Log head-specific losses (always log change, conditionally log others)
    if metrics.get("change_loss", 0) > 0:
        writer.add_scalar("InsulaAgent/change_loss", metrics["change_loss"], action_counter)

    # Only log change_momentum loss if momentum head is enabled
    if config and config.use_change_momentum_head:
        if metrics.get("change_momentum_loss", 0) > 0:
            writer.add_scalar("InsulaAgent/change_momentum_loss", metrics["change_momentum_loss"], action_counter)

    # Only log completion loss if completion head is enabled
    if config and config.use_completion_head:
        if metrics.get("completion_loss", 0) > 0:
            writer.add_scalar("InsulaAgent/completion_loss", metrics["completion_loss"], action_counter)

    # Only log gameover loss if gameover head is enabled
    if config and config.use_gameover_head:
        if metrics.get("gameover_loss", 0) > 0:
            writer.add_scalar("InsulaAgent/gameover_loss", metrics["gameover_loss"], action_counter)
