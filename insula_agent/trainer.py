"""
InsulaAgent Training Module

This module contains all training-related functions for InsulaAgent,
including hierarchical context windows, temporal hindsight traces,
and importance-weighted replay.

Uses online supervised learning with self-generated labels from game outcomes.
"""

from typing import Any
import random
from collections import deque, defaultdict
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np


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
        "all_completion_rewards": all_completion_rewards,
        "all_gameover_rewards": all_gameover_rewards,
    }


def group_sequences_by_length(
    sequences: list[dict[str, torch.Tensor]]
) -> dict[int, list[dict[str, torch.Tensor]]]:
    """Group sequences by their actual length for efficient batching.

    Args:
        sequences: List of sequence dictionaries (variable lengths)

    Returns:
        Dictionary mapping length → list of sequences with that length
        Example: {80: [seq1, seq2, ...], 95: [seq3, seq4, ...], 100: [...]}
    """
    grouped = defaultdict(list)
    for seq in sequences:
        seq_len = len(seq["actions"])  # Number of past actions
        grouped[seq_len].append(seq)

    return dict(grouped)


def create_change_sequences(
    experience_buffer: deque,
    config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    """Create change sequences: random sampling, fixed length, no variation.

    Args:
        experience_buffer: Deque of experience dictionaries
        config: Configuration dictionary with:
            - "change_replay_size": int (default 16)
            - "change_context_len": int (default 15)

    Returns:
        List of sequence dictionaries, all with same length (change_context_len).
        Returns empty list if buffer too small.
    """
    replay_size = config["change_replay_size"]
    context_len = config["change_context_len"]
    buffer_len = len(experience_buffer)

    # Need at least context_len + 1 experiences (k past actions + 1 current state)
    if buffer_len < context_len + 1:
        return []

    sequences = []
    # Random sampling
    for _ in range(replay_size):
        max_start_idx = buffer_len - context_len - 1
        start_idx = random.randint(0, max_start_idx)
        sequence = extract_sequence(experience_buffer, start_idx, context_len)
        sequences.append(sequence)

    return sequences


def create_completion_sequences(
    experience_buffer: deque,
    config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    """Create completion sequences: outcome-anchored, variable length, multiple batches.

    Args:
        experience_buffer: Deque of experience dictionaries
        config: Configuration dictionary with:
            - "completion_replay_size": int (default 80)
            - "completion_context_len": int (default 100)
            - "replay_variation_min": float (default 0.8)
            - "replay_variation_max": float (default 1.0)

    Returns:
        List of sequence dictionaries with variable lengths (80%-100% of completion_context_len).
        Returns empty list if no completion events found.
    """
    replay_size = config["completion_replay_size"]
    target_context_len = config["completion_context_len"]
    variation_min = config["replay_variation_min"]
    variation_max = config["replay_variation_max"]

    # Find all completion events in buffer
    completion_idx = [
        i for i, exp in enumerate(experience_buffer)
        if exp["completion_reward"] == 1.0
    ]

    if not completion_idx:
        return []  # No completion events yet

    # Determine sampling strategy: WITH or WITHOUT replacement
    if len(completion_idx) < replay_size:
        # SPARSE: Sample WITH replacement (oversample)
        sampled_indices = np.random.choice(completion_idx, size=replay_size, replace=True)
    else:
        # ABUNDANT: Sample WITHOUT replacement (subsample)
        sampled_indices = np.random.choice(completion_idx, size=replay_size, replace=False)

    sequences = []
    for completion_idx in sampled_indices:
        # Apply variation: random length between 80% and 100% of target
        variation_factor = random.uniform(variation_min, variation_max)
        actual_context_len = int(target_context_len * variation_factor)
        actual_context_len = min(actual_context_len, completion_idx)  # Can't go before buffer start

        # Extract sequence ENDING at completion_idx
        start_idx = completion_idx - actual_context_len
        sequence = extract_sequence(experience_buffer, start_idx, actual_context_len)
        sequences.append(sequence)

    return sequences


def create_gameover_sequences(
    experience_buffer: deque,
    config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    """Create GAME_OVER sequences: outcome-anchored, variable length, multiple batches.

    Args:
        experience_buffer: Deque of experience dictionaries
        config: Configuration dictionary with:
            - "gameover_replay_size": int (default 160)
            - "gameover_context_len": int (default 160)
            - "replay_variation_min": float (default 0.8)
            - "replay_variation_max": float (default 1.0)

    Returns:
        List of sequence dictionaries with variable lengths (80%-100% of gameover_context_len).
        Returns empty list if no GAME_OVER events found.
    """
    replay_size = config["gameover_replay_size"]
    target_context_len = config["gameover_context_len"]
    variation_min = config["replay_variation_min"]
    variation_max = config["replay_variation_max"]

    # Find all GAME_OVER events in buffer (gameover_reward == 0.0 means GAME_OVER occurred)
    gameover_idx = [
        i for i, exp in enumerate(experience_buffer)
        if exp["gameover_reward"] == 0.0
    ]

    if not gameover_idx:
        return []  # No GAME_OVER events yet

    # Determine sampling strategy: WITH or WITHOUT replacement
    if len(gameover_idx) < replay_size:
        # SPARSE: Sample WITH replacement (oversample)
        sampled_indices = np.random.choice(gameover_idx, size=replay_size, replace=True)
    else:
        # ABUNDANT: Sample WITHOUT replacement (subsample)
        sampled_indices = np.random.choice(gameover_idx, size=replay_size, replace=False)

    sequences = []
    for gameover_idx in sampled_indices:
        # Apply variation: random length between 80% and 100% of target
        variation_factor = random.uniform(variation_min, variation_max)
        actual_context_len = int(target_context_len * variation_factor)
        actual_context_len = min(actual_context_len, gameover_idx)  # Can't go before buffer start

        # Extract sequence ENDING at gameover_idx
        start_idx = gameover_idx - actual_context_len
        sequence = extract_sequence(experience_buffer, start_idx, actual_context_len)
        sequences.append(sequence)

    return sequences


# ============================================================================
# Loss Computation Functions
# ============================================================================

def compute_head_loss(
    head_logits: torch.Tensor,
    target_actions: torch.Tensor,
    rewards: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    """Compute loss for a single head WITHOUT temporal credit (only final action).

    This function includes diversity regularization to encourage exploration.

    Args:
        head_logits: [batch, 4101] - Predicted logits for this head
        target_actions: [batch] - Final action indices (scalar per sequence)
        rewards: [batch] - Final rewards for this head (1.0 or 0.0)
        config: Configuration dictionary with:
            - "action_entropy_coeff": float (default 0.0001)
            - "coord_entropy_coeff": float (default 0.00001)

    Returns:
        loss: Scalar loss tensor with diversity regularization
    """
    # Gather only the logits for final actions
    selected_logits = head_logits.gather(
        dim=1, index=target_actions.unsqueeze(1)
    ).squeeze(1)  # [batch]

    # Binary cross-entropy with rewards as binary labels
    bce_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

    # Add action diversity regularization (encourage exploring more actions)
    probs = torch.sigmoid(head_logits)  # [batch, 4101]

    # Split into action and coordinate spaces
    action_probs = probs[:, :5]      # [batch, 5]
    coord_probs = probs[:, 5:]       # [batch, 4096]

    # Calculate diversity bonus (mean probability)
    # Higher mean = model considers more actions viable = more diversity
    action_diversity = action_probs.mean(dim=1).mean(dim=0)
    coord_diversity = coord_probs.mean(dim=1).mean(dim=0)

    # Diversity coefficients (configurable)
    action_coeff = config.get("action_entropy_coeff", 0.0001)
    coord_coeff = config.get("coord_entropy_coeff", 0.00001)

    # Total loss with diversity regularization
    # Subtracting encourages higher mean probability = more action diversity
    loss = bce_loss - action_coeff * action_diversity - coord_coeff * coord_diversity

    return loss


def compute_head_loss_with_temporal_credit(
    logits: torch.Tensor,
    all_action_indices: torch.Tensor,
    all_rewards: torch.Tensor,
    eligibility_decay: float,
    config: dict[str, Any],
) -> torch.Tensor:
    """Compute loss for a single head with temporal credit assignment.

    This is the main loss function used in hierarchical training.
    Each head (change/completion/gameover) has its own eligibility decay rate.
    Includes diversity regularization to encourage exploration.

    Args:
        logits: [batch, 4101] - Predicted logits for this head
        all_action_indices: [batch, seq_len] - ALL actions in sequence
        all_rewards: [batch, seq_len] - Rewards for this head (1.0 or 0.0)
        eligibility_decay: Head-specific decay rate (0.7, 0.8, or 0.9)
        config: Configuration dictionary with:
            - "action_entropy_coeff": float (default 0.0001)
            - "coord_entropy_coeff": float (default 0.00001)

    Returns:
        loss: Scalar loss tensor with diversity regularization
    """
    batch_size = logits.shape[0]
    seq_len = all_action_indices.shape[1]

    # Accumulate weighted losses
    total_loss = 0.0
    total_weight = 0.0

    # Loop through sequence - gather() for each action
    for t in range(seq_len):
        # Compute eligibility weight (exponential decay from end)
        steps_from_end = seq_len - 1 - t
        time_weight = eligibility_decay ** steps_from_end

        # Get action at timestep t
        action_t = all_action_indices[:, t]  # [batch]

        # Use gather() to select logits for THIS action
        selected_logits_t = logits.gather(
            dim=1, index=action_t.unsqueeze(1)
        ).squeeze(1)  # [batch]

        # Get rewards for this action
        reward_t = all_rewards[:, t]  # [batch]

        # Compute BCE loss for this action
        loss_t = F.binary_cross_entropy_with_logits(
            selected_logits_t,
            reward_t,
            reduction='none'
        )  # [batch]

        # Accumulate with temporal weight
        total_loss += (loss_t * time_weight).sum()
        total_weight += time_weight * batch_size

    # Normalize by cumulative weight
    bce_loss = total_loss / total_weight

    # Add action diversity regularization (same as compute_head_loss)
    probs = torch.sigmoid(logits)  # [batch, 4101]

    # Split into action and coordinate spaces
    action_probs = probs[:, :5]      # [batch, 5]
    coord_probs = probs[:, 5:]       # [batch, 4096]

    # Calculate diversity bonus (mean probability)
    # Higher mean = model considers more actions viable = more diversity
    action_diversity = action_probs.mean(dim=1).mean(dim=0)
    coord_diversity = coord_probs.mean(dim=1).mean(dim=0)

    # Diversity coefficients (configurable)
    action_coeff = config.get("action_entropy_coeff", 0.0001)
    coord_coeff = config.get("coord_entropy_coeff", 0.00001)

    # Total loss with diversity regularization
    # Subtracting encourages higher mean probability = more action diversity
    loss = bce_loss - action_coeff * action_diversity - coord_coeff * coord_diversity

    return loss


# ============================================================================
# Training Functions
# ============================================================================

def train_head_batch(
    model: torch.nn.Module,
    sequences: list[dict[str, torch.Tensor]],
    head_type: str,
    config: dict[str, Any],
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
        eligibility_decay = config["change_eligibility_decay"]
    elif head_type == "completion":
        all_rewards = torch.stack([seq["all_completion_rewards"] for seq in sequences]).to(device)  # [B, seq_len+1]
        eligibility_decay = config["completion_eligibility_decay"]
    elif head_type == "gameover":
        all_rewards = torch.stack([seq["all_gameover_rewards"] for seq in sequences]).to(device)  # [B, seq_len+1]
        eligibility_decay = config["gameover_eligibility_decay"]
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    # Forward pass
    logits_dict = model(states_batch, actions_batch)

    # Extract logits for this head
    head_logits = logits_dict[f"{head_type}_logits"]  # [B, 4101]

    # Compute loss based on temporal_credit config
    if config.get("temporal_credit", True):
        # Use temporal credit assignment (all actions with eligibility decay)
        loss = compute_head_loss_with_temporal_credit(
            logits=head_logits,
            all_action_indices=all_action_indices,
            all_rewards=all_rewards,
            eligibility_decay=eligibility_decay,
            config=config,
        )
    else:
        # Use only final action (no temporal credit)
        target_actions = all_action_indices[:, -1]  # [B] - final action
        rewards = all_rewards[:, -1]  # [B] - final reward

        loss = compute_head_loss(
            head_logits=head_logits,
            target_actions=target_actions,
            rewards=rewards,
            config=config,
        )

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
    config: dict[str, Any],
    device: torch.device,
    writer: SummaryWriter,
    logger: logging.Logger,
    game_id: str,
    action_counter: int,
) -> None:
    """Train Insula with hierarchical context windows and gradient accumulation.

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
    if len(experience_buffer) < config["min_buffer_size"]:
        return None

    # === STEP 1: Create head-specific sequences ===
    # Always create change sequences (required)
    change_sequences = create_change_sequences(experience_buffer, config)

    # Conditionally create completion sequences (optional)
    if config.get("use_completion_head", True):
        completion_sequences = create_completion_sequences(experience_buffer, config)
    else:
        completion_sequences = []

    # Conditionally create gameover sequences (optional)
    if config.get("use_gameover_head", True):
        gameover_sequences = create_gameover_sequences(experience_buffer, config)
    else:
        gameover_sequences = []

    # Skip training if no sequences
    if not change_sequences and not completion_sequences and not gameover_sequences:
        return None

    # === STEP 2: Group completion/gameover by length for batching ===
    completion_batches = group_sequences_by_length(completion_sequences) if completion_sequences else {}
    gameover_batches = group_sequences_by_length(gameover_sequences) if gameover_sequences else {}

    # Change sequences all same length → single batch
    change_batches = {config["change_context_len"]: change_sequences} if change_sequences else {}

    # Log training info
    total_sequences = len(change_sequences) + len(completion_sequences) + len(gameover_sequences)
    num_batches = len(change_batches) + len(completion_batches) + len(gameover_batches)
    logger.info(
        f"🎯 Starting Insula training. Game: {game_id}. "
        f"{total_sequences} sequences ({len(change_sequences)} change, "
        f"{len(completion_sequences)} completion, {len(gameover_sequences)} gameover) "
        f"→ {num_batches} batches"
    )

    # === STEP 3: Training loop with gradient accumulation ===
    model.train()

    for epoch in range(config["epochs_per_training"]):
        optimizer.zero_grad()  # Zero gradients at start of epoch

        accumulated_metrics = {
            "change_loss": 0.0,
            "completion_loss": 0.0,
            "gameover_loss": 0.0,
            "total_loss": 0.0,
            "num_batches": 0,
        }

        # === STEP 4: Process change head batches ===
        for length, sequences in change_batches.items():
            loss, metrics = train_head_batch(
                model, sequences, head_type="change", config=config, device=device
            )
            accumulated_metrics["change_loss"] += metrics["loss"]
            accumulated_metrics["num_batches"] += 1

        # === STEP 5: Process completion head batches (if head enabled) ===
        if config.get("use_completion_head", True):
            for length, sequences in completion_batches.items():
                loss, metrics = train_head_batch(
                    model, sequences, head_type="completion", config=config, device=device
                )
                accumulated_metrics["completion_loss"] += metrics["loss"]
                accumulated_metrics["num_batches"] += 1

        # === STEP 6: Process gameover head batches (if head enabled) ===
        if config.get("use_gameover_head", True):
            for length, sequences in gameover_batches.items():
                loss, metrics = train_head_batch(
                    model, sequences, head_type="gameover", config=config, device=device
                )
                accumulated_metrics["gameover_loss"] += metrics["loss"]
                accumulated_metrics["num_batches"] += 1

        # === STEP 7: Single optimizer step (accumulated gradients) ===
        if config["gradient_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["gradient_clip_norm"],
            )

        optimizer.step()

        # === STEP 8: Log metrics ===
        # Average losses across batches
        accumulated_metrics["total_loss"] = (
            accumulated_metrics["change_loss"] +
            accumulated_metrics["completion_loss"] +
            accumulated_metrics["gameover_loss"]
        ) / max(accumulated_metrics["num_batches"], 1)

        log_hierarchical_dt_metrics(writer, accumulated_metrics, action_counter, config)

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
    config: dict[str, Any] = None,
) -> None:
    """Log hierarchical training metrics to tensorboard.

    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary with:
            - "total_loss": float
            - "num_batches": int
            - "change_loss": float (optional)
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

    # Only log completion loss if completion head is enabled
    if config and config.get("use_completion_head", True):
        if metrics.get("completion_loss", 0) > 0:
            writer.add_scalar("InsulaAgent/completion_loss", metrics["completion_loss"], action_counter)

    # Only log gameover loss if gameover head is enabled
    if config and config.get("use_gameover_head", True):
        if metrics.get("gameover_loss", 0) > 0:
            writer.add_scalar("InsulaAgent/gameover_loss", metrics["gameover_loss"], action_counter)
