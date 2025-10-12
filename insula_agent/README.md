# Decision Transformer Agent for ARC-AGI-3

A unified transformer-based agent for playing ARC-AGI-3 games using sequential decision-making with configurable learning strategies.

## Overview

The Decision Transformer (DT) agent is an end-to-end transformer model that directly predicts actions from state-action sequence histories. It inherits from the base `Agent` class (agents/agent.py:22) to integrate seamlessly with the ARC-AGI-3 game framework.

## Architecture

### Core Components

#### 1. **ViTStateEncoder** (dt_agent.py:107-266)
- **Purpose**: Encodes 64×64 grid states into vector representations using Vision Transformer with learned cell embeddings
- **Architecture**: Pure transformer with patch-based encoding and learned color representations
  - **Learned cell embeddings**: Each color value (0-15) → learned embedding vector (cell_embed_dim)
  - **Patch extraction**: 8×8 patches (creates 64 patches from 64×64 grid)
    - Each patch contains 64 integer cell values (8×8)
    - **Memory efficient**: 64 cells per patch vs 1024 values with one-hot encoding (16× reduction)
  - **Cell aggregation**: Mean pooling over 64 cell embeddings per patch
  - **Patch embedding**: Aggregated cell embeddings (cell_embed_dim) → embed_dim via linear projection
  - **2D positional embeddings**: Learnable 8×8 grid positions added to patch embeddings
  - **CLS token**: Prepended special token for global aggregation (optional: can use global pooling)
  - **Transformer encoder**: 4 layers, 8 attention heads (configurable)
  - **Global representation**: Extract CLS token or average pooling
  - **Final normalization**: LayerNorm for stable outputs
- **Key advantages**:
  - **Learned color representations**: Network learns semantic meaning of each color (0-15)
  - **Memory efficient**: 8× smaller input tensors vs one-hot encoding
  - **Standard transformer approach**: Similar to word embeddings in NLP transformers
  - **Global attention from layer 1**: Critical for non-local causality in ARC-AGI-3
  - **Architectural consistency**: Hierarchical transformers (ViT spatial + Transformer temporal)
  - **Learned spatial relationships**: Attention discovers which positions matter
  - **Computational efficiency**: Only 64 patches (vs 4096 cells)
- **Input**: `[batch, 64, 64]` integer grids with cell values 0-15
- **Output**: `[batch, embed_dim]` state representations

**Note**: Uses learned embeddings instead of one-hot encoding for better efficiency and alignment with transformer best practices.

#### 2. **ActionEmbedding** (dt_agent.py:103-118)
- **Purpose**: Embeds discrete actions into continuous space
- **Vocabulary**: 4101 actions
  - Indices 0-4: ACTION1-ACTION5 (discrete game actions)
  - Indices 5-4100: Coordinate-based actions (64�64 grid positions)
- **Output**: `[batch, seq_len, embed_dim]` action embeddings

#### 3. **DecisionTransformer** (dt_agent.py:121-234)
- **Purpose**: End-to-end transformer for action prediction
- **Architecture**:
  - **Positional Encoding**: Learnable position embeddings for temporal context
  - **Transformer Decoder**: GPT-style autoregressive transformer
    - Configurable layers (default: 4)
    - Multi-head attention (default: 8 heads)
    - Feed-forward dimension: 4� embedding dimension
  - **Dual-Head Output**:
    - **Action Head**: Predicts discrete actions (ACTION1-5) � `[batch, 5]`
    - **Coordinate Head**: Predicts spatial actions (64�64 grid) � `[batch, 4096]`
- **Input**: Interleaved state-action sequences
- **Output**: Combined action logits `[batch, 4101]`

#### 4. **PureDTAgent** (dt_agent.py:238-933)
- **Purpose**: Self-contained agent with full game loop integration
- **Responsibilities**:
  - Action selection via transformer predictions
  - Experience collection and buffer management
  - Periodic model training
  - Visualization and logging

## Learning Flow

### 1. Experience Collection (dt_agent.py:860-886)

```
For each game step:
1. Execute action in environment
2. Observe frame change (reward signal)
3. Store experience: (state, action_idx, reward)
   - state: [64, 64] integer grid with cell values 0-15
   - action_idx: unified index (0-4100)
   - reward: 1.0 if frame changed, 0.0 otherwise
4. Deduplicate experiences using MD5 hash
5. Add to circular buffer (max 50,000 experiences)
```

**Key Implementation**:
- Unique experience tracking prevents redundant training (dt_agent.py:365-372)
- Buffer cleared when agent advances to new level (dt_agent.py:625-630)

### 2. Sequence Building (dt_agent.py:436-480)

Training sequences are constructed from experience buffer:

```
Input: Buffer with N experiences
Output: Batch of sequences

For each training sample:
1. Sample random starting index i
2. Extract k consecutive experiences [i, i+k)
3. Build state sequence: [s�, s�, ..., s�] (k+1 states)
4. Build action sequence: [a�, a�, ..., a���] (k actions)
5. Target: action a��� and reward r���
```

**Context Length**: Configurable (default: 5 steps for CPU, 20 for GPU)

### 3. Forward Pass (dt_agent.py:163-234)

The transformer processes sequences autoregressively:

```
1. Build interleaved sequence:
   [s�, a�, s�, a�, ..., s���, a���, s�]

2. Encode states via StateEncoder
3. Embed actions via ActionEmbedding
4. Add positional encodings
5. Apply causal attention mask (prevent future leakage)
6. Transform sequence through decoder layers
7. Extract final representation (position -1)
8. Dual-head prediction:
   - Action logits: [batch, 5]
   - Coordinate logits: [batch, 4096]
9. Concatenate � [batch, 4101]
```

### 4. Loss Computation (dt_agent.py:482-576)

Four configurable loss strategies:

#### **Cross-Entropy Loss** (Dense Updates)
```python
loss = F.cross_entropy(action_logits, targets)
```
- Updates all action probabilities every training step
- Fast convergence but may overfit to wrong actions

#### **Selective Loss** (Sparse Updates)
```python
positive_mask = rewards > 0
loss = F.cross_entropy(action_logits[positive_mask], targets[positive_mask])
```
- Only updates on actions that caused frame changes
- Conservative but slower learning

#### **Hybrid Loss** (Confidence-Based)
```python
alpha = confidence_fraction  # High-confidence samples
loss = alpha * selective_loss + (1 - alpha) * ce_loss
```
- Interpolates between dense and sparse based on model confidence
- Balances exploration and exploitation

#### **Bandit Loss** (Exploration-Focused)
```python
# Binary cross-entropy on selected actions only
selected_logits = action_logits.gather(1, targets.unsqueeze(1))
main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

# Entropy regularization for exploration
loss = main_loss - action_coeff * action_entropy - coord_coeff * coord_entropy
```
- Uses binary cross-entropy on selected actions
- Encourages exploration through entropy bonuses

### 5. Training Loop (dt_agent.py:374-434)

```
Triggered every N actions (default: 5)

Prerequisites:
- Buffer size e min_buffer_size (default: 5)
- Sample rate � buffer_size e context_length + 1

Procedure:
1. Sample experiences and create sequences
2. For each epoch (configurable: 1-5):
   a. Forward pass
   b. Compute loss (based on loss_type)
   c. Backward pass
   d. Gradient clipping (norm: 1.0)
   e. Optimizer step (Adam)
   f. Log metrics to TensorBoard
3. Clear GPU cache
```

### 6. Action Selection (dt_agent.py:613-827)

```
At inference time:
1. Convert current frame to tensor
2. Build inference sequence from recent buffer
3. Get action logits from transformer
4. Sample action with temperature and masking:
   - Split into action logits [5] and coord logits [4096]
   - Apply sigmoid activation
   - Mask invalid actions based on available_actions
   - Normalize and sample from combined distribution
5. Execute selected action
6. Store for next experience
```

**Cold Start**: Random valid action if buffer is empty. When building sequences with no history, ACTION1 (index 0) is used as a dummy action to satisfy model input requirements.

**Action Masking**: Respects game constraints by masking unavailable actions (dt_agent.py:841-861)

## Key Features

### 1. **Vision Transformer State Encoding with Learned Cell Embeddings**
- **Learned color representations**: Each cell value (0-15) learns a semantic embedding vector
- **Memory efficient**: 8× smaller input tensors vs one-hot encoding (64×64 integers vs 16×64×64 floats)
- **Standard transformer approach**: Cell embeddings analogous to word embeddings in NLP
- **Hierarchical transformer architecture**: Pure transformers all the way (ViT for spatial + Transformer for temporal)
- **Global attention**: Non-local causality handled from layer 1 (critical for ARC-AGI-3 dynamics)
- **Learned spatial relationships**: Self-attention discovers which grid positions interact
- **Efficient**: Only 64 patches (8×8 grid) vs 4096 cells for attention computation
- **Flexible**: Configurable patch sizes, layers, heads, and cell embedding dimensions for different compute budgets

### 2. **Unified Action Space**
- Single model handles both discrete (ACTION1-5) and spatial (coordinates) actions
- Dual-head architecture maintains separate reasoning paths
- Combined sampling ensures fair probability distribution

### 3. **Configurable Loss Functions**
Four strategies with different exploration-exploitation trade-offs:
- `cross_entropy`: Dense, fast convergence
- `selective`: Sparse, conservative
- `hybrid`: Adaptive confidence-based
- `bandit`: Exploration-focused with entropy bonuses

### 4. **Experience Deduplication**
- MD5 hashing prevents redundant state-action pairs
- Improves training efficiency and diversity

### 5. **Level-Aware Reset**
- Automatically clears buffer and resets model when advancing to new level
- Prevents negative transfer between different game stages

### 6. **Integrated Visualization**
- TensorBoard logging for training metrics
- Action probability heatmaps for spatial reasoning
- Optional visualization saves every N steps

## Configuration

### Device-Specific Configs

**CPU Configuration** (config.py:65-71):
```python
embed_dim: 128
num_layers: 2
context_length: 10
max_training_experiences: 30
```

**GPU Configuration** (config.py:73-79):
```python
embed_dim: 512
num_layers: 6
context_length: 20
max_training_experiences: 100
```

### Loss-Specific Configs

**Cross-Entropy** (config.py:82-87):
```python
loss_type: 'cross_entropy'
learning_rate: 1e-4
epochs_per_training: 1
```

**Selective** (config.py:89-94):
```python
loss_type: 'selective'
learning_rate: 5e-4  # Higher for sparse gradients
epochs_per_training: 3
```

**Hybrid** (config.py:96-102):
```python
loss_type: 'hybrid'
selective_confidence_threshold: 0.8
learning_rate: 2e-4
epochs_per_training: 2
```

**Bandit** (config.py:104-112):
```python
loss_type: 'bandit'
learning_rate: 1e-4
action_entropy_coeff: 0.0001
coord_entropy_coeff: 0.00001
```

### Environment Variable Overrides

```bash
export PURE_DT_LEARNING_RATE=1e-3
export PURE_DT_CONTEXT_LENGTH=15
export PURE_DT_LOSS_TYPE=hybrid
export PURE_DT_TEMPERATURE=0.8
```

## Usage

### Basic Usage

```python
from insula_agent import PureDTAgent

# Initialize agent (inherits from agents.agent.Agent)
agent = PureDTAgent(
        card_id='example_card',
        game_id='example_game',
        agent_name='PureDT_v1',
        ROOT_URL='https://arc-agi-3.com',
        record=True,
        tags=['pure_dt', 'transformer']
)

# Run game loop (inherited from base Agent class)
agent.main()
```

### Custom Configuration

```python
from insula_agent import load_pure_dt_config, PureDTAgent

# Load custom config
config = load_pure_dt_config(device='cuda', loss_type='hybrid')

# Agent automatically uses loaded config
agent = PureDTAgent(...)
```

### Integration with ARC-AGI-3 Framework

The agent implements the required `Agent` interface (agents/agent.py:22):

```python
class PureDTAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        """Check if game is complete"""

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Select next action using transformer"""
```

## Performance Characteristics

### Memory Requirements
- **Model Size**: ~10-50M parameters (depending on config)
- **Experience Buffer**: ~50,000 experiences � 65KB H 3.2GB
- **Training Batch**: Configurable (30-100 sequences)

### Training Speed
- **CPU**: ~0.5-1 second per training cycle
- **GPU**: ~0.1-0.2 seconds per training cycle
- **Frequency**: Every 5 actions (configurable)

### Action Selection Speed
- **Inference**: ~10-50ms per action (GPU)
- **Cold Start**: Random action (negligible overhead)

## Comparison with Base Agent

| Feature | Base Agent (agent.py) | PureDTAgent |
|---------|----------------------|-------------|
| Action Selection | Abstract method | Transformer-based |
| Learning | Not specified | Online learning with experience buffer |
| State Representation | Raw frames | CNN-encoded embeddings |
| Temporal Reasoning | Per-step | Sequence-based (context window) |
| Training | Not specified | Configurable loss strategies |
| Exploration | Not specified | Temperature sampling + entropy bonuses |

## Files

- `dt_agent.py` - Main agent implementation (933 lines)
- `config.py` - Configuration management (234 lines)
- `example_usage.py` - Usage examples and demos
- `__init__.py` - Module exports
- `README.md` - This file

## References

**Base Agent Interface**: `agents/agent.py:22-212`
- Provides game loop (`main()`)
- Handles API communication (`do_action_request()`)
- Manages recording and cleanup

**Decision Transformer Concepts**:
- Sequences of (state, action) tuples
- Causal transformer architecture
- Autoregressive action prediction
- Trajectory-based learning

## License

Part of the ARC-AGI-3 agent framework.
