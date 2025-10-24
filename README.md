# AgentMoosh

**A basic Insula-inspired non-LLM system learning to play the ARC-AGI-3 games efficiently**

AgentMoosh's goal is to create a starting point for empirically exploring approaches to developing Insula-inspired non-LLM systems that can learn to play the ARC-3 games efficiently.

A preliminary test run put AgentMoosh at the third position on the [unverified ARC-AGI-3 leaderboard](https://three.arcprize.org/leaderboard). We look forward to having AgentMoosh verified on private games once the verification process is open again. It's likely that the leaderboard position will change quickly as more competitive agents join while we continue to develop the system and perform further experiments. 

**Key Features:**

  - Decision process with multi-level spatial-temporal integration 
  - Oversampling replay with temporal update weights
  - Trajectory reward revaluation during replay
  - Hierarchical action sampling covers both discrete and coordinate actions
  - Online supervised learning  
  - Extensive configurations for ablation studies and empirical evaluation

---

## People

Advisor: [Dan V. Nicolau](https://www.linkedin.com/in/dan-nicolau-384661219?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)

Support: Tram Dang, Ngan Dinh, [Bang Dao](https://www.linkedin.com/in/daotranbang?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app), [Steve Bickley](https://www.linkedin.com/in/steve-bickley/), [David Nasonov](https://www.linkedin.com/in/david-nasonov-323767250?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), [Aidan Saldanha](https://www.linkedin.com/in/aidandsaldanha?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), [Panalogy Lab](https://panalogy-lab.com), Insular AI

Author: [Son Tran](https://github.com/s27183)

---

## Instructions

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone the arc-3-insula repo and enter the directory.

```bash
git clone --recurse-submodules https://github.com/s27183/ARC-3-InsulaT
cd ARC-3-InsulaT
```

### 2. Setup the environment

```bash
cd ARC-AGI-3-Agents
cp .env.example .env
cd ..
make setup
```

Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it as an environment variable in the .env file.

### 3. Update submodule

Add the following code to `ARC-AGI-3-Agents/agents/__init__.py` (just before `load_dotenv()`):

```bash
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from insula_agent import *
```

**Pillow 12.0.0 compatibility issue in the submodule**

If you encounter this error when running the agent: `AttributeError: module 'PIL.ImageDraw' has no attribute 'Coords'`:

  1. Open `ARC-AGI-3-Agents/agents/templates/langgraph_thinking/vision.py`

  2. Find line 225 in the `add_highlight` function

  3. Replace `coords: ImageDraw.Coords` with `coords: tuple[tuple[int, int], tuple[int, int]]`

### 4. Run the agent

```bash
make play
# Or with tags
make play tags="test,configs"
```

### 5. View tensorboard logs

```bash
make tensorboard
```

## Configurations

Various configuration otpions can be set in the `insula_agent/config.py` file

---

## Project Structure

```
ARC-3-InsulaT/
├── ARC-AGI-3-Agents/          # Submodule: ARC-AGI-3 game interface and agent templates
├── insula_agent/              # InsulaT agent implementation
│   ├── __init__.py            # Module initialization and exports
│   ├── config.py              # Configuration settings and hyperparameters
│   ├── models.py              # Neural network architectures (ViT, Decision Transformer, prediction heads)
│   ├── play.py                # Main agent class (InsulaT) and game interaction logic
│   ├── trainer.py             # Training functions (experience replay, loss computation, gradient updates)
│   ├── utils.py               # Utility functions (logging setup, etc.)
│   ├── logs/                  # Training logs and TensorBoard outputs
├── recordings/                # Game recordings (trajectory data from agent runs)
├── Makefile                   # Build commands (play, setup, tensorboard, clean)
├── pyproject.toml             # Python project metadata and dependencies
├── uv.lock                    # Locked dependency versions
├── main.py                    # Entry point (redirects to submodule)
└── README.md                  # This file
```

**Key Modules:**

- **`models.py`**: Contains the core neural network components
  - `CellPositionEncoder`: Encodes cell positions (lines 7-68)
  - `InsularCellIntegration`: Integrates color and position features (lines 71-144)
  - `ViTStateEncoder`: Vision Transformer for spatial processing (lines 147-342)
  - `DecisionModel`: Main transformer-based decision model with multiple prediction heads (lines 361-838)

- **`play.py`**: Main agent implementation
  - `InsulaT`: Agent class implementing the full online supervised learning loop (lines 58-954)
  - Action selection with multiplicative head combination (lines 722-888)
  - Experience storage and replay buffer management (lines 549-633)

- **`trainer.py`**: Training pipeline
  - Sequence creation functions for different heads (lines 35-350)
  - Loss computation with temporal credit assignment (lines 357-451)
  - Main training loop with gradient accumulation (lines 540-698)

- **`config.py`**: Configuration dataclass with all hyperparameters and settings

