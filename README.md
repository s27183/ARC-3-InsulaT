# InsulaT

**A basic Insula-inspired non-LLM system learning to play the ARC-3 game efficiently**

InsulaT's goal is to create starting point for empirically exploring approaches to developing Insula-inspired systems that can learn to play the ARC-3 games efficiently.

**Key Features:**

  - **Insula-inspired multi-level integration** via Vision Transformer (ViT) and Decision Transformer (DT) for spatial-temporal processing
  - **Multi-timestep forward prediction** trains on all timesteps in sequence (past + present) for sample efficiency
  - **Hippocampal-inspired oversampling** with head-specific replay sizes prioritizing rare events (completion/gameover)
  - **Memory reconsolidation with reward revaluation** assigns trajectory-level rewards during replay without corrupting episodic memory
  - **Multiple prediction heads** for simultaneous learning of productivity (change), quality (momentum), goal completion, and safety (gameover avoidance)
  - **Hierarchical action sampling** combining discrete actions (ACTION1-7) and coordinate-based actions via multiplicative probability combination
  - **Importance-weighted replay** prioritizes critical events matching hippocampal memory consolidation
  - **Online supervised learning** generates labels on-the-fly from game outcomes without reward engineering
  - **Hierarchical temporal weighting** via head-specific decay rates creating multi-timescale learning (5/10/20 action focus windows)
  - **Attention as neural integration** implementing biological integration principles through transformer layers
  - **Configurable transfer learning** with optional model/optimizer reset on level completion for ablation studies

---

## Team

Advisor: [Dan V. Nicolau]()

Support: Dang Huu Tai, Ngan Dinh, Bang Dao, Steve Bickley, Aidan, David, [Panalogy Lab](), Insular AI

Author: [Son Tran]()




---

## Run The Agent

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.

1. Clone the arc-3-insula repo and enter the directory.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. Copy .env.example to .env

```bash
cp .env.example .env
```

3. Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it as an environment variable in your .env file.

```bash
export ARC_API_KEY="your_api_key_here"
```

4. Set up the environment

```bash
uv venv
uv sync
```
**Note:** If you encounter warning about incorrect `VIRTUAL_ENV`, make sure to run the following command before running `uv venv`:

```bash
unset VIRTUAL_ENV
```
5. Run the agent

```bash
uv run main.py --agent=insula
```

6. View tensorboard logs

```bash
tensorboard --logdir=insula_agent/logs
```
## Configurations

Various configuration otpions can be set in the `insula_agent/config.py` file

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
