# Agent Moosh

**A basic Insula-inspired non-LLM system learning to play the ARC-3 games efficiently**

Agent Moosh's goal is to create a starting point for empirically exploring approaches to developing Insula-inspired non-LLM systems that can learn to play the ARC-3 games efficiently.

Agent Moosh is currently ranked third on the [unverified ARC-AGI-3 leadboard](https://three.arcprize.org/leaderboard) (see user `Son Tran`) in one of our test run configurations. We look forward to having it verified on private games once the verification process is open again. It's likely that the leaderboard position will change as more competitive agents will join the competition while we continue to develop the system and perform further experiments. 

**Key Features:**

  - **Insula-inspired multi-level integration** via Vision Transformer (ViT) and Decision Transformer (DT) for spatial-temporal processing
  - **Multi-timestep forward prediction** trains on all timesteps in sequence (past + present) for sample efficiency
  - **Hippocampal-inspired oversampling** with head-specific replay sizes prioritizing rare events (completion/gameover)
  - **Memory reconsolidation with reward revaluation** assigns trajectory-level rewards during replay without corrupting episodic memory
  - **Multiple prediction heads** for simultaneous learning of productivity (change), quality (momentum), goal completion, and safety (gameover avoidance)
  - **Hierarchical action sampling** combining all available discrete actions and coordinate-based actions via multiplicative probability combination
  - **Importance-weighted replay** prioritizes critical events
  - **Online supervised learning** generates labels on-the-fly from game outcomes
  - **Hierarchical temporal weighting** via head-specific decay rates creating multi-timescale learning
  - **Attention as neural integration** implementing biological integration principles through transformer operations
  - **Configurable transfer learning** with optional model/optimizer reset on level completion for ablation studies
  - **Selected configurations** for ablation studies and empirical evaluation

---

## People

Advisor: [Dan V. Nicolau]()

Support: Dang Huu Tai, Ngan Dinh, [Bang Dao](https://www.linkedin.com/in/daotranbang?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app), [Steve Bickley](https://www.linkedin.com/in/steve-bickley/), [David Nasonov](https://www.linkedin.com/in/david-nasonov-323767250?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), [Aidan Saldanha](https://www.linkedin.com/in/aidandsaldanha?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), [Panalogy Lab](https://panalogy-lab.com), Insular AI

Author: [Son Tran](https://github.com/s27183)

---

## Run The Agent

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.

1. Clone the arc-3-insula repo and enter the directory.

```bash
git clone https://github.com/s27183/ARC-3-InsulaT
cd ARC-3-InsulaT
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
uv run main.py --agent=insulat
```

6. View tensorboard logs

```bash
tensorboard --logdir=insula_agent/logs
```
## Configurations

Various configuration otpions can be set in the `insula_agent/config.py` file

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
