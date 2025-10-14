# ARC-3 Insula Agent

**Insula-inspired agent combining visual and decision transformers:**

## Key Features:


  - Online supervised learning with ViT + Decision Transformer backbone and hierarchical prediction heads
  - Spatial-temporal integration encodes game state and action dynamics through multi-level attention
  - Configurable hierarchical temporal contexts match biological timescales: immediate feedback (e.g. 5 steps), goal sequences (e.g. 20 steps), and failure chains (e.g. 40 steps)
  - Configurable multi-head prediction learns multiple objectives simultaneously (productivity, goal completion, safety) over different temporal horizons
  - Configurable temporal hindsight tracing with learnable decay rates re-evaluates past decisions with present knowledge
  - Importance-weighted replay prioritizes critical events (1:5:10 ratio for
  change:completion:failure), mirroring hippocampal memory consolidation

---

## People

**Advisor**: [Dan V. Nicolau]()

**Supporters**: Dang Huu Tai, [Panalogy Lab]()

**Author**: [Son Tran]()



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
