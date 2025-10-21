play:
	uv run ARC-AGI-3-Agents/main.py --agent=insulat

setup:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	cd ..

tensorboard:
	.venv/bin/tensorboard --logdir=insula_agent/logs

clean:
	rm -r ./insula_agent/logs/*

