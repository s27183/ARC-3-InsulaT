setup:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	cd ..

play:
	uv run ARC-AGI-3-Agents/main.py --agent=insulat $(if ($tags),--tags="$(tags)")

tensorboard:
	.venv/bin/tensorboard --logdir=insula_agent/logs

clean:
	rm -r ./insula_agent/logs/*

