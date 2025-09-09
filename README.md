## Rota Generator RL

Train a rota-generation agent with reinforcement learning using the [[ART](https://github.com/openpipe/art)] framework and Qwen2.5-3B-Instruct, backed by a SkyPilot-managed GPU cluster. The agent learns to produce valid, constraint-satisfying staff rotas from natural-language scenarios.

Key pieces:
- Training loop: `src/rota_generator/train.py` (ART + SkyPilot backend)
- Rollout and self-judging: `src/rota_generator/rollout.py`
- Training inputs: `data/training_inputs.json` (JSON array of scenario strings)

## Setup

Requirements:
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management
- SkyPilot configured with a GPU provider (e.g., RunPod, AWS, GCP)

Install deps (dev group includes SkyPilot and tooling):

```bash
uv sync --dev
```

Create a `.env` in repo root (you can copy `.env.example`). Common variables:
- OPENROUTER_API_KEY (required) – used for data generation and judge models via OpenRouter
- WANDB_API_KEY (optional) – enables logging via Weights & Biases (see code calling `weave.init`)
- OPENPIPE_API_KEY (optional) – logs completions to OpenPipe if set
- For optional S3 backups/benchmarks: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BACKUP_BUCKET (see CONFIGURING_AWS.md)

Provisioning: follow SkyPilot docs for your provider. RunPod integration guide: https://docs.runpod.io/integrations/skypilot/

## Data: training inputs

`data/training_inputs.json` contains a JSON array of training scenarios (one string per entry). If you set `FORCE_REGENERATE_TRAINING_INPUTS=1`, the training script will synthesize inputs and overwrite this file. The file has been normalized so each scenario is one line-entry string.

## Train

```bash
uv run python src/rota_generator/train.py
```

What happens:
- SkyPilot cluster is created (H100 by default in code) and prepared
- Model registered with ART, training inputs loaded/generated (target ~150)
- Trajectories collected via `rollout.py` and judged by a separate model
- Model trained with ART; checkpoints optionally pushed to S3

Notes:
- You can force input regeneration by exporting `FORCE_REGENERATE_TRAINING_INPUTS=1`.
- `TARGET_TRAINING_INPUTS` is defined in `train.py` (edit there to change count).
- Ensure your OpenRouter API key has access to the models referenced in code (e.g., `deepseek/deepseek-r1-0528`, `moonshotai/kimi-k2`).

## Stopping the cluster

From CLI:

```bash
uv run sky down <cluster-name>
```

Or in code, call your backend teardown (see SkyPilot docs). Keeping clusters warm reduces spin-up times for subsequent runs.

## Troubleshooting

- Server idle/low GPU usage with only health checks in logs (SkyPilot): usually no requests are hitting the workload endpoint yet. Verify the caller URL/port/path, security group rules, and add logs at the start of your handler. Probe the service with a simple client.
- Missing packages: the code references `litellm` and `weave`. If you plan to use those paths, ensure they are installed in your environment.
- OpenRouter errors: confirm `OPENROUTER_API_KEY` is set and models used in code are accessible for your account.

## Benchmarks

The included `benchmarks/` artifacts are illustrative. They’re not wired to the rota task by default. If you export results to S3, configure AWS in `.env` and adapt the scripts accordingly.
