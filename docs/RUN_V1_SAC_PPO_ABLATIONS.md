# Restored-v1 final experiment suite

This suite uses the restored v1 reward/scoring baseline and keeps all outputs
under `experiments\v1_suite`. It is resumable: re-running the same command skips
completed runs and continues unfinished stages.

## Server smoke test

Run this first after pulling the GitHub branch on the server:

```powershell
cd E:\IslandTest_remote_work
python run_v1_experiment_suite.py --stage all --fast-dev --resume --output-root experiments\v1_suite_smoke
```

This is only a chain test. It uses tiny episode counts and tiny VAE assets, so
do not use these scores as formal results. A successful smoke run should produce
`suite_summary.csv`, `suite_aggregate.csv`, comparison PNGs, and per-run
`final_summary.json` files.

## Formal run

After the smoke test passes, start the full suite:

```powershell
cd E:\IslandTest_remote_work
python run_v1_experiment_suite.py --stage all --resume
```

The full suite runs these stages:

1. SAC screening: four 1500-episode SAC variants.
2. SAC long runs: automatically select the best two SAC variants and run 4000 episodes.
3. PPO ablations: latent state, latent dimension, and network depth across seeds 42/43/44.
4. VAE ablations: full v1, no structure loss, no metric alignment, and connectivity supervision.
5. Reward ablations: full reward, absolute score only, no best bonus, and no success bonus.
6. Baseline ablations: zero action, random action, CMA-ES, PPO, and SAC best if available.

If the archived v1 checkpoint directory exists, it is reused. On a fresh GitHub
checkout without archived results, the script rebuilds the latent-64 v1 assets
from `configs\optuna_best_latent64_3k_v3_voronoi_drop_connectivity.json`.

## Run one stage

```powershell
python run_v1_experiment_suite.py --stage sac-screen --resume
python run_v1_experiment_suite.py --stage sac-long --resume
python run_v1_experiment_suite.py --stage ppo-ablation --resume
python run_v1_experiment_suite.py --stage vae-ablation --resume
python run_v1_experiment_suite.py --stage reward-ablation --resume
python run_v1_experiment_suite.py --stage baseline-ablation --resume
```

Preview every command without starting training:

```powershell
python run_v1_experiment_suite.py --stage all --fast-dev --dry-run
```

## TensorBoard

Install only TensorBoard in the active environment if it is missing:

```powershell
python -m pip install tensorboard
```

Start TensorBoard while training is running:

```powershell
tensorboard --logdir E:\IslandTest_remote_work\experiments\v1_suite
```

Open `http://localhost:6006` in a browser. If TensorBoard is unavailable, the
training still writes scalar CSV files and the exporter generates PNGs from CSV.

## Outputs

Important files:

- `experiments\v1_suite\suite_summary.csv`: every individual run.
- `experiments\v1_suite\suite_aggregate.csv`: mean and standard deviation by variant.
- `experiments\v1_suite\suite_comparison_ppo.png`: PPO ablation comparison.
- `experiments\v1_suite\suite_comparison_sac.png`: SAC tuning comparison.
- `experiments\v1_suite\suite_comparison_vae.png`: VAE representation ablation.
- `experiments\v1_suite\suite_comparison_reward.png`: reward shaping ablation.
- `experiments\v1_suite\suite_comparison_baseline.png`: zero/random/CMA-ES/PPO/SAC baseline comparison.
- `<run>\tensorboard_plots\`: per-run overview and individual curves.

The archived v1 directory is read-only input and is never overwritten.
