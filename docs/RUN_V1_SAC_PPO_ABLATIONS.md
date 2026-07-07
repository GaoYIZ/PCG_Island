# Restored-v1 SAC tuning and PPO ablations

The experiment suite uses the restored `50da2c9` reward/scoring baseline and
reuses the trained v1 VAE assets whenever possible.

## Run everything

```powershell
cd E:\IslandTest_remote_work
python run_v1_experiment_suite.py --stage all --resume
```

The stages run in this order:

1. Four 1500-episode SAC screening runs.
2. Automatic selection of the best two SAC variants and two 4000-episode runs.
3. PPO latent-state, latent-capacity, and network-depth ablations for seeds 42/43/44.

Completed runs are skipped by default. Re-running the same command continues
the suite without overwriting finished experiments.

If the archived v1 checkpoint directory is present, it is reused directly. On
a fresh machine containing only the Git repository, the script automatically
rebuilds the latent-64 v1 assets from the tracked v1 Optuna configuration before
starting SAC or PPO.

## Run one stage

```powershell
python run_v1_experiment_suite.py --stage sac-screen --resume
python run_v1_experiment_suite.py --stage sac-long --resume
python run_v1_experiment_suite.py --stage ppo-ablation --resume
```

Preview every command without starting training:

```powershell
python run_v1_experiment_suite.py --stage all --dry-run
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

Open `http://localhost:6006` in a browser. Every variant and seed appears as a
separate run.

## Export figures and raw scalars

The suite exports figures automatically after a completed stage. It can also
be run manually:

```powershell
python export_tensorboard_plots.py --run-dir experiments\v1_suite --recursive
```

Important outputs:

- `experiments\v1_suite\suite_summary.csv`: every individual run.
- `experiments\v1_suite\suite_aggregate.csv`: mean and standard deviation by variant.
- `experiments\v1_suite\suite_comparison_ppo.png`: PPO ablation comparison.
- `experiments\v1_suite\suite_comparison_sac.png`: SAC tuning comparison.
- `<run>\tensorboard_plots\`: per-run overview and individual curves.

The archived v1 directory is read-only input and is never overwritten.
