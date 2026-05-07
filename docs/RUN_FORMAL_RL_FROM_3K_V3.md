# RL 启动说明：基于 `3k_v3` 最优 VAE 参数

当前仓库内已固化一份最佳 VAE 参数配置：

- `/E:/IslandTest/configs/optuna_best_latent64_3k_v3.json`

该配置来自正式 `3k_v3` 实验，适用于：

- `map_size = 64`
- `latent_dim = 64`
- `formal-rl` 主流程

## Colab / Notebook 典型启动方式

```bash
python formal_experiment.py \
  --formal-rl \
  --optuna-best-trial /content/PCG_Island/configs/optuna_best_latent64_3k_v3.json \
  --output-dir formal_rl_from_3k_v3 \
  --map-size 64 \
  --latent-dim 64 \
  --dataset-samples 1000 \
  --min-clean-samples 3000 \
  --max-dataset-samples 12000 \
  --sampling-profile island \
  --vae-epochs 40 \
  --batch-size 32 \
  --vae-train-ratio 0.70 \
  --vae-val-ratio 0.15 \
  --ppo-episodes 60 \
  --sac-episodes 60 \
  --ppo-max-steps 30 \
  --eval-islands 24 \
  --seed 42
```

如果想跑更正式一点的版本，可以把：

- `--ppo-episodes 60` 提高到 `100`
- `--sac-episodes 60` 提高到 `100`
- `--eval-islands 24` 提高到 `32`

## 输出重点

输出目录中应重点查看：

- `final_summary.json`
- `ppo_training_curve.png`
- `sac_training_curve.png`
- `zero_generated_islands.png`
- `random_generated_islands.png`
- `ppo_generated_islands.png`
- `sac_generated_islands.png`

以及 VAE 固定表征部分：

- `train_split/vae_reconstruction.png`
- `test_split/vae_latent_predictiveness.png`
