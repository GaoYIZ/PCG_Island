# Google Colab 正式实验运行说明

## 推荐运行方式

当前项目已经补了正式实验脚本，建议不要只跑旧版 notebook。

在 Google Colab 中建议直接运行：

```bash
python formal_experiment.py
```

这样会按下面的顺序自动执行：

1. 数据集构建
2. 数据清洗与统计评估
3. VAE 训练
4. latent 提取与归一化拟合
5. PPO 正式训练
6. 最终评估与中文结果输出
7. 可选 SAC 补充训练

## Colab 操作步骤

### 1. 新建 Colab Notebook

建议运行时选择：

- `运行时 -> 更改运行时类型 -> T4 GPU`

### 2. 第一格：克隆项目

```bash
!git clone -b codex/normalization-reward-pipeline https://github.com/GaoYIZ/PCG_Island.git
%cd PCG_Island
```

### 3. 第二格：安装依赖

```bash
!pip install -q -r requirements.txt
```

### 4. 第三格：先跑测试

```bash
!python -m unittest test_modules.py
```

如果看到 `OK`，说明核心模块可以运行。

### 5. 第四格：正式实验，小规模先跑一版

```bash
!python formal_experiment.py \
  --output-dir colab_formal_outputs \
  --dataset-samples 120 \
  --vae-epochs 10 \
  --latent-dim 16 \
  --batch-size 16 \
  --ppo-episodes 20 \
  --eval-islands 8 \
  --sac-episodes 0
```

这版适合先验证流程和拿第一轮结果。

### 6. 第五格：如果小规模没问题，再跑标准版

```bash
!python formal_experiment.py \
  --output-dir colab_formal_outputs_full \
  --dataset-samples 300 \
  --vae-epochs 20 \
  --latent-dim 16 \
  --batch-size 32 \
  --ppo-episodes 60 \
  --eval-islands 12 \
  --sac-episodes 0
```

### 7. 第六格：如果你还想补 SAC 对比

```bash
!python formal_experiment.py \
  --output-dir colab_formal_outputs_with_sac \
  --dataset-samples 300 \
  --vae-epochs 20 \
  --latent-dim 16 \
  --batch-size 32 \
  --ppo-episodes 60 \
  --eval-islands 12 \
  --sac-episodes 40
```

## 结果会输出到哪里

实验结果默认保存在你指定的 `output-dir` 目录，比如：

```text
colab_formal_outputs/
```

里面主要有：

- `dataset_summary_raw.json`
- `dataset_summary_clean.json`
- `dataset_samples.png`
- `vae_training_curve.png`
- `vae_reconstruction.png`
- `ppo_training_curve.png`
- `ppo_evaluation_summary.json`
- `final_summary.json`

如果启用了 SAC，还会多出：

- `sac_training_curve.png`
- `sac_evaluation_summary.json`

## 你应该优先看的结果

先重点看：

1. `dataset_summary_clean.json`
2. `vae_training_curve.png`
3. `vae_reconstruction.png`
4. `ppo_training_curve.png`
5. `ppo_evaluation_summary.json`
6. `final_summary.json`

## 当前最推荐的汇报版本

如果你只是先给师兄汇报第一轮正式结果，建议先跑：

```bash
!python formal_experiment.py \
  --output-dir colab_formal_outputs \
  --dataset-samples 120 \
  --vae-epochs 10 \
  --latent-dim 16 \
  --batch-size 16 \
  --ppo-episodes 20 \
  --eval-islands 8 \
  --sac-episodes 0
```

这版最稳，成本也最低。
