# 基于强化学习与生成模型的室外岛屿地图自动生成

## 项目简介

本项目实现了基于β-VAE和SAC强化学习的岛屿地图自动生成系统，能够生成高质量、多样化且结构合理的岛屿地形。

## 系统架构

```
PCG基座 (Simplex噪声) 
    ↓
β-VAE表征学习 (隐空间提取)
    ↓
结构评估模块 (连通性、可导航性等)
    ↓
SAC强化学习 (参数优化)
    ↓
高质量岛屿地图
```

## 项目结构

```
IslandTest/
├── pcg_generator.py          # PCG基座模块
├── structure_evaluator.py    # 结构评估模块
├── vae_model.py              # β-VAE模型
├── sac_agent.py              # SAC强化学习智能体
├── rl_environment.py         # RL环境封装
├── demo_island_generation.ipynb  # Demo版Notebook（快速演示）
├── full_experiment.ipynb     # 完整版Notebook（精细训练）
├── requirements.txt          # 依赖包
└── README.md                 # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 方式1: Demo版本（推荐先运行）

Demo版本使用简化的Q-Learning算法，无需GPU即可快速体验完整流程。

```bash
jupyter notebook demo_island_generation.ipynb
```

**特点**：
- ⚡ 运行速度快（5-10分钟）
- 💻 无需GPU
- 🎯 展示完整流程
- 📊 基础可视化

### 方式2: 完整版本

完整版包含β-VAE和SAC算法，需要更多计算资源。

```bash
jupyter notebook full_experiment.ipynb
```

**特点**：
- 🔬 完整实验流程
- 🧠 β-VAE表征学习
- 🎮 SAC强化学习
- 📈 消融实验
- ⏱️ 运行时间较长（1-2小时，取决于硬件）

## 模块说明

### 1. PCG基座模块 (`pcg_generator.py`)

基于Simplex噪声的分数布朗运动（fBm）生成基础高度图。

**核心功能**：
- Simplex噪声生成
- fBm多八度叠加
- Domain Warping坐标扭曲
- Radial Falloff径向衰减

**可调参数**：
- `f`: 基础频率 (1-100)
- `A`: 振幅缩放 (0.5-2.0)
- `N_octaves`: 八度数 (1-8)
- `persistence`: 持久性 (0-1)
- `lacunarity`: 间隙度 (1.5-2.5)
- `warp_strength`: 扭曲强度 (0-1)
- `warp_frequency`: 扭曲频率 (1-10)
- `falloff_radius`: 衰减半径
- `falloff_exponent`: 衰减指数 (1-4)

### 2. 结构评估模块 (`structure_evaluator.py`)

提取高度图的物理拓扑特征。

**评估指标**：
- **连通性**: 单连通分量检测
- **可导航比例**: 坡度<30°的单元格占比
- **海岸复杂度**: 周长-面积比
- **地形方差**: 高程标准差
- **路径可达性**: BFS验证可达性

### 3. β-VAE模块 (`vae_model.py`)

将高维高度图映射为低维隐变量。

**网络结构**：
- 编码器: Conv2d → Flatten → Linear
- 隐空间: 重参数化技巧
- 解码器: Linear → ConvTranspose2d

**关键参数**：
- `latent_dim`: 隐空间维度 (默认32)
- `beta`: KL散度权重 (默认4.0)

### 4. SAC智能体 (`sac_agent.py`)

Soft Actor-Critic算法实现。

**组件**：
- QNetwork (Critic): 双Q网络
- PolicyNetwork (Actor): 高斯策略
- ReplayBuffer: 经验回放

**超参数**：
- `learning_rate`: 3e-4
- `gamma`: 0.99
- `tau`: 0.005 (软更新系数)
- `alpha`: 0.2 (熵温度)

### 5. RL环境 (`rl_environment.py`)

Gymnasium环境封装。

**状态空间**: [connectivity, navigable_ratio, coast_complexity, terrain_variance, path_reachability]

**动作空间**: Δθ (9维连续动作，调整PCG参数)

**奖励函数**: 
```
R = 0.5*R_struct + 0.3*R_reach + 0.2*R_novelty
```

## 实验流程

### Demo版流程

1. **PCG生成测试**: 生成单个岛屿并可视化
2. **批量生成**: 生成9个不同参数的岛屿
3. **结构评估**: 计算质量指标
4. **RL环境测试**: 随机动作测试
5. **简单训练**: Q-Learning训练20 episodes
6. **结果展示**: 最终岛屿可视化

### 完整版流程

1. **数据集生成**: 生成500+个随机岛屿
2. **β-VAE训练**: 训练隐空间表征模型
3. **VAE重建测试**: 验证重建质量
4. **SAC训练**: 训练100+ episodes
5. **训练曲线**: 可视化损失和奖励
6. **测试生成**: 生成20个测试岛屿
7. **统计分析**: 指标分布分析
8. **消融实验**: 对比Random PCG vs SAC
9. **最佳展示**: 详细可视化最佳岛屿
10. **模型保存**: 保存训练好的模型

## 输出文件

运行Notebook后会生成以下文件：

### Demo版输出
- `demo_pcg.png`: PCG生成示例
- `demo_multiple_islands.png`: 多个岛屿展示
- `demo_training_curve.png`: 训练曲线
- `demo_training_curve_smooth.png`: 平滑训练曲线
- `demo_final_result.png`: 最终结果

### 完整版输出
- `dataset_samples.png`: 数据集样本
- `vae_training_curve.png`: VAE训练曲线
- `vae_reconstruction.png`: VAE重建效果
- `sac_training_curves.png`: SAC训练曲线
- `metrics_distribution.png`: 指标分布
- `generated_islands_grid.png`: 生成岛屿网格
- `ablation_comparison.png`: 消融实验对比
- `final_best_island.png`: 最佳岛屿展示
- `saved_models/`: 保存的模型文件

## 性能建议

### CPU运行
- Demo版: ✅ 完全可行
- 完整版: ⚠️ 较慢，建议减少episodes和数据集大小

### GPU运行（推荐）
- Demo版: ⚡ 极快
- 完整版: ⚡⚡⚡ 推荐配置

**修改设备设置**：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 加速技巧

1. **减少数据集大小**: `n_samples = 500` → `200`
2. **减少训练episodes**: `n_episodes = 100` → `50`
3. **减小地图尺寸**: `map_size = 64` → `32`
4. **增大批大小**: `batch_size = 32` → `64` (如果有足够显存)

## 常见问题

### Q1: 运行时出现内存不足错误
**A**: 减少`n_samples`和`batch_size`，或使用更小的`map_size`。

### Q2: 训练速度太慢
**A**: 
- 使用GPU
- 减少episodes数量
- 增大批大小

### Q3: 生成的岛屿质量不佳
**A**: 
- 增加训练episodes
- 调整奖励函数权重
- 修改目标值（如navigable_ratio的目标从0.7调整）

### Q4: VAE重建效果差
**A**: 
- 增加VAE训练epochs
- 调整β值（较小的β保留更多信息）
- 增加隐空间维度

## 扩展方向

根据文档中的第8节，可以扩展：

1. **Diffusion替代PCG**: 使用扩散模型提升真实感
2. **GMVAE聚类**: 自动发现岛屿类型
3. **多模态输入**: 自然语言控制生成
4. **图结构扩展**: 室内场景生成
5. **多智能体环境**: Unity/Unreal集成

## 参考文献

详见原始文档《基于强化学习与生成模型的室外岛屿地图自动生成方法.md》中的参考文献汇总部分。

## 许可证

本项目仅供学术研究和教育用途。

## 联系方式

如有问题，请提Issue或联系项目维护者。
