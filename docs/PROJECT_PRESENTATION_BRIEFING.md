# IslandTest 项目讲解稿

这份文档是为**明天项目讲解**准备的版本，目标不是单纯“介绍代码”，而是帮助你清楚地讲明白：

1. 这个项目到底要解决什么问题  
2. 系统整体是怎么工作的  
3. 最近这段时间你主要做了哪些工作  
4. 试错过程是什么、改进了什么、当前效果如何  
5. 下一步还准备继续优化什么

建议你讲的时候以“**系统目标 -> 模块结构 -> 最近优化 -> 当前结果 -> 下一步**”这个顺序展开，这样既专业，也不会显得自己对项目不熟。

---

## 1. 一句话概括项目

> 这是一个“**参数化地形生成 + 结构评价 + 表征学习 + 强化学习优化**”的完整系统。  
> 它不是直接用神经网络画地图，而是让模型学会**如何调整 PCG 参数**，从而生成结构更合理、可导航性更好、海岸更自然的岛屿高度图。

---

## 2. 先讲总流程

![项目总览](/E:/IslandTest/docs/presentation_assets/presentation_01_overview.png)

你可以按下面这段讲：

> 整个系统从参数化 PCG 开始。  
> 我们先输入一组地形参数 `theta`，PCG 生成岛屿高度图 `H`。  
> 然后这张高度图会进入两个分支：  
> 一个分支由结构评价器提取可解释指标 `f(H)`，比如连通性、可导航比例、海岸复杂度等；  
> 另一个分支由 Beta-VAE 编码成 latent 表征 `z`，用于描述地图的整体语义结构。  
> 最后把 `theta`、`z` 和结构指标拼成 RL 的状态，交给策略网络决定下一步该如何调整参数。

项目里对应的主流程是：

```text
theta -> PCG Generator -> heightmap H
H -> Structure Evaluator -> f(H)
H -> Beta-VAE Encoder -> z
[theta_norm, z_norm, metrics_norm] -> RL Agent
RL Agent -> action delta theta
delta theta -> update theta -> next map
```

---

## 3. 每一部分到底干什么

![模块分工](/E:/IslandTest/docs/presentation_assets/presentation_02_modules.png)

### 3.1 PCG 生成器

对应文件：[/E:/IslandTest/pcg_generator.py](/E:/IslandTest/pcg_generator.py)

作用：

- 用一组参数生成一张岛屿高度图
- 地图不是直接“画出来”的，而是通过噪声、扭曲、衰减等步骤逐像素算出来的

当前的 10 个参数：

| 参数 | 含义 | 当前基础范围 |
|---|---|---|
| `f` | 基础频率 | `1.0 ~ 100.0` |
| `A` | 振幅缩放 | `0.5 ~ 2.0` |
| `N_octaves` | fBM 层数 | `3 ~ 6` |
| `persistence` | 高频衰减系数 | `0.3 ~ 0.7` |
| `lacunarity` | 频率增长系数 | `1.5 ~ 2.5` |
| `warp_strength` | 坐标扭曲强度 | `0.0 ~ 1.0` |
| `warp_frequency` | 扭曲频率 | `1.0 ~ 10.0` |
| `falloff_radius` | 岛屿半径控制 | `0.30*map_size ~ 0.80*map_size` |
| `falloff_exponent` | 边缘衰减指数 | `1.0 ~ 4.0` |
| `seed` | 随机种子 | 整数实例控制 |

正式采样时常用 `island` profile，它比完全均匀采样更偏向“像岛的地图”。

### 3.2 结构评价器

对应文件：[/E:/IslandTest/structure_evaluator.py](/E:/IslandTest/structure_evaluator.py)

作用：

- 把高度图变成几个可解释的结构指标
- 这些指标既用于数据清洗，也用于奖励函数和 VAE 监督

当前主指标：

| 指标 | 解释 |
|---|---|
| `connectivity` | 最大连通陆地块占总陆地面积的比例 |
| `navigable_ratio` | 可通行陆地占总陆地面积的比例 |
| `coast_complexity` | 海岸线复杂程度 |
| `terrain_variance` | 岛屿高程起伏程度 |
| `path_reachability` | 可通行区域中路径能否走通 |
| `land_ratio` | 陆地面积占整张地图的比例 |

另外，当前最新版本里还把：

```text
component_count = 连通块数量
```

加入为 **VAE 专用的辅助监督目标**，因为它比单纯的 `connectivity ratio` 更敏感，更适合帮助 latent 学“裂成几块”的信息。

### 3.3 Beta-VAE

对应文件：[/E:/IslandTest/vae_model.py](/E:/IslandTest/vae_model.py)

作用：

- 把高度图压缩成 latent 向量 `z`
- 同时尽量保证：  
  1. 能重建原图  
  2. latent 能表达结构指标

当前 VAE 不是普通的“图像重建模型”，而是一个**结构感知表征模型**。

它有两条输出支路：

```text
heightmap -> Encoder -> latent z
z -> Decoder -> reconstruction
z -> Structure Predictor -> structure metrics
```

当前损失函数由 4 部分组成：

```text
Total Loss =
重建损失
+ beta * KL 损失
+ 结构监督损失
+ metric alignment loss
```

含义分别是：

- **重建损失**：保证图像重建质量
- **KL 损失**：保证 latent 空间规整
- **结构监督损失**：逼 latent 学会指标
- **metric alignment loss**：让 latent 空间几何关系和结构指标空间几何关系尽量一致

### 3.4 RL 部分

对应文件：

- [/E:/IslandTest/rl_environment.py](/E:/IslandTest/rl_environment.py)
- [/E:/IslandTest/map_scoring.py](/E:/IslandTest/map_scoring.py)
- [/E:/IslandTest/sac_agent.py](/E:/IslandTest/sac_agent.py)
- [/E:/IslandTest/ppo_baseline.py](/E:/IslandTest/ppo_baseline.py)

作用：

- 不是直接生成地图
- 而是学习**怎么改参数**

当前 RL 状态定义：

```text
s = [theta_norm, z_norm, metrics_norm]
```

动作定义：

```text
a = delta theta_norm
```

奖励由三部分组成：

- 结构质量
- 路径可达性
- 新颖性

其中 **新颖性不在 VAE 训练里算**，而是在 RL 阶段根据 latent 历史 buffer 计算，这样更合理。

---

## 4. 最近你的主要工作量：可以怎么讲

![试错过程](/E:/IslandTest/docs/presentation_assets/presentation_03_trials.png)

这一段建议你不要讲成“我瞎试了很多超参”，而要讲成**有目标、有层次的迭代优化过程**。

### 阶段 A：先把图像重建做好

问题：

- 早期重建图很容易变成“平均圆岛”
- 海岸线细节丢失严重

做法：

- 改了 VAE 的 decoder 细化策略
- 加了海岸/陆地重点重建权重
- 用更稳定的 deterministic reconstruction 去评估

结果：

- 重建图从“模板化”变成了更像真实岛屿
- 像素误差、陆地区域误差、海岸带误差都显著降低

### 阶段 B：把重心从 MSE 转向“结构可读性”

问题：

- 图像已经能重建得不错
- 但 latent 对结构指标的表达很弱

做法：

- 加入结构预测头
- 加入 metric alignment loss
- 对结构监督做加权设计

结果：

- latent 对 `navigable_ratio / coast_complexity / terrain_variance / land_ratio` 的可读性明显变强

### 阶段 C：扩大有效数据集

问题：

- 小数据集下，val/test 波动大
- 某些结构指标学不稳

做法：

- 把 clean 数据从几百提升到 1000+，再提升到 3000+

结果：

- 泛化稳定性明显变好
- train / val / test 之间差距明显缩小

### 阶段 D：专项优化 hardest metrics

问题：

- `connectivity`
- `path_reachability`

这两个指标最难学。

做法：

- 把它们的监督权重单独接入 Optuna
- 不再只调总的 `structure_loss_weight`

结果：

- `path_reachability` 已经明显提升
- `connectivity` 也改善了，但仍然最难

### 阶段 E：重新审视 connectivity 定义

问题：

- 当前 `connectivity = 最大连通块占比`
- 在 clean 数据里太容易接近 `1`
- 信息量不够，latent 很难学得稳

新的做法：

- 保留 `connectivity ratio` 作为评分/奖励主线
- 额外加入 `component_count` 作为 VAE 辅助监督

意义：

> 不是推翻现有指标，而是在不破坏主线的前提下，给 latent 一个更敏感、更容易学的连通性信号。

---

## 5. 当前效果：你明天最值得讲的结果

![当前结果](/E:/IslandTest/docs/presentation_assets/presentation_04_results.png)

当前最有代表性的正式实验是：

- `map_size = 64`
- `latent_dim = 64`
- clean 样本数：`3151`
- 划分：`train 2206 / val 473 / test 472`

### 5.1 图像重建效果

在 test split 上：

- `pixel_mae = 0.0081`
- `land_pixel_mae = 0.0221`
- `coast_band_mae = 0.0230`

这说明：

```text
重建质量已经很强，
而且海岸保真度也已经比较高。
```

### 5.2 latent 对结构指标的表达能力

当前 test 上的 `latent_predictive_r2`：

| 指标 | R² |
|---|---:|
| `navigable_ratio` | `0.932` |
| `coast_complexity` | `0.910` |
| `terrain_variance` | `0.918` |
| `land_ratio` | `0.955` |
| `path_reachability` | `0.316` |
| `connectivity` | `0.006` |

最合理的表述方式不是说“做得全都很好”，而是：

> 目前 latent 对大多数连续结构指标已经非常稳定，尤其是可导航性、海岸复杂度、地形起伏和陆地规模；  
> 路径可达性也已经从早期的失败项提升到了可用水平；  
> 剩下最难的是连通性，这也是当前最新工作的重点。

### 5.3 泛化情况

这一点要重点讲，因为它能体现实验的可信度。

当前 `train / val / test` 结果差距很小，说明：

```text
模型不是只记住训练集，
而是真的学到了比较稳定的地图结构表征。
```

---

## 6. 为什么 connectivity 现在还是最难

你可以直接这么讲：

> 当前 connectivity 的实现本身没有明显 bug，它的定义是“最大连通块占总陆地面积的比例”。  
> 这个定义用于 RL 奖励是合理的，因为它是一个平滑的软评分。  
> 但问题在于，在清洗后的单岛数据里，这个值太容易接近 1，分布太窄，导致它作为 VAE 的监督目标时信息量不够，所以 latent 对它的可读性一直不如其它指标。

所以当前采取的策略是：

1. RL 和评分主线继续保留 `connectivity ratio`
2. VAE 额外加入 `component_count`
3. 用更敏感的辅助信号去补 latent 对连通结构的表达能力

---

## 7. 你明天可以怎么讲“项目结构”

### 版本一：3 分钟讲法

> 这个项目本质上是一个参数化岛屿地图优化系统。  
> PCG 负责从参数生成岛屿高度图，结构评价器把地图转成可解释指标，Beta-VAE 把地图压缩成 latent 语义表征，RL 则根据参数、latent 和结构指标一起决定下一步应该怎么改参数。  
> 最近这段时间我的主要工作不再是单纯提升重建图像，而是把 VAE 从图像重建模型改造成结构感知表征模型。  
> 目前 4 个核心连续结构指标已经稳定学会，路径可达性也有明显提升，当前剩下最难的是连通性，因此最新版本又加入了 component_count 作为辅助监督，准备继续优化这一点。

### 版本二：5 到 8 分钟讲法

你可以按这个顺序展开：

1. 项目目标  
2. 总流程图  
3. PCG 参数和生成机制  
4. 结构评价器的 6 个指标  
5. VAE 怎么从“图像重建”走到“结构表征”  
6. 最近几轮试错过程  
7. 当前 3k clean 结果  
8. connectivity 为什么最难  
9. 当前最新改动和下一步计划

---

## 8. 你可以直接复述的“最近工作总结”

这一段你可以几乎原样讲：

> 最近我的工作重点主要有三类。  
> 第一类是把 VAE 的重建质量做扎实，尤其是海岸线和陆地区域的保真；  
> 第二类是把优化目标从单纯的像素误差，转向 latent 对结构指标的可读性，因此我加入了结构预测头、metric alignment loss 和更系统的 Optuna 搜索；  
> 第三类是把实验从小规模验证推进到正式规模评估，包括 clean 数据集扩充到 1000+ 和 3000+，并做 train/val/test 的正式划分。  
> 目前来看，大多数连续结构指标已经学得比较稳定，路径可达性也明显改善，剩下最难的是连通性，因此最新一轮又加入了 component_count 辅助监督，目的是进一步补足这一项。

---

## 9. 当前项目状态：你可以怎么定性

我建议你把当前项目定性为：

```text
主链路已经打通，
VAE 已经从“能重建图像”提升到“能表达结构语义”，
当前处于专项打磨 hardest metric（connectivity）的阶段。
```

这个说法既真实，也显得你对项目进度判断很清楚。

---

## 10. 明天如果别人追问，你可以怎么答

### Q1：为什么不用纯生成模型直接画地图？

答：

> 因为这个项目更强调“可控性”和“结构目标”。  
> 直接画图很难稳定满足连通性、路径可达性、海岸形态等要求，而参数化 PCG + RL 更适合做结构优化。

### Q2：为什么要加 VAE？

答：

> 因为单纯的结构指标虽然可解释，但表达能力有限。  
> VAE 提供了一个更紧凑的地图语义表征，既能服务 RL 状态，也能服务新颖性度量。

### Q3：为什么 connectivity 这么难？

答：

> 因为当前 `connectivity ratio` 在 clean 单岛数据里很容易接近 1，分布太窄，不像海岸复杂度或地形方差那样有足够的信息量。  
> 所以我们现在加了 `component_count` 作为辅助监督，专门强化这一项。

---

## 11. 最后一句收尾建议

你最后可以这么收：

> 目前这个项目已经从“能跑流程”走到了“能做正式实验、能解释指标、能针对性优化难点”的阶段。  
> 当前最主要的剩余工作，就是把 connectivity 的表征进一步补强，使整个 VAE 在结构层面更加完整。

