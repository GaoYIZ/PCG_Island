# 快速启动指南

## 🚀 5分钟快速开始

### 第一步：安装依赖（1分钟）

```bash
cd E:\IslandTest
pip install -r requirements.txt
```

### 第二步：运行测试验证（2分钟）

```bash
python test_modules.py
```

如果看到 "🎉 所有测试通过！"，说明环境配置正确。

### 第三步：启动Demo Notebook（2分钟）

```bash
jupyter notebook demo_island_generation.ipynb
```

在浏览器中打开Notebook，按顺序执行所有单元格即可。

---

## 📋 两个版本的选择

### Demo版 (`demo_island_generation.ipynb`)

**适合人群**：
- ✅ 第一次使用本项目
- ✅ 想快速了解整体流程
- ✅ CPU环境或计算资源有限
- ✅ 教学演示

**特点**：
- ⚡ 运行时间：5-10分钟
- 💻 无需GPU
- 🎯 简化版RL（Q-Learning）
- 📊 基础可视化

**内容**：
1. PCG生成测试
2. 结构评估
3. 多岛屿生成
4. RL环境测试
5. 简单训练（20 episodes）
6. 结果展示

---

### 完整版 (`full_experiment.ipynb`)

**适合人群**：
- ✅ 需要进行完整实验
- ✅ 有GPU或较强CPU
- ✅ 学术研究或项目应用
- ✅ 需要详细分析和对比

**特点**：
- 🔬 完整实验流程
- 🧠 β-VAE表征学习
- 🎮 SAC强化学习
- 📈 消融实验
- ⏱️ 运行时间：1-2小时（GPU）或更长（CPU）

**内容**：
1. 数据集生成（500个样本）
2. β-VAE训练（50 epochs）
3. VAE重建测试
4. SAC训练（100 episodes）
5. 训练曲线可视化
6. 测试生成（20个岛屿）
7. 统计分析
8. 消融实验
9. 最佳岛屿展示
10. 模型保存

---

## 💡 使用建议

### 首次使用推荐流程

```
1. 运行 test_modules.py 验证环境
   ↓
2. 运行 demo_island_generation.ipynb 了解流程
   ↓
3. 根据需要调整参数
   ↓
4. 运行 full_experiment.ipynb 进行完整实验
```

### 加速技巧

#### 如果你有GPU：
```python
# 在Notebook开头确认使用了GPU
import torch
print(torch.cuda.is_available())  # 应该输出 True
```

#### 如果你想快速测试完整版：
修改以下参数：
```python
n_samples = 100      # 原500
vae_epochs = 20      # 原50
n_episodes = 30      # 原100
map_size = 32        # 原64
```

#### 如果你只有CPU：
- Demo版：直接运行，无需修改
- 完整版：建议减少数据集和episodes数量

---

## 📊 预期输出

### Demo版输出文件
运行完成后会生成：
- `demo_pcg.png` - PCG示例
- `demo_multiple_islands.png` - 多岛屿展示
- `demo_training_curve.png` - 训练曲线
- `demo_final_result.png` - 最终结果

### 完整版输出文件
运行完成后会生成：
- `dataset_samples.png` - 数据集样本
- `vae_training_curve.png` - VAE训练曲线
- `vae_reconstruction.png` - VAE重建效果
- `sac_training_curves.png` - SAC训练曲线
- `metrics_distribution.png` - 指标分布
- `generated_islands_grid.png` - 生成岛屿网格
- `ablation_comparison.png` - 消融实验对比
- `final_best_island.png` - 最佳岛屿
- `saved_models/` - 保存的模型

---

## ❓ 常见问题

### Q1: Jupyter Notebook打不开
```bash
# 尝试指定端口
jupyter notebook --port 8888

# 或者使用JupyterLab
jupyter lab
```

### Q2: 运行时出现内存错误
- 减少`n_samples`（完整版）
- 减小`map_size`（64→32）
- 增大批大小`batch_size`（如果有足够显存）

### Q3: 训练太慢
- 使用GPU
- 减少episodes数量
- 使用更小的地图尺寸

### Q4: 想看中间结果
在Notebook中每个主要步骤后都有可视化代码，可以直接查看。

---

## 🎯 下一步

完成Demo后，你可以：

1. **调整参数**：修改PCG参数范围、奖励函数权重等
2. **扩展功能**：参考README中的扩展方向
3. **集成应用**：将生成的地图导出到游戏引擎
4. **学术研究**：运行消融实验，收集数据

---

## 📞 获取帮助

如果遇到问题：
1. 检查`test_modules.py`是否全部通过
2. 查看Notebook中的错误信息
3. 参考README.md的常见问题部分
4. 检查Python版本（建议3.7+）和依赖包版本

---

**祝使用愉快！** 🎉
