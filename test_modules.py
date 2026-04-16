"""
快速测试脚本 - 验证所有模块是否正常工作
"""

import sys
import os

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        import numpy as np
        print("✅ numpy 导入成功")
    except ImportError as e:
        print(f"❌ numpy 导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✅ pytorch 导入成功 (版本: {torch.__version__})")
    except ImportError as e:
        print(f"❌ pytorch 导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib 导入成功")
    except ImportError as e:
        print(f"❌ matplotlib 导入失败: {e}")
        return False
    
    try:
        from scipy import ndimage
        print("✅ scipy 导入成功")
    except ImportError as e:
        print(f"❌ scipy 导入失败: {e}")
        return False
    
    print()
    return True


def test_pcg_generator():
    """测试PCG生成器"""
    print("=" * 60)
    print("测试2: PCG基座模块")
    print("=" * 60)
    
    try:
        from pcg_generator import PCGIslandGenerator
        
        generator = PCGIslandGenerator(map_size=64)
        
        params = {
            'f': 10,
            'A': 1.0,
            'N_octaves': 4,
            'persistence': 0.5,
            'lacunarity': 2.0,
            'seed': 42,
            'warp_strength': 0.5,
            'warp_frequency': 2,
            'falloff_radius': 32,
            'falloff_exponent': 2
        }
        
        heightmap = generator.generate_heightmap(params)
        
        assert heightmap.shape == (64, 64), f"形状错误: {heightmap.shape}"
        assert heightmap.min() >= 0.0, f"最小值错误: {heightmap.min()}"
        assert heightmap.max() <= 1.0, f"最大值错误: {heightmap.max()}"
        
        print(f"✅ 高度图生成成功")
        print(f"   形状: {heightmap.shape}")
        print(f"   范围: [{heightmap.min():.4f}, {heightmap.max():.4f}]")
        print(f"   均值: {heightmap.mean():.4f}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ PCG生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_structure_evaluator():
    """测试结构评估器"""
    print("=" * 60)
    print("测试3: 结构评估模块")
    print("=" * 60)
    
    try:
        from pcg_generator import PCGIslandGenerator
        from structure_evaluator import StructureEvaluator
        
        generator = PCGIslandGenerator(map_size=64)
        evaluator = StructureEvaluator(map_size=64)
        
        # 生成测试地图
        params = {
            'f': 10, 'A': 1.0, 'N_octaves': 4, 'persistence': 0.5,
            'lacunarity': 2.0, 'seed': 42, 'warp_strength': 0.5,
            'warp_frequency': 2, 'falloff_radius': 32, 'falloff_exponent': 2
        }
        heightmap = generator.generate_heightmap(params)
        
        # 评估
        metrics = evaluator.evaluate(heightmap)
        
        print("✅ 结构评估成功")
        print("   指标:")
        for key, value in metrics.items():
            print(f"     {key}: {value:.4f}")
        
        # 测试特征向量
        feature_vec = evaluator.get_feature_vector(heightmap)
        assert feature_vec.shape == (5,), f"特征向量形状错误: {feature_vec.shape}"
        print(f"   特征向量形状: {feature_vec.shape}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 结构评估器测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_vae_model():
    """测试VAE模型"""
    print("=" * 60)
    print("测试4: β-VAE模型")
    print("=" * 60)
    
    try:
        import torch
        from vae_model import BetaVAE
        
        # 创建模型
        vae = BetaVAE(map_size=64, latent_dim=32, beta=4.0)
        
        # 创建测试数据
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 64, 64)
        
        # 前向传播
        x_recon, mu, logvar = vae(test_input)
        
        assert x_recon.shape == (batch_size, 1, 64, 64), f"重建形状错误: {x_recon.shape}"
        assert mu.shape == (batch_size, 32), f"mu形状错误: {mu.shape}"
        assert logvar.shape == (batch_size, 32), f"logvar形状错误: {logvar.shape}"
        
        # 测试损失函数
        losses = vae.loss_function(x_recon, test_input, mu, logvar)
        
        print("✅ VAE模型测试成功")
        print(f"   输入形状: {test_input.shape}")
        print(f"   重建形状: {x_recon.shape}")
        print(f"   隐变量形状: {mu.shape}")
        print(f"   总损失: {losses['total_loss'].item():.4f}")
        print(f"   重建损失: {losses['recon_loss'].item():.4f}")
        print(f"   KL散度: {losses['kl_divergence'].item():.4f}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ VAE模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_rl_environment():
    """测试RL环境"""
    print("=" * 60)
    print("测试5: RL环境")
    print("=" * 60)
    
    try:
        from rl_environment import IslandGenerationEnv
        
        env = IslandGenerationEnv(map_size=64, max_steps=10)
        
        # 重置
        state, info = env.reset(seed=42)
        
        assert state.shape == (5,), f"状态形状错误: {state.shape}"
        
        print("✅ 环境重置成功")
        print(f"   状态形状: {state.shape}")
        print(f"   动作空间: {env.action_space}")
        print(f"   观测空间: {env.observation_space}")
        
        # 测试步进
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        assert next_state.shape == (5,), f"下一状态形状错误: {next_state.shape}"
        assert isinstance(reward, float), f"奖励类型错误: {type(reward)}"
        
        print("✅ 环境步进成功")
        print(f"   奖励: {reward:.4f}")
        print(f"   终止: {terminated}")
        print()
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ RL环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_sac_agent():
    """测试SAC智能体"""
    print("=" * 60)
    print("测试6: SAC智能体")
    print("=" * 60)
    
    try:
        import numpy as np
        from sac_agent import SACAgent, ReplayBuffer
        
        state_dim = 5
        action_dim = 9
        
        agent = SACAgent(state_dim, action_dim, hidden_dim=128)
        
        # 测试动作选择
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        
        assert action.shape == (action_dim,), f"动作形状错误: {action.shape}"
        
        print("✅ 动作选择成功")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   动作范围: [{action.min():.4f}, {action.max():.4f}]")
        
        # 测试更新
        replay_buffer = ReplayBuffer(capacity=1000)
        
        for _ in range(100):
            s = np.random.randn(state_dim)
            a = np.random.randn(action_dim) * 0.1
            r = np.random.randn()
            s_next = np.random.randn(state_dim)
            d = False
            replay_buffer.push(s, a, r, s_next, d)
        
        losses = agent.update(replay_buffer, batch_size=32)
        
        print("✅ 网络更新成功")
        print(f"   Q损失: {losses['q_loss']:.4f}")
        print(f"   策略损失: {losses['policy_loss']:.4f}")
        print(f"   Alpha: {losses['alpha']:.4f}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ SAC智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始模块测试")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("PCG基座", test_pcg_generator()))
    results.append(("结构评估", test_structure_evaluator()))
    results.append(("β-VAE模型", test_vae_model()))
    results.append(("RL环境", test_rl_environment()))
    results.append(("SAC智能体", test_sac_agent()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:15s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以运行Notebook了。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
