"""
VQ-HC-SAC 训练结果分析脚本

专注于分析训练好的模型和训练历史数据：
- 加载checkpoint和训练历史
- 可视化训练曲线（回报、损失、Episode长度）
- 打印训练统计摘要
- 测试模型性能（使用与训练相同的环境配置）
- 统计环境中效用收集情况

使用方法：
    1. 先运行 train_demo.py 完成训练
    2. 再运行此脚本进行分析: python examples/analyze_training.py
"""

import sys
import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import AdversarialCollectionEnv
from algorithms.vq_hc_sac import VQHCSACAgent


def plot_training_curves(history, save_dir='./figures'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VQ-HC-SAC Training Progress', fontsize=16)
    
    # 1. Episode Returns
    if history['episode_returns']:
        ax = axes[0, 0]
        returns = history['episode_returns']
        episodes = list(range(len(returns)))
        ax.plot(episodes, returns, alpha=0.3, label='Raw')
        
        # 移动平均
        if len(returns) > 10:
            window = min(100, len(returns) // 10)
            moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax.plot(range(len(moving_avg)), moving_avg, linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Episode Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    if history['episode_lengths']:
        ax = axes[0, 1]
        lengths = history['episode_lengths']
        episodes = list(range(len(lengths)))
        ax.plot(episodes, lengths, alpha=0.3)
        
        if len(lengths) > 10:
            window = min(100, len(lengths) // 10)
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(len(moving_avg)), moving_avg, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
    
    # 3. Evaluation Returns
    if history['eval_returns']:
        ax = axes[1, 0]
        eval_returns = history['eval_returns']
        ax.plot(eval_returns, marker='o', linewidth=2)
        ax.set_xlabel('Evaluation #')
        ax.set_ylabel('Mean Return')
        ax.set_title('Evaluation Returns')
        ax.grid(True, alpha=0.3)
    
    # 4. Losses
    if history['losses']:
        ax = axes[1, 1]
        losses = history['losses']
        
        # 提取各类loss
        critic_losses = [l.get('critic_loss', 0) for l in losses if 'critic_loss' in l]
        actor_losses = [l.get('actor_loss', 0) for l in losses if 'actor_loss' in l]
        
        if critic_losses:
            ax.plot(critic_losses, label='Critic Loss', alpha=0.7)
        if actor_losses:
            ax.plot(actor_losses, label='Actor Loss', alpha=0.7)
        
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        
        # 只在有数据时显示图例
        if critic_losses or actor_losses:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def print_training_summary(history):
    """打印训练摘要"""
    print("\n" + "="*70)
    print("训练摘要")
    print("="*70)
    
    if history['episode_returns']:
        returns = history['episode_returns']
        print(f"\n回报统计:")
        print(f"  总Episode数: {len(returns)}")
        print(f"  平均回报: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  最小回报: {np.min(returns):.2f}")
        print(f"  最大回报: {np.max(returns):.2f}")
        
        # 最后100个episode
        if len(returns) >= 100:
            recent = returns[-100:]
            print(f"\n最近100个Episode:")
            print(f"  平均回报: {np.mean(recent):.2f} ± {np.std(recent):.2f}")
    
    if history['episode_lengths']:
        lengths = history['episode_lengths']
        print(f"\nEpisode长度统计:")
        print(f"  平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  最小长度: {np.min(lengths)}")
        print(f"  最大长度: {np.max(lengths)}")
    
    if history['eval_returns']:
        eval_returns = history['eval_returns']
        print(f"\n评估回报:")
        print(f"  评估次数: {len(eval_returns)}")
        print(f"  最佳评估回报: {np.max(eval_returns):.2f}")
        print(f"  最终评估回报: {eval_returns[-1]:.2f}")
    
    if history['losses']:
        losses = history['losses']
        print(f"\nLoss统计:")
        print(f"  更新次数: {len(losses)}")
        
        # 提取最后的loss
        if losses:
            last_loss = losses[-1]
            print(f"  最终Critic Loss: {last_loss.get('critic_loss', 'N/A')}")
            print(f"  最终Actor Loss: {last_loss.get('actor_loss', 'N/A')}")
            if 'vq_total' in last_loss:
                print(f"  最终VQ Loss: {last_loss.get('vq_total', 'N/A')}")


def load_env_config(checkpoint_dir):
    """加载环境配置"""
    env_config_path = checkpoint_dir / 'env_config.pkl'
    
    if env_config_path.exists():
        try:
            with open(env_config_path, 'rb') as f:
                env_config = pickle.load(f)
            print(f"✓ 环境配置已加载: {env_config_path}")
            return env_config
        except Exception as e:
            print(f"⚠ 加载环境配置失败: {e}")
    else:
        print(f"⚠ 未找到环境配置文件，使用默认配置")
    
    # 默认配置（与train_demo.py一致）
    return {
        'n_agents': 3,
        'n_nodes': 6,
        'graph_type': 'custom',
        'edge_list': [(0, i) for i in range(1, 6)],
        'max_steps': 50,
        'use_pbrs': True,
        'utility_range': (5.0, 20.0),
        'risk_range': (0.1, 0.5)
    }


def test_model_performance(checkpoint_path, env_config, n_episodes=10):
    """测试训练好的模型并统计效用收集"""
    print("\n" + "="*70)
    print("测试训练好的模型")
    print("="*70)
    
    # 使用保存的环境配置创建环境
    env = AdversarialCollectionEnv(**env_config)
    
    # 加载模型
    try:
        agent = VQHCSACAgent.load(checkpoint_path, device='cpu')
        print(f"✓ 模型加载成功: {checkpoint_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试统计
    episode_returns = []
    episode_lengths = []
    total_utilities_collected = []
    
    print(f"\n运行 {n_episodes} 个测试Episode...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0
        steps = 0
        done = False
        
        # 记录初始效用总量
        initial_utility = sum(env.node_states[node][0] 
                            for node in env.graph.task_nodes)
        
        while not done and steps < 100:
            # 转换观察格式
            if isinstance(obs, tuple):
                states = np.array([np.array(o) for o in obs], dtype=np.float32)
            else:
                states = obs
            
            # 选择动作（确定性）
            actions = agent.select_actions(states, deterministic=True)
            actions_list = [int(a) for a in actions]
            
            # 环境步进
            obs, rewards, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated
            
            # 累积奖励
            if isinstance(rewards, dict):
                episode_return += sum(rewards.values())
            else:
                episode_return += np.sum(rewards)
            
            steps += 1
        
        # 计算收集的效用
        remaining_utility = sum(env.node_states[node][0] 
                               for node in env.graph.task_nodes)
        collected_utility = initial_utility - remaining_utility
        
        episode_returns.append(episode_return)
        episode_lengths.append(steps)
        total_utilities_collected.append(collected_utility)
        
        print(f"  Episode {ep+1}: 回报={episode_return:.2f}, "
              f"步数={steps}, 收集效用={collected_utility:.2f}/{initial_utility:.2f}")
    
    # 打印统计
    print("\n" + "-"*70)
    print("测试结果统计")
    print("-"*70)
    print(f"\n回报:")
    print(f"  平均: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"  最小: {np.min(episode_returns):.2f}")
    print(f"  最大: {np.max(episode_returns):.2f}")
    
    print(f"\nEpisode长度:")
    print(f"  平均: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    print(f"\n效用收集:")
    print(f"  平均收集: {np.mean(total_utilities_collected):.2f} ± "
          f"{np.std(total_utilities_collected):.2f}")
    print(f"  最佳收集: {np.max(total_utilities_collected):.2f}")
    print(f"  收集率: {np.mean(total_utilities_collected)/initial_utility*100:.1f}%")
    
    return {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'utilities_collected': total_utilities_collected
    }


def analyze_checkpoint(checkpoint_path):
    """分析checkpoint内容"""
    print("\n" + "="*70)
    print("Checkpoint信息")
    print("="*70)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"\n训练进度:")
        print(f"  总步数: {checkpoint.get('total_steps', 'N/A')}")
        print(f"  总Episode数: {checkpoint.get('episode_count', 'N/A')}")
        print(f"  最佳评估回报: {checkpoint.get('best_eval_return', 'N/A')}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n配置:")
            print(f"  角色数: {config.get('n_roles', 'N/A')}")
            print(f"  嵌入维度: {config.get('embedding_dim', 'N/A')}")
            print(f"  批量大小: {config.get('batch_size', 'N/A')}")
            print(f"  学习率 (Actor): {config.get('actor_lr', 'N/A')}")
            print(f"  学习率 (Critic): {config.get('critic_lr', 'N/A')}")
        
        print(f"\n网络组件:")
        for key in checkpoint.keys():
            if 'state_dict' in str(type(checkpoint[key])) or isinstance(checkpoint[key], dict):
                if key not in ['config']:
                    print(f"  ✓ {key}")
        
    except Exception as e:
        print(f"❌ 加载checkpoint失败: {e}")


def main():
    print("="*70)
    print("VQ-HC-SAC 训练结果分析")
    print("="*70)
    
    # 查找最新的checkpoint
    checkpoint_dir = Path('./checkpoints/demo')
    
    if not checkpoint_dir.exists():
        print(f"\n❌ 未找到checkpoint目录: {checkpoint_dir}")
        return
    
    # 优先使用quick_demo.pt，否则使用best_model.pt
    checkpoint_files = ['quick_demo.pt', 'best_model.pt']
    checkpoint_path = None
    
    for fname in checkpoint_files:
        path = checkpoint_dir / fname
        if path.exists():
            checkpoint_path = str(path)
            break
    
    if checkpoint_path is None:
        print(f"\n❌ 未找到checkpoint文件")
        return
    
    print(f"\n使用checkpoint: {checkpoint_path}")
    
    # 1. 分析checkpoint
    analyze_checkpoint(checkpoint_path)
    
    # 2. 加载环境配置
    env_config = load_env_config(checkpoint_dir)
    
    # 3. 加载并可视化训练历史
    history_path = checkpoint_dir / 'history.pkl'
    if history_path.exists():
        try:
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            print(f"\n✓ 训练历史已加载: {history_path}")
            
            # 打印训练摘要
            print_training_summary(history)
            
            # 绘制训练曲线
            plot_training_curves(history)
            
        except Exception as e:
            print(f"\n⚠ 加载训练历史失败: {e}")
    else:
        print(f"\n⚠ 未找到训练历史文件: {history_path}")
        print("提示: 运行train_demo.py时会自动保存训练历史")
    
    # 4. 测试模型性能
    test_results = test_model_performance(checkpoint_path, env_config, n_episodes=10)
    
    print("\n✓ 分析完成!")


if __name__ == '__main__':
    main()
