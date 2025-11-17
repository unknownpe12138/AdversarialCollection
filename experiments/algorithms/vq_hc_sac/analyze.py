"""
VQ-HC-SAC结果分析脚本

提供run_analysis()函数供实验框架调用
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import pickle
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.environment import AdversarialCollectionEnv
from algorithms.vq_hc_sac import VQHCSACAgent


def plot_training_curves(history: dict, save_path: Path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VQ-HC-SAC Training Progress', fontsize=16)
    
    # 1. Episode Returns
    if history['episode_returns']:
        ax = axes[0, 0]
        returns = history['episode_returns']
        episodes = list(range(len(returns)))
        ax.plot(episodes, returns, alpha=0.3, label='Raw', color='blue')
        
        # 移动平均
        if len(returns) > 10:
            window = min(100, max(10, len(returns) // 10))
            moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax.plot(range(len(moving_avg)), moving_avg, linewidth=2, 
                   label=f'MA({window})', color='orange')
        
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
        ax.plot(episodes, lengths, alpha=0.3, color='green')
        
        if len(lengths) > 10:
            window = min(100, max(10, len(lengths) // 10))
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(len(moving_avg)), moving_avg, linewidth=2, color='darkgreen')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
    
    # 3. Evaluation Returns
    if history.get('eval_returns'):
        ax = axes[1, 0]
        eval_returns = history['eval_returns']
        ax.plot(eval_returns, marker='o', linewidth=2, color='red')
        ax.set_xlabel('Evaluation #')
        ax.set_ylabel('Mean Return')
        ax.set_title('Evaluation Returns')
        ax.grid(True, alpha=0.3)
    
    # 4. Training Losses
    if history.get('losses'):
        ax = axes[1, 1]
        losses = history['losses']
        
        # 提取各类loss
        critic_losses = [l.get('critic_loss') for l in losses if l.get('critic_loss') is not None]
        actor_losses = [l.get('actor_loss') for l in losses if l.get('actor_loss') is not None]
        
        if critic_losses:
            ax.plot(critic_losses, label='Critic Loss', alpha=0.7, color='blue')
        if actor_losses:
            ax.plot(actor_losses, label='Actor Loss', alpha=0.7, color='orange')
        
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        if critic_losses or actor_losses:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 训练曲线: {save_path}")


def print_training_summary(history: dict):
    """打印训练摘要"""
    print("\n" + "-"*70)
    print("训练统计摘要")
    print("-"*70)
    
    if history['episode_returns']:
        returns = history['episode_returns']
        print(f"\n回报统计:")
        print(f"  Episode数: {len(returns)}")
        print(f"  平均回报: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  最小/最大: {np.min(returns):.2f} / {np.max(returns):.2f}")
        
        # 最近episodes
        if len(returns) >= 100:
            recent = returns[-100:]
            print(f"  最近100个Episode: {np.mean(recent):.2f} ± {np.std(recent):.2f}")
        elif len(returns) >= 10:
            recent = returns[-10:]
            print(f"  最近10个Episode: {np.mean(recent):.2f} ± {np.std(recent):.2f}")
    
    if history['episode_lengths']:
        lengths = history['episode_lengths']
        print(f"\nEpisode长度:")
        print(f"  平均: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  最小/最大: {np.min(lengths)} / {np.max(lengths)}")
    
    if history.get('eval_returns'):
        eval_returns = history['eval_returns']
        print(f"\n评估回报:")
        print(f"  评估次数: {len(eval_returns)}")
        print(f"  最佳: {np.max(eval_returns):.2f}")
        print(f"  最终: {eval_returns[-1]:.2f}")


def test_model(checkpoint_path: Path, env_config: dict, n_episodes: int = 10):
    """测试训练好的模型"""
    print("\n" + "-"*70)
    print("测试模型性能")
    print("-"*70)
    
    # 创建环境
    env = AdversarialCollectionEnv(**env_config)
    
    # 加载模型
    try:
        agent = VQHCSACAgent.load(str(checkpoint_path), device='cpu')
        print(f"  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return None
    
    # 测试
    episode_returns = []
    episode_lengths = []
    utilities_collected = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0
        steps = 0
        
        # 初始效用
        initial_utility = sum(env.node_states[node][0] for node in env.graph.task_nodes)
        
        while steps < env.max_steps:
            # 转换观察
            if isinstance(obs, tuple):
                states = np.array([np.array(o) for o in obs], dtype=np.float32)
            else:
                states = obs
            
            # 选择动作
            actions = agent.select_actions(states, deterministic=True)
            actions_list = [int(a) for a in actions]
            
            # 环境步进
            obs, rewards, terminated, truncated, _ = env.step(actions_list)
            
            # 累积回报
            if isinstance(rewards, dict):
                episode_return += sum(rewards.values())
            else:
                episode_return += np.sum(rewards)
            
            steps += 1
            
            if terminated or truncated:
                break
        
        # 收集的效用
        remaining_utility = sum(env.node_states[node][0] for node in env.graph.task_nodes)
        collected = initial_utility - remaining_utility
        
        episode_returns.append(episode_return)
        episode_lengths.append(steps)
        utilities_collected.append(collected)
    
    # 统计
    print(f"\n测试结果 ({n_episodes} episodes):")
    print(f"  回报: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"  长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  效用收集: {np.mean(utilities_collected):.2f} ± {np.std(utilities_collected):.2f}")
    print(f"  收集率: {np.mean(utilities_collected)/initial_utility*100:.1f}%")
    
    return {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'utilities_collected': utilities_collected,
        'collection_rate': np.mean(utilities_collected) / initial_utility
    }


def run_analysis(exp_dir: Path):
    """
    VQ-HC-SAC分析入口函数
    
    此函数被实验框架调用，执行完整的分析流程
    
    Args:
        exp_dir: 实验目录路径
    """
    exp_dir = Path(exp_dir)
    
    print("\n" + "="*70)
    print("VQ-HC-SAC结果分析")
    print("="*70)
    print(f"实验目录: {exp_dir}\n")
    
    # 1. 加载训练历史
    history_path = exp_dir / 'data' / 'history.pkl'
    if history_path.exists():
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        print(f"✓ 训练历史已加载")
    else:
        print(f"⚠ 未找到训练历史: {history_path}")
        return
    
    # 2. 打印训练摘要
    print_training_summary(history)
    
    # 3. 绘制训练曲线
    figures_dir = exp_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print(f"\n生成图表...")
    plot_training_curves(history, figures_dir / 'training_curves.png')
    
    # 4. 测试模型
    checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = exp_dir / 'checkpoints' / 'final_model.pt'
    
    env_config_path = exp_dir / 'data' / 'env_config.pkl'
    
    if checkpoint_path.exists() and env_config_path.exists():
        with open(env_config_path, 'rb') as f:
            env_config = pickle.load(f)
        
        test_results = test_model(checkpoint_path, env_config, n_episodes=10)
        
        # 保存测试结果
        if test_results:
            test_results_path = exp_dir / 'data' / 'test_results.pkl'
            with open(test_results_path, 'wb') as f:
                pickle.dump(test_results, f)
            print(f"\n  ✓ 测试结果已保存: {test_results_path}")
    else:
        print(f"\n⚠ 跳过模型测试 (找不到checkpoint或环境配置)")
    
    print("\n" + "="*70)
    print("分析完成")
    print("="*70)


if __name__ == '__main__':
    print("VQ-HC-SAC分析脚本")
    print("请使用 run_experiment.py 来运行实验和分析")
