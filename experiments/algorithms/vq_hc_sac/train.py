"""
VQ-HC-SAC训练脚本

提供run_training()函数供实验框架调用
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import pickle

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.environment import AdversarialCollectionEnv
from algorithms.vq_hc_sac import VQHCSACTrainer, VQHCSACConfig


def create_environment(env_config: dict):
    """
    根据配置创建环境
    
    Args:
        env_config: 环境配置字典
    
    Returns:
        环境实例
    """
    print("\n" + "-"*70)
    print("创建环境...")
    print("-"*70)
    
    # 提取参数
    n_agents = env_config['n_agents']
    n_nodes = env_config['n_nodes']
    graph_type = env_config['graph_type']
    max_steps = env_config['max_steps']
    use_pbrs = env_config['use_pbrs']
    
    # 可选参数
    edge_list = env_config.get('edge_list', None)
    grid_size = env_config.get('grid_size', None)
    utility_range = tuple(env_config.get('utility_range', [5.0, 20.0]))
    risk_range = tuple(env_config.get('risk_range', [0.1, 0.5]))
    death_penalty = env_config.get('death_penalty', 10.0)
    gamma = env_config.get('gamma', 0.99)
    edge_probability = env_config.get('edge_probability', 0.3)
    graph_seed = env_config.get('graph_seed', None)
    
    # 创建环境
    env = AdversarialCollectionEnv(
        n_agents=n_agents,
        n_nodes=n_nodes,
        graph_type=graph_type,
        edge_list=edge_list,
        grid_size=grid_size,
        max_steps=max_steps,
        use_pbrs=use_pbrs,
        utility_range=utility_range,
        risk_range=risk_range,
        death_penalty=death_penalty,
        gamma=gamma,
        edge_probability=edge_probability,
        graph_seed=graph_seed
    )
    
    print(f"✓ 环境创建成功:")
    print(f"  智能体数量: {env.n_agents}")
    print(f"  节点数量: {env.graph.n_nodes}")
    print(f"  图类型: {graph_type}")
    if graph_type == 'random':
        print(f"  边概率: {edge_probability}")
        print(f"  图种子: {graph_seed if graph_seed is not None else 'None (每次随机)'}")
    print(f"  最大步数: {env.max_steps}")
    print(f"  PBRS: {env.use_pbrs}")
    print(f"  效用范围: {utility_range}")
    print(f"  风险范围: {risk_range}")
    
    return env


def create_trainer(env, save_dir: Path, device: str = 'cpu'):
    """
    创建VQ-HC-SAC训练器
    
    Args:
        env: 环境实例
        save_dir: 保存目录
        device: 训练设备
    
    Returns:
        训练器实例
    """
    print("\n" + "-"*70)
    print("创建VQ-HC-SAC训练器...")
    print("-"*70)
    
    # 配置（根据环境规模动态调整）
    n_agents = env.n_agents
    n_nodes = env.graph.n_nodes
    
    # 小规模: <10个智能体
    if n_agents < 10:
        config = VQHCSACConfig(
            n_roles=2,
            embedding_dim=32,
            encoder_hidden_dims=[64, 64],
            actor_hidden_dims=[128, 128],
            critic_hidden_dims=[128, 128],
            batch_size=32,
            buffer_size=50000,
            warmup_steps=500,
        )
    # 中等规模: 10-20个智能体
    elif n_agents < 20:
        config = VQHCSACConfig(
            n_roles=3,
            embedding_dim=64,
            encoder_hidden_dims=[128, 128],
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            batch_size=64,
            buffer_size=100000,
            warmup_steps=1000,
        )
    # 大规模: >=20个智能体
    else:
        config = VQHCSACConfig(
            n_roles=4,
            embedding_dim=128,
            encoder_hidden_dims=[256, 256],
            actor_hidden_dims=[512, 512],
            critic_hidden_dims=[512, 512],
            batch_size=128,
            buffer_size=200000,
            warmup_steps=2000,
        )
    
    # 创建训练器
    trainer = VQHCSACTrainer(
        env=env,
        config=config,
        save_dir=str(save_dir / "checkpoints"),
        device=device
    )
    
    print(f"✓ 训练器创建成功:")
    print(f"  设备: {trainer.device}")
    print(f"  角色数: {config.n_roles}")
    print(f"  嵌入维度: {config.embedding_dim}")
    print(f"  批量大小: {config.batch_size}")
    print(f"  预热步数: {config.warmup_steps}")
    
    return trainer, config


def run_training(env_config: dict,
                save_dir: Path,
                total_steps: int = 100000,
                eval_interval: int = 10000,
                log_interval: int = 1000,
                seed: int = 42,
                device: str = 'cpu'):
    """
    VQ-HC-SAC训练入口函数
    
    此函数被实验框架调用，执行完整的训练流程
    
    Args:
        env_config: 环境配置字典
        save_dir: 保存目录路径
        total_steps: 总训练步数
        eval_interval: 评估间隔
        log_interval: 日志间隔
        seed: 随机种子
        device: 训练设备 ('cpu' or 'cuda')
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print("\n" + "="*70)
    print("VQ-HC-SAC训练")
    print("="*70)
    
    # 1. 创建环境
    env = create_environment(env_config)
    
    # 2. 创建训练器
    trainer, config = create_trainer(env, save_dir, device)
    
    # 3. 执行训练
    print("\n" + "="*70)
    print(f"开始训练 (总步数: {total_steps:,})")
    print("="*70)
    
    try:
        history = trainer.train(
            total_steps=total_steps,
            eval_interval=eval_interval,
            log_interval=log_interval
        )
        
        print(f"\n✓ 训练完成!")
        print(f"  总步数: {trainer.total_steps:,}")
        print(f"  总Episode: {trainer.episode_count}")
        
        if trainer.episode_returns:
            recent_returns = trainer.episode_returns[-min(100, len(trainer.episode_returns)):]
            print(f"  平均回报 (最近): {np.mean(recent_returns):.2f} ± {np.std(recent_returns):.2f}")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 4. 保存训练历史
    history_path = save_dir / 'data' / 'history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\n✓ 训练历史已保存: {history_path}")
    
    # 5. 保存环境配置
    env_config_path = save_dir / 'data' / 'env_config.pkl'
    with open(env_config_path, 'wb') as f:
        pickle.dump({
            'n_agents': env.n_agents,
            'n_nodes': env.n_nodes,
            'graph_type': env_config['graph_type'],
            'edge_list': env_config.get('edge_list'),
            'max_steps': env.max_steps,
            'use_pbrs': env.use_pbrs,
            'utility_range': env.utility_range,
            'risk_range': env.risk_range
        }, f)
    print(f"✓ 环境配置已保存: {env_config_path}")
    
    # 6. 保存最终模型
    trainer.save_checkpoint('final_model.pt')
    print(f"✓ 最终模型已保存: {save_dir / 'checkpoints' / 'final_model.pt'}")
    
    print("\n" + "="*70)
    print("训练流程完成")
    print("="*70)


if __name__ == '__main__':
    # 测试脚本
    print("VQ-HC-SAC训练脚本")
    print("请使用 run_experiment.py 来运行实验")
