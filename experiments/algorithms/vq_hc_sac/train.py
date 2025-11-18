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
import yaml

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


def load_config_from_yaml(config_path: str = None):
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径，默认使用config_template.yaml
    
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认使用config_template.yaml
        config_path = Path(__file__).parent / "config_template.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️  配置文件不存在: {config_path}")
        print("   将使用默认配置")
        return None
    
    print(f"📋 加载配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def create_trainer(env, save_dir: Path, device: str = 'cpu', config_path: str = None):
    """
    创建VQ-HC-SAC训练器
    
    Args:
        env: 环境实例
        save_dir: 保存目录
        device: 训练设备
        config_path: 配置文件路径（可选）
    
    Returns:
        训练器实例
    """
    print("\n" + "-"*70)
    print("创建VQ-HC-SAC训练器...")
    print("-"*70)
    
    # 尝试从配置文件加载
    yaml_config = load_config_from_yaml(config_path)
    
    if yaml_config is not None:
        # 使用配置文件的设置
        print("✅ 使用配置文件中的设置")
        
        # 从YAML配置创建VQHCSACConfig
        config = VQHCSACConfig(
            # 角色配置
            n_roles=yaml_config['roles']['n_roles'],
            embedding_dim=yaml_config['roles']['embedding_dim'],
            
            # 网络架构
            encoder_hidden_dims=yaml_config['networks']['encoder']['hidden_dims'],
            actor_hidden_dims=yaml_config['networks']['actor']['hidden_dims'],
            critic_hidden_dims=yaml_config['networks']['critic']['hidden_dims'],
            
            # 训练参数
            batch_size=yaml_config['training']['batch_size'],
            buffer_size=yaml_config['training']['buffer_size'],
            warmup_steps=yaml_config['training']['warmup_steps'],
            gamma=yaml_config['training']['gamma'],
            tau=yaml_config['training']['tau'],
            
            # 学习率
            actor_lr=yaml_config['learning_rates']['actor'],
            critic_lr=yaml_config['learning_rates']['critic'],
            encoder_lr=yaml_config['learning_rates']['encoder'],
            alpha_lr=yaml_config['learning_rates']['alpha'],
            
            # VQ模块
            vq_beta=yaml_config['vq']['beta'],
            use_ema_codebook=yaml_config['vq']['use_ema'],
            
            # 更新策略
            update_frequency=yaml_config['updates']['frequency'],
            updates_per_step=yaml_config['updates']['updates_per_step'],
            clip_grad_norm=yaml_config['updates'].get('clip_grad_norm', 1.0),
        )
        
    else:
        # 使用原来的动态配置逻辑
        print("⚠️  使用默认配置（根据环境规模动态调整）")
        n_agents = env.n_agents
        n_nodes = env.graph.n_nodes
        
        # 小规模: <10个智能体
        if n_agents < 10:
            # 小规模问题建议使用CPU（GPU利用率低）
            # 如果非要用GPU，需要更大的batch size
            batch_size = 128 if (device == 'cuda' and n_agents >= 5) else 32
            config = VQHCSACConfig(
                n_roles=2,
                embedding_dim=32,
                encoder_hidden_dims=[64, 64],
                actor_hidden_dims=[128, 128],
                critic_hidden_dims=[128, 128],
                batch_size=batch_size,
                buffer_size=50000,
                warmup_steps=500,
            )
        # 中等规模: 10-20个智能体
        elif n_agents < 20:
            batch_size = 256 if device == 'cuda' else 64
            config = VQHCSACConfig(
                n_roles=3,
                embedding_dim=64,
                encoder_hidden_dims=[128, 128],
                actor_hidden_dims=[256, 256],
                critic_hidden_dims=[256, 256],
                batch_size=batch_size,
                buffer_size=100000,
                warmup_steps=1000,
            )
        # 大规模: >=20个智能体
        else:
            batch_size = 512 if device == 'cuda' else 128
            config = VQHCSACConfig(
                n_roles=4,
                embedding_dim=128,
                encoder_hidden_dims=[256, 256],
                actor_hidden_dims=[512, 512],
                critic_hidden_dims=[512, 512],
                batch_size=batch_size,
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
                device: str = 'cpu',
                algorithm_config: dict = None):
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
        algorithm_config: 算法配置字典（可选，来自统一配置文件）
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
    # 配置优先级：
    # 1. algorithm_config参数（来自run_experiment.py的统一配置文件）
    # 2. save_dir中的algorithm_config.yaml（如果存在）
    # 3. 算法目录中的config_template.yaml
    # 4. 如果都不存在，使用默认配置
    
    config_path = None
    
    # 如果有传入的算法配置，优先使用
    if algorithm_config:
        # 将算法配置保存为临时文件供create_trainer读取
        temp_config = save_dir / "algorithm_config.yaml"
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(algorithm_config, f, default_flow_style=False, allow_unicode=True)
        config_path = str(temp_config)
        print(f"📋 使用传入的算法配置")
    else:
        # 检查save_dir中是否有配置文件
        custom_config = save_dir / "algorithm_config.yaml"
        if custom_config.exists():
            config_path = str(custom_config)
            print(f"📋 找到自定义配置文件: {custom_config}")
        else:
            # 使用默认模板
            default_config = Path(__file__).parent / "config_template.yaml"
            if default_config.exists():
                config_path = str(default_config)
                print(f"📋 使用默认配置模板: {default_config}")
    
    trainer, config = create_trainer(env, save_dir, device, config_path)
    
    # 保存实际使用的配置
    actual_config_path = save_dir / "actual_config.yaml"
    with open(actual_config_path, 'w', encoding='utf-8') as f:
        # 将配置转换为字典格式
        config_dict = {
            'algorithm': {
                'name': 'vq_hc_sac',
                'version': '1.0.0'
            },
            'roles': {
                'n_roles': config.n_roles,
                'embedding_dim': config.embedding_dim
            },
            'networks': {
                'encoder': {'hidden_dims': config.encoder_hidden_dims},
                'actor': {'hidden_dims': config.actor_hidden_dims},
                'critic': {'hidden_dims': config.critic_hidden_dims}
            },
            'training': {
                'batch_size': config.batch_size,
                'buffer_size': config.buffer_size,
                'warmup_steps': config.warmup_steps,
                'gamma': config.gamma,
                'tau': config.tau
            },
            'learning_rates': {
                'actor': config.actor_lr,
                'critic': config.critic_lr,
                'encoder': config.encoder_lr,
                'alpha': config.alpha_lr
            },
            'vq': {
                'beta': config.vq_beta,
                'use_ema': config.use_ema_codebook
            },
            'updates': {
                'frequency': config.update_frequency,
                'updates_per_step': config.updates_per_step,
                'clip_grad_norm': config.clip_grad_norm
            }
        }
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"📝 实际使用的配置已保存到: {actual_config_path}")
    
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
