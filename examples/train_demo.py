"""
VQ-HC-SAC 训练脚本

专注于训练VQ-HC-SAC模型并保存必要的数据：
- 创建和配置训练环境
- 配置VQ-HC-SAC训练器
- 执行训练循环
- 保存模型checkpoint、训练历史和环境配置

训练完成后，使用 analyze_training.py 进行结果分析和模型测试。
"""

import sys
import os
import numpy as np
import torch
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import AdversarialCollectionEnv
from algorithms.vq_hc_sac import VQHCSACTrainer, VQHCSACConfig


def create_simple_environment():
    """创建一个简单的测试环境"""
    print("\n" + "="*70)
    print("创建环境...")
    print("="*70)
    
    # 创建简单的星型图: 1个基地 + 5个任务节点
    n_nodes = 6
    edge_list = [(0, i) for i in range(1, 6)]  # 基地连接到所有任务节点
    
    # 创建环境
    env = AdversarialCollectionEnv(
        n_agents=3,  # 3个智能体
        n_nodes=n_nodes,
        graph_type='custom',
        edge_list=edge_list,
        max_steps=50,
        use_pbrs=True  # 使用PBRS奖励塑形
    )
    
    print(f"✓ 环境创建成功:")
    print(f"  智能体数量: {env.n_agents}")
    print(f"  节点数量: {env.graph.n_nodes}")
    print(f"  最大步数: {env.max_steps}")
    print(f"  PBRS: {env.use_pbrs}")
    
    return env




def create_vq_hc_sac_trainer(env):
    """创建VQ-HC-SAC训练器"""
    print("\n" + "="*70)
    print("创建VQ-HC-SAC训练器...")
    print("="*70)
    
    # 配置 (小规模测试配置)
    config = VQHCSACConfig(
        # 角色配置
        n_roles=2,  # 2个角色: Explorer和Collector
        embedding_dim=32,  # 较小的嵌入维度
        
        # 网络配置
        encoder_hidden_dims=[64, 64],
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        
        # 训练配置
        batch_size=32,  # 小批量
        buffer_size=5000,  # 小缓冲区
        warmup_steps=500,  # 较短预热
        gamma=0.99,
        tau=0.005,
        
        # 学习率
        actor_lr=3e-4,
        critic_lr=3e-4,
        encoder_lr=3e-4,
        alpha_lr=3e-4,
        
        # VQ配置
        vq_beta=0.25,
        
        # 更新配置
        update_frequency=1,
        updates_per_step=1,
        
        # 其他
        clip_grad_norm=1.0,
        eval_episodes=3,
        save_interval=5000
    )
    
    # 创建训练器
    trainer = VQHCSACTrainer(
        env=env,
        config=config,
        save_dir='./checkpoints/demo',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"✓ 训练器创建成功")
    print(f"  设备: {trainer.device}")
    print(f"  角色数: {config.n_roles}")
    print(f"  批量大小: {config.batch_size}")
    
    return trainer, config


def quick_training(trainer, total_steps=2000):
    """快速训练测试"""
    print("\n" + "="*70)
    print(f"开始训练 (共 {total_steps} 步)...")
    print("="*70)
    
    try:
        history = trainer.train(
            total_steps=total_steps,
            eval_interval=1000,
            log_interval=500
        )
        
        print(f"\n✓ 训练完成!")
        print(f"  总步数: {trainer.total_steps}")
        print(f"  总episode: {trainer.episode_count}")
        
        if trainer.episode_returns:
            print(f"  平均回报 (最近100ep): {np.mean(trainer.episode_returns):.2f}")
        
        return history
    
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return None




def main():
    """主函数"""
    print("\n" + "="*70)
    print("VQ-HC-SAC 完整训练流程演示")
    print("="*70)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. 创建环境
    env = create_simple_environment()
    
    # 2. 创建训练器
    trainer, config = create_vq_hc_sac_trainer(env)
    
    # 3. 执行训练
    print("\n提示: 这是一个快速演示，只训练2000步")
    print("      完整训练建议100K-1M步")
    
    history = quick_training(trainer, total_steps=2000)
    
    if history is None:
        print("\n❌ 训练失败!")
        return
    
    # 4. 保存模型和训练历史
    checkpoint_path = './checkpoints/demo/quick_demo.pt'
    trainer.save_checkpoint('quick_demo.pt')
    print(f"\n✓ 模型已保存: {checkpoint_path}")
    
    # 保存训练历史
    history_path = './checkpoints/demo/history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"✓ 训练历史已保存: {history_path}")
    
    # 保存环境配置（用于后续分析）
    env_config = {
        'n_agents': env.n_agents,
        'n_nodes': env.n_nodes,
        'graph_type': 'custom',
        'edge_list': [(0, i) for i in range(1, 6)],
        'max_steps': env.max_steps,
        'use_pbrs': env.use_pbrs,
        'utility_range': env.utility_range,
        'risk_range': env.risk_range
    }
    env_config_path = './checkpoints/demo/env_config.pkl'
    with open(env_config_path, 'wb') as f:
        pickle.dump(env_config, f)
    print(f"✓ 环境配置已保存: {env_config_path}")
    
    # 5. 总结
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print("\n保存的文件:")
    print(f"  模型: {checkpoint_path}")
    print(f"  训练历史: {history_path}")
    print(f"  环境配置: {env_config_path}")
    print("\n下一步:")
    print("  1. 运行 analyze_training.py 查看训练结果和测试模型")
    print("  2. 增加训练步数进行完整训练 (100K-1M步)")
    print("  3. 调整超参数优化性能")
    print("="*70)


if __name__ == '__main__':
    main()
