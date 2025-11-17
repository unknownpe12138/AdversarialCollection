"""
统一实验运行入口

自动创建实验目录并运行训练和分析

用法:
    python scripts/run_experiment.py \
        --algorithm vq_hc_sac \
        --env-config configs/environments/3vs6.yaml \
        --total-steps 100000 \
        --exp-name baseline
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_experiment_dir(algorithm: str, n_agents: int, n_nodes: int, 
                         exp_name: str) -> Path:
    """
    动态创建实验目录
    
    格式: results/{algorithm}/{n_agents}vs{n_nodes}_{exp_name}/run_{timestamp}/
    
    Args:
        algorithm: 算法名称
        n_agents: 智能体数量
        n_nodes: 节点数量
        exp_name: 实验名称
    
    Returns:
        实验目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = f"{n_agents}vs{n_nodes}"
    
    # 构造目录路径
    exp_dir = Path(__file__).parent.parent / "results" / algorithm / f"{env_name}_{exp_name}" / f"run_{timestamp}"
    
    # 创建必要的子目录
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    
    return exp_dir


def load_env_config(config_path: str) -> dict:
    """加载环境配置文件"""
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_experiment_config(exp_dir: Path, args, env_config: dict):
    """保存完整的实验配置"""
    full_config = {
        'experiment': {
            'algorithm': args.algorithm,
            'exp_name': args.exp_name,
            'timestamp': datetime.now().isoformat(),
            'exp_dir': str(exp_dir),
            'seed': args.seed,
            'device': args.device
        },
        'training': {
            'total_steps': args.total_steps,
            'eval_interval': args.eval_interval,
            'log_interval': args.log_interval
        },
        'environment': env_config
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(full_config, f, indent=2)
    
    return full_config


def main():
    parser = argparse.ArgumentParser(
        description='运行单个实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--algorithm', type=str, required=True,
                       help='算法名称，如: vq_hc_sac')
    parser.add_argument('--env-config', type=str, required=True,
                       help='环境配置文件路径，如: configs/environments/3vs6.yaml')
    
    # 训练参数
    parser.add_argument('--total-steps', type=int, default=100000,
                       help='总训练步数')
    parser.add_argument('--eval-interval', type=int, default=10000,
                       help='评估间隔（步数）')
    parser.add_argument('--log-interval', type=int, default=1000,
                       help='日志记录间隔（步数）')
    
    # 实验参数
    parser.add_argument('--exp-name', type=str, default='baseline',
                       help='实验名称（用于目录命名），如: baseline, scale, no_pbrs')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='训练设备')
    
    # 可选参数
    parser.add_argument('--skip-analysis', action='store_true',
                       help='跳过训练后的自动分析')
    
    args = parser.parse_args()
    
    # 加载环境配置
    try:
        env_config = load_env_config(args.env_config)
    except Exception as e:
        print(f"❌ 加载环境配置失败: {e}")
        sys.exit(1)
    
    n_agents = env_config['n_agents']
    n_nodes = env_config['n_nodes']
    
    # 创建实验目录
    exp_dir = create_experiment_dir(args.algorithm, n_agents, n_nodes, args.exp_name)
    
    # 打印实验信息
    print("\n" + "="*70)
    print("实验开始")
    print("="*70)
    print(f"算法:      {args.algorithm}")
    print(f"环境:      {n_agents}智能体 × {n_nodes}节点")
    print(f"实验名:    {args.exp_name}")
    print(f"训练步数:  {args.total_steps:,}")
    print(f"随机种子:  {args.seed}")
    print(f"设备:      {args.device}")
    print(f"保存目录:  {exp_dir}")
    print("="*70 + "\n")
    
    # 保存实验配置
    full_config = save_experiment_config(exp_dir, args, env_config)
    print(f"✓ 实验配置已保存: {exp_dir / 'config.json'}\n")
    
    # 动态导入并运行训练
    try:
        # 导入训练模块（从experiments目录开始的完整路径）
        train_module_path = f'experiments.algorithms.{args.algorithm}.train'
        train_module = __import__(train_module_path, fromlist=['run_training'])
        
        if not hasattr(train_module, 'run_training'):
            raise ImportError(f"{train_module_path} 缺少 run_training() 函数")
        
        # 运行训练
        print(f"开始训练 {args.algorithm}...\n")
        train_module.run_training(
            env_config=env_config,
            save_dir=exp_dir,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            seed=args.seed,
            device=args.device
        )
        
        print("\n" + "="*70)
        print("✓ 训练完成！")
        print("="*70)
        print(f"结果保存在: {exp_dir}\n")
        
        # 自动运行分析（除非跳过）
        if not args.skip_analysis:
            try:
                print("="*70)
                print("开始分析结果...")
                print("="*70 + "\n")
                
                analyze_module_path = f'experiments.algorithms.{args.algorithm}.analyze'
                analyze_module = __import__(analyze_module_path, fromlist=['run_analysis'])
                
                if hasattr(analyze_module, 'run_analysis'):
                    analyze_module.run_analysis(exp_dir)
                    print("\n✓ 分析完成！\n")
                else:
                    print(f"⚠ {analyze_module_path} 缺少 run_analysis() 函数，跳过分析\n")
                    
            except ImportError as e:
                print(f"⚠ 无法导入分析模块: {e}")
                print("跳过自动分析\n")
        
        # 最终总结
        print("="*70)
        print("实验完成！")
        print("="*70)
        print(f"\n生成的文件:")
        print(f"  配置:      {exp_dir / 'config.json'}")
        print(f"  模型:      {exp_dir / 'checkpoints/'}")
        print(f"  数据:      {exp_dir / 'data/'}")
        print(f"  日志:      {exp_dir / 'logs/'}")
        print(f"  图表:      {exp_dir / 'figures/'}")
        print(f"\n实验目录: {exp_dir}\n")
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print(f"\n提示: 确保算法模块存在于 algorithms/{args.algorithm}/")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
