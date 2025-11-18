"""
统一实验运行入口

通过YAML配置文件运行实验，自动创建实验目录并执行训练和分析

用法:
    python scripts/run_experiment.py --config configs/example_full_config.yaml
    
    # 跳过自动分析
    python scripts/run_experiment.py --config configs/small_scale_test.yaml --skip-analysis
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




def main():
    parser = argparse.ArgumentParser(
        description='运行实验 - 使用YAML配置文件',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数：配置文件路径
    parser.add_argument('--config', type=str, required=True,
                       help='实验配置文件路径，如: configs/example_full_config.yaml')
    
    # 可选参数
    parser.add_argument('--skip-analysis', action='store_true',
                       help='跳过训练后的自动分析')
    
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        
        # 验证配置文件包含必需的部分
        required_sections = ['experiment', 'training', 'environment', 'algorithm_config']
        for section in required_sections:
            if section not in full_config:
                raise ValueError(f"配置文件缺少必需的部分: {section}")
        
        # 从配置文件提取参数
        args.algorithm = full_config['experiment']['algorithm']
        args.exp_name = full_config['experiment']['name']
        args.seed = full_config['experiment']['seed']
        args.device = full_config['experiment']['device']
        
        args.total_steps = full_config['training']['total_steps']
        args.eval_interval = full_config['training']['eval_interval']
        args.log_interval = full_config['training']['log_interval']
        
        env_config = full_config['environment']
        algorithm_config = full_config['algorithm_config']
        
        print(f"📋 已加载配置文件: {args.config}")
        
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ 配置文件缺少必需的字段: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
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
    
    # 动态导入并运行训练
    try:
        # 导入训练模块（从experiments目录开始的完整路径）
        train_module_path = f'experiments.algorithms.{args.algorithm}.train'
        train_module = __import__(train_module_path, fromlist=['run_training'])
        
        if not hasattr(train_module, 'run_training'):
            raise ImportError(f"{train_module_path} 缺少 run_training() 函数")
        
        # 运行训练
        print(f"开始训练 {args.algorithm}...\n")
        # 传递algorithm_config（如果有的话）
        run_training_kwargs = {
            'env_config': env_config,
            'save_dir': exp_dir,
            'total_steps': args.total_steps,
            'eval_interval': args.eval_interval,
            'log_interval': args.log_interval,
            'seed': args.seed,
            'device': args.device
        }
        
        # 如果有算法配置，也传递给训练函数
        if algorithm_config:
            run_training_kwargs['algorithm_config'] = algorithm_config
        
        train_module.run_training(**run_training_kwargs)
        
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
