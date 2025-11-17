"""
批量运行实验

支持多个环境配置、多个种子的批量实验

用法:
    # 运行所有环境配置
    python scripts/batch_run.py \
        --algorithm vq_hc_sac \
        --env-configs configs/environments/*.yaml \
        --total-steps 100000
    
    # 使用多个种子
    python scripts/batch_run.py \
        --algorithm vq_hc_sac \
        --env-configs configs/environments/3vs6.yaml \
        --seeds 42 43 44 \
        --total-steps 50000
"""

import argparse
import subprocess
import sys
from pathlib import Path
from glob import glob


def run_single_experiment(algorithm, env_config, exp_name, total_steps, 
                         seed, device, skip_analysis):
    """运行单个实验"""
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--algorithm", algorithm,
        "--env-config", env_config,
        "--exp-name", exp_name,
        "--total-steps", str(total_steps),
        "--seed", str(seed),
        "--device", device
    ]
    
    if skip_analysis:
        cmd.append("--skip-analysis")
    
    print("\n" + ">"*70)
    print(f"运行命令: {' '.join(cmd)}")
    print(">"*70 + "\n")
    
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='批量运行实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--algorithm', type=str, required=True,
                       help='算法名称')
    parser.add_argument('--env-configs', nargs='+', required=True,
                       help='环境配置文件路径（支持通配符）')
    
    # 训练参数
    parser.add_argument('--total-steps', type=int, default=100000,
                       help='总训练步数')
    parser.add_argument('--eval-interval', type=int, default=10000,
                       help='评估间隔')
    parser.add_argument('--log-interval', type=int, default=1000,
                       help='日志间隔')
    
    # 实验参数
    parser.add_argument('--exp-name', type=str, default='baseline',
                       help='实验名称')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='随机种子列表（支持多个）')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    # 可选参数
    parser.add_argument('--skip-analysis', action='store_true',
                       help='跳过自动分析')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='遇到错误时停止（默认继续）')
    
    args = parser.parse_args()
    
    # 展开通配符
    env_configs = []
    for pattern in args.env_configs:
        matched = glob(pattern)
        if matched:
            env_configs.extend(matched)
        else:
            env_configs.append(pattern)  # 如果不是通配符，直接添加
    
    # 计算总实验数
    total_experiments = len(env_configs) * len(args.seeds)
    
    print("\n" + "="*70)
    print("批量实验")
    print("="*70)
    print(f"算法:        {args.algorithm}")
    print(f"环境配置:    {len(env_configs)} 个")
    print(f"随机种子:    {len(args.seeds)} 个")
    print(f"总实验数:    {total_experiments}")
    print(f"训练步数:    {args.total_steps:,}")
    print("="*70 + "\n")
    
    print("环境配置列表:")
    for i, config in enumerate(env_configs, 1):
        print(f"  {i}. {config}")
    print()
    
    print(f"随机种子列表: {args.seeds}\n")
    
    # 询问确认
    response = input(f"确认运行 {total_experiments} 个实验? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 运行所有实验
    success_count = 0
    failed_experiments = []
    
    experiment_idx = 0
    for env_config in env_configs:
        for seed in args.seeds:
            experiment_idx += 1
            
            print("\n" + "="*70)
            print(f"实验 {experiment_idx}/{total_experiments}")
            print(f"环境: {env_config}")
            print(f"种子: {seed}")
            print("="*70)
            
            success = run_single_experiment(
                algorithm=args.algorithm,
                env_config=env_config,
                exp_name=args.exp_name,
                total_steps=args.total_steps,
                seed=seed,
                device=args.device,
                skip_analysis=args.skip_analysis
            )
            
            if success:
                success_count += 1
                print(f"\n✓ 实验 {experiment_idx} 完成")
            else:
                failed_experiments.append((env_config, seed))
                print(f"\n❌ 实验 {experiment_idx} 失败")
                
                if args.stop_on_error:
                    print("\n遇到错误，停止批量运行")
                    break
        
        if args.stop_on_error and failed_experiments:
            break
    
    # 总结
    print("\n" + "="*70)
    print("批量实验完成")
    print("="*70)
    print(f"成功: {success_count}/{total_experiments}")
    print(f"失败: {len(failed_experiments)}/{total_experiments}")
    
    if failed_experiments:
        print("\n失败的实验:")
        for env_config, seed in failed_experiments:
            print(f"  - {env_config}, seed={seed}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
