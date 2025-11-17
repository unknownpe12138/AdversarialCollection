"""
算法基类接口

所有算法必须实现这个接口以确保兼容性
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path


class BaseAlgorithm(ABC):
    """
    所有多智能体强化学习算法的基类
    
    新算法需要继承此类并实现所有抽象方法
    """
    
    @abstractmethod
    def __init__(self, env, config: Dict[str, Any], device: str = 'cpu'):
        """
        初始化算法
        
        Args:
            env: Gymnasium环境实例
            config: 算法配置字典
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.config = config
        self.device = device
    
    @abstractmethod
    def train(self, 
              total_steps: int,
              eval_interval: int = 10000,
              log_interval: int = 1000) -> Dict[str, List]:
        """
        训练算法
        
        Args:
            total_steps: 总训练步数
            eval_interval: 评估间隔（步数）
            log_interval: 日志记录间隔（步数）
        
        Returns:
            history: 训练历史字典，必须包含以下键：
                - 'episode_returns': List[float] - 每个episode的总回报
                - 'episode_lengths': List[int] - 每个episode的长度
                - 'losses': List[Dict] - 训练损失记录
                - 'eval_returns': List[float] - 评估回报（可选）
        
        示例:
            {
                'episode_returns': [10.5, 12.3, ...],
                'episode_lengths': [45, 50, ...],
                'losses': [{'critic_loss': 0.5, 'actor_loss': 0.2}, ...],
                'eval_returns': [15.2, 18.7, ...]
            }
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, filename: str):
        """
        保存模型checkpoint
        
        Args:
            filename: checkpoint文件名（相对于save_dir）
        
        应保存:
            - 所有网络的state_dict
            - 优化器状态
            - 训练进度（steps, episodes）
            - 配置信息
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, filename: str):
        """
        加载模型checkpoint
        
        Args:
            filename: checkpoint文件路径
        """
        pass
    
    @abstractmethod
    def select_actions(self, observations, deterministic: bool = False):
        """
        根据观察选择动作
        
        Args:
            observations: 观察（numpy array或list）
                - 单智能体: shape (obs_dim,)
                - 多智能体: shape (n_agents, obs_dim) 或 list of arrays
            deterministic: 是否使用确定性策略（用于评估）
        
        Returns:
            actions: 动作列表或数组
                - 离散动作: List[int] 或 ndarray of ints
                - 连续动作: ndarray of floats
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        返回算法名称（用于目录命名和识别）
        
        Returns:
            算法名称字符串（小写，下划线分隔）
            
        示例:
            'vq_hc_sac', 'qmix', 'mappo', 'independent_dqn'
        """
        pass
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        评估当前策略（可选实现）
        
        Args:
            n_episodes: 评估episode数量
        
        Returns:
            评估指标字典
            
        示例:
            {
                'mean_return': 25.5,
                'std_return': 5.2,
                'mean_length': 48.3,
                'success_rate': 0.8
            }
        """
        episode_returns = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done:
                actions = self.select_actions(obs, deterministic=True)
                obs, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                
                # 处理rewards（可能是dict或array）
                if isinstance(rewards, dict):
                    episode_return += sum(rewards.values())
                else:
                    episode_return += sum(rewards) if hasattr(rewards, '__iter__') else rewards
                
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        import numpy as np
        return {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }


class AlgorithmFactory:
    """算法工厂类，用于动态创建算法实例"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class):
        """注册算法"""
        cls._registry[name] = algorithm_class
    
    @classmethod
    def create(cls, name: str, env, config: Dict, device: str = 'cpu') -> BaseAlgorithm:
        """创建算法实例"""
        if name not in cls._registry:
            raise ValueError(f"Unknown algorithm: {name}. "
                           f"Available: {list(cls._registry.keys())}")
        
        return cls._registry[name](env, config, device)
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """列出所有已注册的算法"""
        return list(cls._registry.keys())
