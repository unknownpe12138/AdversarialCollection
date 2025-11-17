"""
配置文件解析工具
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置文件是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
    
    Returns:
        是否验证通过
    """
    missing_keys = []
    
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"配置文件缺少必需的键: {missing_keys}")
    
    return True


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    
    Args:
        *configs: 多个配置字典
    
    Returns:
        合并后的配置
    """
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    return merged


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
