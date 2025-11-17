"""
日志记录工具
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = 'experiment', 
                log_file: Path = None,
                level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（如果指定）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """实验日志记录类"""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)
        self.log_dir = self.exp_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'train_{timestamp}.log'
        
        # 设置logger
        self.logger = setup_logger('train', log_file)
    
    def info(self, msg: str):
        """记录信息"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """记录警告"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """记录错误"""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """记录调试信息"""
        self.logger.debug(msg)
