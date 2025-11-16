"""
Logging utilities for environment interactions.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """
    Custom logger for environment events.
    """
    
    def __init__(
        self,
        name: str = 'AdversarialEnv',
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (None for console only)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def episode_start(self, episode: int, config: dict = None):
        """Log episode start."""
        msg = f"Episode {episode} started"
        if config:
            msg += f" with config: {config}"
        self.logger.info(msg)
    
    def episode_end(self, episode: int, summary: dict):
        """Log episode end with summary."""
        self.logger.info(
            f"Episode {episode} ended - "
            f"Steps: {summary.get('steps', 'N/A')}, "
            f"Reward: {summary.get('team_reward', 'N/A'):.2f}, "
            f"Survived: {summary.get('survival_rate', 'N/A'):.1%}"
        )
    
    def action(self, agent_id: int, action: str, details: str = ""):
        """Log agent action."""
        msg = f"Agent {agent_id}: {action}"
        if details:
            msg += f" ({details})"
        self.logger.debug(msg)
    
    def event(self, event_type: str, details: dict):
        """Log custom event."""
        self.logger.info(f"Event [{event_type}]: {details}")
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error."""
        self.logger.error(message)
        if exception:
            self.logger.exception(exception)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)


def create_experiment_logger(experiment_name: str, log_dir: str = 'logs') -> Logger:
    """
    Create a logger for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs
        
    Returns:
        Configured Logger instance
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    logger = Logger(
        name=experiment_name,
        log_file=log_file,
        level=logging.DEBUG
    )
    
    logger.info(f"Experiment '{experiment_name}' started")
    logger.info(f"Log file: {log_file}")
    
    return logger
