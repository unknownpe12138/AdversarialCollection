"""
Demonstration of environment monitoring and logging capabilities.

Shows:
1. Episode tracking
2. Multi-episode statistics
3. Environment state monitoring
4. Logging functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AdversarialCollectionEnv
from core.agent import Action, ActionType
from utils.monitoring import EnvironmentMonitor, print_environment_info
from utils.logger import create_experiment_logger
import numpy as np
from datetime import datetime
from pathlib import Path


def simple_greedy_policy(env, agent):
    """
    Simple greedy policy for demonstration.
    
    Strategy:
    1. If at base with utility, stay
    2. If carrying utility > threshold, return to base
    3. If at node with high utility and low risk, collect
    4. Otherwise, move toward nearest valuable node
    """
    if not agent.state.is_alive:
        return Action(ActionType.STAY_DEAD)
    
    current_pos = agent.state.position
    utility, risk = env.node_states.get(current_pos, (0.0, 0.0))
    is_base = env.graph.is_base_node(current_pos)
    
    # If carrying enough, return to base
    if agent.state.carried_utility > 15.0 and not is_base:
        # Find path to base
        neighbors = env.graph.get_neighbors(current_pos)
        distances = [env.graph.get_distance(n, 0) for n in neighbors]
        closest_neighbor = neighbors[np.argmin(distances)]
        return Action(ActionType.MOVE, closest_neighbor)
    
    # If at valuable node, try to collect
    if utility > 10.0 and risk < 0.3 and not is_base:
        return Action(ActionType.COLLECT)
    
    # Find nearest valuable node
    best_node = None
    best_value = -float('inf')
    
    for node in env.graph.task_nodes:
        node_utility, node_risk = env.node_states.get(node, (0.0, 0.0))
        if node_utility > 0:
            distance = env.graph.get_distance(current_pos, node)
            # Value = utility / (risk * distance)
            value = node_utility / (max(node_risk, 0.1) * max(distance, 1))
            if value > best_value:
                best_value = value
                best_node = node
    
    # Move toward best node
    if best_node is not None:
        neighbors = env.graph.get_neighbors(current_pos)
        distances = [env.graph.get_distance(n, best_node) for n in neighbors]
        closest_neighbor = neighbors[np.argmin(distances)]
        return Action(ActionType.MOVE, closest_neighbor)
    
    # Default: skip
    return Action(ActionType.SKIP)


def run_monitored_episode(env, monitor, logger, episode_num, render_every=10):
    """
    Run a single episode with full monitoring.
    
    Args:
        env: Environment instance
        monitor: EnvironmentMonitor
        logger: Logger instance
        episode_num: Episode number
        render_every: Render frequency
        
    Returns:
        Episode summary
    """
    # Start tracking
    monitor.start_episode()
    logger.episode_start(episode_num)
    
    # Reset environment
    observations, info = env.reset()
    
    # Log initial state
    logger.info(f"Initial utility: {info['total_utility_remaining']:.2f}")
    
    step = 0
    while True:
        step += 1
        
        # Get actions using greedy policy
        actions = [simple_greedy_policy(env, agent) for agent in env.agents]
        
        # Log actions
        for i, action in enumerate(actions):
            if env.agents[i].state.is_alive:
                logger.action(i, str(action))
        
        # Execute step
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Record in monitor
        monitor.record_step(step, rewards, info['agent_statistics'], actions)
        
        # Render periodically
        if step % render_every == 0:
            env.render(mode='human')
            state = env.get_environment_state()
            print_environment_info(state)
        
        # Log important events
        for agent_id, reward in rewards.items():
            if reward > 0:
                logger.event('reward_gained', {
                    'agent': agent_id,
                    'reward': reward,
                    'step': step
                })
            elif reward < 0:
                logger.event('agent_died', {
                    'agent': agent_id,
                    'penalty': reward,
                    'step': step
                })
        
        # Check termination
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            logger.info(f"Episode ended ({reason}) at step {step}")
            break
    
    # End episode and get summary
    summary = monitor.end_episode()
    logger.episode_end(episode_num, summary)
    
    return summary


def main():
    """Main demonstration function."""
    
    print("="*70)
    print("Environment Monitoring & Logging Demonstration")
    print("="*70)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path('logs') / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment logger
    logger = create_experiment_logger('monitoring_demo', log_dir=str(run_dir))
    
    # Create environment
    env_config = {
        'n_agents': 4,
        'n_nodes': 16,
        'graph_type': 'grid',
        'grid_size': (4, 4),
        'utility_range': (10.0, 30.0),
        'risk_range': (0.1, 0.4),
        'use_pbrs': True,
        'gamma': 0.99,
        'max_steps': 100,
        'seed': 42
    }
    
    logger.info(f"Environment configuration: {env_config}")
    env = AdversarialCollectionEnv(**env_config)
    
    # Create monitor
    monitor = EnvironmentMonitor(n_agents=env_config['n_agents'])
    
    # Run multiple episodes
    n_episodes = 5
    
    for episode in range(1, n_episodes + 1):
        print(f"\n{'#'*70}")
        print(f"Episode {episode}/{n_episodes}")
        print(f"{'#'*70}")
        
        summary = run_monitored_episode(
            env, monitor, logger, episode,
            render_every=20
        )
        
        # Print episode summary
        print(f"\n--- Episode {episode} Summary ---")
        print(f"Steps: {summary['steps']}")
        print(f"Team reward: {summary['team_reward']:.2f}")
        print(f"Utility collected: {summary['total_collected']:.2f}")
        print(f"Utility delivered: {summary['total_delivered']:.2f}")
        print(f"Survival rate: {summary['survival_rate']:.1%}")
        
        if summary['agents_died'] > 0:
            print(f"Agents died: {summary['death_list']}")
    
    # Print aggregated statistics
    print(f"\n\n{'='*70}")
    print("Multi-Episode Statistics")
    print(f"{'='*70}")
    
    monitor.print_statistics()
    
    # Save monitoring data
    json_file = run_dir / 'monitoring_results.json'
    monitor.save_to_file(str(json_file))
    
    # Test windowed statistics
    print("\n--- Last 3 Episodes Statistics ---")
    monitor.print_statistics(last_n=3)
    
    logger.info("Demonstration completed successfully")
    
    print(f"\n{'='*70}")
    print("Demonstration Complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {run_dir}")
    print("  - monitoring_results.json (monitoring data)")
    print("  - monitoring_demo_*.log (detailed logs)")


if __name__ == '__main__':
    main()
