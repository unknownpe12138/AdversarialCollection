"""
Basic usage example for Adversarial Collection Environment.

This demonstrates:
1. Environment creation
2. Random agent interactions
3. Environment state monitoring
4. Episode completion
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AdversarialCollectionEnv
from core.agent import Action, ActionType
import numpy as np


def random_policy(env, observations):
    """
    Simple random policy for demonstration.
    
    Args:
        env: Environment instance
        observations: Current observations
        
    Returns:
        List of random actions
    """
    actions = []
    
    for agent in env.agents:
        if not agent.state.is_alive:
            actions.append(Action(ActionType.STAY_DEAD))
            continue
        
        # Get valid actions
        current_pos = agent.state.position
        neighbors = env.graph.get_neighbors(current_pos)
        utility, _ = env.node_states.get(current_pos, (0.0, 0.0))
        is_base = env.graph.is_base_node(current_pos)
        
        valid_actions = agent.get_valid_actions(neighbors, utility, is_base)
        
        # Choose random action
        action = np.random.choice(valid_actions)
        actions.append(action)
    
    return actions


def run_episode(env, max_steps=100, verbose=True):
    """
    Run a single episode with random policy.
    
    Args:
        env: Environment instance
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        
    Returns:
        Episode statistics
    """
    observations, info = env.reset()
    
    total_rewards = {agent.agent_id: 0.0 for agent in env.agents}
    episode_length = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Episode {env.episode_count}")
        print(f"{'='*60}")
        print(f"Agents: {env.n_agents}, Nodes: {env.n_nodes}")
        print(f"Total initial utility: {sum(env.node_states.get(n, (0,0))[0] for n in env.graph.task_nodes):.2f}")
    
    for step in range(max_steps):
        # Get random actions
        actions = random_policy(env, observations)
        
        # Execute step
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
        
        episode_length += 1
        
        # Optional: render every N steps
        if verbose and step % 10 == 0:
            env.render(mode='human')
        
        # Check termination
        if terminated or truncated:
            if verbose:
                reason = "Terminated" if terminated else "Truncated"
                print(f"\nEpisode ended: {reason} at step {episode_length}")
            break
    
    # Final statistics
    stats = {
        'episode_length': episode_length,
        'total_rewards': total_rewards,
        'team_reward': sum(total_rewards.values()),
        'agents_survived': info['agents_alive'],
        'utility_remaining': info['total_utility_remaining'],
        'agent_stats': info['agent_statistics']
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Episode Summary")
        print(f"{'='*60}")
        print(f"Length: {stats['episode_length']} steps")
        print(f"Team reward: {stats['team_reward']:.2f}")
        print(f"Agents survived: {stats['agents_survived']}/{env.n_agents}")
        print(f"Utility remaining: {stats['utility_remaining']:.2f}")
        print(f"\nIndividual rewards:")
        for agent_id, reward in total_rewards.items():
            print(f"  Agent {agent_id}: {reward:.2f}")
    
    return stats


def main():
    """Main demonstration function."""
    
    print("="*60)
    print("Adversarial Collection Environment - Basic Usage Demo")
    print("="*60)
    
    # Create environment with different configurations
    configs = [
        {
            'name': 'Small Grid (3 agents, 3x3 grid)',
            'n_agents': 3,
            'n_nodes': 9,
            'graph_type': 'grid',
            'grid_size': (3, 3),
            'use_pbrs': False
        },
        {
            'name': 'Medium Grid with PBRS (5 agents, 4x4 grid)',
            'n_agents': 5,
            'n_nodes': 16,
            'graph_type': 'grid',
            'grid_size': (4, 4),
            'use_pbrs': True
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*60}")
        print(f"Configuration {i+1}: {config['name']}")
        print(f"{'#'*60}")
        
        # Extract config name for display
        config_copy = config.copy()
        config_name = config_copy.pop('name')
        
        # Create environment
        env = AdversarialCollectionEnv(**config_copy, seed=42)
        
        # Run single episode
        stats = run_episode(env, max_steps=50, verbose=True)
        
        # Environment state inspection
        print(f"\n--- Environment State Inspection ---")
        state = env.get_environment_state()
        print(f"Current step: {state['step']}")
        print(f"Graph edges: {len(state['graph_info']['edges'])} connections")
        
        env.close()
    
    print(f"\n\n{'='*60}")
    print("Demo completed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
