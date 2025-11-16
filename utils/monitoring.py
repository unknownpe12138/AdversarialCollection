"""
Environment monitoring and tracking utilities.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json


class EpisodeTracker:
    """
    Tracks statistics for a single episode.
    """
    
    def __init__(self, n_agents: int):
        """
        Initialize episode tracker.
        
        Args:
            n_agents: Number of agents
        """
        self.n_agents = n_agents
        self.reset()
    
    def reset(self):
        """Reset tracker for new episode."""
        self.step_count = 0
        self.total_rewards = {i: 0.0 for i in range(self.n_agents)}
        self.step_rewards = []
        self.utility_collected = {i: 0.0 for i in range(self.n_agents)}
        self.utility_delivered = {i: 0.0 for i in range(self.n_agents)}
        self.deaths = []
        self.actions_taken = defaultdict(lambda: defaultdict(int))
        self.positions_history = defaultdict(list)
    
    def record_step(
        self,
        step: int,
        rewards: Dict[int, float],
        agent_states: List[Dict],
        actions: Optional[List] = None
    ):
        """
        Record information for a step.
        
        Args:
            step: Current step number
            rewards: Rewards for each agent
            agent_states: List of agent state dictionaries
            actions: List of actions taken (optional)
        """
        self.step_count = step
        self.step_rewards.append(rewards.copy())
        
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            self.total_rewards[agent_id] += reward
        
        # Track agent states
        for state in agent_states:
            agent_id = state['agent_id']  # Fixed: use 'agent_id' instead of 'id'
            
            # Record position
            self.positions_history[agent_id].append(state['position'])
            
            # Track utilities
            self.utility_collected[agent_id] = state['total_collected']
            self.utility_delivered[agent_id] = state['total_delivered']
            
            # Track deaths
            if not state['is_alive'] and agent_id not in self.deaths:
                self.deaths.append(agent_id)
        
        # Track actions
        if actions:
            for agent_id, action in enumerate(actions):
                action_type = str(action.action_type.value) if hasattr(action, 'action_type') else str(action)
                self.actions_taken[agent_id][action_type] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get episode summary statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'steps': self.step_count,
            'total_rewards': self.total_rewards,
            'team_reward': sum(self.total_rewards.values()),
            'mean_agent_reward': np.mean(list(self.total_rewards.values())),
            'utility_collected': self.utility_collected,
            'utility_delivered': self.utility_delivered,
            'total_collected': sum(self.utility_collected.values()),
            'total_delivered': sum(self.utility_delivered.values()),
            'agents_died': len(self.deaths),
            'survival_rate': 1.0 - len(self.deaths) / self.n_agents,
            'death_list': self.deaths,
            'actions_distribution': dict(self.actions_taken)
        }


class EnvironmentMonitor:
    """
    Monitors environment over multiple episodes.
    Provides comprehensive statistics and tracking.
    """
    
    def __init__(self, n_agents: int):
        """
        Initialize environment monitor.
        
        Args:
            n_agents: Number of agents
        """
        self.n_agents = n_agents
        self.episode_trackers: List[EpisodeTracker] = []
        self.current_episode = None
    
    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode = EpisodeTracker(self.n_agents)
    
    def record_step(
        self,
        step: int,
        rewards: Dict[int, float],
        agent_states: List[Dict],
        actions: Optional[List] = None
    ):
        """Record step in current episode."""
        if self.current_episode is None:
            self.start_episode()
        
        self.current_episode.record_step(step, rewards, agent_states, actions)
    
    def end_episode(self) -> Dict[str, Any]:
        """
        End current episode and get summary.
        
        Returns:
            Episode summary
        """
        if self.current_episode is None:
            return {}
        
        summary = self.current_episode.get_summary()
        self.episode_trackers.append(self.current_episode)
        self.current_episode = None
        
        return summary
    
    def get_statistics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregated statistics across episodes.
        
        Args:
            last_n: Only consider last N episodes (None for all)
            
        Returns:
            Aggregated statistics
        """
        if not self.episode_trackers:
            return {}
        
        trackers = self.episode_trackers[-last_n:] if last_n else self.episode_trackers
        
        # Collect metrics
        episode_rewards = [t.get_summary()['team_reward'] for t in trackers]
        episode_lengths = [t.step_count for t in trackers]
        survival_rates = [t.get_summary()['survival_rate'] for t in trackers]
        utilities_delivered = [t.get_summary()['total_delivered'] for t in trackers]
        
        stats = {
            'episodes': len(trackers),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_survival_rate': np.mean(survival_rates),
            'mean_utility_delivered': np.mean(utilities_delivered),
            'total_utility_delivered': np.sum(utilities_delivered)
        }
        
        return stats
    
    def print_statistics(self, last_n: Optional[int] = None):
        """
        Print formatted statistics.
        
        Args:
            last_n: Only consider last N episodes
        """
        stats = self.get_statistics(last_n)
        
        if not stats:
            print("No episodes tracked yet.")
            return
        
        n_episodes = stats['episodes']
        window = f"Last {last_n}" if last_n else "All"
        
        print(f"\n{'='*60}")
        print(f"Environment Statistics ({window} {n_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Reward:")
        print(f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"\nPerformance:")
        print(f"  Mean episode length: {stats['mean_episode_length']:.1f} steps")
        print(f"  Mean survival rate: {stats['mean_survival_rate']:.1%}")
        print(f"  Mean utility delivered: {stats['mean_utility_delivered']:.2f}")
        print(f"  Total utility delivered: {stats['total_utility_delivered']:.2f}")
        print(f"{'='*60}\n")
    
    def save_to_file(self, filepath: str):
        """
        Save monitoring data to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'n_agents': self.n_agents,
            'n_episodes': len(self.episode_trackers),
            'statistics': self.get_statistics(),
            'episodes': [tracker.get_summary() for tracker in self.episode_trackers]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Monitoring data saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """
        Load monitoring data from JSON file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {data['n_episodes']} episodes from {filepath}")
        # Note: This loads statistics but not the full tracker objects
        return data


def print_environment_info(env_state: Dict):
    """
    Print formatted environment state information.
    
    Args:
        env_state: Environment state dictionary from get_environment_state()
    """
    print(f"\n{'='*60}")
    print(f"Environment State - Step {env_state['step']}")
    print(f"{'='*60}")
    
    # Graph info
    graph_info = env_state['graph_info']
    print(f"Graph: {graph_info['n_nodes']} nodes, {len(graph_info['edges'])} edges")
    print(f"Base node: {graph_info['base_node']}")
    
    # Node states
    print(f"\nTask Nodes:")
    node_states = env_state['node_states']
    for node_id, (utility, risk) in sorted(node_states.items()):
        if node_id == 0:  # Skip base node
            continue
        if utility > 0:
            print(f"  Node {node_id}: utility={utility:.2f}, risk={risk:.2%}")
    
    # Agent states
    print(f"\nAgents:")
    for agent_state in env_state['agent_states']:
        status = "Alive" if agent_state['alive'] else "Dead"
        print(f"  Agent {agent_state['id']}: {status}, pos={agent_state['position']}, "
              f"carrying={agent_state['carried']:.2f}")
    
    print(f"{'='*60}\n")
