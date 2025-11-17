"""
Core environment class for adversarial multi-agent value collection.
Implements Gymnasium interface for RL training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy

from .graph import EnvironmentGraph
from .agent import Agent, Action, ActionType, AgentState
from .rewards import RewardCalculator, PBRSRewardShaper, TeamRewardAggregator


class AdversarialCollectionEnv(gym.Env):
    """
    Multi-agent adversarial value collection environment.
    
    MDP: M = <S, A, P, R, γ>
    
    State: s = (s_env, s_1, ..., s_N)
        - s_env = {(u_v, r_v) | v ∈ V\{v_0}}
        - s_k = (v_k, U_k^carried, is_alive)
    
    Action: a = (a_1, ..., a_N)
        - Move(v_j), Collect, Skip, Stay_Dead
    
    Transitions: 
        - Move: deterministic
        - Collect: stochastic (success prob = 1 - r_v)
    
    Rewards:
        - Deliver: R_k = U_k^carried
        - Collect fail: R_k = -C
        - Others: R_k = 0
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        n_agents: int = 3,
        n_nodes: int = 10,
        graph_type: str = 'grid',
        grid_size: Optional[Tuple[int, int]] = None,
        edge_list: Optional[List[Tuple[int, int]]] = None,
        utility_range: Tuple[float, float] = (5.0, 20.0),
        risk_range: Tuple[float, float] = (0.1, 0.5),
        death_penalty: float = 10.0,
        gamma: float = 0.99,
        use_pbrs: bool = True,
        max_steps: int = 200,
        edge_probability: float = 0.3,
        graph_seed: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize environment.
        
        Args:
            n_agents: Number of agents N
            n_nodes: Total number of nodes (including base)
            graph_type: 'grid', 'random', or 'complete'
            grid_size: For grid type, (rows, cols)
            edge_list: Custom edges (overrides graph_type)
            utility_range: (min, max) for node utilities
            risk_range: (min, max) for node risks
            death_penalty: Penalty C for dying
            gamma: Discount factor
            use_pbrs: Whether to use PBRS reward shaping
            max_steps: Maximum steps per episode
            edge_probability: For random graph, edge creation probability (default: 0.3)
            graph_seed: For random graph, seed for reproducibility (default: None)
            seed: Random seed for environment dynamics
        """
        super().__init__()
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        self._seed = seed
        
        # Environment parameters
        self.n_agents = n_agents
        self.n_nodes = n_nodes
        self.utility_range = utility_range
        self.risk_range = risk_range
        self.gamma = gamma
        self.use_pbrs = use_pbrs
        self.max_steps = max_steps
        
        # Create graph structure
        self.graph = EnvironmentGraph(
            n_nodes=n_nodes,
            edge_list=edge_list,
            graph_type=graph_type,
            grid_size=grid_size,
            edge_probability=edge_probability,
            random_seed=graph_seed
        )
        
        # Create agents
        self.agents = [Agent(agent_id=i, initial_position=0) 
                      for i in range(n_agents)]
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(death_penalty=death_penalty)
        self.reward_shaper = PBRSRewardShaper(gamma=gamma) if use_pbrs else None
        self.team_aggregator = TeamRewardAggregator()
        
        # Environment state: {node_id: (utility, risk)}
        self.node_states: Dict[int, Tuple[float, float]] = {}
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Initialize environment
        self.reset()
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Action space: For each agent, discrete actions
        # We'll use a simple discrete space where each action is encoded
        # This is simplified; real implementation might use MultiDiscrete or Dict
        
        # For now: max actions = max_neighbors + 2 (collect, skip)
        max_neighbors = max(len(self.graph.get_neighbors(node)) 
                           for node in range(self.n_nodes))
        max_actions = max_neighbors + 2
        
        # Multi-agent: Tuple of discrete spaces
        self.action_space = spaces.Tuple([
            spaces.Discrete(max_actions) for _ in range(self.n_agents)
        ])
        
        # Observation space: continuous box
        # Each agent observes: [position, carried_utility, is_alive] 
        # + environment state (all nodes: utility, risk)
        agent_obs_dim = 3  # position (normalized), carried, alive
        env_obs_dim = (self.n_nodes - 1) * 2  # (u_v, r_v) for task nodes
        total_obs_dim = agent_obs_dim + env_obs_dim
        
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(total_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observations: Initial observations for all agents
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset agents
        for agent in self.agents:
            agent.reset(initial_position=0)
        
        # Initialize node states (utility, risk)
        self.node_states = {}
        for node in self.graph.task_nodes:
            utility = np.random.uniform(*self.utility_range)
            risk = np.random.uniform(*self.risk_range)
            self.node_states[node] = (utility, risk)
        
        # Base node has no utility/risk
        self.node_states[0] = (0.0, 0.0)
        
        # Reset tracking
        self.current_step = 0
        self.episode_count += 1
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(
        self,
        actions: Union[List[Action], List[int]]
    ) -> Tuple[Any, Dict[int, float], bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            actions: List of actions for each agent (Action objects or integers)
            
        Returns:
            observations: Next observations
            rewards: Individual rewards for each agent
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self.current_step += 1
        
        # Convert integer actions to Action objects if needed
        if actions and isinstance(actions[0], int):
            actions = [self._decode_action(agent_id, action_int) 
                      for agent_id, action_int in enumerate(actions)]
        
        # Store states before actions (for PBRS)
        states_before = [agent.state.copy() for agent in self.agents]
        
        # Execute actions and calculate rewards
        individual_rewards = {}
        original_rewards = {}
        
        for agent, action, state_before in zip(self.agents, actions, states_before):
            reward = self._execute_action(agent, action)
            original_rewards[agent.agent_id] = reward
            
            # Apply PBRS if enabled
            if self.use_pbrs and self.reward_shaper:
                reward = self.reward_shaper.shape_reward(
                    reward, state_before, agent.state
                )
            
            individual_rewards[agent.agent_id] = reward
            agent.step()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get observations and info
        observations = self._get_observations()
        info = self._get_info()
        info['original_rewards'] = original_rewards
        info['shaped_rewards'] = individual_rewards if self.use_pbrs else original_rewards
        
        return observations, individual_rewards, terminated, truncated, info
    
    def _execute_action(self, agent: Agent, action: Action) -> float:
        """
        Execute single agent action and return reward.
        
        Args:
            agent: Agent executing action
            action: Action to execute
            
        Returns:
            Reward for this action
        """
        if not agent.state.is_alive:
            return 0.0
        
        current_pos = agent.state.position
        
        if action.action_type == ActionType.MOVE:
            # Move action
            target = action.target_node
            
            # Validate move
            neighbors = self.graph.get_neighbors(current_pos)
            if target not in neighbors:
                # Invalid move, treat as skip
                return 0.0
            
            # Execute move
            agent.move(target)
            
            # Check if moved to base (deliver)
            if self.graph.is_base_node(target):
                delivered = agent.deliver()
                return self.reward_calculator.calculate_move_reward(
                    to_base=True, carried_utility=delivered
                )
            else:
                return self.reward_calculator.calculate_move_reward(
                    to_base=False, carried_utility=0
                )
        
        elif action.action_type == ActionType.COLLECT:
            # Collect action
            if self.graph.is_base_node(current_pos):
                # Can't collect at base
                return 0.0
            
            utility, risk = self.node_states.get(current_pos, (0.0, 0.0))
            
            if utility <= 0:
                # No utility to collect
                return 0.0
            
            # Stochastic collection
            success_prob = 1.0 - risk
            success = np.random.random() < success_prob
            
            if success:
                # Success: collect utility
                agent.collect(utility)
                self.node_states[current_pos] = (0.0, 0.0)  # Clear node
                return self.reward_calculator.calculate_collect_reward(success=True)
            else:
                # Failure: agent dies
                agent.die()
                return self.reward_calculator.calculate_collect_reward(success=False)
        
        elif action.action_type == ActionType.SKIP:
            # Skip action
            return self.reward_calculator.calculate_skip_reward()
        
        else:  # STAY_DEAD
            return 0.0
    
    def _decode_action(self, agent_id: int, action_int: int) -> Action:
        """
        Decode integer action to Action object.
        
        Args:
            agent_id: Agent index
            action_int: Integer action code
            
        Returns:
            Action object
        """
        agent = self.agents[agent_id]
        
        if not agent.state.is_alive:
            return Action(ActionType.STAY_DEAD)
        
        current_pos = agent.state.position
        neighbors = self.graph.get_neighbors(current_pos)
        utility, _ = self.node_states.get(current_pos, (0.0, 0.0))
        
        # Action encoding: 0..len(neighbors)-1 = move, next = collect, last = skip
        if action_int < len(neighbors):
            return Action(ActionType.MOVE, neighbors[action_int])
        elif action_int == len(neighbors) and utility > 0 and not self.graph.is_base_node(current_pos):
            return Action(ActionType.COLLECT)
        else:
            return Action(ActionType.SKIP)
    
    def _get_observations(self) -> Tuple:
        """Get observations for all agents."""
        observations = []
        
        for agent in self.agents:
            obs = self._get_single_observation(agent)
            observations.append(obs)
        
        return tuple(observations)
    
    def _get_single_observation(self, agent: Agent) -> np.ndarray:
        """
        Get observation for a single agent.
        
        Observation: [agent_state, environment_state]
        - Agent: [position (normalized), carried_utility, is_alive]
        - Environment: [(u_v, r_v) for all task nodes]
        """
        # Agent state
        pos_normalized = agent.state.position / (self.n_nodes - 1)
        carried = agent.state.carried_utility / self.utility_range[1]  # Normalize
        alive = 1.0 if agent.state.is_alive else 0.0
        
        agent_obs = [pos_normalized, carried, alive]
        
        # Environment state (all task nodes)
        env_obs = []
        for node in self.graph.task_nodes:
            utility, risk = self.node_states.get(node, (0.0, 0.0))
            env_obs.extend([
                utility / self.utility_range[1],  # Normalize
                risk
            ])
        
        observation = np.array(agent_obs + env_obs, dtype=np.float32)
        return observation
    
    def _is_terminated(self) -> bool:
        """
        Check if episode is terminated.
        
        Terminated when:
        1. All task nodes have zero utility, OR
        2. All agents are dead
        """
        # Check if all utilities are zero
        all_utilities_zero = all(
            self.node_states.get(node, (0, 0))[0] == 0 
            for node in self.graph.task_nodes
        )
        
        # Check if all agents dead
        all_agents_dead = all(not agent.state.is_alive for agent in self.agents)
        
        return all_utilities_zero or all_agents_dead
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        # Aggregate rewards
        individual_rewards = {agent.agent_id: 0.0 for agent in self.agents}
        
        info = {
            'step': self.current_step,
            'episode': self.episode_count,
            'agents_alive': sum(1 for agent in self.agents if agent.state.is_alive),
            'total_utility_remaining': sum(
                self.node_states.get(node, (0, 0))[0] 
                for node in self.graph.task_nodes
            ),
            'agent_statistics': [agent.get_statistics() for agent in self.agents]
        }
        
        return info
    
    def get_environment_state(self) -> Dict:
        """
        Get full environment state for monitoring/visualization.
        
        Returns:
            Dictionary with complete state information
        """
        return {
            'step': self.current_step,
            'episode': self.episode_count,
            'node_states': copy.deepcopy(self.node_states),
            'agent_states': [
                {
                    'id': agent.agent_id,
                    'position': agent.state.position,
                    'carried': agent.state.carried_utility,
                    'alive': agent.state.is_alive,
                    'stats': agent.get_statistics()
                }
                for agent in self.agents
            ],
            'graph_info': {
                'n_nodes': self.n_nodes,
                'base_node': self.graph.base_node,
                'edges': list(self.graph.graph.edges())
            }
        }
    
    def render(self, mode: str = 'human'):
        """Render environment (placeholder for visualization module)."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Agents alive: {sum(1 for a in self.agents if a.state.is_alive)}/{self.n_agents}")
            for agent in self.agents:
                print(f"  {agent}")
            print(f"Utility remaining: {sum(self.node_states.get(n, (0,0))[0] for n in self.graph.task_nodes):.2f}")
        
        return None
    
    def close(self):
        """Clean up resources."""
        pass
