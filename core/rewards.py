"""
Reward calculation and shaping mechanisms.
Implements original rewards and PBRS (Potential-Based Reward Shaping).
"""

from typing import Dict, Tuple
from .agent import AgentState


class RewardCalculator:
    """
    Calculates original rewards R_k based on MDP definition.
    
    Reward rules:
    - Move to base (deliver): R_k = U_k^carried
    - Other moves: R_k = 0
    - Collect success: R_k = 0
    - Collect failure (die): R_k = -C
    - Skip/Stay: R_k = 0
    """
    
    def __init__(self, death_penalty: float = 10.0):
        """
        Initialize reward calculator.
        
        Args:
            death_penalty: Penalty C when agent dies
        """
        self.death_penalty = death_penalty
    
    def calculate_move_reward(self, to_base: bool, carried_utility: float) -> float:
        """
        Calculate reward for move action.
        
        Args:
            to_base: Whether moving to base node
            carried_utility: Amount being carried
            
        Returns:
            Reward value
        """
        if to_base:
            return carried_utility
        return 0.0
    
    def calculate_collect_reward(self, success: bool) -> float:
        """
        Calculate reward for collect action.
        
        Args:
            success: Whether collection succeeded
            
        Returns:
            Reward value (0 if success, -C if failure)
        """
        if success:
            return 0.0
        else:
            return -self.death_penalty
    
    def calculate_skip_reward(self) -> float:
        """Calculate reward for skip action (always 0)."""
        return 0.0


class PBRSRewardShaper:
    """
    Potential-Based Reward Shaping (PBRS) for dense rewards.
    
    Uses potential function: Φ(s_k) = U_k^carried
    
    Shaped reward: R'_k = R_k + γ * Φ(s'_k) - Φ(s_k)
    
    This transfers credit from Deliver (R'_k=0) to Collect (R'_k ≈ +γ * u_v)
    """
    
    def __init__(self, gamma: float = 0.99):
        """
        Initialize PBRS reward shaper.
        
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
    
    def potential(self, agent_state: AgentState) -> float:
        """
        Compute potential function Φ(s_k) = U_k^carried.
        
        Args:
            agent_state: Agent state
            
        Returns:
            Potential value
        """
        return agent_state.carried_utility
    
    def shape_reward(
        self,
        original_reward: float,
        state_before: AgentState,
        state_after: AgentState
    ) -> float:
        """
        Apply PBRS to get shaped reward.
        
        R'_k = R_k + γ * Φ(s'_k) - Φ(s_k)
        
        Args:
            original_reward: Original reward R_k
            state_before: State before action
            state_after: State after action
            
        Returns:
            Shaped reward R'_k
        """
        phi_before = self.potential(state_before)
        phi_after = self.potential(state_after)
        
        shaped = original_reward + self.gamma * phi_after - phi_before
        return shaped
    
    def get_shaping_info(
        self,
        state_before: AgentState,
        state_after: AgentState
    ) -> Dict[str, float]:
        """
        Get detailed shaping information for debugging.
        
        Returns:
            Dictionary with potential values and shaping delta
        """
        phi_before = self.potential(state_before)
        phi_after = self.potential(state_after)
        delta = self.gamma * phi_after - phi_before
        
        return {
            'potential_before': phi_before,
            'potential_after': phi_after,
            'shaping_delta': delta
        }


class TeamRewardAggregator:
    """
    Aggregates individual rewards into team reward.
    
    Team reward: R(s,a) = Σ_k R_k(s,a)
    """
    
    @staticmethod
    def aggregate(individual_rewards: Dict[int, float]) -> float:
        """
        Sum individual rewards to get team reward.
        
        Args:
            individual_rewards: Dict mapping agent_id to reward
            
        Returns:
            Total team reward
        """
        return sum(individual_rewards.values())
    
    @staticmethod
    def get_statistics(individual_rewards: Dict[int, float]) -> Dict[str, float]:
        """
        Get reward statistics.
        
        Returns:
            Dictionary with team total, mean, min, max
        """
        rewards = list(individual_rewards.values())
        
        return {
            'team_total': sum(rewards),
            'mean_reward': sum(rewards) / len(rewards) if rewards else 0.0,
            'min_reward': min(rewards) if rewards else 0.0,
            'max_reward': max(rewards) if rewards else 0.0
        }
