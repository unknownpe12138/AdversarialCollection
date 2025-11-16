"""
Agent state and action management.
"""

from dataclasses import dataclass
from typing import Optional, List, Union
from enum import Enum


class ActionType(Enum):
    """Action types for agents."""
    MOVE = "move"
    COLLECT = "collect"
    SKIP = "skip"
    STAY_DEAD = "stay_dead"


@dataclass
class Action:
    """Represents an agent action."""
    action_type: ActionType
    target_node: Optional[int] = None  # For MOVE actions
    
    def __repr__(self) -> str:
        if self.action_type == ActionType.MOVE:
            return f"Move({self.target_node})"
        return self.action_type.value.capitalize()


@dataclass
class AgentState:
    """
    Individual agent state: s_k = (v_k, U_k^carried, is_alive)
    
    Attributes:
        position: Current node v_k
        carried_utility: Accumulated utility U_k^carried
        is_alive: Survival status
    """
    position: int
    carried_utility: float = 0.0
    is_alive: bool = True
    
    def copy(self) -> 'AgentState':
        """Create a deep copy of the state."""
        return AgentState(
            position=self.position,
            carried_utility=self.carried_utility,
            is_alive=self.is_alive
        )


class Agent:
    """
    Represents a single agent in the environment.
    """
    
    def __init__(self, agent_id: int, initial_position: int = 0):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier
            initial_position: Starting position (default: base node 0)
        """
        self.agent_id = agent_id
        self.state = AgentState(position=initial_position)
        
        # Statistics
        self.total_utility_collected = 0.0
        self.total_utility_delivered = 0.0
        self.collection_attempts = 0
        self.collection_successes = 0
        self.steps_alive = 0
    
    def reset(self, initial_position: int = 0):
        """Reset agent to initial state."""
        self.state = AgentState(position=initial_position)
        self.total_utility_collected = 0.0
        self.total_utility_delivered = 0.0
        self.collection_attempts = 0
        self.collection_successes = 0
        self.steps_alive = 0
    
    def get_valid_actions(
        self, 
        neighbors: List[int],
        current_node_utility: float,
        is_base: bool
    ) -> List[Action]:
        """
        Get valid actions for current state.
        
        Args:
            neighbors: Neighboring nodes
            current_node_utility: Utility at current node
            is_base: Whether at base node
            
        Returns:
            List of valid actions
        """
        if not self.state.is_alive:
            return [Action(ActionType.STAY_DEAD)]
        
        actions = []
        
        # Move actions to all neighbors
        for neighbor in neighbors:
            actions.append(Action(ActionType.MOVE, neighbor))
        
        # Collect action (if not at base and node has utility)
        if not is_base and current_node_utility > 0:
            actions.append(Action(ActionType.COLLECT))
        
        # Skip action (always available)
        actions.append(Action(ActionType.SKIP))
        
        return actions
    
    def move(self, target_position: int):
        """Move to target position."""
        if self.state.is_alive:
            self.state.position = target_position
    
    def collect(self, utility: float):
        """Collect utility from current node."""
        if self.state.is_alive:
            self.state.carried_utility += utility
            self.total_utility_collected += utility
            self.collection_attempts += 1
            self.collection_successes += 1
    
    def deliver(self) -> float:
        """Deliver carried utility at base. Returns delivered amount."""
        if self.state.is_alive:
            delivered = self.state.carried_utility
            self.total_utility_delivered += delivered
            self.state.carried_utility = 0.0
            return delivered
        return 0.0
    
    def die(self):
        """Mark agent as dead and lose carried utility."""
        self.state.is_alive = False
        self.state.carried_utility = 0.0
    
    def step(self):
        """Increment step counter."""
        if self.state.is_alive:
            self.steps_alive += 1
    
    def get_statistics(self) -> dict:
        """Get agent statistics."""
        success_rate = (self.collection_successes / self.collection_attempts 
                       if self.collection_attempts > 0 else 0.0)
        
        return {
            'agent_id': self.agent_id,
            'is_alive': self.state.is_alive,
            'position': self.state.position,
            'carried_utility': self.state.carried_utility,
            'total_collected': self.total_utility_collected,
            'total_delivered': self.total_utility_delivered,
            'collection_success_rate': success_rate,
            'steps_alive': self.steps_alive
        }
    
    def __repr__(self) -> str:
        status = "Alive" if self.state.is_alive else "Dead"
        return (f"Agent{self.agent_id}({status}, pos={self.state.position}, "
                f"carrying={self.state.carried_utility:.2f})")
