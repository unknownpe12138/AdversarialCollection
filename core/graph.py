"""
Graph structure management for the environment.
Manages the topology G = (V, E) with base node v_0 and task nodes.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set


class EnvironmentGraph:
    """
    Manages the environment topology as an undirected graph.
    
    Graph structure:
    - v_0: base node (position 0)
    - v_i (i>0): task nodes with utility and risk
    """
    
    def __init__(
        self,
        n_nodes: int,
        edge_list: Optional[List[Tuple[int, int]]] = None,
        graph_type: str = 'grid',
        grid_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize environment graph.
        
        Args:
            n_nodes: Total number of nodes (including base node)
            edge_list: List of edges as (node_i, node_j) pairs
            graph_type: 'custom', 'grid', 'random', or 'complete'
            grid_size: For grid type, (rows, cols)
        """
        self.n_nodes = n_nodes
        self.base_node = 0
        self.task_nodes = list(range(1, n_nodes))
        
        # Create graph structure
        if edge_list is not None:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(n_nodes))
            self.graph.add_edges_from(edge_list)
        else:
            self.graph = self._create_graph(graph_type, grid_size)
        
        # Validate graph
        if not nx.is_connected(self.graph):
            raise ValueError("Graph must be connected!")
        
        # Node positions for visualization
        self.node_positions = self._compute_layout()
        
    def _create_graph(self, graph_type: str, grid_size: Optional[Tuple[int, int]]) -> nx.Graph:
        """Create graph based on type."""
        if graph_type == 'grid':
            if grid_size is None:
                # Auto-determine grid size
                side = int(np.ceil(np.sqrt(self.n_nodes)))
                grid_size = (side, side)
            
            rows, cols = grid_size
            if rows * cols < self.n_nodes:
                raise ValueError(f"Grid size {grid_size} too small for {self.n_nodes} nodes")
            
            # Create grid graph
            G = nx.grid_2d_graph(rows, cols)
            # Convert to simple integer labels
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            # Keep only n_nodes
            nodes_to_remove = list(range(self.n_nodes, len(G.nodes())))
            G.remove_nodes_from(nodes_to_remove)
            return G
            
        elif graph_type == 'random':
            # Random connected graph
            while True:
                G = nx.erdos_renyi_graph(self.n_nodes, p=0.3, seed=None)
                if nx.is_connected(G):
                    return G
                    
        elif graph_type == 'complete':
            return nx.complete_graph(self.n_nodes)
        
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    
    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        """Compute node positions for visualization."""
        return nx.spring_layout(self.graph, seed=42)
    
    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        return list(self.graph.neighbors(node))
    
    def is_base_node(self, node: int) -> bool:
        """Check if node is the base."""
        return node == self.base_node
    
    def is_task_node(self, node: int) -> bool:
        """Check if node is a task node."""
        return node in self.task_nodes
    
    def get_distance(self, node_a: int, node_b: int) -> int:
        """Get shortest path distance between two nodes."""
        try:
            return nx.shortest_path_length(self.graph, node_a, node_b)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_shortest_path(self, node_a: int, node_b: int) -> List[int]:
        """Get shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, node_a, node_b)
        except nx.NetworkXNoPath:
            return []
    
    def __repr__(self) -> str:
        return (f"EnvironmentGraph(nodes={self.n_nodes}, "
                f"edges={self.graph.number_of_edges()}, "
                f"base={self.base_node})")
