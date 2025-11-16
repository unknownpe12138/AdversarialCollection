"""
Quick test script to verify environment functionality.
Run: python test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core import AdversarialCollectionEnv
    from core.agent import Action, ActionType
    
    print("✓ Successfully imported core modules")
    
    # Test 1: Environment creation
    print("\n--- Test 1: Environment Creation ---")
    env = AdversarialCollectionEnv(
        n_agents=2,
        n_nodes=5,
        graph_type='complete',
        seed=42
    )
    print(f"✓ Environment created: {env.n_agents} agents, {env.n_nodes} nodes")
    
    # Test 2: Reset
    print("\n--- Test 2: Environment Reset ---")
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observations shape: {len(obs)} agents")
    print(f"  Info keys: {list(info.keys())}")
    
    # Test 3: Single step with manual actions
    print("\n--- Test 3: Step Execution ---")
    # Create simple actions: both agents skip
    actions = [Action(ActionType.SKIP), Action(ActionType.SKIP)]
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"✓ Step executed successfully")
    print(f"  Rewards: {rewards}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    # Test 4: Environment state
    print("\n--- Test 4: Environment State ---")
    state = env.get_environment_state()
    print(f"✓ State retrieved successfully")
    print(f"  Current step: {state['step']}")
    print(f"  Agents alive: {sum(1 for a in state['agent_states'] if a['alive'])}/{env.n_agents}")
    print(f"  Total utility: {sum(state['node_states'].get(n, (0,0))[0] for n in env.graph.task_nodes):.2f}")
    
    # Test 5: Graph structure
    print("\n--- Test 5: Graph Structure ---")
    print(f"✓ Graph info:")
    print(f"  Base node: {env.graph.base_node}")
    print(f"  Task nodes: {env.graph.task_nodes[:3]}..." if len(env.graph.task_nodes) > 3 else f"  Task nodes: {env.graph.task_nodes}")
    print(f"  Edges: {env.graph.graph.number_of_edges()}")
    
    # Test 6: Run a few random steps
    print("\n--- Test 6: Multi-step Execution ---")
    import numpy as np
    np.random.seed(42)
    
    env.reset()
    for i in range(5):
        # Random actions
        actions = []
        for agent in env.agents:
            if agent.state.is_alive:
                valid_actions = agent.get_valid_actions(
                    env.graph.get_neighbors(agent.state.position),
                    env.node_states.get(agent.state.position, (0,0))[0],
                    env.graph.is_base_node(agent.state.position)
                )
                actions.append(np.random.choice(valid_actions))
            else:
                actions.append(Action(ActionType.STAY_DEAD))
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"✓ Completed {i+1} steps successfully")
    
    # Test 7: Render
    print("\n--- Test 7: Rendering ---")
    env.render(mode='human')
    print("✓ Rendering successful")
    
    # Final summary
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
    print("\nEnvironment is ready for RL training!")
    print("Next steps:")
    print("  1. Run examples/basic_usage.py for detailed demo")
    print("  2. Integrate with your RL algorithm")
    print("  3. Customize environment parameters")
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"✗ Test Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
