# 快速入门指南

## 🚀 安装与设置

### 1. 安装依赖

```bash
cd "/Users/lei/GraduationProject/AdversarialCollection "
pip install -r requirements.txt
```

核心依赖：
- `gymnasium` - RL环境标准接口
- `networkx` - 图结构处理
- `numpy` - 数值计算

### 2. 验证安装

```bash
python test_env.py
```

如果看到 "ALL TESTS PASSED ✓"，说明环境已正确安装！

## 📖 基础使用

### 创建环境

```python
from core import AdversarialCollectionEnv

# 创建环境
env = AdversarialCollectionEnv(
    n_agents=3,          # 智能体数量
    n_nodes=9,           # 节点总数
    graph_type='grid',   # 图类型：grid/random/complete
    grid_size=(3, 3),    # 网格大小
    use_pbrs=True,       # 使用PBRS奖励塑形
    gamma=0.99,          # 折扣因子
    seed=42              # 随机种子
)
```

### 环境交互

```python
# 重置环境
observations, info = env.reset()

# 执行动作
from core.agent import Action, ActionType

# 方式1：使用Action对象
actions = [
    Action(ActionType.MOVE, target_node=1),  # Agent 0: 移动到节点1
    Action(ActionType.COLLECT),               # Agent 1: 采集
    Action(ActionType.SKIP)                   # Agent 2: 跳过
]

# 方式2：使用整数编码（更简单）
actions = [0, 1, 2]  # 环境会自动解码

# 执行步骤
obs, rewards, terminated, truncated, info = env.step(actions)

print(f"Rewards: {rewards}")
print(f"Terminated: {terminated}")
```

### 查看环境状态

```python
# 获取完整环境状态
state = env.get_environment_state()

print(f"当前步数: {state['step']}")
print(f"存活智能体: {sum(1 for a in state['agent_states'] if a['alive'])}")
print(f"剩余效用: {sum(state['node_states'].get(n, (0,0))[0] for n in env.graph.task_nodes)}")

# 渲染环境
env.render(mode='human')
```

## 🎯 运行示例

### 示例1：基础使用

```bash
python examples/basic_usage.py
```

展示：
- 环境创建和配置
- 随机策略交互
- 基础状态监控

### 示例2：监控与日志

```bash
python examples/monitoring_demo.py
```

展示：
- 多回合训练
- 统计数据追踪
- 日志记录功能
- 贪心策略示例

## 📊 环境监控

### 使用EnvironmentMonitor

```python
from utils.monitoring import EnvironmentMonitor, print_environment_info

# 创建监控器
monitor = EnvironmentMonitor(n_agents=3)

# 开始新回合
monitor.start_episode()

# 在每步记录数据
for step in range(max_steps):
    obs, rewards, done, truncated, info = env.step(actions)
    monitor.record_step(step, rewards, info['agent_statistics'], actions)
    
    if done or truncated:
        break

# 结束回合并获取统计
summary = monitor.end_episode()
print(f"回合奖励: {summary['team_reward']}")
print(f"存活率: {summary['survival_rate']}")

# 打印多回合统计
monitor.print_statistics()

# 保存数据
monitor.save_to_file('results.json')
```

### 使用Logger

```python
from utils.logger import create_experiment_logger

# 创建实验日志
logger = create_experiment_logger('my_experiment')

# 记录事件
logger.episode_start(episode=1, config={'n_agents': 3})
logger.action(agent_id=0, action='Move(1)')
logger.event('collection_success', {'agent': 0, 'utility': 15.0})
logger.episode_end(episode=1, summary=summary)
```

## 🎮 动作空间说明

### 动作类型

1. **Move(target_node)** - 移动到相邻节点
   - 必须是当前位置的邻居
   - 移动到基地时自动交付效用

2. **Collect** - 采集当前节点资源
   - 仅在任务节点可用
   - 随机结果：成功(1-risk)或失败(risk)
   - 失败会导致智能体永久失效

3. **Skip** - 原地等待
   - 总是可用
   - 无状态变化，奖励为0

4. **Stay_Dead** - 失效后唯一动作
   - 自动分配给失效智能体

### 动作编码

整数编码方式：
- `0` 到 `len(neighbors)-1`: 移动到对应邻居
- `len(neighbors)`: 采集（如果可用）
- 最后一个编号: 跳过

## 🏆 奖励机制

### 原始奖励

- **返回基地**: `R = U_carried` (交付的效用)
- **采集成功**: `R = 0`
- **采集失败**: `R = -C` (死亡惩罚)
- **其他动作**: `R = 0`

### PBRS奖励塑形

当 `use_pbrs=True` 时：

```
势能函数: Φ(s) = U_carried
塑形奖励: R' = R + γ·Φ(s') - Φ(s)
```

效果：
- 采集成功时立即获得正奖励 ≈ +γ·u_v
- 交付时奖励变为0（因为势能下降）
- 更好的信用分配，加速学习

## 📈 环境配置指南

### 简单环境（测试用）

```python
env = AdversarialCollectionEnv(
    n_agents=2,
    n_nodes=5,
    graph_type='complete',  # 全连接图
    utility_range=(5, 10),
    risk_range=(0.1, 0.2),
    max_steps=50
)
```

### 中等难度

```python
env = AdversarialCollectionEnv(
    n_agents=4,
    n_nodes=16,
    graph_type='grid',
    grid_size=(4, 4),
    utility_range=(10, 30),
    risk_range=(0.2, 0.4),
    use_pbrs=True,
    max_steps=100
)
```

### 高难度（挑战）

```python
env = AdversarialCollectionEnv(
    n_agents=10,
    n_nodes=25,
    graph_type='grid',
    grid_size=(5, 5),
    utility_range=(10, 50),
    risk_range=(0.3, 0.6),
    death_penalty=20.0,
    use_pbrs=True,
    max_steps=200
)
```

## 🔧 常见问题

### Q1: 如何实现自定义策略？

```python
def my_policy(env, agent):
    """自定义策略函数"""
    if not agent.state.is_alive:
        return Action(ActionType.STAY_DEAD)
    
    # 你的策略逻辑
    current_pos = agent.state.position
    neighbors = env.graph.get_neighbors(current_pos)
    
    # 示例：随机选择动作
    import random
    target = random.choice(neighbors)
    return Action(ActionType.MOVE, target)

# 使用策略
actions = [my_policy(env, agent) for agent in env.agents]
env.step(actions)
```

### Q2: 如何保存和加载环境配置？

```python
import json

# 保存配置
config = {
    'n_agents': 3,
    'n_nodes': 9,
    'graph_type': 'grid',
    'grid_size': (3, 3),
    # ... 其他参数
}
with open('env_config.json', 'w') as f:
    json.dump(config, f)

# 加载配置
with open('env_config.json', 'r') as f:
    config = json.load(f)
env = AdversarialCollectionEnv(**config)
```

### Q3: 如何与RL库集成？

环境完全兼容Gymnasium接口，可以直接与主流RL库使用：

```python
# Stable-Baselines3 示例（需要适配多智能体）
# 或使用RLlib等支持多智能体的框架

# 简化为单智能体问题的包装器示例
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env, agent_id=0):
        super().__init__(env)
        self.agent_id = agent_id
        # 修改observation_space和action_space为单智能体
```

## 📚 下一步

1. **阅读完整文档**: 查看 `README.md`
2. **运行示例**: 尝试 `examples/` 中的所有示例
3. **实现策略**: 基于环境实现你的RL算法
4. **可视化**: 添加可视化模块（可选）
5. **VQ-HC-SAC**: 实现论文中的算法（高级）

## 💡 提示

- 使用 `seed` 参数确保可重复性
- 使用 `use_pbrs=True` 加速学习
- 监控 `survival_rate` 和 `utility_delivered` 评估性能
- 从简单环境开始，逐步增加难度

## 🤝 需要帮助？

- 查看示例代码：`examples/`
- 运行测试：`python test_env.py`
- 查看源码注释：所有模块都有详细文档

祝您使用愉快！🎉
