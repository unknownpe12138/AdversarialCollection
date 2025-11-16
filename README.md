# Adversarial Multi-Agent Value Collection Environment

基于强化学习的多智能体对抗环境价值采集系统。

## 📋 项目概述

本项目实现了一个多智能体马尔可夫决策过程(MDP)环境，用于研究在对抗性环境中的价值采集问题。环境建模为无向图结构，智能体需要在风险和收益之间做出权衡决策。

### 核心特性

- ✅ **标准Gymnasium接口**: 完全兼容OpenAI Gym/Gymnasium标准
- ✅ **多智能体支持**: 支持N个智能体并发交互
- ✅ **图拓扑结构**: 基于NetworkX的灵活图结构
- ✅ **随机环境动态**: 基于风险的随机采集机制
- ✅ **奖励塑形**: 支持PBRS(Potential-Based Reward Shaping)
- ✅ **状态监控**: 完整的环境状态追踪和可视化接口

## 🎯 问题建模

### MDP定义: M = ⟨S, A, P, R, γ⟩

#### 状态空间 S

全局状态 `s = (s_env, s_1, ..., s_N)`:

- **环境状态** `s_env = {(u_v, r_v) | v ∈ V\{v_0}}`: 所有任务节点的效用和风险
- **个体状态** `s_k = (v_k, U_k^carried, is_alive)`: 位置、携带效用、存活状态

#### 动作空间 A

联合动作 `a = (a_1, ..., a_N)`:

- **Move(v_j)**: 移动到相邻节点
- **Collect**: 在当前节点采集(随机结果)
- **Skip**: 原地等待
- **Stay_Dead**: 失效智能体的唯一动作

#### 状态转移 P

- **移动**: 确定性转移
- **采集**: 随机转移
  - 成功(概率 1-r_v): 获得效用，清空节点
  - 失败(概率 r_v): 智能体永久失效

#### 奖励函数 R

- **返回基地**: R_k = U_k^carried (交付携带的效用)
- **采集成功**: R_k = 0
- **采集失败**: R_k = -C (死亡惩罚)
- **其他动作**: R_k = 0

#### PBRS奖励塑形

势能函数: `Φ(s_k) = U_k^carried`

塑形奖励: `R'_k = R_k + γ·Φ(s'_k) - Φ(s_k)`

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from core import AdversarialCollectionEnv

# 创建环境
env = AdversarialCollectionEnv(
    n_agents=3,          # 智能体数量
    n_nodes=9,           # 节点总数
    graph_type='grid',   # 图类型
    grid_size=(3, 3),    # 网格大小
    use_pbrs=True,       # 使用PBRS奖励塑形
    gamma=0.99           # 折扣因子
)

# 重置环境
observations, info = env.reset()

# 执行动作
actions = [...]  # 智能体动作列表
observations, rewards, terminated, truncated, info = env.step(actions)

# 获取环境状态
state = env.get_environment_state()

# 渲染环境
env.render(mode='human')
```

### 运行示例

```bash
python examples/basic_usage.py
```

## 📁 项目结构

```
AdversarialCollection/
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── environment.py         # 主环境类
│   ├── agent.py              # 智能体定义
│   ├── graph.py              # 图结构管理
│   └── rewards.py            # 奖励机制
├── examples/                  # 使用示例
│   └── basic_usage.py        # 基础使用示例
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明
```

## 🔧 环境配置参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `n_agents` | int | 3 | 智能体数量 |
| `n_nodes` | int | 10 | 节点总数 |
| `graph_type` | str | 'grid' | 图类型: grid/random/complete |
| `grid_size` | Tuple | None | 网格大小(行,列) |
| `utility_range` | Tuple | (5.0, 20.0) | 节点效用范围 |
| `risk_range` | Tuple | (0.1, 0.5) | 节点风险范围 |
| `death_penalty` | float | 10.0 | 死亡惩罚 C |
| `gamma` | float | 0.99 | 折扣因子 |
| `use_pbrs` | bool | True | 是否使用PBRS |
| `max_steps` | int | 200 | 最大步数 |

## 🎮 环境接口

### 标准Gymnasium接口

```python
# 重置环境
observations, info = env.reset(seed=42)

# 执行步骤
observations, rewards, terminated, truncated, info = env.step(actions)

# 渲染
env.render(mode='human')

# 关闭
env.close()
```

### 扩展接口

```python
# 获取完整环境状态
state = env.get_environment_state()
# 返回: {step, episode, node_states, agent_states, graph_info}

# 获取智能体统计
for agent in env.agents:
    stats = agent.get_statistics()
    # 返回: {agent_id, is_alive, position, carried_utility, 
    #        total_collected, total_delivered, success_rate, steps_alive}
```

## 📊 监控功能

环境提供多层次的状态监控：

1. **实时状态**: 每步的完整环境状态
2. **智能体统计**: 个体性能指标
3. **团队指标**: 聚合的团队表现
4. **历史追踪**: 完整的交互历史

## 🔬 RL训练集成

环境完全兼容主流RL库：

- **Stable-Baselines3**: 标准Gymnasium接口
- **RLlib**: 多智能体扩展
- **Custom algorithms**: 灵活的状态/动作接口

## 🎯 下一步开发

- [ ] 可视化模块(Matplotlib/Pygame)
- [ ] Web监控界面(Streamlit/Flask)
- [ ] VQ-HC-SAC算法实现
- [ ] 分布式训练支持
- [ ] 性能基准测试

## 📝 引用

```bibtex
@misc{adversarial_collection_env,
  title={Adversarial Multi-Agent Value Collection Environment},
  year={2025},
  author={Your Name}
}
```

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request!
