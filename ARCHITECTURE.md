# 系统架构文档

## 🏗️ 整体架构

```
AdversarialCollection/
├── core/                          # 核心环境实现
│   ├── environment.py            # 主环境类 (Gymnasium接口)
│   ├── agent.py                  # 智能体状态与动作
│   ├── graph.py                  # 图结构管理
│   └── rewards.py                # 奖励计算与塑形
├── utils/                         # 工具模块
│   ├── monitoring.py             # 环境监控
│   └── logger.py                 # 日志系统
└── examples/                      # 使用示例
    ├── basic_usage.py            # 基础示例
    └── monitoring_demo.py        # 监控示例
```

## 📦 核心模块详解

### 1. Environment (`core/environment.py`)

**主类**: `AdversarialCollectionEnv`

**继承**: `gymnasium.Env`

**职责**:
- 实现标准Gym接口 (`reset`, `step`, `render`, `close`)
- 管理MDP状态转移 `P(s'|s,a)`
- 计算奖励 `R(s,a)`
- 协调多智能体交互

**核心方法**:

```python
class AdversarialCollectionEnv(gym.Env):
    def reset(seed, options) -> (observations, info)
        """初始化环境状态"""
        
    def step(actions) -> (observations, rewards, terminated, truncated, info)
        """执行一步交互"""
        
    def _execute_action(agent, action) -> reward
        """执行单个智能体动作并返回奖励"""
        
    def get_environment_state() -> dict
        """获取完整环境状态（用于监控）"""
```

**状态表示**:
- **全局状态**: `s = (s_env, s_1, ..., s_N)`
- **环境状态**: `s_env = {node_id: (utility, risk)}`
- **个体状态**: `s_k = (position, carried_utility, is_alive)`

**动作空间**:
- 多智能体元组: `(a_1, ..., a_N)`
- 每个智能体: 离散动作空间

**观察空间**:
- 多智能体元组观察
- 每个观察: 连续向量 `[agent_info, env_info]`

---

### 2. Agent (`core/agent.py`)

**主类**: `Agent`, `AgentState`, `Action`

**职责**:
- 维护智能体状态
- 管理有效动作集合
- 追踪统计信息

**核心组件**:

```python
@dataclass
class AgentState:
    """智能体状态 s_k"""
    position: int              # 当前节点 v_k
    carried_utility: float     # 携带效用 U_k^carried
    is_alive: bool            # 存活状态

class Agent:
    """智能体实体"""
    def get_valid_actions(neighbors, node_utility, is_base) -> List[Action]
        """获取当前有效动作"""
        
    def move(target_position)
        """移动到目标位置"""
        
    def collect(utility)
        """采集资源"""
        
    def deliver() -> float
        """交付资源，返回交付量"""
        
    def die()
        """标记为失效"""
        
    def get_statistics() -> dict
        """获取统计信息"""
```

**Action类型**:
```python
class ActionType(Enum):
    MOVE = "move"           # 移动
    COLLECT = "collect"     # 采集
    SKIP = "skip"          # 跳过
    STAY_DEAD = "stay_dead" # 失效后
```

---

### 3. Graph (`core/graph.py`)

**主类**: `EnvironmentGraph`

**职责**:
- 管理环境拓扑结构 `G = (V, E)`
- 提供图操作接口
- 计算路径和距离

**核心方法**:

```python
class EnvironmentGraph:
    """基于NetworkX的图结构"""
    
    def __init__(n_nodes, edge_list, graph_type, grid_size)
        """创建图: grid/random/complete"""
        
    def get_neighbors(node) -> List[int]
        """获取邻居节点"""
        
    def get_distance(node_a, node_b) -> int
        """计算最短路径距离"""
        
    def get_shortest_path(node_a, node_b) -> List[int]
        """获取最短路径"""
```

**图类型**:
- **Grid**: 网格拓扑（可配置行列）
- **Random**: 随机连通图（Erdős–Rényi）
- **Complete**: 全连接图
- **Custom**: 自定义边列表

---

### 4. Rewards (`core/rewards.py`)

**主类**: `RewardCalculator`, `PBRSRewardShaper`

**职责**:
- 计算原始奖励 `R(s,a)`
- 实现PBRS奖励塑形 `R'(s,a)`
- 聚合团队奖励

**核心组件**:

```python
class RewardCalculator:
    """原始奖励计算"""
    
    def calculate_move_reward(to_base, carried_utility) -> float
        """移动奖励（仅返回基地时非零）"""
        
    def calculate_collect_reward(success) -> float
        """采集奖励（成功=0, 失败=-C）"""
        
    def calculate_skip_reward() -> float
        """跳过奖励（总是0）"""

class PBRSRewardShaper:
    """势能奖励塑形"""
    
    def potential(agent_state) -> float
        """势能函数: Φ(s) = U_carried"""
        
    def shape_reward(original_reward, state_before, state_after) -> float
        """塑形奖励: R' = R + γ·Φ(s') - Φ(s)"""
```

**PBRS原理**:
```
势能函数: Φ(s_k) = U_k^carried
塑形奖励: R'_k = R_k + γ·Φ(s'_k) - Φ(s_k)

效果：
- Collect成功: R=0 → R'≈+γ·u_v (立即正奖励)
- Deliver: R=U → R'≈0 (势能下降抵消)
- 保证最优策略不变（Ng et al. 1999）
```

---

## 🔧 工具模块

### 5. Monitoring (`utils/monitoring.py`)

**主类**: `EnvironmentMonitor`, `EpisodeTracker`

**职责**:
- 追踪回合统计
- 聚合多回合数据
- 导出分析结果

```python
class EpisodeTracker:
    """单回合追踪"""
    def record_step(step, rewards, agent_states, actions)
    def get_summary() -> dict

class EnvironmentMonitor:
    """多回合监控"""
    def start_episode()
    def end_episode() -> summary
    def get_statistics(last_n) -> dict
    def save_to_file(filepath)
```

**追踪指标**:
- 回合长度
- 累计奖励（个体/团队）
- 采集/交付效用
- 存活率
- 动作分布

---

### 6. Logger (`utils/logger.py`)

**主类**: `Logger`

**职责**:
- 记录环境事件
- 输出到文件/控制台
- 支持多级日志

```python
class Logger:
    def episode_start(episode, config)
    def episode_end(episode, summary)
    def action(agent_id, action, details)
    def event(event_type, details)
    def error/info/debug/warning(message)
```

---

## 🔄 数据流

### 初始化流程

```
1. 创建环境
   ↓
2. 构建图结构 (EnvironmentGraph)
   ↓
3. 创建智能体 (Agent × N)
   ↓
4. 初始化奖励计算器 (RewardCalculator, PBRSRewardShaper)
   ↓
5. 定义观察/动作空间
```

### 交互循环

```
reset()
   ↓
   ├─ 初始化节点状态 {(u_v, r_v)}
   ├─ 重置智能体状态
   └─ 返回初始观察

step(actions)
   ↓
   ├─ 对每个智能体：
   │   ├─ 验证动作
   │   ├─ 执行动作 (移动/采集/跳过)
   │   ├─ 更新状态
   │   └─ 计算奖励
   │
   ├─ 应用PBRS塑形（如果启用）
   ├─ 检查终止条件
   └─ 返回 (obs, rewards, terminated, truncated, info)

终止条件
   ↓
   ├─ 所有节点效用清零, OR
   ├─ 所有智能体失效, OR
   └─ 达到最大步数
```

### 状态转移

```
Move(v_j):
   确定性: v_k ← v_j
   如果 v_j == base:
      R_k = U_k^carried
      U_k^carried ← 0

Collect:
   随机: p = 1 - r_v
   
   成功 (p):
      U_k^carried += u_v
      (u_v, r_v) ← (0, 0)
      R_k = 0
   
   失败 (1-p):
      is_alive ← False
      U_k^carried ← 0
      R_k = -C

Skip:
   无变化, R_k = 0
```

---

## 🎯 设计模式

### 1. 组合模式
- `AdversarialCollectionEnv` 组合了 `Graph`, `Agent`, `RewardCalculator`
- 各组件独立可测试

### 2. 策略模式
- `RewardCalculator` vs `PBRSRewardShaper`
- 可切换奖励计算策略

### 3. 观察者模式
- `EnvironmentMonitor` 观察环境状态
- 不侵入核心逻辑

### 4. 工厂模式
- 图创建工厂 (`_create_graph`)
- 支持多种图类型

---

## 🔌 扩展点

### 1. 自定义图拓扑

```python
# 方式1: 提供边列表
edge_list = [(0,1), (1,2), (2,3), ...]
env = AdversarialCollectionEnv(edge_list=edge_list)

# 方式2: 继承EnvironmentGraph
class CustomGraph(EnvironmentGraph):
    def _create_graph(self, graph_type, grid_size):
        # 自定义图生成逻辑
        return custom_graph
```

### 2. 自定义奖励函数

```python
class CustomRewardCalculator(RewardCalculator):
    def calculate_collect_reward(self, success):
        # 自定义采集奖励
        return custom_reward

env = AdversarialCollectionEnv(...)
env.reward_calculator = CustomRewardCalculator()
```

### 3. 添加可视化

```python
# 在environment.py中扩展render方法
def render(self, mode='human'):
    if mode == 'human':
        # 使用matplotlib/pygame绘制
        visualize_graph(self.graph, self.agents, self.node_states)
    elif mode == 'rgb_array':
        return render_to_array(...)
```

### 4. 多智能体包装器

```python
# 包装为RLlib MultiAgentEnv
from ray.rllib.env import MultiAgentEnv

class RLlibWrapper(MultiAgentEnv):
    def __init__(self, config):
        self.env = AdversarialCollectionEnv(**config)
        # 适配接口...
```

---

## 📊 性能考虑

### 时间复杂度
- `reset()`: O(N + V)
- `step()`: O(N) - 线性于智能体数
- 图操作: O(V + E) - 使用NetworkX缓存

### 空间复杂度
- 状态存储: O(N + V)
- 图结构: O(V + E)
- 监控数据: O(T × N) - T为步数

### 优化建议
- 使用`max_steps`限制回合长度
- 大规模实验时关闭详细日志
- 考虑批量环境并行

---

## 🧪 测试策略

### 单元测试
```python
# 测试图创建
def test_graph_creation():
    graph = EnvironmentGraph(n_nodes=5, graph_type='complete')
    assert graph.n_nodes == 5
    assert nx.is_connected(graph.graph)

# 测试智能体动作
def test_agent_collect():
    agent = Agent(0)
    agent.collect(10.0)
    assert agent.state.carried_utility == 10.0
```

### 集成测试
```python
# 测试完整交互
def test_full_episode():
    env = AdversarialCollectionEnv(...)
    obs, info = env.reset()
    
    for _ in range(10):
        actions = random_policy(env)
        obs, rewards, done, _, info = env.step(actions)
        if done:
            break
    
    assert env.current_step > 0
```

---

## 📝 代码规范

### 命名约定
- 类: `PascalCase`
- 函数/方法: `snake_case`
- 常量: `UPPER_CASE`
- 私有方法: `_leading_underscore`

### 类型提示
- 所有公开API使用类型提示
- 复杂类型使用 `typing` 模块

### 文档字符串
- 所有类和公开方法包含docstring
- 格式: Google style

---

## 🔮 未来规划

### 短期 (v1.1)
- [ ] Matplotlib可视化
- [ ] 更多图类型（小世界、无标度）
- [ ] 性能优化

### 中期 (v1.5)
- [ ] Pygame实时渲染
- [ ] Web监控界面
- [ ] 分布式训练支持

### 长期 (v2.0)
- [ ] VQ-HC-SAC算法实现
- [ ] 预训练模型库
- [ ] 基准测试套件

---

## 📚 参考资料

### 理论基础
- Ng et al. (1999): "Policy invariance under reward transformations"
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"

### 技术框架
- Gymnasium: https://gymnasium.farama.org/
- NetworkX: https://networkx.org/
- NumPy: https://numpy.org/

---

**文档版本**: 1.0  
**最后更新**: 2025-11-16  
**维护者**: Lei
