# VQ-HC-SAC 训练和分析工作流

## 📋 概述

训练和分析分为两个独立的步骤，由两个解耦的脚本完成：

1. **train_demo.py** - 训练脚本
2. **analyze_training.py** - 分析脚本

---

## 🔄 完整工作流

### Step 1: 训练模型

```bash
python examples/train_demo.py
```

**功能：**
- ✅ 创建训练环境
- ✅ 配置VQ-HC-SAC训练器
- ✅ 执行训练循环
- ✅ 保存训练数据

**输出文件：**
```
checkpoints/demo/
├── quick_demo.pt        # 模型checkpoint
├── history.pkl          # 训练历史（回报、损失等）
└── env_config.pkl       # 环境配置（确保测试环境一致）
```

**训练参数：**
- 默认训练步数: 2000 步（快速演示）
- 环境: 3个智能体，6个节点（星型拓扑）
- 角色数: 2
- 批量大小: 32

### Step 2: 分析结果

```bash
python examples/analyze_training.py
```

**功能：**
- ✅ 加载checkpoint和训练历史
- ✅ 打印训练统计摘要
- ✅ 生成训练曲线图
- ✅ 测试模型性能（10个episode）
- ✅ 统计效用收集情况

**输出文件：**
```
figures/
└── training_curves.png  # 训练曲线图（4个子图）
```

**分析内容：**
1. Checkpoint信息（训练步数、配置等）
2. 训练摘要（回报统计、Episode长度、损失）
3. 训练曲线图（回报、长度、评估、损失）
4. 模型测试结果（平均回报、效用收集率）

---

## 📊 输出示例

### 训练输出 (train_demo.py)

```
======================================================================
VQ-HC-SAC 完整训练流程演示
======================================================================

======================================================================
创建环境...
======================================================================
✓ 环境创建成功:
  智能体数量: 3
  节点数量: 6
  最大步数: 50
  PBRS: True

======================================================================
创建VQ-HC-SAC训练器...
======================================================================
✓ 训练器创建成功
  设备: cuda
  角色数: 2
  批量大小: 32

提示: 这是一个快速演示，只训练2000步
      完整训练建议100K-1M步

======================================================================
开始训练 (共 2000 步)...
======================================================================
...
✓ 训练完成!
  总步数: 2000
  总episode: XX

✓ 模型已保存: ./checkpoints/demo/quick_demo.pt
✓ 训练历史已保存: ./checkpoints/demo/history.pkl
✓ 环境配置已保存: ./checkpoints/demo/env_config.pkl

======================================================================
训练完成!
======================================================================

保存的文件:
  模型: ./checkpoints/demo/quick_demo.pt
  训练历史: ./checkpoints/demo/history.pkl
  环境配置: ./checkpoints/demo/env_config.pkl

下一步:
  1. 运行 analyze_training.py 查看训练结果和测试模型
  2. 增加训练步数进行完整训练 (100K-1M步)
  3. 调整超参数优化性能
======================================================================
```

### 分析输出 (analyze_training.py)

```
======================================================================
VQ-HC-SAC 训练结果分析
======================================================================

使用checkpoint: checkpoints\demo\quick_demo.pt

======================================================================
Checkpoint信息
======================================================================

训练进度:
  总步数: 2000
  总Episode数: XX
  最佳评估回报: X.XX

配置:
  角色数: 2
  嵌入维度: 32
  批量大小: 32
  学习率 (Actor): 0.0003
  学习率 (Critic): 0.0003

✓ 环境配置已加载: checkpoints\demo\env_config.pkl

✓ 训练历史已加载: checkpoints\demo\history.pkl

======================================================================
训练摘要
======================================================================

回报统计:
  总Episode数: XX
  平均回报: X.XX ± X.XX
  最小回报: X.XX
  最大回报: X.XX

最近100个Episode:
  平均回报: X.XX ± X.XX

Episode长度统计:
  平均长度: XX.X ± X.X

评估回报:
  评估次数: X
  最佳评估回报: X.XX
  最终评估回报: X.XX

Loss统计:
  更新次数: XXX
  最终Critic Loss: X.XX
  最终Actor Loss: X.XX
  最终VQ Loss: X.XX

✓ 训练曲线已保存: figures/training_curves.png

======================================================================
测试训练好的模型
======================================================================
✓ 模型加载成功: checkpoints\demo\quick_demo.pt

运行 10 个测试Episode...
  Episode 1: 回报=X.XX, 步数=XX, 收集效用=XX.XX/XX.XX
  Episode 2: 回报=X.XX, 步数=XX, 收集效用=XX.XX/XX.XX
  ...
  Episode 10: 回报=X.XX, 步数=XX, 收集效用=XX.XX/XX.XX

----------------------------------------------------------------------
测试结果统计
----------------------------------------------------------------------

回报:
  平均: X.XX ± X.XX
  最小: X.XX
  最大: X.XX

Episode长度:
  平均: XX.X ± X.X

效用收集:
  平均收集: XX.XX ± X.XX
  最佳收集: XX.XX
  收集率: XX.X%

✓ 分析完成!
```

---

## 🎯 为什么要解耦？

### 优势

1. **职责分离**
   - 训练脚本专注于训练和保存
   - 分析脚本专注于加载和分析

2. **灵活性**
   - 可以多次分析同一个checkpoint
   - 可以比较不同checkpoint的性能
   - 不需要重新训练就能查看结果

3. **可维护性**
   - 代码结构清晰
   - 便于修改和扩展
   - 避免重复代码

4. **效率**
   - 训练和分析可独立运行
   - 分析不会占用训练时间
   - 可以并行进行多次分析

---

## 🔧 自定义训练

### 修改训练参数

编辑 `train_demo.py` 中的配置：

```python
history = quick_training(trainer, total_steps=100000)  # 增加训练步数
```

### 修改环境配置

```python
env = AdversarialCollectionEnv(
    n_agents=5,              # 更多智能体
    n_nodes=10,              # 更多节点
    max_steps=100,           # 更多步数
    utility_range=(10, 30),  # 更高效用
    risk_range=(0.05, 0.3)   # 更低风险
)
```

### 修改算法配置

```python
config = VQHCSACConfig(
    n_roles=3,               # 更多角色
    embedding_dim=64,        # 更大嵌入
    batch_size=256,          # 更大批量
    buffer_size=100000,      # 更大缓冲区
    warmup_steps=10000       # 更长预热
)
```

---

## 🔍 高级分析

### 比较多个checkpoint

修改 `analyze_training.py`，对比不同训练阶段：

```python
checkpoints = [
    'checkpoints/demo/checkpoint_1000.pt',
    'checkpoints/demo/checkpoint_5000.pt',
    'checkpoints/demo/checkpoint_10000.pt'
]

for ckpt in checkpoints:
    print(f"\n分析: {ckpt}")
    analyze_checkpoint(ckpt)
    test_results = test_model_performance(ckpt, env_config, n_episodes=20)
```

### 导出训练数据

```python
import pandas as pd

# 加载历史
with open('checkpoints/demo/history.pkl', 'rb') as f:
    history = pickle.load(f)

# 导出为CSV
df = pd.DataFrame({
    'episode': range(len(history['episode_returns'])),
    'return': history['episode_returns'],
    'length': history['episode_lengths']
})
df.to_csv('training_data.csv', index=False)
```

---

## 📚 相关文档

- **ANALYSIS_GUIDE.md** - 详细的分析指南
- **examples/README.md** - 所有示例脚本说明
- **PHASE3_SUMMARY.md** - Phase 3实现总结

---

## ✅ 快速检查清单

训练前：
- [ ] 已安装所有依赖 (`pip install -r requirements.txt`)
- [ ] 环境测试通过 (`quick_integration_test.py`)

训练后：
- [ ] checkpoint文件已生成 (`quick_demo.pt`)
- [ ] 训练历史已保存 (`history.pkl`)
- [ ] 环境配置已保存 (`env_config.pkl`)

分析前：
- [ ] 所有训练输出文件存在
- [ ] 已安装matplotlib (`pip install matplotlib`)

分析后：
- [ ] 训练曲线图已生成 (`training_curves.png`)
- [ ] 模型测试完成
- [ ] 效用收集率统计完成

---

**祝训练和分析顺利！** 🚀
