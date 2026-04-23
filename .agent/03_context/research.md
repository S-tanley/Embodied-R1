# Embodied-R1 深度研究报告

> **项目**: Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation  
> **发表**: ICLR 2026  
> **模型**: IffYuan/Embodied-R1-3B-v1 (Hugging Face)  
> **数据集**: IffYuan/Embodied-R1-Dataset (Hugging Face)  
> **论文**: [arXiv:2508.13998](http://arxiv.org/abs/2508.13998)  
> **代码**: https://github.com/pickxiguapi/Embodied-R1  

---

## 1. 项目概述

**Embodied-R1** 是一个面向通用机器人操作的 3B 参数视觉-语言模型（Vision-Language Model, VLM）。其核心目标是通过创新的 **"Pointing"（指向）机制** 和 **Reinforced Fine-Tuning (RFT)** 训练方法，弥合机器人领域中 "seeing-to-doing"（从感知到执行）的鸿沟，实现卓越的零样本泛化能力。

该项目已被 **ICLR 2026** 接收，作者团队来自多个研究机构。项目完全开源，包括预训练模型、训练数据集、训练代码和评测基准。

### 1.1 核心能力

Embodied-R1 将机器人操作任务统一为 **2D 坐标点预测**问题，支持四种核心指向能力：

| 能力缩写 | 全称 | 说明 | 典型应用 |
|---------|------|------|---------|
| **VTG** | Visual Trace Generation | 生成以被操作物体为中心的轨迹路点（固定8个点） | 机器人运动规划 |
| **RRG** | Region Referring Grounding | 标记操作任务的目标位置 | 目标放置定位 |
| **REG** | Referring Expression Grounding | 根据自然语言描述定位物体区域 | 物体识别与定位 |
| **OFG** | Object Affordance Grounding | 根据功能描述定位物体的 affordance 区域 | 功能区域识别 |

### 1.2 统一输出范式

所有任务共享统一的结构化输出格式：

```
<think> reasoning process here </think>
<answer><point>[[x1, y1], [x2, y2], ...]</point></answer>
```

这种 **Think-Then-Answer** 范式使得模型在进行像素级定位之前，先进行显式的空间推理，从而提高了定位的准确性和可解释性。

---

## 2. 研究背景与动机

### 2.1 机器人领域的 "Seeing-to-Doing" 鸿沟

传统的机器人操作管线通常分为三个阶段：
1. **感知（Seeing）**：视觉模型识别物体、分割场景
2. **推理（Reasoning）**：理解任务指令、空间关系
3. **执行（Doing）**：生成具体的操作轨迹或控制信号

问题在于，感知和推理模块往往产生高层语义输出（如边界框、类别标签），而执行模块需要低层几何信息（如像素坐标、3D 位姿）。这种 **语义-几何鸿沟** 导致系统复杂、误差累积、泛化能力差。

### 2.2 Embodied-R1 的解决方案

Embodied-R1 的核心思想是 **将操作任务统一为指向（Pointing）问题**：
- 不输出边界框或分割掩码，而是输出精确的 2D/3D 坐标点
- 通过显式推理（`<think>`）建立从自然语言指令到像素坐标的映射
- 使用强化学习（GRPO）训练模型，直接优化指向精度

这种方法的优势在于：
- **统一性**：所有操作任务都可以用点预测表示
- **简洁性**：无需复杂的后处理管线
- **泛化性**：RL 训练使模型能够零样本泛化到新场景

---

## 3. 技术方案详解

### 3.1 基础模型架构

Embodied-R1 基于 **Qwen2.5-VL-3B-Instruct** 构建：
- **视觉编码器**：ViT 架构，处理输入图像
- **视觉-语言投影器**：将视觉特征映射到语言空间
- **语言模型**：3B 参数 Decoder-only Transformer
- **多模态位置编码**：支持 3D 旋转位置编码（mRoPE），处理图像/视频 token 的特殊位置

模型使用 `transformers>=4.49.0` 和 `vllm>=0.7.3`，支持 Flash Attention 2 加速。

### 3.2 Pointing 机制

Pointing 是 Embodied-R1 的核心创新。它将复杂的机器人操作任务分解为两种基本能力：

#### (1) 定位能力（Where）
- **REG**："bring me the camel model" → 定位骆驼模型的像素坐标
- **OFG**："loosening stuck bolts" → 定位适合拧松螺栓的 affordance 区域

#### (2) 轨迹能力（How）
- **RRG**："put pepper in pan" → 标记胡椒应该被放置的目标位置
- **VTG**："put the red block on top of the yellow block" → 生成红方块从起点到终点的8个轨迹路点

### 3.3 推理-回答分离范式

```
输入: [图像] + [任务指令]
      ↓
<think> 空间推理过程（自然语言）
  - 识别目标物体
  - 分析空间关系
  - 确定操作策略
</think>
      ↓
<answer><point>[[x1,y1], [x2,y2], ...]</point></answer>
      ↓
输出: 精确的像素坐标点
```

这种分离使得：
- 推理过程可解释、可调试
- 回答部分结构化，便于程序化解析
- RL 奖励函数可以分别评估推理质量和定位精度

---

## 4. 训练方法：两阶段 RFT

Embodied-R1 采用 **两阶段强化学习训练策略**，基于 **GRPO (Group Relative Policy Optimization)** 算法。

### 4.1 训练框架：verl/

项目使用基于 ByteDance **veRL** 框架修改的分布式 RL 训练框架，核心技术栈：
- **Ray**：分布式计算调度
- **FSDP (Fully Sharded Data Parallel)**：模型分片训练
- **vLLM**：高吞吐序列生成（Rollout）
- **HybridEngine**：训练（FSDP）和推理（vLLM）之间的无缝权重切换

### 4.2 阶段一：空间语义理解（Stage 1）

**目标**：建立空间推理和问答能力的基础。

**训练数据**（QA 任务）：

| 数据集 | 任务类型 | 规模（训练/测试） | 说明 |
|--------|---------|----------------|------|
| whatsup_rft_spatial_qa | spatial_qa | 4,000 / 138 | 空间关系问答 |
| SAT_rft_spatial_qa | spatial_qa | 80,000 / 4,000 | 空间推理问答 |
| ViRL_rft_general_qa | general_qa | 17,831 / - | 通用视觉推理问答 |

**关键超参数**：
- 算法：GRPO (`adv_estimator: grpo`)
- KL 约束：`kl_coef=1e-2`, `kl_penalty: low_var_kl`
- Actor 学习率：`lr=1e-6`
- 全局批次大小：128
- Rollout 采样：`n=5`（每组采样5个 response）
- Tensor Parallelism：2
- 训练轮数：`total_episodes=15`
- GPU：8 卡

**奖励函数**（Stage 1）：
```
reward = 0.1 * thinking_format_reward + 0.9 * accuracy_reward
```

### 4.3 阶段二：像素级定位（Stage 2）

**目标**：在 Stage 1 的基础上，强化像素级指向能力。

**训练数据**（Point 任务）：

| 数据集 | 任务类型 | 规模（训练/测试） | 说明 |
|--------|---------|----------------|------|
| robopoint_rft_point_ref | point_ref | 40,000 / 2,000 | 点参考定位 |
| FSD_points_rft_fsd_free_point | fsd_free_point | 32,790 / 300 | 自由点预测 |
| FSD_visual_trace_rft_fsd_visual_trace | fsd_visual_trace | 32,790 / 300 | 视觉轨迹生成 |
| roborefit_rft_point_rec | point_rec | 35,000 / 1,000 | 点推荐/定位 |
| refcoco_rft_point_rec | point_rec | 20,000 / 189 | 指代表达定位 |
| handal_rft_grounding_rec | grounding_rec | 40,000 / 1,000 | affordance 定位 |

Stage 2 从 Stage 1 的 checkpoint 继续训练，配置与 Stage 1 基本一致，但更换了数据集。

### 4.4 奖励函数设计（核心）

奖励函数文件：`verl/utils/reward_score/embodiedr1.py`

奖励函数是多任务统一的，根据 `ground_truth` 中的 `<type>` 标签自动路由到对应的评估逻辑。

#### 通用格式奖励

| 奖励项 | 评估内容 | 权重 |
|--------|---------|------|
| `thinking_format_reward` | 是否匹配 `<think>...</think>\s*<answer>...</answer>` | 0.1 |
| `point_format_reward` | 是否包含 `<point>...</point>` | 0.1 |

#### 各任务奖励公式

**spatial_qa / general_qa**（Stage 1）：
```
overall = 0.1 * format + 0.9 * accuracy
```
- `accuracy`：使用 `mathruler.grader.grade_answer` 评估答案正确性

**point_ref**（RoboPoint 定位）：
```
overall = 0.1 * format + 0.1 * point_fmt + 0.8 * point_l1_reward
```
- `point_l1_reward`：预测点与 GT 点之间的最小 L1 距离
  - ≤10px：reward=1.0
  - ≥50px：reward=0.0
  - 之间：线性插值

**fsd_free_point**（自由点预测）：
```
overall = 0.1 * format + 0.1 * point_fmt + 0.6 * in_box + 0.2 * dist
```
- `in_box`：预测点在 GT bbox 内的比例
- `dist`：预测点与关键点的 L1 距离

**fsd_visual_trace**（轨迹生成）：
```
overall = 0.1 * format + 0.1 * point_fmt + 0.1 * num + 0.3 * frechet + 0.4 * rmse
```
- `num`：是否输出恰好 8 个点
- `frechet`：基于离散 Fréchet 距离的轨迹相似度（阈值 10-100px）
- `rmse`：插值对齐后的轨迹 RMSE（阈值 5-50px）

**point_rec / grounding_rec**（多边形区域定位）：
```
overall = 0.1 * format + 0.1 * point_fmt + 0.8 * in_polygon
```
- `in_polygon`：使用射线法（ray casting）判断点是否在分割多边形内

**3d_position**（3D 空间定位）：
```
overall = 0.1 * format + 0.1 * point_fmt + 0.8 * position
```
- 将 2D 像素点 + 深度值通过相机内参/外参反投影到 3D 世界坐标
- 评估空间关系正确性（left/right/front/behind/top/between/center）

---

## 5. 训练框架架构（verl/）

### 5.1 整体架构

```
Driver (Ray Trainer)
    ├── DataLoader → RLHFDataset → DataProto
    ├── ActorRolloutWorker (FSDP + vLLM)
    │   ├── generate_sequences()  [vLLM Rollout]
    │   ├── compute_log_probs()   [Actor Forward]
    │   └── update_actor()        [PPO Policy Update]
    ├── RefPolicyWorker (FSDP, frozen)
    │   └── compute_ref_log_probs()
    ├── CriticWorker (FSDP, optional)
    │   ├── compute_values()
    │   └── update_critic()
    └── RewardFunction
        └── embodiedr1_compute_score()
```

### 5.2 核心组件

#### DataProto (`protocol.py`)

框架的核心数据结构，封装了：
- `batch` (TensorDict)：张量数据（input_ids, attention_mask, responses, old_log_probs, advantages 等）
- `non_tensor_batch` (Dict)：非张量数据（ground_truth, multi_modal_data 等）
- `meta_info` (Dict)：元信息（temperature, eos_token_id 等）

支持分片、合并、重排序等操作，是模块间数据交换的统一协议。

#### RayPPOTrainer (`trainer/ray_trainer.py`)

主训练循环 `fit()` 的流程：

```python
for episode in range(total_episodes):
    for batch in dataloader:
        # 1. 生成序列
        gen_batch = actor_rollout_wg.generate_sequences(batch)
        
        # 2. 计算奖励
        reward_tensor, metrics = reward_fn(gen_batch)
        
        # 3. 序列长度负载均衡（Karmarkar-Karp）
        _balance_batch(gen_batch)
        
        # 4. 计算策略 log prob
        old_log_probs = actor_rollout_wg.compute_log_probs(gen_batch)
        
        # 5. 计算参考策略 log prob（KL 约束）
        ref_log_probs = ref_policy_wg.compute_ref_log_probs(gen_batch)
        
        # 6. 计算 Critic values（GAE 需要）
        values = critic_wg.compute_values(gen_batch)
        
        # 7. 计算优势（Advantage）
        # GRPO: 组内相对优势
        # GAE: 广义优势估计
        # RLOO: REINFORCE Leave-One-Out
        
        # 8. 更新 Critic
        critic_wg.update_critic(gen_batch)
        
        # 9. 更新 Actor（PPO Clip + Dual Clip）
        actor_rollout_wg.update_actor(gen_batch)
```

#### FSDPWorker (`workers/fsdp_workers.py`)

统一的 FSDP 模型管理类，一个 worker 根据 `role` 扮演不同角色：
- `actor` / `actor_rollout`：策略网络 + vLLM 生成
- `critic`：价值网络（Token Classification head）
- `ref`：参考策略（冻结权重）

#### HybridEngine：FSDP ↔ vLLM 权重同步

`FSDPVLLMShardingManager` 实现训练（FSDP）和推理（vLLM）之间的权重切换：

```python
# 进入生成阶段
sharding_manager.__enter__()
    torch.cuda.empty_cache()
    state_dict = fsdp_model.state_dict()      # 获取 FSDP 权重
    vllm_model.wake_up(tags=["weights"])
    vllm_model.load_weights(state_dict)        # 加载到 vLLM
    vllm_model.wake_up(tags=["kv_cache"])

# 生成序列
sequences = vllm.generate(...)

# 退出生成阶段
sharding_manager.__exit__()
    vllm_model.sleep(level=1)                  # 释放权重和 KV Cache
    # 恢复 FSDP 训练模式
```

这种设计使得：
- 推理时使用 vLLM 的高性能生成
- 训练时使用 FSDP 的高效分布式训练
- 显存利用率高（vLLM sleep 时释放显存）

#### 序列并行：Ulysses SP

支持 DeepSpeed Ulysses Sequence Parallelism：
- 通过 all-to-all 通信在序列维度上切分数据
- 支持长序列训练
- 与 FSDP 结合使用（`FSDPUlyssesShardingManager`）

### 5.3 支持的 Advantage Estimator

| 方法 | 需要 Critic | 说明 |
|------|------------|------|
| **GAE** | 是 | 完整的 PPO，使用广义优势估计 |
| **GRPO** | 否 | Group Relative Policy Optimization，组内归一化（最常用） |
| **RLOO** | 否 | REINFORCE Leave-One-Out，每组留一法 baseline |
| **REMAX** | 否 | 基于 reward baseline 的方差缩减 |
| **REINFORCE++** | 否 | 带 gamma 折扣的 REINFORCE |

Embodied-R1 使用 **GRPO**（默认），每组采样 `n=5` 或 `n=8` 个 response，组内归一化优势。

### 5.4 PPO 策略更新（Dual Clip）

```python
# PPO loss with Dual-Clip
pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = compute_policy_loss(
    old_log_probs, log_probs, advantages, response_mask,
    clip_ratio_low=0.2,      # 标准 PPO clip 下限
    clip_ratio_high=0.3,     # 标准 PPO clip 上限
    clip_ratio_dual=3.0      # 双重 clip，防止策略过度偏离
)

# KL loss（可选）
if use_kl_loss:
    kl_loss = compute_kl(log_probs, ref_log_probs)
    pg_loss += kl_loss * kl_coef
```

---

## 6. 评估体系

项目提供了 **11 个评估脚本**，覆盖 2D/3D 空间推理、affordance 理解、轨迹生成、视觉问答等多个维度。

### 6.1 评估方法概览

所有评估脚本采用统一的技术栈：
- **模型**: `Qwen2_5_VLForConditionalGeneration` (bf16)
- **分布式**: `accelerate` (FSDP/DP)
- **输出解析**: 正则提取 `<answer>...</answer>` → 提取 `<point>...</point>`
- **推理参数**: `temperature=0, do_sample=False, max_new_tokens=2048`

### 6.2 各基准测试详解

#### 1. VABench-Point (`hf_inference_vabench_point.py`)
- **任务**: 自由点预测（在 bbox 内预测操作点）
- **数据**: FSD free-point Parquet 测试集
- **评估**: 预测点是否在 GT bbox 内
- **指标**: `points_in_bbox / total_points`

#### 2. VABench-VisualTrace (`hf_inference_vabench_visual_trace.py`)
- **任务**: 视觉轨迹生成（VTG）
- **数据**: FSD visual-trace Parquet 测试集
- **评估**: 将预测轨迹和 GT 轨迹插值到相同长度，计算 RMSE 和 MAE
- **指标**: Average RMSE, Average MAE
- **可视化**: 渐变色彩绘制预测轨迹（红→蓝）

#### 3. Where2Place (`hf_inference_where2place.py`)
- **任务**: 物体放置位置预测
- **数据**: `FlagEval/Where2Place`
- **评估**: 预测点是否在 GT mask 内
- **指标**: `points_in_mask / total_points`

#### 4. Part-Affordance-2K (`hf_inference_affordance.py`)
- **任务**: affordance 区域定位（如"握持位置"）
- **数据**: `IffYuan/Part-Affordance-2K`
- **评估**: 预测点是否在 GT mask 的 foreground 内
- **指标**: `points_in_mask / total_points`

#### 5. RoboRefit (`hf_inference_roborefit.py`)
- **任务**: 机器人指令跟随中的目标物体定位
- **数据**: `eval/roborefit_test.json`（自然语言指令 + bbox）
- **评估**: 预测点是否在 GT bbox 内
- **指标**: `points_in_box / total_points`
- **可视化**: 绘制 bbox 和预测点（绿=命中，蓝=未命中）

#### 6. Open6DoR-Custom (`hf_inference_3d.py`)
- **任务**: 3D 空间关系定位
- **数据**: `eval/3d_dataset.json`（IsaacSim 渲染的 RGB-D 图像）
- **评估**: 将 2D 像素点通过相机内参/外参反投影到 3D 世界坐标，判断空间关系（left/right/front/behind/top/between/center）是否正确
- **指标**: 二元准确率

#### 7. BLINK (`hf_inference_blink.py`)
- **任务**: 多模态视觉推理多选题
- **子任务**: Multi-view Reasoning, Object Localization, Relative Depth, Spatial Relation
- **数据**: `BLINK-Benchmark/BLINK`
- **评估**: 选项字母匹配
- **指标**: 总体准确率 + 子任务准确率

#### 8. CRPE (`hf_inference_crpe.py`)
- **任务**: 组合式空间关系推理（Compositional Relative Position Evaluation）
- **评估**: 单图多选题，选项字母匹配
- **指标**: 总体准确率 + 按 category 分组

#### 9. CV-Bench (`hf_inference_cvbench.py`)
- **任务**: 计算机视觉综合基准
- **数据**: `nyu-visionx/CV-Bench`
- **评估**: 多选题匹配
- **指标**: 总体准确率 + 按 task 分组

#### 10. EmbSpatial-Bench (`hf_inference_embspatial.py`)
- **任务**: 空间关系理解选择题（A/B/C/D 四选一）
- **数据**: `FlagEval/EmbSpatial-Bench`
- **指标**: 总体准确率

#### 11. SAT (`hf_inference_sat.py`)
- **任务**: 空间推理问答（Spatial Awareness Test）
- **数据**: `array/SAT`
- **格式**: 多图问答，自然语言短语答案
- **评估**: 字符串包含匹配
- **指标**: 总体准确率

---

## 7. 推理示例

### 7.1 inference_example.py

项目的核心推理脚本，支持四种模式：

```python
# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'IffYuan/Embodied-R1-3B-v1',
    device_map='auto'
)
processor = AutoProcessor.from_pretrained('IffYuan/Embodied-R1-3B-v1')

# 构建消息
messages = [{
    'role': 'user',
    'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt}
    ]
}]

# 生成
response = generate_response(model, processor, messages)
# 输出: <think>...</think><answer><point>[[x1,y1],...]</point></answer>

# 解析
think_content, answer_content, coordinates = _extract_model_output_parts(response)

# 可视化
visual_image = _visualize_coordinates_on_image(image, coordinates, mode)
```

### 7.2 可视化细节

- **非 VTG 模式**：浅蓝色圆点（白边），标记定位点
- **VTG 模式**：
  - 使用 SciPy 三次插值生成平滑曲线
  - 渐变色彩（红→紫→蓝）表示轨迹方向
  - 终点：蓝色方块
  - 中间路点：紫色圆圈

---

## 8. 代码结构分析

### 8.1 项目文件树

```
Embodied-R1/
├── inference_example.py          # 主推理入口（支持4种模式）
├── README.md                     # 项目说明
├── requirements.txt              # 训练依赖
├── pyproject.toml / setup.py     # 包配置
│
├── scripts/                      # 训练脚本与配置
│   ├── config_stage1.yaml        # Stage 1 数据配置
│   ├── config_stage2.yaml        # Stage 2 数据配置
│   ├── stage_1_embodied_r1.sh    # Stage 1 启动脚本
│   ├── stage_2_embodied_r1.sh    # Stage 2 启动脚本
│   └── model_merger.py           # FSDP checkpoint → HF 格式合并
│
├── eval/                         # 11个评估基准脚本
│   ├── hf_inference_3d.py
│   ├── hf_inference_affordance.py
│   ├── hf_inference_blink.py
│   ├── hf_inference_crpe.py
│   ├── hf_inference_cvbench.py
│   ├── hf_inference_embspatial.py
│   ├── hf_inference_roborefit.py
│   ├── hf_inference_sat.py
│   ├── hf_inference_vabench_point.py
│   ├── hf_inference_vabench_visual_trace.py
│   ├── hf_inference_where2place.py
│   ├── 3d_dataset.json
│   └── roborefit_test.json
│
├── verl/                         # 核心 RL 训练框架
│   ├── protocol.py               # DataProto 数据协议
│   ├── trainer/
│   │   ├── main.py               # 训练入口
│   │   ├── ray_trainer.py        # RayPPOTrainer 主循环
│   │   ├── core_algos.py         # PPO/GRPO/RLOO/REMAX 算法
│   │   ├── config.py             # 配置类
│   │   └── metrics.py            # 指标计算
│   ├── workers/
│   │   ├── fsdp_workers.py       # FSDPWorker 统一管理
│   │   ├── actor/dp_actor.py     # Actor 策略更新
│   │   ├── critic/dp_critic.py   # Critic 价值更新
│   │   ├── rollout/vllm_rollout_spmd.py  # vLLM 生成
│   │   ├── reward/custom.py      # 奖励函数调度
│   │   └── sharding_manager/     # FSDP ↔ vLLM / Ulysses 同步
│   ├── single_controller/        # Ray 分布式控制器
│   │   ├── base/                 # Worker/WorkerGroup 基类
│   │   └── ray/                  # RayWorkerGroup 实现
│   ├── models/                   # 模型补丁
│   │   └── transformers/         # FlashAttention + Ulysses SP
│   └── utils/
│       ├── dataset.py            # RLHFDataset（多模态支持）
│       ├── reward_score/         # 奖励函数
│       │   ├── embodiedr1.py     # Embodied-R1 核心奖励
│       │   ├── math.py
│       │   └── r1v.py
│       ├── torch_functional.py   # log_prob / AdamW
│       ├── ulysses.py            # Ulysses SP 实现
│       ├── fsdp_utils.py         # FSDP 工具
│       └── checkpoint/           # Checkpoint 管理
│
├── assets/                       # 展示图片
├── example_data/                 # 推理示例输入图片
└── .agent/                       # Agent 工作目录
```

### 8.2 关键文件说明

| 文件 | 行数 | 核心功能 |
|------|------|---------|
| `inference_example.py` | 408 | 推理入口、4种模式、可视化 |
| `verl/utils/reward_score/embodiedr1.py` | 679 | 多任务统一奖励函数 |
| `verl/trainer/ray_trainer.py` | ~500 | 主训练循环、GRPO/GAE/RLOO |
| `verl/workers/fsdp_workers.py` | ~400 | FSDP Worker 统一初始化 |
| `verl/workers/actor/dp_actor.py` | ~300 | Actor 策略更新、Dual-Clip PPO |
| `verl/workers/rollout/vllm_rollout_spmd.py` | ~200 | vLLM 序列生成 |
| `verl/workers/sharding_manager/fsdp_vllm.py` | ~150 | HybridEngine 权重同步 |
| `verl/utils/dataset.py` | ~300 | 多模态数据集加载 |
| `verl/protocol.py` | ~300 | DataProto 数据协议 |

---

## 9. 数据集与数据格式

### 9.1 训练数据格式（Parquet）

每个样本包含：
```json
{
  "problem": "用户输入的图像和文本 prompt",
  "answer": "<type>task_type</type> ground_truth_answer",
  "images": ["path/to/image.png"]
}
```

`answer` 中的 `<type>` 标签用于奖励函数路由到对应的评估逻辑。

### 9.2 各任务类型的 GT 格式

| 任务类型 | GT 格式示例 |
|---------|------------|
| `general_qa` / `spatial_qa` | 纯文本答案 |
| `point_ref` | `[[x1,y1], [x2,y2], ...]` |
| `fsd_free_point` | `{"bbox": [x1,y1,x2,y2], "free_points": [[x1,y1],...]}` |
| `fsd_visual_trace` | `{"trajectory": [[x1,y1],...]}` |
| `point_rec` / `grounding_rec` | `{"segmentation": [[x1,y1],...]}` |
| `3d_position` | `{"object": [[x,y,z],...], "direction": "behind"}` |

### 9.3 评估数据格式

**Open6DoR** (`3d_dataset.json`)：
```json
{
  "selected_obj_names": ["box", "apple"],
  "target_obj_name": "apple",
  "position_instruction": "Place the apple behind the box on the table.",
  "position_tag": "behind",
  "answer": {"object": [[0.537, 0.018, 0.300]], "direction": "behind"},
  "images": ["rgb.png", "depth.npy"]
}
```

**RoboRefit** (`roborefit_test.json`)：
```json
{
  "id": 0,
  "problem": "will you please pass me the glue stick",
  "image": "image_0.png",
  "bbox": [175.0, 168.0, 233.0, 217.0]
}
```

---

## 10. 关键技术创新点总结

### 10.1 方法论创新

1. **Pointing 统一范式**：将复杂的机器人操作任务统一为 2D 坐标点预测，简化了从视觉感知到动作执行的映射
2. **Think-Then-Answer 结构化输出**：显式分离推理过程和答案，提高可解释性和 RL 训练效率
3. **两阶段渐进式训练**：Stage 1 建立空间语义理解 → Stage 2 强化像素级定位，符合认知递进规律

### 10.2 技术实现创新

4. **多任务统一奖励函数**：通过 `<type>` 标签路由，一套奖励函数覆盖 QA、点定位、轨迹生成、3D 定位等7种任务类型
5. **HybridEngine (FSDP + vLLM)**：训练时 FSDP 分片、推理时 vLLM 生成，通过权重同步实现无缝切换
6. **Ulysses Sequence Parallelism**：支持长序列多模态训练
7. **序列长度负载均衡**：Karmarkar-Karp 算法均衡各 DP rank 的计算负载

### 10.3 工程实践创新

8. **全面的评估覆盖**：11 个评估基准覆盖 2D/3D 空间推理、affordance、轨迹生成、VQA 等维度
9. **完整开源生态**：模型、数据、训练代码、评估代码全部开源

---

## 11. 局限性与未来方向

### 11.1 当前局限

1. **2D 限制**：当前版本主要输出 2D 像素坐标，虽然支持 3D 评估（Open6DoR），但核心能力仍是 2D 指向
2. **固定路点数**：VTG 模式固定输出 8 个点，可能不适用于所有轨迹长度需求
3. **单一模型规模**：仅发布了 3B 版本，未探索更大/更小模型的 scaling law
4. **依赖 Qwen2.5-VL**：受限于基础模型的能力上限

### 11.2 潜在扩展方向

1. **3D Pointing**：直接输出 3D 世界坐标，摆脱对相机参数的依赖
2. **动态路点数**：根据任务复杂度自适应输出不同数量的路点
3. **多模态动作输出**：除了坐标点，输出姿态、力矩、夹爪开合等更丰富的动作信息
4. **Real-World 部署**：与真实机器人平台（如 Franka、UR5）集成
5. **Online RL**：在仿真环境中进行在线强化学习，而非仅离线训练

---

## 12. 相关资源

| 资源 | 链接 |
|------|------|
| 项目主页 | https://embodied-r1.github.io |
| arXiv 论文 | http://arxiv.org/abs/2508.13998 |
| ICLR 2026 版本 | https://openreview.net/forum?id=i5wlozMFsQ |
| 预训练模型 | https://huggingface.co/IffYuan/Embodied-R1-3B-v1 |
| 训练数据集 | https://huggingface.co/datasets/IffYuan/Embodied-R1-Dataset |
| HuggingFace 合集 | https://huggingface.co/collections/IffYuan/embodied-r1-684a8474b3a49210995f9081 |
| VABench-P | https://huggingface.co/datasets/IffYuan/VABench-P |
| VABench-V | https://huggingface.co/datasets/IffYuan/vabench-v |

---

## 13. 引用

```bibtex
@article{yuan2026embodied,
  title={Embodied-r1: Reinforced embodied reasoning for general robotic manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Huang, Yaoting and Chen, Yibin and Ni, Fei and Dong, Zibin and Li, Pengyi and Zheng, Yan and Tang, Hongyao and Hao, Jianye},
  journal={The Fourteenth International Conference on Learning Representations},
  year={2026}
}

@article{yuan2026seeing,
  title={From seeing to doing: Bridging reasoning and decision for robotic manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Chen, Yibin and Dong, Zibin and Ni, Fei and Kou, Longxin and Liu, Jinyi and Li, Pengyi and Zheng, Yan and Hao, Jianye},
  journal={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

---

*本报告基于对 Embodied-R1 项目代码库的全面分析生成，涵盖项目架构、训练方法、评估体系和技术细节。*
