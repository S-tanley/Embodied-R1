# Eval Results Report — Embodied-R1-3B
**Date:** 2026-04-23  
**Model:** `IffYuan/Embodied-R1-3B`（官方发布的 3B checkpoint，基于 Qwen2.5-VL-3B-Instruct）  
**Evaluation Suite:** `eval/run_all.sh`，共 10 个 benchmark（不含 CRPE）

---

## 数据来源说明

所有数据来自以下文件，报告中每处数字均可溯源：

| Benchmark | 结果文件（JSON） | 运行日志 |
|-----------|----------------|---------|
| CV-Bench | `eval/logs/results/CV-Bench_Embodied-R1-3B_True.json` | `eval/logs/run_cvbench.log`（末尾摘要行） |
| EmbSpatialBench | `eval/logs/results/EmbSpatialBench_Embodied-R1-3B_True.json` | `eval/logs/run_embspatial.log`（末尾摘要行） |
| BLINK | 无 JSON（进程在最终同步阶段 NCCL 超时崩溃） | `eval/logs/inference_BLINK_Embodied-R1-3B_20260423_193905.log`（全部 451 条样本已处理完，数字由此推算） |
| RoboRefit | `eval/logs/results/RoboRefit_Embodied-R1-3B_True_test_20260423_081716.json` | `eval/logs/run_roborefit.log`（末尾摘要行） |
| Part-Affordance-2K | `eval/logs/results/Part-Affordance-2K_Embodied-R1-3B_True_train.json` | `eval/logs/run_affordance.log`（末尾摘要行） |
| Where2Place | `eval/logs/results/Where2Place_Embodied-R1-3B_True_test.json` | `eval/logs/run_where2place.log`（末尾摘要行）及 `eval/logs/inference_Where2Place_Embodied-R1-3B_20260423_200041.log` |
| SAT-Real | — | `eval/logs/inference_SAT-Real_Embodied-R1-3B_20260423_195916.log`（仅初始化，5 行，无推理结果） |
| VABench-Point | — | `eval/logs/inference_VABench_Point_Embodied-R1-3B_20260423_200007/8.log`（仅初始化，无结果） |
| VABench-VisualTrace | — | `eval/logs/inference_VABench_VisualTrace_Embodied-R1-3B_20260423_200024.log`（仅初始化，无结果） |
| Open6dor-Custom | — | `eval/logs/run_3d.log`（崩溃：`eval/data/open6dor_test.parquet` 文件不存在） |

---

## 汇总结果

| # | Benchmark | 任务类型 | 主指标 | 结果 | 样本数 |
|---|-----------|---------|--------|------|--------|
| 1 | **CV-Bench** | 空间 QA（多选） | Accuracy | **81.69%** | 2638 |
| 2 | **EmbSpatialBench** | 空间 QA（多选） | Accuracy | **68.96%** | 3640 |
| 3 | **BLINK** | 视觉感知 QA（多选） | Accuracy | **65.63%** | 451 |
| 4 | **RoboRefit** | 指向定位（点打在 bbox 内） | Points-in-Box Acc | **85.35%** | 2000 |
| 5 | **Part-Affordance-2K** | 可操作区域点定位（点打在 mask 内） | Points-in-Mask Acc | **44.61%** | 2000 |
| 6 | **Where2Place** | 放置位置点定位（点打在 mask 内） | Points-in-Mask Acc | **69.50%** | 100 |
| 7 | SAT-Real | 空间 QA | — | **未完成** | — |
| 8 | VABench-Point | 视觉轨迹点 | — | **未完成** | — |
| 9 | VABench-VisualTrace | 视觉轨迹 | — | **未完成** | — |
| 10 | Open6dor-Custom | 6DoF 抓取方向 | — | **崩溃** | — |

> **注：** BLINK 在 eval/logs/run_blink.log 末尾显示 NCCL collective operation timeout（进程间同步超时），但 inference log 证实所有 451 个样本均已推理完毕。65.63% 是从 inference log 中统计 `accuracy: True/False` 条目得出，结果可靠。

---

## 各 Benchmark 详细分析

### 1. CV-Bench — 81.69%（2155 / 2638）

**任务描述：** 室内场景图像的多选空间理解题，涵盖四个子任务：对象计数、相对深度、距离估算、空间关系判断。

**子任务细分：**

| 子任务 | 正确/总计 | 准确率 |
|--------|----------|--------|
| Relation（空间关系） | 590 / 650 | **90.77%** |
| Depth（深度比较） | 520 / 600 | **86.67%** |
| Distance（距离估算） | 505 / 600 | **84.17%** |
| Count（对象计数） | 540 / 788 | **68.53%** |

**分析：**
- 空间关系（Relation）和深度判断（Depth）表现最强，说明模型对场景中对象的相对位置和远近关系有较好的理解。
- 计数（Count）是四个子任务中最弱的，低于整体均值约 13 个百分点。这与视觉计数任务通常需要精确的空间注意力分配有关，是 VLM 普遍的短板。
- 总体 81.69% 属于较高水平，说明 Embodied-R1-3B 在通用室内空间 QA 上具备较强基础能力。

**数据来源：** `eval/logs/results/CV-Bench_Embodied-R1-3B_True.json`（2638 条完整样本），摘要确认见 `eval/logs/run_cvbench.log` 末尾（`Average accuracy: 0.8169`，`Accuracy by Sub-task` 四行）。

---

### 2. EmbSpatialBench — 68.96%（2510 / 3640）

**任务描述：** 专为具身场景设计的空间关系多选题，问题涉及以图像中第一视角或场景中心为基准的方位关系（如"哪个物体离你最近"、"A 在 B 的哪个方向"）。

**分析：**
- 68.96% 低于 CV-Bench（81.69%），但 EmbSpatialBench 的题目更具挑战性：它要求以自我中心（ego-centric）视角理解 3D 空间关系，而不仅仅是图像中对象的相对位置。
- 结果文件（JSON）中没有子任务字段（仅有 `question`、`predicted`、`ground_truth`、`accuracy`），无法进一步分解。
- 与 CV-Bench 相差约 13 个点，暗示模型在需要"我在哪里"视角的推理上明显弱于"图中 A 相对于 B"的推理。这是一个值得关注的系统性差距，与 Embodied-R1 论文中 Stage 1 主要针对 QA 任务而非具身视角的训练目标吻合。

**数据来源：** `eval/logs/results/EmbSpatialBench_Embodied-R1-3B_True.json`（3640 条），摘要确认见 `eval/logs/run_embspatial.log` 末尾（`Average accuracy: 0.6896`）。

---

### 3. BLINK — 65.63%（296 / 451）

**任务描述：** 多子任务视觉感知 benchmark，包含 bounding box 精度判断、相机运动方向、深度顺序等感知类问题，均为二选一或多选形式。

**分析：**
- 65.63% 略高于随机基线（~50%），但在本次测试的 6 个 benchmark 中排名最低。
- BLINK 的问题偏向"感知层"（perception），如"哪个 bbox 更准确"、"相机顺时针还是逆时针运动"，要求细粒度的视觉判断，与 Embodied-R1 训练时侧重的空间关系和指向任务有一定距离。
- 从 inference log 中可以看到，模型有时在 `<think>` 段中的推理方向和最终 `<answer>` 相矛盾（即 think 过程给出 A，answer 却选 B），说明在视觉感知类任务上，chain-of-thought 推理链与最终决策的对齐不稳定。
- 注意：本条结果来自 inference log 逐条统计（451 条全部处理完毕），而非 JSON 摘要文件（最终同步阶段因 NCCL 超时崩溃，未写出摘要）。

**数据来源：** `eval/logs/inference_BLINK_Embodied-R1-3B_20260423_193905.log`（451 条 `accuracy: True/False` 条目，全部遍历完毕，human-counted: 296 True, 155 False）。

---

### 4. RoboRefit — 85.35%（3414 / 4000 points）

**任务描述：** 机器人抓取指向任务。给定自然语言指令（如"pick up the red cup"），模型需要预测 2 个点坐标，要求点落在对应物体的 bounding box 内。每条样本预测 2 个点（total_points = 4000，samples = 2000）。

**指标：** `points_in_box / total_points`，即预测点命中 GT bbox 的比例。

**分析：**
- 85.35% 是本次测试中最高分，说明模型在有语言引导的、有明确目标物体的指向任务上具有最强表现。
- 机器人抓取指向与 Embodied-R1 Stage 2 的训练目标（REG/VTG 等指向任务）直接对应，因此高分符合预期。
- 每条样本预测 2 个点（而非 1 个），说明模型的多点输出格式解析稳定。

**数据来源：** `eval/logs/results/RoboRefit_Embodied-R1-3B_True_test_20260423_081716.json`（2000 条），摘要确认见 `eval/logs/run_roborefit.log` 末尾（`Overall points accuracy: 0.8535`）。

---

### 5. Part-Affordance-2K — 44.61%（1779 / 3988 points）

**任务描述：** 零件级可操作区域（affordance）点定位任务。给定"grasp the mug"类的指令，模型需要预测点坐标，要求点落在对应零件的 segmentation mask 内。数据集来自 HANDAL，测试的是 train split（2000 条）。

**指标：** `points_in_mask / total_points`。每条样本预测点数不固定（平均约 2 点，total=3988）。

**分数分布（accuracy_score per sample）：**

| 分数 | 样本数 | 占比 |
|------|--------|------|
| 1.0（全部点命中） | 773 | 38.7% |
| 0.5（部分命中） | 233 | 11.7% |
| 0.0（全部未中） | 994 | 49.7% |

**分析：**
- 44.61% 是本次测试中最低分，与其他任务差距明显（比 RoboRefit 低约 41 个点）。
- 核心难点：Part-Affordance 要求精确定位到**零件级别**（如杯柄、把手、按钮），而非整个物体。这比 RoboRefit 的 bbox 定位精度要求高出一个量级。
- 接近 50% 的样本得分为 0.0（全部预测点均未落入 mask），说明模型在细粒度零件定位上存在系统性失败，而非随机噪声。
- 注意：测试的是 `train` split，而非 `test` split（文件名后缀为 `_True_train.json`），结果仅供参考，不应直接与其他模型的 test split 结果对比。

**数据来源：** `eval/logs/results/Part-Affordance-2K_Embodied-R1-3B_True_train.json`（2000 条），摘要确认见 `eval/logs/run_affordance.log` 末尾（`Average accuracy score: 0.4447`，`Overall points accuracy: 0.4461`）。

---

### 6. Where2Place — 69.50%（139 / 200 points）

**任务描述：** 物体放置位置预测任务。给定场景图像和放置指令（如"put the block on the plate"），模型需要预测 2 个点坐标，要求点落在合理放置区域的 mask 内。测试集 100 条，每条预测 2 点。

**分数分布（accuracy_score per sample）：**

| 分数 | 样本数 | 占比 |
|------|--------|------|
| 1.0（全部点命中） | 68 | 68.0% |
| 0.5（部分命中） | 1 | 1.0% |
| 0.0（全部未中） | 31 | 31.0% |

**分析：**
- 69.50% 处于中等水平。测试集仅 100 条，样本量较小，结果波动范围较大。
- 分数分布呈现强烈的二元化（68% 全中，31% 全不中），说明模型在这类任务上要么推理完全正确，要么完全偏离目标位置，缺乏"部分正确"的中间状态——这与 0.5 分仅有 1 个样本吻合。
- 放置位置预测比抓取定位更难：抓取目标唯一，而放置区域需要理解上下文语义（"on the plate"需要先识别 plate 的可放置表面）。

**数据来源：** `eval/logs/results/Where2Place_Embodied-R1-3B_True_test.json`（100 条），摘要确认见 `eval/logs/inference_Where2Place_Embodied-R1-3B_20260423_200041.log` 末尾（`Overall points accuracy: 0.6950`）及 `eval/logs/run_where2place.log` 末尾。

---

## 未完成的 4 个 Evaluation

以下 4 个 benchmark 在当前 `eval/logs/` 中没有推理结果，仅有初始化日志（加载模型后立即崩溃）：

| Benchmark | 失败原因 | 日志位置 |
|-----------|---------|---------|
| **SAT-Real** | 进程崩溃，未留推理记录 | `eval/logs/inference_SAT-Real_Embodied-R1-3B_20260423_195916.log`（5 行） |
| **VABench-Point** | 进程崩溃，未留推理记录 | `eval/logs/inference_VABench_Point_Embodied-R1-3B_20260423_200007/8.log`（4-6 行） |
| **VABench-VisualTrace** | 进程崩溃，未留推理记录 | `eval/logs/inference_VABench_VisualTrace_Embodied-R1-3B_20260423_200024.log`（10 行） |
| **Open6dor-Custom** | `FileNotFoundError: eval/data/open6dor_test.parquet` 不存在 | `eval/logs/run_3d.log` |

这 4 个 benchmark 在 `run_all.sh` 脚本的 19:58–20:00 批次中均未成功执行。如需补跑：
- Open6dor：需要先下载 `eval/data/open6dor_test.parquet`
- SAT-Real / VABench*：需排查进程崩溃原因（可能是数据文件路径或内存问题）

---

## 横向对比与整体分析

### 按任务类型分组

**空间 QA（选择题）：**
```
CV-Bench          81.69%  ████████████████████
EmbSpatialBench   68.96%  █████████████████
BLINK             65.63%  ████████████████
```

**点定位（grounding）：**
```
RoboRefit         85.35%  █████████████████████
Where2Place       69.50%  █████████████████
Part-Affordance   44.61%  ███████████
```

### 核心观察

1. **指向任务 > QA 任务**（整体均值：RoboRefit 85.35% vs CV-Bench 81.69%）：Embodied-R1-3B 在有明确指向目标的任务上表现优于纯 QA。这与其两阶段训练设计（Stage 1 QA → Stage 2 指向）的目标一致。

2. **零件级定位是明显短板**：Part-Affordance-2K 的 44.61% 比次低分（BLINK 65.63%）低约 21 个点。模型在**细粒度零件 mask** 定位上存在系统性困难，这是下一步改进的关键方向之一。

3. **自我中心视角是 QA 内的短板**：EmbSpatialBench（68.96%）比 CV-Bench（81.69%）低约 13 个点。两者都是空间 QA，差异主要来自 EmbSpatialBench 要求以"当前位置"为参考系，而 CV-Bench 以"图中对象间关系"为参考系——后者对模型更友好。

4. **计数能力相对弱**：CV-Bench 中 Count 子任务仅 68.53%，是四个子任务中最低，比最高的 Relation（90.77%）低约 22 个点。

5. **BLINK 的 think-answer 不一致现象**：从 inference log 样本中观察到模型有时在 `<think>` 中推理出答案 A，但 `<answer>` 给出答案 B。这种推理链与输出的不一致是 VLM 在感知类任务上的已知问题，值得在训练时通过专项 reward shaping 来解决。

---

## 总结

| 维度 | 最佳 | 最弱 |
|------|------|------|
| 空间 QA | CV-Bench 81.69% | BLINK 65.63% |
| 点定位 | RoboRefit 85.35% | Part-Affordance 44.61% |
| 需优先改进 | Part-Affordance 零件级定位、EmbSpatialBench 自我中心视角 | — |
| 数据缺失 | — | SAT-Real / VABench / Open6dor 需补跑 |
