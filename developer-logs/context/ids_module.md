# IDS 模块研究文档

## 概述

IDS（Instruction Disambiguation System）是一个轻量级的指令歧义消解模块，位于 `disambiguation/` 目录。其核心职责是：在机器人执行抓取/操控指令之前，判断该指令是否存在歧义，并在必要时生成澄清问题而不是盲目执行。

**模块入口**：`disambiguation/__init__.py`，对外暴露三个核心类：

```
InstructionResolver      # 主入口，负责决策
MCPStateTracker          # 基于结构化场景状态做候选检索
R1CandidateExtractor     # 基于 VLM 做视觉候选提取（需要 GPU）
```

---

## 架构设计

IDS 采用**两层歧义检测**策略，两层可以独立工作，也可以组合：

### 第一层：MCP State-based 检索（无需 GPU）

`MCPStateTracker` 接收结构化的场景状态（object_id、category、color、size、location、state），通过文本匹配检索候选目标。

**决策逻辑（`MCPStateTracker.retrieve_candidates`）**：
1. 从指令中提取被提及的 category（支持单复数变形）
2. 若指令含 "object" / "thing"，所有 category 均视为候选
3. 若提及多个 category，只保留场景中出现超过一次的（避免对唯一目标误判为歧义）
4. 用颜色/尺寸/位置/状态约束进一步过滤候选列表

**状态约束推断（`_state_constraints`）** 是亮点设计：
- `"open the box"` → 推断目标是 `open=False` 的 box（需要被打开的）
- `"close the bottle that is open"` → 推断目标是 `open=True` 的 bottle
- `"turn on the switch that is off"` → 推断目标是 `on=False` 的 switch

这样即使场景中有多个同类物体，也可以通过动作语义自动消解。

**位置约束的特殊处理**：当指令中出现 `"left of"` / `"on top of"` 这类关系短语时，不将 left/top/right 等词视为位置约束（因为它们描述的是关系，不是目标属性）。

### 第二层：R1 VLM 视觉候选提取（需要 GPU）

`R1CandidateExtractor` 使用 Qwen2.5-VL 模型，将图像和指令输入 VLM，让模型列出图像中所有可能满足指令的候选目标，输出结构化 JSON：

```json
{
  "target_query": "...",
  "candidates": [
    {"name": "...", "visual_attributes": "...", "location": "..."}
  ]
}
```

**触发条件**（`_should_check_candidates`）：
- 指令中的泛化名词（cup/bottle/box/...）出现 2+ 次且无属性词
- mode 为 REG 或 OFG 且没有任何消歧线索（颜色、空间词、关系词）

---

## 主决策流（`InstructionResolver.resolve`）

```
输入：image, instruction, mode
     ↓
1. 归一化指令（去除多余空格）
2. 空指令 → status="invalid"
3. MCPStateTracker.retrieve_candidates() 有结果？
   → 走 _result_from_candidates()
4. 场景中只有一种同类目标？
   → status="resolved"（单一 category 不歧义）
5. 是否需要视觉候选检查？（_should_check_candidates）
   否 → status="resolved"
6. 无 candidate_extractor？
   → status="ambiguous", needs_clarification=True
7. 调用 R1CandidateExtractor，走 _result_from_candidates()
```

**`_result_from_candidates` 的三条出口**：
- 候选数 == 1 → `status="resolved"`，resolved_instruction 补充 "referring to X"
- 候选数 > 1 → `status="ambiguous"`，生成选择问题
- 候选数 == 0 → `status="unknown"`，无法识别目标

---

## 状态枚举

| status | 含义 | needs_clarification |
|--------|------|---------------------|
| `resolved` | 唯一目标已确定，可执行 | False |
| `ambiguous` | 多个候选，需用户选择 | True |
| `unknown` | 目标在场景中不存在 | True |
| `invalid` | 指令为空或格式无效 | True |

---

## 评估脚本

### `experiments/evaluate_ids_behavior.py`
- **目的**：验证 IDS 基本逻辑的正确性
- **场景**：固定的 14 个物体（cup、box、bottle、switch、mug、lighter、glue stick、apple），23 条测试指令
- **典型测试点**：
  - 颜色/尺寸/位置消歧（单个属性足够 → resolved）
  - 状态消歧（open/close/turn on/off 的语义推断）
  - 关系任务（"place A behind B" → ambiguous，因为 A 有多个候选）
  - 未知目标（"pick up the purple cup"，紫色不存在 → unknown）
  - 空指令 → invalid
- **运行**：`PYTHONPATH=. python3 experiments/evaluate_ids_behavior.py`
- **输出**：`output_results/ids_behavior_eval.json`

**关键指标**：
- `status_accuracy`：指令状态分类正确率
- `resolved_target_accuracy`：resolved 类指令中候选集匹配正确率
- `ambiguous_candidate_recall`：ambiguous 类指令候选集召回率

### `experiments/evaluate_ids_synthetic_mcp.py`
- **目的**：大规模随机评估（500+ cases），证明 IDS 在 MCP-style 结构化输入下的泛化性
- **场景生成**：随机生成颜色/尺寸/位置组合的场景，8 种 case 类型循环：
  - `category_ambiguous`：同类 2 个，无属性 → ambiguous
  - `color_resolved`：指定颜色 → resolved
  - `location_resolved`：指定位置 → resolved
  - `size_resolved`：指定尺寸 → resolved
  - `state_on_resolved`：turn off the switch that is on → resolved
  - `state_closed_resolved`：open the box that is closed → resolved
  - `unknown_attribute`：指定不存在的颜色 → unknown
  - `invalid_empty`：空指令 → invalid
- **运行**：`PYTHONPATH=. python3 experiments/evaluate_ids_synthetic_mcp.py --num-cases 500 --seed 7`
- **输出**：`output_results/ids_synthetic_mcp_eval.json`

**额外指标**：
- `unknown_detection_accuracy`：unknown 类检测率
- `false_ambiguity_rate_on_resolved`：误判为 ambiguous 的比率（越低越好）
- `unsafe_pointing_rate_ids_on_ambiguous`：IDS 对歧义指令的错误执行率（设计上固定为 0）

### `experiments/build_ids_original_subset.py`
- **目的**：从真实数据集（roborefit_test.json、3d_dataset.json）中提取歧义风险样本，构建评估子集
- **风险判定规则**（`infer_risk`）：
  - `high`：指令提及了场景中重复出现的同名物体（最可靠的歧义信号）
  - `medium`：指令含泛化名词且无属性词（潜在歧义）
  - `low`：无上述信号（默认排除，除非 `--include-low-risk`）
- **标注逻辑**：high-risk + 提及重复物体 → `ambiguity_label="ambiguous"`，`expected_ids_behavior="ask_clarification"`；其余 → `unlabeled`
- **运行**：`PYTHONPATH=. python3 experiments/build_ids_original_subset.py --max-samples 200`
- **输出**：`output_results/ids_original_ambiguity_subset.json`

---

## 关键设计约束

1. **决策逻辑不在模型中**：VLM（R1CandidateExtractor）只负责列候选，最终 ambiguous/resolved/unknown 判断全部在 `InstructionResolver` 代码中。这保证了决策的可解释性和稳定性。

2. **无 VLM 也可运行**：若不传 `candidate_extractor`，系统仍可通过 MCP state 做判断；完全无状态时对泛化名词指令兜底标注 `ambiguous`。

3. **状态约束优先**：同类多个物体但指令含动作语义（open/close/turn on/off）时，优先通过状态过滤，避免不必要的澄清问题。

4. **关系短语不作为位置约束**：`"left of"` 中的 `left` 描述的是目标与参照物的关系，不是目标自身的位置属性，`_location_constraints` 会排除这种情况。

---

## 数据流示意

```
用户指令 + 场景状态
        ↓
MCPStateTracker.retrieve_candidates()
        ↓
   有候选？── 是 ──→ _result_from_candidates()
        ↓否
单一 category？─ 是 ─→ resolved
        ↓否
需要视觉检查？─ 否 ─→ resolved（preserved）
        ↓是
R1CandidateExtractor.extract_candidates()  [需要图像 + GPU]
        ↓
_result_from_candidates()
        ↓
DisambiguationResult { status, needs_clarification, clarification_question, ... }
```
