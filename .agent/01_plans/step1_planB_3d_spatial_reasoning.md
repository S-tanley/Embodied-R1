# Plan B：进阶 3D 空间推理（Embodied-R1）

> 目标：在现有 Embodied-R1 基础上，补齐论文中提到的 3D 能力短板，重点降低 Depth 幻觉并实现 3D RRG 输出（X, Y, Depth）。
> 
> 结论先行：
> - 方案可行。
> - 但需要分两层推进：
>   1) 先做“现框架兼容”的 3D 升级（短期可落地）。
>   2) 再做“真实 Dual-Tower/点云”升级（中期高工程量）。

---

## 1. 可行性判断

## 1.1 为什么可行（已有基础）

当前代码已经具备 3D 升级的关键基础：

1. 训练数据管道支持多图输入（同一条样本可带多张 image，适配 RGB + Depth）。
2. Stage 脚本已预留 `worker.rollout.limit_images=2`，说明多图路径在训练链路中是考虑过的。
3. 奖励函数 `embodiedr1` 已内置 `3d_position` 类型，且支持 `<point>` + `<depth>` 联合格式。
4. 训练框架支持可插拔奖励函数与多任务混训，可把 3D 任务并入 Stage2/新 Stage3。

## 1.2 现阶段硬约束（必须先处理）

1. 当前 3D eval 脚本默认把深度固定为 1，且只使用第 1 张图（RGB），没有真正消费模型预测深度与 depth 图信息。
2. 仓库内暂未提供可直接用于训练的 3D train parquet（仅看到 3D 测试集路径）。
3. `embodiedr1_3d.py` 独立奖励文件存在可用性风险（有未定义依赖），短期不建议直接切到该 reward。
4. 训练框架对 `hybrid_engine=false` 路径未实现，意味着 rollout 强依赖 vLLM。
5. 若做“真正新增网络结构的 Dual-Tower”，大概率会遇到 vLLM 兼容问题（这是中期工程，不是立刻可跑通）。

## 1.3 2 张 H200 的结论

1. 做 3B 模型的 RGB-D 对齐微调 + 3D RL 训练是足够的。
2. 建议先走 vLLM 兼容路线（不破坏主模型结构），在 2xH200 上快速迭代。
3. 真正 Dual-Tower/点云分支建议作为第二阶段，不与第一阶段耦合上线。

---

## 2. 总体路线（两阶段）

## 阶段 A：兼容式 3D 升级（推荐先做，2-4 周）

目标：
1. 在不破坏 vLLM 兼容性的前提下，让模型学会输出 `(X, Y, Depth)`。
2. 建立“可验证”的 3D 指标闭环（训练 reward 与评测一致）。

做法：
1. 使用 RGB+Depth 双图输入（Depth 先转伪彩色图或归一化灰度图，作为第 2 张 image）。
2. 统一输出格式：
   - `<answer><point>[[x,y],... ]</point><depth>[d1,...]</depth></answer>`
3. 训练 reward 先走 `embodiedr1` 的 `3d_position` 路由，不直接用 `embodiedr1_3d.py`。
4. 修正 3D eval：真正解析 `<depth>`，并与 depth 图/相机参数联合评估。

## 阶段 B：真实 Dual-Tower/点云增强（中期，4-8 周）

目标：
1. 引入专门的 Depth/3D tower，与 RGB tower 做跨模态融合。
2. 支持点云（或点云投影特征）输入，提升几何一致性。

注意：
1. 这一步会触发 vLLM 兼容工程。
2. 建议在 A 阶段收益跑出来后，再决定是否投入。

---

## 3. 详细执行计划

## Phase 0：基线冻结与数据契约（2-3 天）

产出：
1. 3D 数据规范文档（train/val/test 字段一致）。
2. 评测协议 v2（必须包含 depth 解析，不再固定 depth=1）。

任务：
1. 定义训练样本 schema：
   - `problem`：明确包含两处 `<image>` 占位（RGB + Depth）。
   - `images`：长度为 2，均为可直接 decode 的 image bytes。
   - `answer`：`<type>3d_position</type>{"object": ..., "direction": ...}`。
2. 固定深度单位（建议 mm）与归一化策略。
3. 建立深度合法范围（例如 0~3000mm）用于幻觉率统计。

Go/No-Go：
1. 若 schema 仍不统一，后续训练与评测会持续错位，必须先停在本阶段修正。

---

## Phase 1：评测闭环修复（3-5 天）

产出：
1. 3D eval v2：真实使用 `<depth>` + depth 图，不再固定深度。
2. 指标面板：
   - 3D relation accuracy
   - depth RMSE / MAE
   - depth hallucination rate
   - output format pass rate

任务：
1. 改 `hf_inference_3d.py`：
   - 解析 `<depth>`。
   - 使用双图输入（RGB + Depth）。
   - 支持 depth 缺失或非法输出的鲁棒统计。
2. 按关系类型分桶统计（left/right/front/behind/top/between/center）。
3. 产出 baseline 分数（当前 ckpt + 新评测协议）。

Go/No-Go：
1. 若无稳定评测闭环，严禁进入训练改模阶段。

---

## Phase 2：3D 对齐微调（Modality Alignment）（1-2 周）

目标：
1. 先让模型“看懂” depth，不急于大规模 RL。
2. 降低 depth 幻觉（非法深度、无效深度分布）。

建议方式：
1. 轻量微调（LoRA/QLoRA 或 full-finetune 的低学习率版本）。
2. 数据以 3D 定位/关系任务为主，混入少量 2D 任务防遗忘。
3. 指令模板强化 depth 显式推理：
   - 输出必须包含 `<point>` 与 `<depth>`，长度一致。

两卡 H200 推荐初始参数（起跑档）：
1. `trainer.n_gpus_per_node=2`
2. `worker.rollout.tensor_parallel_size=1`（先求稳）
3. `data.rollout_batch_size=64`（再根据显存逐步加）
4. `worker.actor.global_batch_size=32`
5. `worker.actor.micro_batch_size_per_device_for_update=1~2`
6. `worker.rollout.n=4`
7. `worker.rollout.limit_images=2`
8. `worker.reward.score_function=embodiedr1`
9. `algorithm.adv_estimator=grpo`

阶段验收：
1. depth 幻觉率相对 baseline 明显下降（建议目标：>20% 相对降幅）。
2. 3D relation accuracy 有统计显著提升。

---

## Phase 3：3D RRG 强化训练（1-2 周）

目标：
1. 将 Phase 2 的对齐能力推进到更强决策表现。
2. 支持稳定输出 `(X, Y, Depth)` 用于机器人下游。

任务：
1. 新建 Stage3 配置（建议独立 yaml），仅/主要包含 3D 数据。
2. 使用 GRPO 继续训练，保留 KL 约束。
3. 增加 reward 项：
   - 几何关系正确性（主项）
   - 深度数值稳定性（辅助项）
   - 格式一致性（硬约束）

建议里程碑：
1. M1：3D-only 训练跑通（无崩溃，loss/reward 正常）。
2. M2：3D relation accuracy 超过当前 baseline。
3. M3：在混合任务上不明显退化 2D 能力。

---

## Phase 4：真实 Dual-Tower 与点云（可选扩展，4-8 周）

目标：
1. 引入独立 depth tower（甚至 point cloud encoder）。
2. 融合后输出更稳健的 3D 表征。

关键工程项：
1. 模型改造：
   - RGB tower + Depth tower + fusion module（cross-attn/gated fusion）。
2. 训练链路：
   - 若架构脱离 vLLM 原生支持，需要新增 rollout 后端路径（当前框架默认强依赖 vLLM）。
3. 推理与部署：
   - 兼容 HuggingFace 保存/加载。
   - 评测脚本与可视化同步升级。

风险：
1. 这一步的主要风险不是算法，而是工程兼容成本（vLLM + 自定义模型）。

---

## 4. 实验矩阵（建议）

至少做以下对照：

1. Baseline-2D：原模型 + 原 3D eval（固定 depth=1）。
2. Baseline-3D-EvalFix：原模型 + 新 3D eval（真实 depth 评估）。
3. Exp-A1：RGB-only 输入但要求输出 depth。
4. Exp-A2：RGB+Depth 双图输入（共享视觉塔）。
5. Exp-A3：在 A2 基础上加入 3D reward 权重调参。
6. Exp-B（可选）：Dual-Tower/点云版本。

核心指标：
1. overall 3D relation accuracy
2. per-relation accuracy
3. depth RMSE/MAE
4. depth hallucination rate
5. output format pass rate
6. 2D 能力回归指标（防灾难遗忘）

---

## 5. 成功标准（毕业设计亮点可直接复用）

满足以下任一组，可视为 Plan B 成功：

1. 在 Open6DoR 类 3D 任务上，relation accuracy 相对当前 baseline 有显著提升，同时 depth 幻觉率下降。
2. 模型能够稳定输出 `(X, Y, Depth)` 并通过闭环评测验证几何一致性。
3. 在不显著牺牲 2D 指向能力的前提下，建立了可复现的 3D 升级训练范式。

---

## 6. 我建议你下一步立刻做的 3 件事

1. 先完成 Phase 1（评测闭环修复），否则后续所有提升都不可证。  
2. 以 Phase 2 的“兼容式 3D 对齐”为主线，先拿到可量化提升。  
3. 等 A 阶段结果稳定后，再评估是否投入 B 阶段的真实 Dual-Tower 工程。

---

## 7. 计划状态

- 当前状态：Draft v1（可执行）
- 依赖条件：3D 训练数据准备 + 3D eval v2 落地
- 预计周期：
  - A 阶段：2-4 周
  - B 阶段（可选）：4-8 周
