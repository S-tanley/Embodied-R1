# Training Log 解读指南（veRL / Embodied-R1）

> 覆盖范围：日志位置、Checkpoint 位置、所有 wandb metric 含义、如何判断训练是否健康。

---

## 1. 日志在哪里

### 1.1 Console 输出
训练启动后，Ray 主进程直接在终端打印 step-level log。格式示例：

```
Episode: 1/2 | Step: 1/1583
  critic/rewards/mean: 0.32 | actor/pg_loss: -0.012 | ...
```

如果你用 `nohup` 或 `tmux` 跑，输出保存在你重定向的文件里（例如 `nohup.out`）。

### 1.2 wandb 在线
训练时终端会打印类似：
```
wandb: 🚀 View run at https://wandb.ai/<entity>/easy_r1/runs/<run_id>
```
用浏览器打开即可看到实时曲线。项目名固定为 `easy_r1`（来自 `scripts/config_stage1.yaml: trainer.project_name`）。

### 1.3 wandb 本地缓存
所有 metric 数据同步到：
```
Embodied-R1/wandb/run-<timestamp>-<run_id>/
├── files/
│   ├── config.yaml          # 本次运行的完整超参
│   ├── output.log           # console 输出的镜像
│   └── wandb-summary.json   # 最终 summary（最后一步的值）
└── run-<run_id>.wandb       # 二进制 event 文件（wandb 用来同步）
```

### 1.4 Checkpoint
保存路径由 `trainer.default_local_dir` 控制，smoke test 里设为 `workdir/smoke_test_stage1`：
```
workdir/smoke_test_stage1/
├── global_step_100/
│   ├── actor/               # 模型权重（FSDP sharded）
│   └── dataloader.pt        # dataloader 状态（用于断点续训）
└── latest_global_step.txt   # 记录最新 step 编号
```

加载 checkpoint 推理：先用 `verl` 自带的合并脚本把 FSDP shard 合成完整 HuggingFace 权重，或直接用 `trainer.default_hdfs_dir` 指向 HF 格式保存路径（需在 yaml 里配置）。

---

## 2. 所有 Metric 含义

### 2.1 reward/ — 奖励函数输出

| Metric | 含义 |
|--------|------|
| `reward/overall` | 综合奖励分，是 thinking_format + spatial_qa_accuracy 的加权结果。**最重要的训练信号**，应随训练单调上升。 |
| `reward/thinking_format` | 格式奖励：模型输出是否包含合法的 `<think>...</think><answer>...</answer>` 结构。格式正确得 1，否则 0（或部分分）。训练早期波动大，中后期应趋近 1.0。 |
| `reward/spatial_qa_accuracy` | 任务精度奖励：对于空间 QA 任务（Whatsup、SAT 数据集），答案是否与 ground truth 一致。Stage1 的核心指标之一。 |
| `reward/general_qa_accuracy` | 任务精度奖励：对于通用视觉推理 QA 任务（ViRL 数据集），答案是否与 ground truth 一致。Stage1 中与 `spatial_qa_accuracy` 并列的另一类 QA 精度指标。 |

代码来源：`verl/utils/reward_score/embodiedr1.py` → `embodiedr1_compute_score()` 返回 dict，然后在 `ray_trainer.py` 加上 `reward/` 前缀。

---

### 2.2 critic/ — 轨迹层面统计量

这些是在 rollout 结束后、policy update 之前对整条生成序列的统计，反映当前 batch 的质量分布。

| Metric | 含义 |
|--------|------|
| `critic/score/mean` | 当前 batch 里每条轨迹的总奖励分均值（即 `token_level_scores.sum(-1)` 的均值）。 |
| `critic/score/max/min` | 同上，最大/最小值。两者差值大说明 batch 内差异大，GRPO 有更多可利用的信号。 |
| `critic/rewards/mean` | KL 惩罚后的实际奖励均值（`= score - kl_coef × KL`）。与 score 接近时说明 KL 惩罚不强。 |
| `critic/rewards/max/min` | 同上的最大/最小值。 |
| `critic/advantages/mean` | GRPO advantage 均值。理论上 GRPO 做 group 内归一化，均值应接近 0。若持续偏离 0 说明归一化异常。 |
| `critic/advantages/max/min` | advantage 的极端值，反映方差。方差太小说明 rollout 结果高度一致，策略已收敛（或退化）。 |
| `critic/returns/mean` | 折扣累计回报均值（GRPO 中等于 advantage，因为无 value baseline）。 |
| `critic/kl` | 当前策略与 ref policy 之间每 token KL 散度的均值。**应保持在合理范围（0.01~0.5）**，过大说明策略跑偏，需检查 kl_coef。 |
| `critic/kl_coef` | 自适应 KL 控制器当前的系数值（如果用了 `adaptive_kl`，会随训练动态调整）。 |

---

### 2.3 actor/ — Policy Update 层面

这些在每次梯度更新时记录，反映优化过程是否健康。

| Metric | 含义 |
|--------|------|
| `actor/pg_loss` | Policy Gradient loss（PPO-clip 损失）。**正常应为小负数或接近 0**。绝对值不断增大说明训练不稳定。 |
| `actor/pg_clipfrac_higher` | 概率比 `π/π_old` **超过** `1+clip_ratio_high` 的 token 比例。代表"过于激进"的更新被截断的比例。健康范围约 0.05~0.3，持续接近 1 说明 lr 太大或 rollout 步数太少。 |
| `actor/pg_clipfrac_lower` | 概率比低于 `1-clip_ratio_low` 被截断的比例。代表"过于保守"的更新。 |
| `actor/entropy_loss` | 策略熵的负值估计（`-mean(log_probs)`）。**应缓慢下降**。若太低（接近 0）说明策略已经退化为确定性输出，探索不足。 |
| `actor/ppo_kl` | PPO 视角下的 KL（从 log_probs - old_log_probs 直接估算），不同于 critic/kl（那个用 ref policy）。用于监控单步更新幅度。 |
| `actor/grad_norm` | 梯度范数（clip 前）。**正常范围约 0.1~5**。持续 >10 说明梯度爆炸，需降 lr 或开启梯度裁剪。 |
| `actor/lr` | 当前学习率（受 warmup/scheduler 影响）。 |
| `actor/kl_loss` | 若开启了 `use_kl_loss=True`，这是加入总 loss 的 KL 项值。默认配置下为 0（KL 通过 reward 惩罚而非 loss 项实现）。 |
| `actor/kl_coef` | 同上，对应的 kl 系数。 |

---

### 2.4 response_length / prompt_length — 长度统计

| Metric | 含义 |
|--------|------|
| `response_length/mean` | 生成回复的平均 token 数。Stage1 正常范围取决于 `max_response_length`（默认 8192）。若训练中持续下降，说明模型学到了"偷懒"输出短回复。 |
| `response_length/clip_ratio` | 生成长度触达 `max_response_length` 的比例。若 clip_ratio 高（>0.2），说明模型经常超限被截断，考虑增大 `max_response_length`。 |
| `prompt_length/mean` | 输入 prompt 的平均 token 数。训练过程中应基本稳定，突变说明数据有问题。 |
| `prompt_length/clip_ratio` | prompt 被截断的比例。若 >0 说明有样本 prompt 超过了 `max_prompt_length`，被截断会导致信息丢失。 |

---

### 2.5 timing_s / timing_per_token_ms — 各阶段耗时

每个 step 内各子阶段的挂钟时间（秒）和每 token 时间（毫秒）。

| Key | 对应阶段 |
|-----|----------|
| `gen` | vLLM 生成（rollout）阶段 |
| `reward` | 奖励函数计算 |
| `ref` | Ref policy 前向（计算 ref log_probs） |
| `old` | Old policy 前向（计算 old log_probs，PPO 需要） |
| `adv` | Advantage 计算 |
| `update_actor` | Actor 梯度更新 |
| `step` | 整个 step 总耗时 |

`gen` 通常占比最大（vLLM 自回归生成慢）。若 `update_actor` 远大于 `gen`，说明 batch size 太大或 micro_batch 配置不合理。

---

### 2.6 perf/ — 性能指标

| Metric | 含义 |
|--------|------|
| `perf/throughput` | 每秒每 GPU 处理的 token 数（tokens/s/GPU）。用于横向对比不同配置的效率。 |
| `perf/time_per_step` | 每 step 总耗时（秒）。 |
| `perf/total_num_tokens` | 当前 batch 总 token 数（prompt + response）。 |
| `perf/mfu_actor` | Actor 的 Model FLOPs Utilization（MFU），越高越好，H200 上通常 30~60%。 |
| `perf/max_memory_allocated_gb` | PyTorch 峰值显存占用（GB）。 |
| `perf/max_memory_reserved_gb` | PyTorch 保留显存（含碎片），通常比 allocated 大。 |
| `perf/cpu_memory_used_gb` | 系统内存使用（GB），Ray 对象存储也计入其中。 |

---

### 2.7 val/ — 验证集指标

验证集评估在每隔 `trainer.test_freq` 步或 episode 结束时触发（如果配置了 val 数据集）。

| Metric | 含义 |
|--------|------|
| `val/reward_score` | 验证集平均奖励分（综合）。**判断模型是否真的在提升，而非只是训练集过拟合**。 |
| `val/overall_reward` | 综合分的 val 版本（对应 reward/overall）。 |
| `val/thinking_format_reward` | 格式得分的 val 版本。 |
| `val/spatial_qa_accuracy_reward` | 空间 QA 精度得分的 val 版本（对应 `reward/spatial_qa_accuracy`）。 |
| `val/general_qa_accuracy_reward` | 通用 QA 精度得分的 val 版本（对应 `reward/general_qa_accuracy`）。 |

---

## 3. 如何判断训练是否健康

### 正常训练的典型曲线

```
reward/overall        ↑ 缓慢上升（初期可能有震荡）
reward/spatial_qa_accuracy  ↑ 上升（Stage1 核心指标）
reward/general_qa_accuracy    ↑ 上升（ViRL 通用推理 QA）
reward/thinking_format     → 趋近 1.0（格式很快收敛）
actor/pg_loss         ↓ 绝对值缓慢减小或平稳
actor/entropy_loss    ↓ 缓慢下降（策略在收敛）
critic/kl             → 稳定在 0.01~0.5
actor/grad_norm       → 稳定，不持续上升
response_length/mean  → 基本稳定（或略微增长）
```

### 危险信号

| 现象 | 可能原因 | 建议处理 |
|------|----------|----------|
| `reward/overall` 不上升甚至下降 | 奖励函数 bug / 数据问题 / lr 太大 | 先检查 reward 函数输出，再看 pg_loss 是否发散 |
| `actor/grad_norm` > 20 且持续 | 梯度爆炸 | 降低 lr；检查 `max_grad_norm` 是否设置 |
| `critic/kl` > 1.0 | 策略跑偏 ref policy 太远 | 增大 `kl_coef`；或降低 lr |
| `response_length/clip_ratio` > 0.3 | 模型经常生成超长回复被截断 | 增大 `max_response_length`；检查是否有死循环输出 |
| `reward/thinking_format` 停留在 0 | 模型完全不输出合法格式 | 检查 prompt template；可能需要先做 SFT 预热 |
| 显存 OOM | batch size 太大 | 减小 `micro_batch_size_per_device_for_update` |

---

## 4. 快速查看当前训练状态的命令

```bash
# 查看最新 wandb summary（训练结束后或中途）
cat wandb/run-<timestamp>-<run_id>/files/wandb-summary.json | python3 -m json.tool

# 查看 console 输出（如果用了 nohup）
tail -f nohup.out

# 查看最新 checkpoint
cat workdir/<exp_name>/latest_global_step.txt
ls workdir/<exp_name>/global_step_$(cat workdir/<exp_name>/latest_global_step.txt)/

# 检查训练是否还在跑（Ray 进程）
ps aux | grep "ray\|verl\|python" | grep -v grep
```

---

## 5. Checkpoint 加载与导出

veRL 保存的 actor 权重是 **FSDP sharded** 格式（每张卡保存一部分），不能直接用 `from_pretrained` 加载。

导出为 HuggingFace 格式的方法：

```bash
# 官方脚本（verl 内置）
python -m verl.utils.hdfs_io \
  --src workdir/<exp_name>/global_step_<N>/actor \
  --dst output_hf/<exp_name>_step<N>

# 或者在训练 yaml 里配置 default_hdfs_dir，训练结束自动导出
```

导出后就可以用 `eval/` 里的脚本直接评测。
