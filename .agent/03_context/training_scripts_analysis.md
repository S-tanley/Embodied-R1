# Embodied-R1 训练脚本深度分析

> 分析日期：2026-04-23  
> 覆盖范围：scripts + verl 训练主干  
> 目标：回答 4 个问题
> 1) 这个项目怎么 training
> 2) 有哪些 training 模式
> 3) 每个模式可调参数有哪些
> 4) 每个模式应该怎么跑

---

## 1. 训练是怎么跑起来的

### 1.1 训练入口

统一入口是：

```bash
python3 -m verl.trainer.main config=... [Hydra/OmegaConf 覆盖参数]
```

核心逻辑：

1. 读取默认结构化配置 `PPOConfig`，再 merge `config=xxx.yaml`，最后 merge 命令行覆盖项。  
2. 启动 Ray（本地未初始化时会自动 `ray.init(...)`）。  
3. `Runner` 在 Ray remote 进程里执行：创建 tokenizer/processor、创建 worker 角色、创建奖励函数、启动 `RayPPOTrainer.fit()`。

结论：这套训练不是传统单进程循环，而是 Ray + FSDP + vLLM 的分布式 PPO/GRPO 框架。

---

### 1.2 训练数据流（每个 step）

训练主循环（简化）是：

1. DataLoader 读 batch（`RLHFDataset`，支持多图多模态）  
2. ActorRollout 生成 responses（vLLM）  
3. 奖励函数计算 token-level 分数（rule-based）  
4. 计算 old log probs（actor）  
5. 计算 ref log probs（如果 KL 开启）  
6. 计算 values（只有 GAE 需要 critic）  
7. 计算 advantage（GRPO/GAE/RLOO/REMAX/REINFORCE++）  
8. 更新 critic（如果启用）  
9. 更新 actor（PPO + dual clip）  
10. 按频率做 validation 和 checkpoint

---

### 1.3 Stage1 / Stage2 的关系

项目官方脚本是两阶段：

1. Stage1：空间问答（QA）强化训练  
2. Stage2：点位/轨迹任务强化训练（从 Stage1 checkpoint 继续）

两者共用同一训练框架，主要差异在数据集与启动脚本参数覆盖。

---

## 2. 训练模式总览

这里把“模式”分成 6 类，因为这个仓库的模式切换不是单一开关，而是多维组合。

### 2.1 模式 A：按训练阶段

1. `Stage1 QA`：`scripts/config_stage1.yaml` 的 QA 数据集。  
2. `Stage2 Pointing`：`scripts/config_stage2.yaml` 的点定位/轨迹数据集。

### 2.2 模式 B：按优势估计器（算法模式）

可选值：

1. `grpo`
2. `gae`
3. `rloo`
4. `remax`
5. `reinforce_plus_plus`

区别：

1. `gae` 需要 critic。  
2. 其余 4 个不需要 critic。  
3. `grpo` 和 `rloo` 强制要求 `worker.rollout.n > 1`。

### 2.3 模式 C：按 KL 处理方式

1. `disable_kl=true`：完全不使用参考策略 KL。  
2. `disable_kl=false + use_kl_loss=false`：KL 作为 reward shaping（加到 token reward）。  
3. `disable_kl=false + use_kl_loss=true`：KL 作为 actor loss 项（策略损失里加 KL）。

额外可调：`kl_penalty`（kl/abs/mse/low_var_kl/full）与 `kl_type`（fixed/adaptive）。

### 2.4 模式 D：按 reward function（任务定义模式）

当前可选打分函数：

1. `math`
2. `r1v`
3. `embodiedr1`
4. `embodiedr1_nothinking`
5. `embodiedr1_3d`

Embodied-R1 训练一般使用 `embodiedr1`。

### 2.5 模式 E：按并行/系统策略

1. `hybrid_engine=true`（默认且必需）：FSDP 训练 + vLLM rollout。  
2. FSDP 分片模式：full shard / shard grad op / hybrid shard（由 `fsdp_size` + `enable_full_shard` 决定）。  
3. vLLM 张量并行：`worker.rollout.tensor_parallel_size`。  
4. Ulysses 序列并行：`ulysses_sequence_parallel_size`。  
5. 参数/优化器 offload：`offload_params`、`offload_optimizer`。

### 2.6 模式 F：按训练流程控制

1. 正常训练（默认）。  
2. `val_before_train=true`：训练前先跑验证。  
3. 周期验证：`val_freq`。  
4. 周期存档：`save_freq`。  
5. 恢复训练：`trainer.load_checkpoint_path=.../global_step_xxx`。  
6. `max_steps` 覆盖 `total_episodes`。

注意：`val_only` 字段虽然在配置里定义了，但当前训练循环没有使用该字段。

---

## 3. 每个模式可调参数（按配置块）

## 3.1 Data 参数

常用参数：

1. `data.train_files` / `data.val_files`：训练/验证 parquet 列表。  
2. `data.prompt_key` / `answer_key` / `image_key`：字段名映射。  
3. `data.max_prompt_length` / `max_response_length`：输入截断与生成长度上限。  
4. `data.rollout_batch_size`：每次 rollout 的 prompt batch。  
5. `data.val_batch_size`：验证 batch（`-1` 可等于全量）。  
6. `data.shuffle` / `seed`。  
7. `data.max_pixels` / `min_pixels`：图像分辨率约束。

关键约束：

1. `rollout_batch_size % actor.global_batch_size == 0`。  
2. `rollout_batch_size * rollout.n % actor.micro_batch_size_per_device_for_experience == 0`。  
3. 若启用 critic，同样要满足 critic 对应整除约束。

---

## 3.2 Algorithm 参数

1. `algorithm.adv_estimator`：`grpo/gae/rloo/remax/reinforce_plus_plus`。  
2. `algorithm.gamma`、`lam`：主要用于 GAE 与 REINFORCE++。  
3. `algorithm.disable_kl`。  
4. `algorithm.use_kl_loss`。  
5. `algorithm.kl_penalty`：`kl/abs/mse/low_var_kl/full`。  
6. `algorithm.kl_coef`。  
7. `algorithm.kl_type`：`fixed/adaptive`。  
8. `algorithm.kl_target`、`kl_horizon`（adaptive KL 时使用）。

---

## 3.3 Actor 参数

1. `worker.actor.global_batch_size`。  
2. `worker.actor.micro_batch_size_per_device_for_update`。  
3. `worker.actor.micro_batch_size_per_device_for_experience`。  
4. `worker.actor.ppo_epochs`。  
5. `worker.actor.max_grad_norm`。  
6. `worker.actor.clip_ratio_low` / `clip_ratio_high` / `clip_ratio_dual`（dual-clip PPO 关键）。  
7. `worker.actor.padding_free`。  
8. `worker.actor.ulysses_sequence_parallel_size`。  
9. `worker.actor.use_torch_compile`。  
10. `worker.actor.model.*`：`model_path`、`tokenizer_path`、`enable_gradient_checkpointing`、`freeze_vision_tower` 等。  
11. `worker.actor.optim.*`：`lr`、`betas`、`weight_decay`、`strategy`、`lr_warmup_ratio`、`min_lr_ratio`。  
12. `worker.actor.fsdp.*`：`enable_full_shard`、`enable_cpu_offload`、`enable_rank0_init`、`fsdp_size`、`mp_*_dtype`。  
13. `worker.actor.offload.*`：`offload_params`、`offload_optimizer`。

---

## 3.4 Critic 参数（仅 GAE 必须）

1. `worker.critic.global_batch_size`。  
2. `worker.critic.micro_batch_size_per_device_for_update`。  
3. `worker.critic.micro_batch_size_per_device_for_experience`。  
4. `worker.critic.ppo_epochs`。  
5. `worker.critic.max_grad_norm`。  
6. `worker.critic.cliprange_value`（value clipping）。  
7. 其余 model/optim/fsdp/offload 与 actor 类似。

---

## 3.5 Rollout（vLLM）参数

1. `worker.rollout.n`：每个 prompt 采样几个 response。  
2. `worker.rollout.temperature`、`top_p`、`top_k`。  
3. `worker.rollout.tensor_parallel_size`。  
4. `worker.rollout.gpu_memory_utilization`。  
5. `worker.rollout.max_num_batched_tokens`、`max_num_seqs`。  
6. `worker.rollout.limit_images`。  
7. `worker.rollout.enforce_eager`、`enable_chunked_prefill`。  
8. `worker.rollout.val_override_config`（验证时覆盖采样策略）。

经验上最常调：`n`、`tensor_parallel_size`、`gpu_memory_utilization`、`temperature`。

---

## 3.6 Reward / Trainer 参数

Reward：

1. `worker.reward.score_function`（最关键）。  
2. `worker.reward.skip_special_tokens`。

Trainer：

1. `trainer.total_episodes` / `max_steps`。  
2. `trainer.n_gpus_per_node` / `nnodes`。  
3. `trainer.critic_warmup`（前若干步只训 critic，不更新 actor）。  
4. `trainer.val_freq`、`val_before_train`、`val_generations_to_log`。  
5. `trainer.save_freq`、`save_limit`。  
6. `trainer.save_checkpoint_path`、`load_checkpoint_path`。  
7. `trainer.logger`（console/wandb）。

---

## 4. 默认 Stage 脚本实际做了什么

## 4.1 Stage1 脚本

文件：`scripts/stage_1_embodied_r1.sh`

关键覆盖（相对 YAML）：

1. 模型切到 `Qwen/Qwen2.5-VL-3B-Instruct`（YAML 里是 7B 默认）。  
2. `worker.rollout.n=8`（YAML 默认 5）。  
3. `worker.reward.score_function=embodiedr1`（YAML 默认 `math`）。  
4. `trainer.total_episodes=2`（这更像快速跑通，不是论文完整训练）。

## 4.2 Stage2 脚本

文件：`scripts/stage_2_embodied_r1.sh`

在 Stage1 基础上：

1. 用 `config_stage2.yaml` 切换到点任务数据集。  
2. 通过 `trainer.load_checkpoint_path` 从 Stage1 checkpoint 继续。  
3. 默认 `trainer.total_episodes=1`（同样偏快速实验）。

---

## 5. 每种模式怎么跑（可直接复制）

以下命令均在仓库根目录执行。

## 5.1 官方两阶段（推荐起点）

```bash
# Stage1
bash scripts/stage_1_embodied_r1.sh

# Stage2（先把脚本里的 EMBODIED_R1_STAGE_1_CHECKPOINT_PATH 改成真实路径）
bash scripts/stage_2_embodied_r1.sh
```

---

## 5.2 手工启动 Stage1（完整训练版示例）

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
RUN_NAME=embodiedr1_stage1_full

python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.max_prompt_length=4096 \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.rollout.tensor_parallel_size=2 \
  worker.rollout.n=8 \
  worker.rollout.gpu_memory_utilization=0.6 \
  worker.rollout.limit_images=2 \
  worker.reward.score_function=embodiedr1 \
  trainer.experiment_name=${RUN_NAME} \
  trainer.n_gpus_per_node=8 \
  trainer.total_episodes=15 \
  trainer.save_checkpoint_path=./workdir/${RUN_NAME} \
  trainer.val_freq=5 \
  trainer.save_freq=5
```

---

## 5.3 Stage2 接 Stage1 checkpoint

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
RUN_NAME=embodiedr1_stage2_full
STAGE1_CKPT=./workdir/embodiedr1_stage1_full/global_step_XXXX

python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.max_prompt_length=4096 \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.rollout.tensor_parallel_size=2 \
  worker.rollout.n=8 \
  worker.rollout.gpu_memory_utilization=0.6 \
  worker.rollout.limit_images=2 \
  worker.reward.score_function=embodiedr1 \
  trainer.experiment_name=${RUN_NAME} \
  trainer.n_gpus_per_node=8 \
  trainer.total_episodes=15 \
  trainer.save_checkpoint_path=./workdir/${RUN_NAME} \
  trainer.load_checkpoint_path=${STAGE1_CKPT} \
  trainer.val_freq=5 \
  trainer.save_freq=5
```

---

## 5.4 切换算法模式（Adv Estimator）

### GRPO（默认）

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.adv_estimator=grpo \
  worker.rollout.n=8
```

### RLOO

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.adv_estimator=rloo \
  worker.rollout.n=8
```

### REMAX

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.adv_estimator=remax
```

### REINFORCE++

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.adv_estimator=reinforce_plus_plus
```

### GAE（需要 critic）

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.adv_estimator=gae \
  worker.rollout.n=1 \
  worker.critic.global_batch_size=128 \
  worker.critic.micro_batch_size_per_device_for_update=4 \
  worker.critic.micro_batch_size_per_device_for_experience=16
```

---

## 5.5 切换 KL 模式

### KL 作为 reward shaping（常见）

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.disable_kl=false \
  algorithm.use_kl_loss=false \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_coef=1e-2
```

### KL 作为 actor loss

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.disable_kl=false \
  algorithm.use_kl_loss=true \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_coef=1e-2
```

### 完全关闭 KL

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage1.yaml \
  algorithm.disable_kl=true \
  algorithm.kl_coef=0
```

---

## 5.6 切换 reward function 模式

### no-thinking 训练

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  worker.reward.score_function=embodiedr1_nothinking
```

### 3D 训练

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  worker.reward.score_function=embodiedr1_3d
```

---

## 5.7 断点续训

```bash
python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  trainer.load_checkpoint_path=./workdir/embodiedr1_stage2_full/global_step_XXXX \
  trainer.save_checkpoint_path=./workdir/embodiedr1_stage2_resume
```

注意：`load_checkpoint_path` 必须直接指向 `global_step_xxx` 目录。

---

## 5.8 导出 HuggingFace 可用模型

训练产出的 FSDP 分片可用下面脚本合并：

```bash
python scripts/model_merger.py \
  --local_dir ./workdir/embodiedr1_stage2_full/global_step_XXXX/actor
```

合并结果在：

```text
./workdir/.../actor/huggingface
```

---

## 6. 你真正开跑前要先确认的 8 件事

1. `datasets/` 目录和 parquet 文件当前仓库里没有，需要你自行准备（YAML 用的是相对路径）。  
2. Stage1/Stage2 shell 脚本默认 episode 很小（2 和 1），更像 smoke test。  
3. Stage 脚本会把 reward 从 `math` 覆盖成 `embodiedr1`，不要漏这个覆盖。  
4. `grpo/rloo` 必须保证 `worker.rollout.n > 1`。  
5. 若你要开 GAE，请重点检查 critic 分支是否与你当前代码版本兼容。  
6. `val_only` 当前未在训练循环里生效，不要依赖这个开关。  
7. 恢复训练路径要填 `.../global_step_xxx`，不是外层目录。  
8. 多卡和 TP 参数要匹配总 GPU 数，尤其是 `n_gpus_per_node` 与 `rollout.tensor_parallel_size`。

---

## 7. 训练相关文件索引

1. 训练入口：`verl/trainer/main.py`  
2. 主循环：`verl/trainer/ray_trainer.py`  
3. 全局配置：`verl/trainer/config.py`  
4. 算法实现：`verl/trainer/core_algos.py`  
5. Stage 启动脚本：`scripts/stage_1_embodied_r1.sh`、`scripts/stage_2_embodied_r1.sh`  
6. Stage 配置：`scripts/config_stage1.yaml`、`scripts/config_stage2.yaml`  
7. Worker 配置：`verl/workers/*/config.py`、`verl/workers/config.py`  
8. FSDP/vLLM 协同：`verl/workers/fsdp_workers.py`、`verl/workers/sharding_manager/fsdp_vllm.py`  
9. 奖励路由：`verl/workers/reward/custom.py`、`verl/utils/reward_score/embodiedr1.py`  
10. Checkpoint 合并：`scripts/model_merger.py`
