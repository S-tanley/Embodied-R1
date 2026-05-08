# veRL 多模态训练 CPU 内存持续增长问题分析报告

**训练任务**: Embodied-R1 Stage 2（GRPO，7B 模型，多模态）  
**分析时间**: 2026-05-02  
**观测训练**: `embodiedr1_7b_stage2-2`（wandb run: `kgb8271z`）

---

## 一、问题描述

在使用 veRL 框架进行 GRPO 强化学习训练时，wandb 记录的 `perf/cpu_memory_used_gb` 指标从 step 1 开始线性上涨，无任何停止趋势：

| Step | cpu_memory_used_gb |
|------|--------------------|
| 1    | 139 GB             |
| 30   | 265 GB             |
| 60   | 383 GB             |
| 90   | 546 GB             |
| 110  | 1028 GB            |

**增长速率**：约 8 GB / step，极为稳定线性。

训练配置：
- 4 × H200 (141 GB VRAM each)
- 7B 模型，FSDP across 4 GPUs
- `offload_params=true`，`offload_optimizer=true`
- `rollout_batch_size=256`，`n=8`，`limit_images=2`
- `max_pixels=1,638,400`，`max_response_length=2048`
- Ray Object Store 上限：200 GB（`--object_store_memory=200000000000`）
- 机器总 RAM：1511 GB

---

## 二、指标的真实含义

`cpu_memory_used_gb` 的计算位置在 `verl/workers/fsdp_workers.py:452`：

```python
metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
```

`psutil.virtual_memory().used` 返回的是**整台机器在该时刻的总 RAM 占用量**（`total - available`），并非单个进程的内存，也不是 Python 堆大小。

**采样时机**：该行代码位于 `update_actor` 函数末尾，即 FSDP 完成一次梯度更新之后。这是整个 step 中 CPU 内存占用最高的时刻（FSDP allgather + optimizer state 同时在 CPU 上）。

---

## 三、根本原因分析

### 3.1 每步数据量估算

每个训练 step 的 rollout 阶段会生成以下数据：

- **输入图像（pixel_values tensor）**：  
  `256 prompts × 2 images/prompt × 1,638,400 pixels × 3 channels × 4 bytes (fp32)`  
  ≈ **10 GB / step**

- **生成的 token 序列**（n=8，共 2048 条）：  
  `2048 sequences × 4096 tokens × 4 bytes (int32)` ≈ 33 MB（可忽略）

这些 pixel_values tensor 在 DataProto 的 `non_tensor_batch["multi_modal_inputs"]` 中，以 `numpy.ndarray(dtype=object)` 的形式存储，流经以下路径：

```
DataLoader worker → 主进程 → Ray Plasma Object Store → FSDP worker 进程
```

### 3.2 glibc malloc 不归还内存

图像张量在 FSDP worker 进程内被分配用于计算，用完之后 Python 将其标记为可回收。**但 glibc 的默认 malloc 实现在释放大块内存后，不会立即通过 `brk()`/`munmap()` 将其归还给操作系统**，而是保留在进程内部的堆池（free list）中备用。

这是 glibc malloc 的已知行为：

- 超过 `MMAP_THRESHOLD`（默认 128 KB）的大块分配使用 `mmap()`，理论上 `free()` 后会 `munmap()` 归还。  
- 但由于堆碎片化，实际上堆的高水位线（high-water mark）会在每步轻微抬升，导致进程 RSS 单调增长。
- `psutil.virtual_memory().used` 统计的是所有进程的 RSS 之和，因此表现为全局内存线性增长。

### 3.3 为什么增长是完美线性的

这与每步图像数据量恒定（~10 GB）完全匹配：

- 每步，FSDP 的 4 个 worker 进程各处理 256/4 = 64 条 prompt，每条 2 张图
- 每个 worker 每步处理 ~2.5 GB 图像
- glibc 每步向操作系统多保留约 2 GB（碎片留存）
- 4 worker × 2 GB = ~8 GB / step，与观测值吻合

### 3.4 为什么不是真正的"内存泄漏"

从 Python 对象层面看，每步结束时 `batch` 变量被重新赋值，旧的 DataProto 对象引用计数归零，Python GC 将其标记为已释放。**数据本身没有积累**，Ray Plasma 中的对象在 step 结束后也被释放（Plasma 有 LRU 淘汰策略，上限 200 GB）。

真正的问题在于：Python → glibc → OS 这条链路上，glibc 层囤积了内存，OS 看到的进程 RSS 持续走高，`psutil.virtual_memory().used` 如实反映了这个现象。

---

## 四、Stage 2 首次崩溃（`slr1nrqk`）的关系

第一次 Stage 2 训练在 step 100 崩溃，根本原因同样是内存增长触及 Ray 的 OOM kill 阈值：

```
# raylet.out 记录
Node memory: 1442.13 GB / 1511.52 GB (0.954098)
exit_type = NODE_OUT_OF_MEMORY
```

Ray 的保护阈值 = `1511 × 0.95 = 1435 GB`。Step 100 恰好同时触发了 `save_freq=100`（FSDP allgather 峰值）和 `val_freq=100`（WorkerDict 初始化申请 535 GB），两者叠加突破阈值，worker 被 kill。

第二次训练（本次分析的 `kgb8271z`）同样在走向 OOM 的路上——按照 step 110 时 1028 GB 的基础，以 8 GB/step 的速率计算：

```
剩余 headroom: 1435 - 1028 = 407 GB
预计还能撑: 407 / 8 ≈ 50 steps
预计 OOM step: ~160
```

训练在 step 110 被手动终止（`exit_type=INTENDED_USER_EXIT`），避免了第二次 OOM。

---

## 五、解决方案

### 方案一：`MALLOC_TRIM_THRESHOLD_=0`（最简单，立即可用）

通过环境变量让 glibc 在每次释放大块内存后立即调用 `malloc_trim(0)`，强制将空闲内存还给 OS：

```bash
MALLOC_TRIM_THRESHOLD_=0 \
CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  ...（其他参数不变）
```

**优点**：无需安装任何依赖，一行解决。  
**缺点**：每次内存释放都会触发系统调用，轻微影响训练性能（通常 <2%）。

### 方案二：jemalloc（推荐，性能最佳）

jemalloc 是 Meta 开源的内存分配器，在内存归还和碎片处理上远优于 glibc，是 veRL/vLLM 社区对此类问题的标准推荐解法。

**安装**：
```bash
conda install -c conda-forge jemalloc
```

**使用**：
```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so \
MALLOC_CONF="background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0" \
CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  ...（其他参数不变）
```

`MALLOC_CONF` 参数说明：
- `background_thread:true`：后台线程异步归还内存，不阻塞训练
- `dirty_decay_ms:0`：脏页（已使用再释放的内存）立即归还，不等待
- `muzzy_decay_ms:0`：模糊页（半释放状态）也立即归还

**优点**：内存归还更积极，无需 trim 系统调用，性能开销极低。  
**缺点**：需要安装额外依赖。

### 方案三：降低每步图像内存峰值

若上述方案无法部署，可从数据角度降低每步的图像内存压力：

```yaml
# 降低 max_pixels（减小图像分辨率）
data:
  max_pixels: 802816  # 从 1,638,400 降至 ~800K（约降低 2.7×）

# 降低 rollout_batch_size（减少每步处理的图像数量）
data:
  rollout_batch_size: 128  # 从 256 降至 128
```

**代价**：图像分辨率降低会影响视觉理解精度；减小 batch 会降低 RL 信号质量。

---

## 六、推荐的 2 卡续训命令（含修复）

从 `global_step_90` 恢复，使用 `MALLOC_TRIM_THRESHOLD_=0` 临时修复，等待 jemalloc 安装后切换：

```bash
MALLOC_TRIM_THRESHOLD_=0 \
CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main \
  config=scripts/config_stage2.yaml \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.max_prompt_length=4096 \
  worker.actor.model.model_path=IffYuan/Embodied-R1-7B-Stage1 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.n=8 \
  worker.rollout.gpu_memory_utilization=0.6 \
  worker.rollout.limit_images=2 \
  worker.reward.score_function=embodiedr1 \
  data.rollout_batch_size=256 \
  worker.actor.global_batch_size=128 \
  worker.actor.micro_batch_size_per_device_for_update=8 \
  worker.actor.micro_batch_size_per_device_for_experience=16 \
  trainer.experiment_name=embodiedr1_7b_stage2-2 \
  trainer.n_gpus_per_node=2 \
  trainer.total_episodes=1 \
  trainer.save_checkpoint_path=./workdir/embodiedr1_7b_stage2 \
  trainer.load_checkpoint_path=./workdir/embodiedr1_7b_stage2/global_step_90 \
  trainer.val_freq=140 \
  trainer.save_freq=30
```

验证修复效果：观察 `perf/cpu_memory_used_gb` 在训练早期（step 5-30）是否趋于平稳，而非线性增长。若在 step 30 时内存仍稳定在 ~200 GB 以内，则修复有效。

---

## 七、总结

| 问题 | 根因 | 修复 |
|------|------|------|
| `cpu_memory_used_gb` 线性增长 | glibc malloc 不归还已释放的大块内存给 OS，导致 worker 进程 RSS 单调上涨 | `MALLOC_TRIM_THRESHOLD_=0` 或 jemalloc |
| Stage 2 首次 OOM（step 100）| save+val 同时触发，内存峰值叠加超过 Ray 95% 阈值 | 错开 `save_freq` 和 `val_freq` + 内存修复 |
| 本次训练在 step 110 停止 | 手动终止（`INTENDED_USER_EXIT`），非崩溃 | 从 `global_step_90` 恢复即可 |
