# Embodied-R1 Agent 工作区状态

> 最后更新：2026-04-23

---

## 当前阶段

**准备阶段**：环境搭建、数据下载、eval baseline 建立。尚未开始训练。

---

## 已完成事项

### 1. 项目理解
- 读完 `research.md`，深度理解了项目架构（Qwen2.5-VL-3B + GRPO + veRL + 两阶段 RFT）。
- 确定毕设方向为 **Plan B：进阶 3D 空间推理**，计划文档见 `01_plans/step1_planB_3d_spatial_reasoning.md`。

### 2. Eval 脚本分析与修复
- 分析了全部 11 个 eval 脚本，记录在 `03_context/eval_scripts_analysis.md`。
- **10 个脚本可直接运行**，1 个暂不可运行（`hf_inference_crpe.py`，需 COCO ~20GB + 私有数据集申请）。
- 修复了 4 个依赖内部服务器路径的脚本，详情见 `02_reports/eval_scripts_fix_report.md`：
  - `hf_inference_vabench_point.py`：改路径
  - `hf_inference_vabench_visual_trace.py`：改路径
  - `hf_inference_roborefit.py`：从 JSON+文件路径迁移到 parquet+内嵌 bytes，修复了 `np.int32` JSON 序列化 bug
  - `hf_inference_3d.py`：从 JSON+文件路径迁移到 parquet+内嵌 bytes

### 3. Eval 数据下载
所有 eval 数据已下载到 `eval/data/`：

| 文件 | 大小 | 样本数 | 来源 |
|------|------|--------|------|
| `eval/data/vabench_point_test.parquet` | 26 MB | 300 | `IffYuan/Embodied-R1-Dataset` |
| `eval/data/vabench_trace_test.parquet` | 26 MB | 300 | `IffYuan/Embodied-R1-Dataset` |
| `eval/data/roborefit_test_0.parquet` | 184 MB | 1000 | `IffYuan/Roborefit` |
| `eval/data/roborefit_test_1.parquet` | 181 MB | 1000 | `IffYuan/Roborefit` |
| `eval/data/open6dor_test.parquet` | 136 MB | 1618 | `IffYuan/Embodied-R1-Dataset` |

### 4. 训练脚本分析
分析了两阶段训练脚本，记录在 `03_context/training_scripts_analysis.md`。
覆盖内容：训练入口、两阶段关系、可调参数、各模式启动命令。

---

## 待完成事项

### 【进行中】训练数据下载
- 已生成 `download_train_data.py` 脚本，放在项目根目录。
- 数据全部来自 `IffYuan/Embodied-R1-Dataset`，下载到 `datasets/` 目录（与 YAML config 路径一致）。
- **运行命令**：`python download_train_data.py`（不要用 `uv run`）
- 预计总大小 10–20 GB。

**Stage 1 数据（3 train + 2 test）**：
- `datasets/whatsup_rft_spatial_qa_train_4000_test_138_0417/`
- `datasets/SAT_rft_spatial_qa_train_80000_test_4000_0428/`
- `datasets/ViRL_rft_general_qa_train_17831_test_0_plus_0428/`

**Stage 2 数据（6 train + 5 test）**：
- `datasets/robopoint_rft_point_ref_train_40000_test_2000_0417/`
- `datasets/FSD_points_rft_fsd_free_point_train_32790_test_300_0425/`
- `datasets/FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/`
- `datasets/roborefit_rft_point_rec_train_35000_test_1000_0502/`
- `datasets/refcoco_rft_point_rec_train_20000_test_189_0502/`
- `datasets/handal_rft_grounding_rec_train_40000_test_1000_0503/`

### 【待做】flash-attn 安装
训练和部分 eval 需要 flash-attn。当前 uv 环境中未安装。
```bash
uv pip install flash-attn --no-build-isolation
```

### 【待做】运行 Eval Baseline
准备好后在 4 张 H200（GPU 2/3/4/5）上跑基线，估计总时间约 3 小时。

**GPU 2**（~171min）：
```bash
CUDA_VISIBLE_DEVICES=2 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54322 eval/hf_inference_vabench_point.py
CUDA_VISIBLE_DEVICES=2 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54323 eval/hf_inference_vabench_visual_trace.py
CUDA_VISIBLE_DEVICES=2 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54324 eval/hf_inference_3d.py
```

**GPU 3**（~134min）：
```bash
CUDA_VISIBLE_DEVICES=3 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54325 eval/hf_inference_roborefit.py
CUDA_VISIBLE_DEVICES=3 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54326 eval/hf_inference_spatial_relationship.py
```

**GPU 4**（~165min）：
```bash
CUDA_VISIBLE_DEVICES=4 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54327 eval/hf_inference_whatsup.py
CUDA_VISIBLE_DEVICES=4 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54328 eval/hf_inference_sat.py
```

**GPU 5**（~146min）：
```bash
CUDA_VISIBLE_DEVICES=5 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54329 eval/hf_inference_refcoco.py
CUDA_VISIBLE_DEVICES=5 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54330 eval/hf_inference_robopoint.py
CUDA_VISIBLE_DEVICES=5 python -m accelerate.commands.launch --num_processes=1 --main_process_port=54331 eval/hf_inference_handal.py
```

### 【待做】Plan B Phase 0：基线冻结与数据契约
见 `01_plans/step1_planB_3d_spatial_reasoning.md` Phase 0 节。
核心：定义 3D 训练样本 schema，统一深度单位。

### 【待做】Plan B Phase 1：3D Eval 闭环修复
修改 `hf_inference_3d.py`：真实解析 `<depth>`，使用 RGB+Depth 双图，建立 depth 幻觉率指标。

---

## 环境状态

| 项目 | 状态 |
|------|------|
| Python venv | `.venv`（uv 管理） |
| flash-attn | **未安装**，需 `uv pip install flash-attn --no-build-isolation` |
| 可用 GPU | H200 × 4，编号 2/3/4/5 |
| 模型（eval 用） | HuggingFace 在线加载 `IffYuan/Embodied-R1-3B-v1` |
| eval 数据 | `eval/data/` 已就绪 |
| 训练数据 | `datasets/` 待下载 |

---

## 文件索引

### 计划文档（01_plans）

**`.agent/01_plans/step1_planB_3d_spatial_reasoning.md`**  
毕设 Plan B 的完整执行计划。分两大阶段：阶段 A（兼容式 3D 升级，2–4 周，主线）和阶段 B（Dual-Tower/点云，可选扩展）。主线细分 Phase 0–3：Phase 0 定义数据 schema 与深度单位，Phase 1 修复 3D eval 闭环（去掉固定 depth=1），Phase 2 做 RGB-D 对齐微调，Phase 3 做 3D GRPO 强化训练。包含 2 张 H200 上的推荐超参、实验矩阵、成功标准，以及当前已知的 4 个硬约束（3D eval 深度固定、训练数据缺失、`embodiedr1_3d.py` 依赖风险、vLLM 兼容限制）。

---

### 修复报告（02_reports）

**`.agent/02_reports/eval_scripts_fix_report.md`**  
记录了 4 个 eval 脚本从「内部服务器路径 + 本地文件图像」迁移到「HuggingFace parquet + 内嵌 bytes」的全部改动。包含：每个脚本的改动类型与具体 diff、数据下载命令（含正确的 HF 子目录路径，以及踩坑记录：`IffYuan/VABench-P` 字段结构不兼容，正确源是 `IffYuan/Embodied-R1-Dataset` 的 FSD 子集）、发现并修复的 `np.int32` 不可 JSON 序列化 bug、端到端 mock 验证结论。

---

### 上下文资料（03_context）

**`.agent/03_context/research.md`**  
项目架构的完整深度分析。覆盖：四种核心指向能力（VTG/RRG/REG/OFG）的定义与区别；统一输出范式 `<think>...</think><answer><point>[[x,y],...]</point></answer>`；两阶段 RFT 训练策略（Stage 1 空间 QA → Stage 2 像素级指向）；veRL 框架（Ray + FSDP + vLLM HybridEngine）的工作原理；GRPO 奖励计算与评测指标体系；论文中 3D 能力的现有局限（eval 深度固定为 1、无真实 depth 输入）。**读懂这个文件是理解整个项目的前提。**

**`.agent/03_context/eval_scripts_analysis.md`**  
对 `eval/` 目录下全部 11 个脚本的系统分析，按脚本逐一记录：任务类型（VTG/REG/REF/3D 等）、评测数据集来源、核心评分指标（bbox 命中率、trajectory DTW、3D relation accuracy 等）、关键可调参数（`reasoning_model`、`max_pixels`、`gen_args`）、推荐运行命令、注意事项。另含 1 个暂不可运行的脚本说明（`hf_inference_crpe.py`：需 COCO ~20GB + 私有数据集 + flash-attn）。

**`.agent/03_context/training_scripts_analysis.md`**  
对训练体系的完整拆解。覆盖：训练入口（`verl.trainer.main` + Hydra 配置）；每个 step 的数据流（DataLoader → vLLM rollout → reward → GRPO advantage → actor update）；Stage1/Stage2 的差异（仅数据集与 checkpoint 路径不同）；6 个维度的模式说明（训练阶段、优势估计器、KL 处理、reward 函数、并行策略、流程控制）；所有关键参数的含义与约束（如 `rollout_batch_size % global_batch_size == 0`）；可直接复制的各场景启动命令（官方两阶段、完整训练版、断点续训、模型导出）；开跑前必确认的 8 件事。

---

### 数据与脚本

**`download_train_data.py`**（项目根目录）  
训练数据下载脚本。从 `IffYuan/Embodied-R1-Dataset` 下载 Stage 1（3 个数据集）和 Stage 2（6 个数据集）的 train/test parquet，保存到 `datasets/` 目录，与 `scripts/config_stage1.yaml` 和 `config_stage2.yaml` 中的相对路径完全对应。运行方式：`python download_train_data.py`（不要用 `uv run`，会触发 flash-attn 编译）。预计总大小 10–20 GB。
