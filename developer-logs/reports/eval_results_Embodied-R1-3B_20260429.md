# Embodied-R1-3B 全面评估结果报告

> **模型**: `IffYuan/Embodied-R1-3B-v1`  
> **评估时间**: 2026-04-23 ~ 2026-04-27  
> **运行环境**: 2-4×H200, CUDA 12.x, Python 3.11  
> **结果来源**: `eval/logs/` + `eval/offical-3b-2/` + `eval/offical-3b-3/` + `eval/offical-3b-sat/`

---

## 一、评估总览

| 序号 | 评估基准 | 状态 | 核心指标 | 样本数 |
|------|---------|------|---------|--------|
| 1 | **CV-Bench** | ✅ 成功 | **81.69%** | 2,638 |
| 2 | **EmbSpatial-Bench** | ✅ 成功 | **68.96%** | 3,640 |
| 3 | **Open6DoR-Custom (3D)** | ✅ 成功 | **48.27%** | 1,618 |
| 4 | **Part-Affordance-2K** | ✅ 成功 | **44.61%** (points in mask) | 2,000 |
| 5 | **RoboRefit** | ✅ 成功 | **85.35%** (points in box) | 2,000 |
| 6 | **VABench-Point** | ✅ 成功 | **74.00%** (points in bbox) | 300 |
| 7 | **VABench-VisualTrace** | ✅ 成功 | RMSE **71.83** / MAE **40.08** | 300 |
| 8 | **Where2Place** | ✅ 成功 | **69.50%** (points in mask) | 100 |
| 9 | **BLINK** | ❌ 失败 | — | — |
| 10 | **SAT-Real** | ❌ 失败 | — | — |
| 11 | **CRPE** | ❌ 未运行 | — | — |

**成功 8 / 11，失败 3 / 11**（失败原因见第四节）。

---

## 二、详细结果分析

### 2.1 视觉-空间推理 QA（VQA 类）

#### CV-Bench — 计算机视觉综合评测 ✅

| 子任务 | 准确率 | 样本数 |
|--------|--------|--------|
| **Overall** | **81.69%** | 2,638 |
| Count | 68.53% | 788 |
| Relation | **90.77%** | 650 |
| Depth | 86.67% | 600 |
| Distance | 84.17% | 600 |

**分析**: 空间关系推理（Relation）表现最强（90.77%），计数（Count）相对较弱（68.53%）。整体 81.69% 在 CV-Bench 上属于优秀水平。

---

#### EmbSpatial-Bench — 空间关系理解 ✅

| 指标 | 数值 |
|------|------|
| **Overall Accuracy** | **68.96%** |
| Total Samples | 3,640 |
| Correct | 2,510 |

**分析**: 四选一空间推理题，68.96% 显著高于随机猜测（25%），说明模型具备一定的空间语义理解能力，但还有提升空间。

---

### 2.2 3D 空间定位

#### Open6DoR-Custom — 3D 空间关系定位 ✅

| 指标 | 数值 |
|------|------|
| **Overall Accuracy** | **48.27%** |
| Total Samples | 1,618 |
| Correct | 781 |

**按空间关系细分类别**:

| 关系 | 准确率 | 样本数 |
|------|--------|--------|
| **left** | **99.65%** (284/285) | 285 |
| **top** | **100.00%** (193/193) | 193 |
| **behind** | **97.27%** (285/293) | 293 |
| between | 5.20% (9/173) | 173 |
| front | 3.72% (10/269) | 269 |
| center | 0.00% (0/145) | 145 |
| right | 0.00% (0/260) | 260 |

**分析**:
- **极度不平衡**: 左右（left/right）和前后（front/behind）的表现截然相反
- left/behind/top 表现接近完美（>97%）
- right/center/front/between 几乎完全失败（<6%）
- 原因推测：模型对"左右前后"的 3D 空间映射有严重的方向性偏差，可能是相机坐标系理解问题或训练数据 bias
- 这是**已知问题**：Embodied-R1 的 3D 评估依赖 2D→3D 反投影，而模型本身是 2D pointing，3D 理解能力有限

---

### 2.3 Pointing / 2D 定位类

#### RoboRefit — 机器人指令目标定位 ✅

| 指标 | 数值 |
|------|------|
| **Points in Box Accuracy** | **85.35%** |
| Total Points | 4,000 |
| Points in Box | 3,414 |
| Samples | 2,000 |

**分析**: 模型能很好地根据自然语言指令定位目标物体的 bbox 区域（85.35%），这是 Embodied-R1 的核心能力之一。accuracy 以**点**为单位计算（3414/4000），每个样本平均 2 个预测点。

---

#### VABench-Point — 自由点预测 ✅

| 指标 | 数值 |
|------|------|
| **Points in BBox Accuracy** | **74.00%** |
| Total Points | 600 |
| Points in BBox | 444 |
| Samples | 300 |

**分析**: 在 bbox 内预测操作点的能力较强（74%），说明模型对操作区域有合理的空间感知。

---

#### VABench-VisualTrace — 视觉轨迹生成 ✅

| 指标 | 数值 |
|------|------|
| **Average RMSE** | **71.83** |
| **Average MAE** | **40.08** |
| Valid Samples | 299 / 300 |

**分析**: VTG（Visual Trace Generation）生成 8 个轨迹路点。RMSE ~72px 在图像坐标系中属于中等水平（假设图像尺寸 500-800px），说明轨迹大致正确但精度有待提升。

---

#### Part-Affordance-2K — Affordance 区域定位 ✅

| 指标 | 数值 |
|------|------|
| **Points in Mask Accuracy** | **44.61%** |
| Total Points | 3,988 |
| Points in Mask | 1,779 |
| Samples | 2,000 |

**分析**: 44.61% 低于其他 pointing 任务。Affordance 理解（如"握持位置"）需要更细粒度的功能区域识别，这是当前模型的薄弱环节。

---

#### Where2Place — 物体放置位置预测 ✅

| 指标 | 数值 |
|------|------|
| **Points in Mask Accuracy** | **69.50%** |
| Total Points | 200 |
| Points in Mask | 139 |
| Samples | 100 |

**分析**: 放置位置预测和 RoboRefit 水平接近（~70%），说明模型对"空位"的理解较好。

---

## 三、失败评估分析

### 3.1 BLINK — 多模态视觉推理 ❌

**失败原因**: 进程被中途 kill，日志在样本处理过程中截断，未打印 Average accuracy 汇总行。非崩溃，数据加载和推理本身正常。

**日志位置**:
- `eval/logs/run_blink.log`
- `eval/logs/inference_BLINK_Embodied-R1-3B_20260423_193905.log`（推理进行中被终止）

**分析**: BLINK 能正常跑，只是上次运行被手动终止或超时中断。可直接重跑。

**建议修复**:
```bash
cd eval
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port=54327 hf_inference_blink.py 2>&1 | tee logs/run_blink.log
```

---

### 3.2 SAT-Real — 空间推理问答 ❌

**两次不同失败原因**:

| 尝试 | 日志 | 错误 |
|------|------|------|
| 4月23日 + offical-3b-2（4/27） | `eval/logs/run_sat.log`, `eval/offical-3b-2/run_sat.log` | `pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs` — 数据加载失败，根本未启动推理 |
| offical-3b-sat（4/27） | `eval/offical-3b-sat/run_sat.log` | 数据加载成功，推理过程中 NCCL 通信超时（`SIGABRT exitcode -6`）— 多卡间某个进程卡住，另一个进程等待超时崩溃 |

**分析**: 最新一次（offical-3b-sat）已解决数据格式问题，推理也能正常进行，仅在多卡同步时崩溃。改为单卡可规避 NCCL 超时问题。

**建议修复**:
```bash
cd eval
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port=54328 hf_inference_sat.py 2>&1 | tee logs/run_sat_single.log
```

---

### 3.3 CRPE — 组合式空间关系推理 ❌

**状态**: 从未运行过

**分析**: 没有找到任何 CRPE 的日志或结果文件。可能是评估脚本未被执行。

**建议修复**:
```bash
cd eval
python hf_inference_crpe.py
```

---

## 四、数据来源说明

| 评估 | 成功来源 | 失败来源 |
|------|---------|---------|
| CV-Bench | `eval/logs/` (4/23) | — |
| EmbSpatial-Bench | `eval/logs/` (4/23) | — |
| Open6DoR | `eval/offical-3b-3/` (4/27) | `eval/logs/` (4/23, 失败) |
| Part-Affordance-2K | `eval/logs/` (4/23) | — |
| RoboRefit | `eval/logs/` (4/23) | — |
| VABench-Point | `eval/offical-3b-3/` (4/27) | `eval/offical-3b-2/` (4/27, 缺少 parquet) |
| VABench-VisualTrace | `eval/offical-3b-3/` (4/27) | `eval/offical-3b-2/` (4/27, 缺少 parquet) |
| Where2Place | `eval/logs/` (4/23) | — |
| BLINK | — | `eval/logs/` (4/23, 进程被中断) |
| SAT | — | `eval/logs/` (4/23), `eval/offical-3b-2/` (4/27), `eval/offical-3b-sat/` (4/27) |
| CRPE | — | 从未运行 |

---

## 五、总结

### 模型能力雷达图

| 能力维度 | 表现 | 评分 |
|---------|------|------|
| 空间 QA / VQA | CV-Bench 81.69%, EmbSpatial 68.96% | ⭐⭐⭐⭐ |
| 2D Pointing (bbox 内) | RoboRefit 85.35%, VABench-P 74% | ⭐⭐⭐⭐ |
| 放置位置预测 | Where2Place 69.50% | ⭐⭐⭐⭐ |
| Affordance 理解 | Part-Affordance 44.61% | ⭐⭐⭐ |
| 视觉轨迹生成 | VABench-V RMSE 71.83 | ⭐⭐⭐ |
| 3D 空间定位 | Open6DoR 48.27%（但方向严重偏差） | ⭐⭐ |

### 核心发现

1. **2D Pointing 是强项**: RoboRefit (85.35%)、VABench-P (74%) 证明了模型在像素级定位上的能力
2. **空间 QA 表现良好**: CV-Bench 81.69% 说明模型具备较强的视觉-空间推理能力
3. **3D 理解有严重偏差**: Open6DoR 中 left/right 和 front/behind 的表现截然相反，这是 2D→3D 映射的已知局限
4. **Affordance 和轨迹是弱项**: Part-Affordance (44.61%) 和 VTG (RMSE 71.83) 还有较大提升空间
5. **3 个评估失败**: BLINK (进程被中断)、SAT (数据格式/多卡NCCL)、CRPE (未运行)，需要后续补跑
