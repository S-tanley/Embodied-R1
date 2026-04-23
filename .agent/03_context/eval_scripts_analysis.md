# Embodied-R1 评估脚本深度分析

> 分析日期：2026-04-23  
> 覆盖脚本：11 个（`eval/` 目录下全部）  
> 基础框架：所有脚本均使用 `accelerate` 多卡并行推理，模型统一为 `Qwen2_5_VLForConditionalGeneration`（bfloat16）

---

## 通用运行方式

所有脚本的标准启动命令：

```bash
# 单卡
python eval/hf_inference_XXX.py

# 多卡（推荐，加速推理）
python -m accelerate.commands.launch \
    --num_processes=<GPU数量> \
    --main_process_port=54321 \
    eval/hf_inference_XXX.py
```

结果输出统一保存到：
- **详细结果**：`logs/results/<task>_<model>_<reasoning_model>_<split>.json`
- **日志文件**：`logs/inference_<task>_<model>_<时间戳>.log`
- **可视化图片**：`logs/visualizations/<task>_<model>/`（部分脚本）

---

## 通用可调参数

所有脚本顶部的 `__main__` 块均暴露以下参数，直接在脚本中修改：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `model_path` | str | 模型路径（本地或 HF Hub ID） |
| `model_name` | str | 模型别名（用于结果文件命名） |
| `reasoning_model` | bool | `True`=强制输出 `<think>...<answer>` 格式；`False`=直接输出 |
| `max_pixels` | int | 图像最大像素数，控制图像分辨率上限 |
| `min_pixels` | int | 图像最小像素数（所有脚本均为 `256*28*28=200704`） |
| `use_flash_attention` | bool | 是否启用 Flash Attention 2（部分脚本支持） |
| `disable_visualization` | bool | 是否禁用可视化保存（部分脚本支持） |
| `gen_args` | dict | 生成参数（见下方详细说明） |

### gen_args 详细说明

```python
gen_args = {
    "temperature": 0,          # 0 = 贪心解码，确定性输出
    "top_p": 1,                # nucleus sampling 阈值（temperature=0 时无效）
    "max_new_tokens": 2048,    # 最大生成 token 数
    "repetition_penalty": 1.05, # 重复惩罚系数（1.0=不惩罚）
    "do_sample": False,        # False=贪心/beam search，不随机采样
}
```

**可以调节的点**：
- `max_new_tokens`：如果只跑点定位任务，512 够用（affordance 脚本就是 512），减小能显著加速
- `temperature` + `do_sample=True`：改为采样模式，用于测试模型不确定性
- `repetition_penalty`：Part-Affordance-2K 脚本设为 1.0（无惩罚），其他都是 1.05

---

## 各脚本详细分析

---

### 1. VABench-Point — `hf_inference_vabench_point.py`

**测试内容**：自由点预测（Free-Point）。给定机器人操作任务图像，预测操作点坐标，验证是否落在 GT Bounding Box 内。

**数据来源**：
```python
dataset_path = "/mnt/path/.../FSD_points_rft_fsd_free_point_train_32790_test_300_.../test.parquet"
# ⚠️ 这是作者服务器上的内部路径，本地运行必须替换！
# HF 数据集：IffYuan/VABench-P（需下载后指定本地 Parquet 路径）
```

**测试集规模**：300 条样本

**评估指标**：
```
Accuracy = points_in_bbox / total_points
```
预测的所有点中，落在 GT bbox 内的比例。

**输出格式要求**：
```
<answer><point>[[x1, y1], [x2, y2], ...]</point></answer>
```

**关键参数**：
```python
max_pixels = 1605632   # ~1.5M 像素
max_new_tokens = 2048
reasoning_model = True
disable_visualization = False  # 会生成可视化图
```

**如何运行**：
```bash
# 1. 先修改脚本中的 dataset_path 为本地 Parquet 文件路径
# 2. 单卡运行
python eval/hf_inference_vabench_point.py

# 多卡运行
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_vabench_point.py
```

**注意事项**：VABench-P 数据集需从 HuggingFace 下载到本地，或在脚本中用 HF 数据集 API 改写加载逻辑。

---

### 2. VABench-VisualTrace — `hf_inference_vabench_visual_trace.py`

**测试内容**：视觉轨迹生成（Visual Trace Generation, VTG）。给定操作任务，预测 8 个路点轨迹，与 GT 轨迹做相似度比较。

**数据来源**：
```python
dataset_path = "/mnt/path/.../FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_.../test.parquet"
# ⚠️ 同样是内部路径，需替换
# HF 数据集：IffYuan/vabench-v
```

**测试集规模**：300 条样本

**评估指标**：
```
Average RMSE（均方根误差）
Average MAE（平均绝对误差）
```

计算逻辑：
1. 将预测轨迹和 GT 轨迹的坐标归一化到 [0, 1000]×[0, 1000] 坐标系
2. 用线性插值将两条轨迹对齐到相同长度（取二者最大长度）
3. 逐点计算欧式距离，汇总为 RMSE 和 MAE

**关键参数**：
```python
max_pixels = 1605632
max_new_tokens = 2048
reasoning_model = True
```

**如何运行**：
```bash
# 修改 dataset_path 后运行
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_vabench_visual_trace.py
```

**注意事项**：
- 指标越低越好（RMSE、MAE 均为误差）
- 脚本会自动处理预测点数不为 8 的情况（通过插值对齐）

---

### 3. Where2Place — `hf_inference_where2place.py`

**测试内容**：物体放置位置预测（Region Referring Grounding, RRG）。给定场景图像和任务，预测目标物体应该被放置的区域坐标。

**数据来源**：
```python
huggingface_dataset_name = "FlagEval/Where2Place"
split = "test"
# ✅ 直接从 HuggingFace 在线加载，无需手动下载
```

**评估指标**：
```
Accuracy = points_in_mask / total_points
```
预测点是否落在 GT 分割 mask 的白色前景区域内。

**关键参数**：
```python
max_pixels = 3200000  # ⚠️ 所有脚本中最大值，图像分辨率上限最高
max_new_tokens = 2048
reasoning_model = True
disable_visualization = False
```

**如何运行**：
```bash
# 无需修改路径，直接运行
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_where2place.py
```

---

### 4. Part-Affordance-2K — `hf_inference_affordance.py`

**测试内容**：物体 Affordance 区域定位（Object Affordance Grounding, OFG）。根据功能描述（如"loosening stuck bolts"），定位工具或物体的可操作区域。

**数据来源**：
```python
huggingface_dataset_name = "IffYuan/Part-Affordance-2K"
split = "train"   # ⚠️ 注意：用的是 train split，没有 test split
# ✅ 直接从 HuggingFace 在线加载
```

**评估指标**：
```
Accuracy = points_in_mask_foreground / total_points
```
预测点是否落在 GT mask 的前景（前景像素值 > 0）区域内。

**关键参数**：
```python
max_pixels = 3200000       # 与 Where2Place 一样大
max_new_tokens = 512       # ⚠️ 比其他脚本短 4 倍！推理速度更快
repetition_penalty = 1.0   # ⚠️ 无重复惩罚（其他脚本是 1.05）
reasoning_model = True
# 可视化保存概率：random.random() < 0.1（约 10% 的样本才保存）
```

**如何运行**：
```bash
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_affordance.py
```

**注意事项**：`split="train"` 意味着跑的是完整训练集（数量较多），如需快速验证可手动限制样本数。

---

### 5. RoboRefit — `hf_inference_roborefit.py`

**测试内容**：机器人自然语言指令跟随中的目标物体定位（REG）。根据"will you please pass me the glue stick"这类口语化指令，在图像中定位目标物体。

**数据来源**：
```python
huggingface_dataset_name = "roborefit_test.json"   # 本地 JSON 文件
# ✅ 文件已包含在 eval/ 目录下
image_base_dir = "/mnt/kaiwu-group-x4-sh/iffyuan/roborefit/test_images/"
# ⚠️ 图像路径硬编码在 process_sample() 函数内，必须修改！
```

**测试集规模**：`eval/roborefit_test.json` 中的全部样本（需查看）

**评估指标**：
```
Accuracy = points_in_box / total_points（按样本平均）
Overall Accuracy = total_points_in_box / total_points（按点计算）
```

**关键参数**：
```python
max_pixels = 1605632
max_new_tokens = 2048
reasoning_model = True
disable_visualization = False
```

**如何运行**：
```bash
# 1. 先修改 process_sample() 函数第 105 行的 image_base_dir 路径
# image_base_dir = "/你的本地路径/roborefit/test_images/"
# 需要下载 RoboRefit 数据集的测试集图像

# 2. 运行（需从 eval/ 目录下运行，因为 roborefit_test.json 用相对路径）
cd /path/to/Embodied-R1
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_roborefit.py
```

---

### 6. Open6DoR-Custom — `hf_inference_3d.py`

**测试内容**：3D 空间关系定位。给定 IsaacSim 渲染的 RGB 图像和"Place the apple **behind** the box"这类指令，预测目标放置点，并通过相机参数反投影到 3D 世界坐标，验证空间关系（left/right/front/behind/top/between/center）。

**数据来源**：
```python
dataset_path = "3d_dataset.json"  # 本地 JSON 文件
# ✅ 文件已包含在 eval/ 目录下
base_path = "/mnt/path/iffyuan/all-seeing/..."
# ⚠️ 图像路径硬编码在 process_sample() 函数第 133 行，必须修改！
```

**评估指标**：
```
Binary Accuracy（按 position_tag 分组）
Overall Accuracy = correct_predictions / total_samples
```
逻辑：将预测的 2D 像素点 + 固定深度值（depth=1）通过相机内外参反投影到 3D 坐标，再判断空间关系是否满足（如 behind: x_pred > x_ref）。

**固定的相机参数**（IsaacSim 默认）：
```python
img_width=2160, img_height=1440
camera_view_matrix_inv = ...   # 相机外参的逆矩阵
camera_proj_matrix = ...       # 相机投影矩阵
```

**关键参数**：
```python
max_pixels = 3110400  # 约 3M 像素，高分辨率
max_new_tokens = 2048
reasoning_model = True
# 无可视化功能
```

**如何运行**：
```bash
# 1. 修改 process_sample() 函数中的 base_path
# 2. 需要下载 Open6DoR 对应的 IsaacSim 渲染图像
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_3d.py
```

**注意事项**：
- 当前实现中深度值固定为 1（`predict_depth = [1 for _ in range(len(points))]`），不使用真实深度图
- 这是论文提到的"3D 推理仍基于 2D 感知"的体现
- 结果按 position_tag 分组输出，可以看到 left/right/front/behind/top/between/center 各方向的准确率

---

### 7. BLINK — `hf_inference_blink.py`

**测试内容**：多图视觉推理多选题。从 BLINK 基准的 4 个子任务中测试空间理解能力。

**数据来源**：
```python
huggingface_dataset_name = "BLINK-Benchmark/BLINK"
split = "val"   # 注意：是 validation split，不是 test
subsets = ["Multi-view_Reasoning", "Object_Localization", "Relative_Depth", "Spatial_Relation"]
# ✅ 直接从 HuggingFace 加载
```

**评估指标**：
```
Overall Accuracy + 各子任务 Accuracy
判断逻辑：answer[1] in predicted or text_answer in predicted
（即选项字母"A"/"B"等 或 完整选项文本 出现在预测结果中）
```

**关键参数**：
```python
max_pixels = 2097152   # 约 2M 像素
max_new_tokens = 2048
reasoning_model = True
# 注意：attn_implementation=None，没有用 Flash Attention（硬编码）
# 每个样本最多支持 4 张图片
```

**Prompt 格式**（不同于 Pointing 类任务）：
```
instruct_following = '...<think>...</think><answer>answer example:(A)</answer>'
# 要求输出选项字母，如 (A) (B) (C) (D)
```

**如何运行**：
```bash
# 无需修改路径
python -m accelerate.commands.launch --num_processes=8 --main_process_port=54321 \
    eval/hf_inference_blink.py
# 注释中建议用 8 卡
```

---

### 8. CRPE — `hf_inference_crpe.py`

**测试内容**：组合式空间关系推理多选题（Compositional Relative Position Evaluation）。

**数据来源**：
```python
dataset_path = ["path/to/crpe_relation.jsonl"]
# ⚠️ 占位符路径，必须替换为本地 CRPE 数据集路径！
# 图像路径也需要修改（COCO 图像 + CRPE 私有图像）：
# image_dir = "path/to/coco/images"   # COCO 图像
# image_dir = "path/to/crpe/images"   # CRPE 专有图像
```

**评估指标**：
```
Overall Accuracy + 按 category 分组的准确率
判断逻辑：answer in predicted（预测结果中包含正确选项字母）
```

**关键参数**：
```python
max_pixels = 1605632
max_new_tokens = 2048
reasoning_model = True
# ⚠️ attn_implementation="flash_attention_2" 硬编码开启（与其他脚本不同）
# 需要 flash-attn 库安装
```

**如何运行**：
```bash
# 1. 下载 CRPE 数据集（需申请访问）
# 2. 修改 dataset_path 和 image_dir
python -m accelerate.commands.launch --num_processes=4 --main_process_port=54321 \
    eval/hf_inference_crpe.py
```

**注意事项**：CRPE 数据集不完全开放，部分图像来自 COCO（可下载），部分来自 CRPE 私有数据集。这是 11 个脚本中**最难直接运行**的一个。

---

### 9. CV-Bench — `hf_inference_cvbench.py`

**测试内容**：计算机视觉综合基准多选题，涵盖多个视觉理解子任务。

**数据来源**：
```python
huggingface_dataset_name = "nyu-visionx/CV-Bench"
split = "test"
# ✅ 直接从 HuggingFace 加载
```

**评估指标**：
```
Overall Accuracy + 按 task 分组的准确率
判断逻辑：answer[1] in predicted or correct_text_answer == predicted
```

**关键参数**：
```python
max_pixels = 1605632
max_new_tokens = 2048
reasoning_model = True
# 无 Flash Attention（硬编码为默认）
```

**如何运行**：
```bash
python -m accelerate.commands.launch --num_processes=4 --main_process_port=54321 \
    eval/hf_inference_cvbench.py
```

---

### 10. EmbSpatial-Bench — `hf_inference_embspatial.py`

**测试内容**：具身空间关系理解多选题（A/B/C/D 四选一）。

**数据来源**：
```python
# 数据集硬编码在 main() 内：
dataset = load_dataset("FlagEval/EmbSpatial-Bench")
test_data = dataset['test']
# ✅ 直接从 HuggingFace 加载，无需手动指定
```

**评估指标**：
```
Overall Accuracy
判断逻辑：answer in predicted（A/B/C/D 在预测结果中）
```

**关键参数**：
```python
max_pixels = 1605632
max_new_tokens = 2048
reasoning_model = True
# attn_implementation=None（无 Flash Attention，硬编码）
```

**Prompt 格式**：
```
instruct_following = '...<think>...</think><answer>your answer here</answer>'
# 注意：这里没有限定格式为 (A)，比 BLINK/CV-Bench 宽松
```

**如何运行**：
```bash
python -m accelerate.commands.launch --num_processes=8 --main_process_port=54321 \
    eval/hf_inference_embspatial.py
# 注释中建议用 8 卡
```

---

### 11. SAT — `hf_inference_sat.py`

**测试内容**：空间感知测试（Spatial Awareness Test）。多图输入，自然语言短语形式回答（非选项题）。

**数据来源**：
```python
huggingface_dataset_name = "array/SAT"
split = "test"
# ✅ 直接从 HuggingFace 加载
```

**评估指标**：
```
Overall Accuracy
判断逻辑：answer in predicted（GT 答案字符串是否被模型预测所包含）
```
注意：答案是自然语言短语（如"left"、"above"），采用字符串包含匹配而非严格相等。

**关键参数**：
```python
max_pixels = 1605632
max_new_tokens = 2048
reasoning_model = True
# attn_implementation=None（无 Flash Attention）
# 每个样本可包含多张图片（image_bytes 字段）
```

**如何运行**：
```bash
python -m accelerate.commands.launch --num_processes=4 --main_process_port=54321 \
    eval/hf_inference_sat.py
# 注释中建议用 4 卡
```

---

## 汇总对比表

| 脚本 | 任务类型 | 数据来源 | 测试量 | 评估指标 | 需修改路径 | max_pixels | Flash Attn |
|------|---------|---------|--------|---------|-----------|-----------|-----------|
| VABench-Point | 自由点预测 | 本地 Parquet | 300 | bbox 内点占比 | ✅ 必须 | 1605632 | 可选 |
| VABench-VisualTrace | 轨迹生成 | 本地 Parquet | 300 | RMSE / MAE ↓ | ✅ 必须 | 1605632 | 可选 |
| Where2Place | 放置点预测 | HF 在线 | - | mask 内点占比 | ❌ 无需 | 3200000 | 可选 |
| Part-Affordance-2K | Affordance 定位 | HF 在线 | - | mask 内点占比 | ❌ 无需 | 3200000 | 可选 |
| RoboRefit | 物体定位 | 本地 JSON+图像 | - | bbox 内点占比 | ✅ 必须（图像） | 1605632 | 可选 |
| Open6DoR-3D | 3D 空间关系 | 本地 JSON+图像 | - | 二元空间关系准确率 | ✅ 必须（图像） | 3110400 | 可选 |
| BLINK | 多图多选题 | HF 在线 | - | 选项准确率 | ❌ 无需 | 2097152 | ❌ 关闭 |
| CRPE | 空间推理多选题 | 本地 JSONL+图像 | - | 选项准确率 | ✅ 必须 | 1605632 | ✅ 强制开启 |
| CV-Bench | 综合视觉多选题 | HF 在线 | - | 选项准确率 | ❌ 无需 | 1605632 | ❌ 关闭 |
| EmbSpatial-Bench | 空间关系多选题 | HF 在线 | - | 选项准确率 | ❌ 无需 | 1605632 | ❌ 关闭 |
| SAT | 多图空间推理 | HF 在线 | - | 字符串包含准确率 | ❌ 无需 | 1605632 | ❌ 关闭 |

---

## 可以直接运行（无需修改路径）的脚本

按难度从易到难排列：

1. **EmbSpatial-Bench** — 最简单，HF 在线数据，单图四选一
2. **Where2Place** — HF 在线数据，点定位任务，与模型核心能力最相关
3. **CV-Bench** — HF 在线数据，综合视觉 benchmark
4. **BLINK** — HF 在线数据，多图推理
5. **SAT** — HF 在线数据，多图 + 自然语言答案
6. **Part-Affordance-2K** — HF 在线数据，Affordance 定位（train split）

---

## 需要修改才能运行的脚本

| 脚本 | 需要修改的内容 | 数据获取方式 |
|------|-------------|-------------|
| **VABench-Point** | `dataset_path`（Parquet 路径） | HF 下载 `IffYuan/VABench-P` |
| **VABench-VisualTrace** | `dataset_path`（Parquet 路径） | HF 下载 `IffYuan/vabench-v` |
| **RoboRefit** | `process_sample()` 中 `image_base_dir` | 下载 RoboRefit 测试集图像 |
| **Open6DoR-3D** | `process_sample()` 中 `base_path` | 下载 Open6DoR IsaacSim 图像 |
| **CRPE** | `dataset_path` + `image_dir`（COCO/CRPE 两处） | 下载 COCO + 申请 CRPE 数据集 |

---

## 测试任务分类

按测试的能力维度分类：

### Pointing 类（核心能力，输出坐标点）
- VABench-Point：OFG/自由点
- VABench-VisualTrace：VTG/轨迹
- Where2Place：RRG/放置点
- Part-Affordance-2K：OFG/affordance
- RoboRefit：REG/物体定位
- Open6DoR-3D：RRG + 3D 投影

### VQA/多选题类（空间推理语言能力）
- BLINK：多图空间推理（Multi-view, Depth, Spatial Relation）
- CRPE：组合空间关系推理
- CV-Bench：综合视觉理解
- EmbSpatial-Bench：具身空间关系
- SAT：空间感知自然语言回答

---

## 重要代码细节与坑

### 1. VABench-VisualTrace 坐标归一化
轨迹评估前，坐标统一归一化到 1000×1000 空间：
```python
x = (x / width) * (1000 - 1)
y = (y / height) * (1000 - 1)
```
所以 RMSE/MAE 的单位是"归一化后 1000×1000 空间内的像素误差"。

### 2. Open6DoR-3D 深度值固定为 1
```python
predict_depth = [1 for _ in range(len(points))]  # 不使用真实深度
```
模型只输出 2D 坐标，深度固定为 1，再通过相机参数转到 3D。这是论文的 limitation 所在。

### 3. Part-Affordance-2K 用 train split
```python
split = "train"  # 没有单独的 test split
```
跑的是完整训练集，样本量较大。

### 4. RoboRefit 图像路径在函数体内
路径不在 `__main__` 顶部，而是写死在 `process_sample()` 函数第 104 行：
```python
image_base_dir = "/mnt/kaiwu-group-x4-sh/iffyuan/roborefit/test_images/"
```
修改时需要进入函数体内修改。

### 5. CRPE 强制开启 Flash Attention
CRPE 脚本中 `attn_implementation="flash_attention_2"` 是硬编码而非参数，需要安装 `flash-attn` 库才能运行。

### 6. reasoning_model 参数的含义
- `True`：期望模型输出 `<think>...</think><answer>...</answer>` 格式，从 `<answer>` 中提取结果
- `False`：直接解析模型原始输出，适合 baseline 非推理模型
