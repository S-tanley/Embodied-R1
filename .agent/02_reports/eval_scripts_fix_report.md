# Eval Scripts 修复报告

> 日期：2026-04-23  
> 涉及文件：`eval/` 目录下 4 个脚本 + 新增 `eval/data/` 数据目录

---

## 背景

11 个 eval 脚本中有 4 个依赖作者服务器上的内部路径，无法直接运行：

| 脚本 | 原始问题 |
|------|---------|
| `hf_inference_vabench_point.py` | `dataset_path` 指向 `/mnt/path/iffyuan/...`（内部服务器） |
| `hf_inference_vabench_visual_trace.py` | `dataset_path` 指向 `/mnt/path/iffyuan/...`（内部服务器） |
| `hf_inference_roborefit.py` | 图像路径 `image_base_dir` 指向 `/mnt/kaiwu-group-x4-sh/...`（内部服务器） |
| `hf_inference_3d.py` | 图像路径 `base_path` 指向 `/mnt/path/iffyuan/...`（内部服务器） |

---

## 数据下载

所有所需数据均可从 HuggingFace 获取，**图像以 bytes 形式内嵌在 parquet 文件中**，无需单独下载图像文件夹。

下载命令（Python，需已登录 HuggingFace）：

```python
from huggingface_hub import hf_hub_download
import shutil

downloads = [
    ('IffYuan/Embodied-R1-Dataset',
     'FSD_points_rft_fsd_free_point_train_32790_test_300_0425/test.parquet',
     'eval/data/vabench_point_test.parquet'),

    ('IffYuan/Embodied-R1-Dataset',
     'FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/test.parquet',
     'eval/data/vabench_trace_test.parquet'),

    ('IffYuan/Roborefit',
     'data/test-00000-of-00002.parquet',
     'eval/data/roborefit_test_0.parquet'),

    ('IffYuan/Roborefit',
     'data/test-00001-of-00002.parquet',
     'eval/data/roborefit_test_1.parquet'),

    ('IffYuan/Embodied-R1-Dataset',
     'Open6DOR_rft_3d_position_train_10k_test_date_0514/test.parquet',
     'eval/data/open6dor_test.parquet'),
]

for repo, remote, local in downloads:
    cached = hf_hub_download(repo, remote, repo_type='dataset')
    shutil.copy(cached, local)
```

下载后文件清单：

| 本地路径 | HuggingFace 来源 | 大小 | 样本数 |
|---------|----------------|------|--------|
| `eval/data/vabench_point_test.parquet` | `IffYuan/Embodied-R1-Dataset` | 26 MB | 300 |
| `eval/data/vabench_trace_test.parquet` | `IffYuan/Embodied-R1-Dataset` | 26 MB | 300 |
| `eval/data/roborefit_test_0.parquet` | `IffYuan/Roborefit` | 184 MB | 1000 |
| `eval/data/roborefit_test_1.parquet` | `IffYuan/Roborefit` | 181 MB | 1000 |
| `eval/data/open6dor_test.parquet` | `IffYuan/Embodied-R1-Dataset` | 136 MB | 1618 |

> **注意**：VABench 数据使用的是 `IffYuan/Embodied-R1-Dataset` 中的 FSD 子集，而非 `IffYuan/VABench-P` / `IffYuan/vabench-v`。后者列结构不同（无 `answer`、`images` 字段），与脚本不兼容。

---

## 脚本改动详情

### 1. `hf_inference_vabench_point.py`

**改动类型**：仅修改路径，1 行。

```python
# 改前（第 282 行）
dataset_path = "/mnt/path/iffyuan/all-seeing/all-seeing-v2/process_rl_data/FSD_points_rft_fsd_free_point_train_32790_test_300_0425/test.parquet"

# 改后
dataset_path = "eval/data/vabench_point_test.parquet"
```

脚本本身已使用 `pd.read_parquet()` 加载数据，parquet 列结构完全匹配（`id`, `problem`, `answer`, `images`, ...），无需其他修改。

---

### 2. `hf_inference_vabench_visual_trace.py`

**改动类型**：仅修改路径，1 行。

```python
# 改前（第 401 行）
dataset_path = "/mnt/path/iffyuan/all-seeing/all-seeing-v2/process_rl_data/FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/test.parquet"

# 改后
dataset_path = "eval/data/vabench_trace_test.parquet"
```

同上，脚本与 parquet 格式完全兼容，无需其他修改。

---

### 3. `hf_inference_roborefit.py`

**改动类型**：数据源从 JSON+文件路径 迁移到 parquet+内嵌 bytes，共 4 处改动。

#### 改动 1：新增 import（文件顶部）

原脚本不需要读 parquet，缺少相关 import：

```python
# 新增
import pandas as pd
from io import BytesIO
```

#### 改动 2：读图方式（`process_sample` 函数）

原脚本从本地文件系统读取图像，图像路径硬编码在函数体内：

```python
# 改前
instruction = sample['problem'].strip()
bbox = sample['bbox']
image_base_dir = "/mnt/kaiwu-group-x4-sh/iffyuan/roborefit/test_images/"
image_path = os.path.join(image_base_dir, sample['image'])
image = Image.open(image_path)
doc_id = sample.get('id', random.randint(0, 100000))
```

parquet 中字段名不同（`question` 而非 `problem`，`question_id` 而非 `id`），且图像以 `{'bytes': b'...', 'path': '...'}` 格式内嵌：

```python
# 改后
instruction = sample['question'].strip()
bbox = [int(x) for x in sample['bbox']]
image = Image.open(BytesIO(sample['image']['bytes']))
doc_id = sample.get('question_id', random.randint(0, 100000))
```

> **`bbox` 转换说明**：parquet 读出的 numpy array 元素类型为 `np.int32`，直接 `list()` 后仍是 `np.int32`。`np.int32` 不可 JSON 序列化，结果写入 `logs/results/*.json` 时会抛 `TypeError`。因此用 `[int(x) for x in ...]` 显式转为 Python 原生 int。

#### 改动 3：数据加载（`main` 函数）

原脚本加载本地 JSON 文件：

```python
# 改前
json_dataset = json.load(open(huggingface_dataset_name, 'r'))
test_data = json_dataset
```

改为合并两个 parquet 分片：

```python
# 改后
dfs = [pd.read_parquet(p) for p in huggingface_dataset_name]
import pandas as _pd
test_data = _pd.concat(dfs, ignore_index=True).to_dict(orient='records')
logger.info(f"Test data samples: {len(test_data)}")
```

#### 改动 4：数据路径配置（`__main__` 块）

```python
# 改前
huggingface_dataset_name = "roborefit_test.json"

# 改后
huggingface_dataset_name = ["eval/data/roborefit_test_0.parquet", "eval/data/roborefit_test_1.parquet"]
```

---

### 4. `hf_inference_3d.py`

**改动类型**：数据源从 JSON+文件路径 迁移到 parquet+内嵌 bytes，共 4 处改动。

#### 改动 1：新增 import（文件顶部）

```python
# 新增
import pandas as pd
from io import BytesIO
```

#### 改动 2：`process_sample` 函数重写（核心改动）

原脚本从文件系统读 RGB 图，字段直接取自 JSON 结构（有 `position_instruction`、`position_tag`、干净的 `answer` 等独立字段）：

```python
# 改前
image_path = sample["images"][0].replace("\\", "/")
base_path = "/mnt/path/iffyuan/all-seeing/all-seeing-v2/process_rl_data"
abs_image_path = os.path.join(base_path, image_path)
with open(abs_image_path, "rb") as image_file:
    image = Image.open(image_file).convert("RGB")

task_instruction = sample["position_instruction"]
question = f"You are currently a robot ... {task_instruction}. ..."
answer = sample["answer"]               # 干净的 JSON dict
position_tag = sample.get("position_tag", "unknown")
doc_id = sample.get("id", "unknown")
```

parquet 字段结构不同，主要差异：

| 字段 | JSON | parquet |
|------|------|---------|
| 图像 | 文件路径字符串 | bytes（`sample["images"][0]`） |
| `answer` | 干净 JSON 字符串 | 带 `<type>3d_position</type>` 前缀 |
| `position_instruction` | 独立字段 | 无，需从 `problem` 字段提取 |
| `position_tag` | 独立字段 | 无，需从 `answer.direction` 提取 |

```python
# 改后
image = Image.open(BytesIO(sample["images"][0])).convert("RGB")

# 剥掉 <type> 标签后解析 answer
answer_raw = re.sub(r"<type>.*?</type>", "", sample["answer"]).strip()
answer = json.loads(answer_raw)

# 从 answer 中提取 position_tag（等价于原 JSON 的 position_tag 字段）
position_tag = answer.get("direction", "unknown")
doc_id = sample.get("id", "unknown")

# 从 problem 字段去掉图像占位符，还原任务指令
task_instruction = re.sub(
    r"RGB image:<image>,?\s*Depth image:<image>\s*", "", sample["problem"]
).strip()
question = task_instruction + '\n' + instruct_following
```

#### 改动 3：数据加载（`main` 函数）

```python
# 改前
with open(dataset_path, 'r') as f:
    test_data = json.load(f)

# 改后
test_data = pd.read_parquet(dataset_path).to_dict(orient='records')
```

#### 改动 4：数据路径配置（`__main__` 块）

```python
# 改前
dataset_path = "3d_dataset.json"

# 改后
dataset_path = "eval/data/open6dor_test.parquet"
```

---

## 验证

### 数据结构验证

下载完成后，对每个 parquet 的关键字段进行验证，确认数据可正常读取：

```
VABench-Point:   300 samples, image (640×480), bbox 正常解析
VABench-Trace:   300 samples, image (640×480), trajectory len=8
RoboRefit:      2000 samples, image (640×480), question + bbox 正常
Open6DoR-3D:    1618 samples, image (720×480), direction + 3D坐标 正常
```

### 端到端逻辑验证（mock 推理）

对 RoboRefit 和 Open6DoR-3D 的 `process_sample` 全流程进行验证（用伪造预测结果替代模型输出），确认：

- 图像从 bytes 解码正常
- 字段映射正确（`question`、`question_id`、`direction` 等）
- bbox/answer 解析无误
- 评分函数（`check_points_in_bbox`、`evaluate_posi`、`pixel_to_world`）正常运行
- 结果 dict 可被 `json.dumps()` 序列化（排查了 `np.int32` 不可序列化的 bug）

### 发现并修复的 Bug

**RoboRefit bbox 序列化问题**：

parquet 中 bbox 字段读出为 numpy array，`list()` 后元素仍为 `np.int32`。`np.int32` 无法被标准 `json.dump()` 序列化，会导致结果文件写入失败。

```python
# 有 bug 的版本
bbox = list(sample['bbox'])   # [np.int32(175), np.int32(168), ...]

# 修复后
bbox = [int(x) for x in sample['bbox']]  # [175, 168, ...]
```

---

## 未处理的脚本

**`hf_inference_crpe.py`** 暂未修复，原因：

1. 需要下载 COCO 图像数据集（~20GB）
2. CRPE 私有图像数据集需单独申请访问权限
3. Flash Attention 2 在脚本中硬编码开启，需确认 `flash-attn` 已安装

---

## 运行方式

修复后，从项目根目录运行（**需先 `cd` 到项目根目录**，因为数据路径使用相对路径）：

```bash
cd /path/to/Embodied-R1

# VABench-Point
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_vabench_point.py

# VABench-VisualTrace
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_vabench_visual_trace.py

# RoboRefit
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_roborefit.py

# Open6DoR-3D
python -m accelerate.commands.launch --num_processes=2 --main_process_port=54321 \
    eval/hf_inference_3d.py
```

结果保存至 `logs/results/`，日志保存至 `logs/`。
