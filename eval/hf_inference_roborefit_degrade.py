"""
用退化指令（去掉颜色/尺寸/位置词）在同一张图上重新跑 VLM，
对比 baseline（具体指令）的准确率，验证 IDS 消歧的价值。

运行命令（从项目根目录）：
    CUDA_VISIBLE_DEVICES=3,4 accelerate launch \\
        --num_processes=2 --main_process_port=54322 \\
        eval/hf_inference_roborefit_degrade.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import pandas as pd
from io import BytesIO
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedType
import json
import re
import time
import logging
import random
import textwrap
from PIL import Image, ImageDraw

from disambiguation.resolver import _ATTRIBUTE_WORDS, _SPATIAL_WORDS


INSTRUCTION_RE = re.compile(r"this sentence describes: (.+?)\. The results")


def degrade_instruction(instr: str) -> str:
    tokens = instr.split()
    kept = [t for t in tokens if t.lower().rstrip(".,") not in (_ATTRIBUTE_WORDS | _SPATIAL_WORDS)]
    return " ".join(kept)


def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for content in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])
    return image_inputs, None


def parse_points_from_output(output):
    points = []
    match = re.search(r"<point>\[(.*?)\]</point>", output)
    if match:
        try:
            pts_str = "[" + match.group(1).replace("'", '"') + "]"
            points = json.loads(pts_str)
        except Exception:
            coords = re.findall(r"\[(\d+\.?\d*),\s*(\d+\.?\d*)\]", match.group(1))
            points = [[float(x), float(y)] for x, y in coords]
    return points


def check_points_in_bbox(points, bbox):
    if not points:
        return 0, 0
    x1, y1, x2, y2 = bbox
    in_box = sum(1 for p in points if len(p) == 2 and x1 <= p[0] <= x2 and y1 <= p[1] <= y2)
    return in_box, len(points)


def build_degraded_index(baseline_path: str) -> dict:
    """
    从 baseline JSON 建立全量退化样本索引。
    每条记录额外标注 ids_detectable：
      True  → 退化后 IDS 能检测为 ambiguous（IDS 会换回具体指令）
      False → 退化后 IDS 也检测不到（IDS 无法介入）
    """
    from disambiguation import InstructionResolver
    resolver = InstructionResolver()

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    index = {}
    for item in baseline:
        m = INSTRUCTION_RE.search(item["question"])
        if not m:
            continue
        original = m.group(1).strip()
        degraded = degrade_instruction(original)
        if degraded == original:
            continue  # 无属性词，不纳入实验

        r_deg = resolver.resolve(image=None, instruction=degraded, mode="REG")
        ids_detectable = r_deg.needs_clarification

        index[item["question_id"]] = {
            "original_instruction": original,
            "degraded_instruction": degraded,
            "original_accuracy": item["accuracy_score"],
            "ids_detectable": ids_detectable,
        }

    detectable = sum(1 for v in index.values() if v["ids_detectable"])
    print(f"[build_degraded_index] degradable samples: {len(index)}")
    print(f"  IDS detectable:   {detectable}  ({detectable/len(index):.1%})")
    print(f"  IDS undetectable: {len(index)-detectable}  ({(len(index)-detectable)/len(index):.1%})")
    return index


def process_sample(sample, model, processor, device, instruct_following, gen_args, degraded_index):
    doc_id = sample.get("question_id")
    if doc_id not in degraded_index:
        return None  # 没有属性词的样本跳过

    entry = degraded_index[doc_id]
    instruction = entry["degraded_instruction"]
    bbox = [int(x) for x in sample["bbox"]]
    image = Image.open(BytesIO(sample["image"]["bytes"]))

    question = (
        f"Provide one or more points coordinate of objects region this sentence describes: "
        f"{instruction}. The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>."
    )
    question = textwrap.dedent(question + "\n" + instruct_following).strip()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_args)

    output = processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    match = re.search(r"<answer>(.*?)<\/answer>", output)
    predicted = match.group(1).strip() if match else ""

    points = parse_points_from_output(predicted)
    points_in_box, total_points = check_points_in_bbox(points, bbox)
    degraded_accuracy = points_in_box / total_points if total_points > 0 else 0

    logger.info(f"qid={doc_id}  original={entry['original_instruction']!r}")
    logger.info(f"  degraded={instruction!r}  degraded_acc={degraded_accuracy:.3f}  original_acc={entry['original_accuracy']:.3f}")

    return {
        "question_id": doc_id,
        "original_instruction": entry["original_instruction"],
        "degraded_instruction": instruction,
        "original_accuracy": entry["original_accuracy"],
        "degraded_accuracy": degraded_accuracy,
        "points_in_box": points_in_box,
        "total_points": total_points,
    }


def main(model_path, max_pixels, min_pixels, gen_args, instruct_following,
         dataset_paths, baseline_path, use_flash_attention=False):
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(42)

    if accelerator.num_processes > 1:
        local_device = torch.device(f"cuda:{accelerator.local_process_index}")
        device_map = f"cuda:{accelerator.local_process_index}"
    else:
        local_device = device
        device_map = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
    )
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels)

    if accelerator.num_processes > 1:
        if accelerator.distributed_type == DistributedType.FSDP:
            model = accelerator.prepare(model)
        else:
            model = accelerator.prepare_model(model, evaluation_mode=True)

    degraded_index = build_degraded_index(baseline_path)
    logger.info(f"Degradable samples: {len(degraded_index)}")

    with accelerator.main_process_first():
        dfs = [pd.read_parquet(p) for p in dataset_paths]
        test_data = pd.concat(dfs, ignore_index=True).to_dict(orient="records")

    process_idx = accelerator.process_index
    num_processes = accelerator.num_processes
    samples_per_process = len(test_data) // num_processes
    start_idx = process_idx * samples_per_process
    end_idx = start_idx + samples_per_process if process_idx < num_processes - 1 else len(test_data)

    from tqdm import tqdm
    iterator = tqdm(range(start_idx, end_idx), desc=f"Process {process_idx}") if accelerator.is_main_process else range(start_idx, end_idx)

    start_time = time.time()
    local_results = []
    for idx in iterator:
        result = process_sample(test_data[idx], model, processor, local_device,
                                instruct_following, gen_args, degraded_index)
        if result is not None:
            local_results.append(result)

    all_results = accelerator.gather_object(local_results)

    if accelerator.is_main_process:
        final = [item for sublist in all_results for item in sublist] if isinstance(all_results[0], list) else all_results

        n = len(final)
        detectable   = [r for r in final if r["ids_detectable"]]
        undetectable = [r for r in final if not r["ids_detectable"]]

        def avg(group, key):
            vals = [r[key] for r in group if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        # Upper bound：全部用具体指令（已知，来自 baseline）
        upper_bound = avg(final, "original_accuracy")

        # New baseline：全部用退化指令（这次跑的）
        new_baseline = avg(final, "degraded_accuracy")

        # With IDS：
        #   IDS 能检测的 → 换回具体指令（original_accuracy）
        #   IDS 检测不到 → 保持退化指令（degraded_accuracy）
        ids_acc_detectable   = avg(detectable,   "original_accuracy")   # IDS 换回后的准确率
        ids_acc_undetectable = avg(undetectable, "degraded_accuracy")   # IDS 无法介入的准确率
        ids_combined = (
            sum(r["original_accuracy"]  for r in detectable) +
            sum(r["degraded_accuracy"]  for r in undetectable)
        ) / n if n else None

        summary = {
            "total_degradable_samples": n,
            "ids_detectable_count": len(detectable),
            "ids_undetectable_count": len(undetectable),
            "---": "---",
            "upper_bound (specific instruction)":  upper_bound,
            "new_baseline (degraded, no IDS)":     new_baseline,
            "with_IDS (detectable→specific, undetectable→degraded)": ids_combined,
            "---2": "---",
            "real_gain_IDS_vs_new_baseline":       (ids_combined - new_baseline) if ids_combined and new_baseline else None,
            "remaining_gap_to_upper_bound":        (upper_bound - ids_combined)  if upper_bound and ids_combined else None,
            "---3": "---",
            "detectable_subset_specific_acc":   ids_acc_detectable,
            "detectable_subset_degraded_acc":   avg(detectable,   "degraded_accuracy"),
            "undetectable_subset_specific_acc": avg(undetectable, "original_accuracy"),
            "undetectable_subset_degraded_acc": ids_acc_undetectable,
        }

        os.makedirs("logs/results", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"logs/results/RoboRefit_degrade_vs_specific_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": final}, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info(f"Degradable samples total: {n}  "
                    f"(IDS detectable: {len(detectable)}, undetectable: {len(undetectable)})")
        logger.info("-" * 60)
        logger.info(f"Upper bound  (specific instruction):          {upper_bound:.4f}")
        logger.info(f"New baseline (degraded, no IDS):              {new_baseline:.4f}")
        logger.info(f"With IDS     (detectable→specific, rest→deg): {ids_combined:.4f}")
        logger.info("-" * 60)
        logger.info(f"Real gain  IDS vs new baseline:   +{ids_combined - new_baseline:.4f}")
        logger.info(f"Remaining gap to upper bound:      {upper_bound - ids_combined:.4f}")
        logger.info("-" * 60)
        logger.info(f"IDS detectable subset ({len(detectable)}):   "
                    f"degraded={avg(detectable, 'degraded_accuracy'):.4f} → specific={ids_acc_detectable:.4f}")
        logger.info(f"IDS undetectable subset ({len(undetectable)}): "
                    f"degraded={ids_acc_undetectable:.4f} (IDS 无法介入)")
        logger.info(f"Saved to {out_path}  |  Time: {time.time()-start_time:.1f}s")
        logger.info("=" * 60)


if __name__ == "__main__":
    model_path = "IffYuan/Embodied-R1-3B-v1"
    baseline_path = "eval/offical-3b/results/RoboRefit_Embodied-R1-3B_True_test_20260423_081716.json"
    dataset_paths = ["eval/data/roborefit_test_0.parquet", "eval/data/roborefit_test_1.parquet"]

    max_pixels = 1605632
    min_pixels = 256 * 28 * 28
    gen_args = {"temperature": 0, "top_p": 1, "max_new_tokens": 2048,
                "repetition_penalty": 1.05, "do_sample": False}

    instruct_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
        r"The answer consists only of several coordinate points, with the overall format being: "
        r"<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - Process %(process)d - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/inference_RoboRefit_degrade_{ts}.log")],
    )
    logger = logging.getLogger("RoboRefit_degrade")
    main(model_path, max_pixels, min_pixels, gen_args, instruct_following,
         dataset_paths, baseline_path)
