from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedType
import json
import os
import sys
from tqdm import tqdm
import time
import logging
import random
import textwrap
import pandas as pd
import PIL
from PIL import Image, ImageDraw
from io import BytesIO
import re
import numpy as np
from scipy.interpolate import interp1d

# Ensure project root is importable regardless of launch directory.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import global memory manager
from EXMem.global_task_memory import GlobalTaskMemoryManager

# Optional: wandb for experiment logging
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for content in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])
    return image_inputs, None


def parse_points_from_output(output):
    points = []

    pattern = r"<point>\[(.*?)\]</point>"
    match = re.search(pattern, output)
    if match:
        try:
            points_str = match.group(1)
            if points_str.strip():
                points_str = points_str.replace("'", '"')
                points_str = f"[{points_str}]"
                try:
                    points = json.loads(points_str)
                except json.JSONDecodeError:
                    coord_pattern = r"\[(\d+\.?\d*),\s*(\d+\.?\d*)\]"
                    coords = re.findall(coord_pattern, points_str)
                    points = [[float(x), float(y)] for x, y in coords]
        except Exception as e:
            logger.error(f"Error parsing points: {e}")
    return points


def interpolate_trajectory(trajectory, new_length):
    if len(trajectory) <= 1 or new_length <= 1:
        return trajectory

    old_indices = np.arange(len(trajectory))
    new_indices = np.linspace(0, len(trajectory) - 1, new_length)

    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]

    x_interpolator = interp1d(old_indices, x_coords, kind='linear')
    y_interpolator = interp1d(old_indices, y_coords, kind='linear')

    new_x_coords = x_interpolator(new_indices)
    new_y_coords = y_interpolator(new_indices)

    return [[x, y] for x, y in zip(new_x_coords, new_y_coords)]


def calculate_rmse_mae(pred_trajectory, ans_trajectory):
    if len(pred_trajectory) != len(ans_trajectory):
        logger.warning(f"Trajectory length mismatch: pred={len(pred_trajectory)}, ans={len(ans_trajectory)}")
        return None, None

    squared_diffs = []
    abs_diffs = []

    for pred_point, ans_point in zip(pred_trajectory, ans_trajectory):
        dx = pred_point[0] - ans_point[0]
        dy = pred_point[1] - ans_point[1]

        squared_diff = dx ** 2 + dy ** 2
        squared_diffs.append(squared_diff)

        abs_diff = (abs(dx) + abs(dy)) / 2
        abs_diffs.append(abs_diff)

    rmse = np.sqrt(np.mean(squared_diffs))
    mae = np.mean(abs_diffs)

    return rmse, mae


def interpolate_color(start_color, end_color, ratio):
    r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
    g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
    b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
    return (r, g, b)


def draw_diamond(draw, center_x, center_y, size, fill_color, outline_color=(255, 255, 255)):
    points = [
        (center_x, center_y - size),
        (center_x + size, center_y),
        (center_x, center_y + size),
        (center_x - size, center_y),
    ]
    draw.polygon(points, fill=fill_color, outline=outline_color, width=2)


def draw_triangle(draw, center_x, center_y, size, fill_color, outline_color=(255, 255, 255)):
    points = [
        (center_x, center_y - size),
        (center_x - size, center_y + size),
        (center_x + size, center_y + size),
    ]
    draw.polygon(points, fill=fill_color, outline=outline_color, width=2)


def visualize_trajectory_and_points(image, pred_trajectory, ans_trajectory, save_path):
    draw = ImageDraw.Draw(image)

    if len(pred_trajectory) > 1:
        start_color = (255, 0, 0)
        end_color = (0, 0, 255)

        for i in range(len(pred_trajectory) - 1):
            ratio = i / (len(pred_trajectory) - 1)
            line_color = interpolate_color(start_color, end_color, ratio)

            start_point = tuple(pred_trajectory[i])
            end_point = tuple(pred_trajectory[i + 1])
            draw.line([start_point, end_point], fill=line_color, width=3)

        start_x, start_y = pred_trajectory[0]
        draw_diamond(draw, int(start_x), int(start_y), 8, (255, 0, 0))

        end_x, end_y = pred_trajectory[-1]
        draw_triangle(draw, int(end_x), int(end_y), 8, (0, 0, 255))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    return save_path


def process_sample(sample, model, processor, device, reasoning_model, instruct_following, gen_args, vis_dir, disable_visualization=True, global_memory: GlobalTaskMemoryManager = None):
    question = sample['problem'].strip()
    question = question + '\n' + instruct_following
    question = textwrap.dedent(question).strip()

    doc_id = sample.get('id', str(hash(question))[:8])
    episode_id = sample.get('episode_id', sample.get('task_id', doc_id))
    step_id = sample.get('step', sample.get('step_id', 0))

    memory_used = False
    retrieval_score = 0.0
    if global_memory is not None:
        image_width = sample.get('image_width', 1000)
        image_height = sample.get('image_height', 1000)
        global_context = global_memory.retrieve_failure_context(
            pred_traj_length=10,
            image_width=image_width,
            image_height=image_height,
        )
        question = global_context + question
        memory_used = bool(global_context)

    answer = sample['answer']
    answer = answer.replace("<type>fsd_visual_trace</type>", "")
    json_answer = json.loads(answer)
    ans_traj = json_answer["trajectory"]

    try:
        image_bytes = BytesIO(sample["images"][0])
        image = PIL.Image.open(image_bytes).convert("RGB")
        width, height = image.size

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_args)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        predicted = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        original_predicted = predicted

        if reasoning_model:
            pattern = r"<answer>(.*?)<\/answer>"
            match = re.search(pattern, predicted)
            if match:
                predicted = match.group(1).strip()
            else:
                logger.info(f"oracle prediction: {predicted}")
                predicted = ""

        pred_traj = parse_points_from_output(predicted)

        ans_traj_scaled = []
        for point in ans_traj:
            x, y = point
            x = (x / width) * (1000 - 1)
            y = (y / height) * (1000 - 1)
            ans_traj_scaled.append([x, y])
        ans_traj = ans_traj_scaled

        pred_traj_scaled = []
        for point in pred_traj:
            x, y = point
            x = (x / width) * (1000 - 1)
            y = (y / height) * (1000 - 1)
            pred_traj_scaled.append([x, y])
        pred_traj = pred_traj_scaled

        if len(pred_traj) > 1 and len(ans_traj) > 1:
            new_length = max(len(pred_traj), len(ans_traj))
            pred_traj_interp = interpolate_trajectory(pred_traj, new_length)
            ans_traj_interp = interpolate_trajectory(ans_traj, new_length)
            rmse, mae = calculate_rmse_mae(pred_traj_interp, ans_traj_interp)
        else:
            rmse, mae = None, None
            pred_traj_interp = pred_traj
            ans_traj_interp = ans_traj

        if global_memory is not None:
            event = {
                'pred_traj': pred_traj,
                'ans_traj': ans_traj,
                'rmse': rmse,
                'mae': mae,
            }
            global_memory.write_event(episode_id, step_id, event)

        pred_traj_visual = []
        for point in pred_traj:
            x, y = point
            x = (x / (1000 - 1)) * width
            y = (y / (1000 - 1)) * height
            pred_traj_visual.append([x, y])

        ans_traj_visual = []
        for point in ans_traj:
            x, y = point
            x = (x / (1000 - 1)) * width
            y = (y / (1000 - 1)) * height
            ans_traj_visual.append([x, y])

        vis_path = None
        if not disable_visualization and len(pred_traj) > 0 and random.random() < 1:
            vis_path = os.path.join(vis_dir, f"sample_{doc_id}_{rmse:.2f}_{mae:.2f}.jpg")
            visualize_trajectory_and_points(image, pred_traj_visual, ans_traj_visual, vis_path)

        logger.info("------------------------------------------------------")
        logger.info(f"id: {doc_id}")
        logger.info(f"question: {question}")
        logger.info(f"predicted: {original_predicted}")
        logger.info(f"parsed trajectory points: {len(pred_traj)}")
        logger.info(f"ground truth trajectory points: {len(ans_traj)}")
        if rmse is not None and mae is not None:
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
        if vis_path:
            logger.info(f"visualization saved to: {vis_path}")
        logger.info("------------------------------------------------------")

        return {
            'question_id': doc_id,
            'question': question,
            'predicted': predicted,
            'original_predicted': original_predicted,
            'pred_traj': pred_traj,
            'ans_traj': ans_traj,
            'rmse': rmse,
            'mae': mae,
            'memory_used': memory_used,
            'retrieval_score': retrieval_score,
            'vis_path': vis_path,
        }

    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return None


def main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, dataset_path, split, vis_dir, disable_visualization=True, use_flash_attention=False):
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"device: {device}")
    set_seed(42)

    if accelerator.num_processes > 1:
        local_device = torch.device(f"cuda:{accelerator.local_process_index}")
        device_map = f"cuda:{accelerator.local_process_index}"
        logger.info(f"Process {accelerator.process_index} using GPU: {accelerator.local_process_index}")
    else:
        local_device = device
        device_map = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
    )

    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels)

    start_time = time.time()

    if accelerator.num_processes > 1:
        if accelerator.distributed_type == DistributedType.FSDP:
            model = accelerator.prepare(model)
        else:
            model = accelerator.prepare_model(model, evaluation_mode=True)

        if accelerator.is_main_process:
            logger.info(f"Using {accelerator.num_processes} devices for data parallel processing")

    global_memory_enabled = True
    global_memory = GlobalTaskMemoryManager(output_dir='logs/results', enabled=global_memory_enabled, top_k=3)

    run_wandb = WANDB_AVAILABLE and accelerator.is_main_process
    if run_wandb:
        try:
            wandb.init(project="embodied_exmem", name=f"{task_name}_{model_name}_v2_1", config={"top_k": 3}, reinit=True)
        except Exception as e:
            logger.warning(f"Failed to init wandb: {e}")
            run_wandb = False

    with accelerator.main_process_first():
        logger.info(f"Loading dataset: {dataset_path}")
        dataset = pd.read_parquet(dataset_path)
        test_data = dataset.to_dict(orient='records')
        logger.info(f"Test data samples: {len(test_data)}")

    process_idx = accelerator.process_index
    num_processes = accelerator.num_processes
    samples_per_process = len(test_data) // num_processes
    start_idx = process_idx * samples_per_process
    end_idx = start_idx + samples_per_process if process_idx < num_processes - 1 else len(test_data)

    local_results = []
    memory_used_count = 0
    retrieval_scores = []

    logged_count = 0
    running_rmses = []
    running_maes = []
    running_memory_used = 0
    running_high_error = 0

    if accelerator.is_main_process:
        iterator = tqdm(range(start_idx, end_idx), desc=f"Process {process_idx}")
    else:
        iterator = range(start_idx, end_idx)

    for idx in iterator:
        result = process_sample(
            test_data[idx], model, processor, local_device,
            reasoning_model, instruct_following, gen_args, vis_dir, disable_visualization, global_memory
        )
        if result is not None:
            local_results.append(result)
            if result.get('memory_used'):
                memory_used_count += 1
            retrieval_scores.append(result.get('retrieval_score', 0.0))

            if accelerator.is_main_process and run_wandb:
                try:
                    valid = result.get('rmse') is not None
                    if valid:
                        running_rmses.append(result['rmse'])
                        running_maes.append(result['mae'])
                    if result.get('memory_used'):
                        running_memory_used += 1
                    if result.get('rmse') is not None and result['rmse'] >= 100:
                        running_high_error += 1
                    logged_count += 1

                    running_avg_rmse = float(np.mean(running_rmses)) if running_rmses else None
                    running_avg_mae = float(np.mean(running_maes)) if running_maes else None
                    running_median_rmse = float(np.median(running_rmses)) if running_rmses else None
                    running_memory_used_rate = running_memory_used / logged_count if logged_count else 0.0

                    wandb.log({
                        "test/sample_idx": idx,
                        "test/rmse": result.get('rmse'),
                        "test/mae": result.get('mae'),
                        "test/memory_used": 1 if result.get('memory_used') else 0,
                        "test/retrieval_score": result.get('retrieval_score'),
                        "test/pred_points": len(result.get('pred_traj', [])),
                        "test/ans_points": len(result.get('ans_traj', [])),
                        "test/valid_sample": 1 if valid else 0,
                        "test/running_avg_rmse": running_avg_rmse,
                        "test/running_avg_mae": running_avg_mae,
                        "test/running_median_rmse": running_median_rmse,
                        "test/running_memory_used_rate": running_memory_used_rate,
                        "test/running_high_error_rate": (running_high_error / logged_count) if logged_count else 0.0,
                    }, step=idx)
                except Exception:
                    logger.exception("WandB logging failed for a sample; continuing.")

    all_results = accelerator.gather_for_metrics(local_results)

    if accelerator.is_main_process:
        if isinstance(all_results[0], list):
            final_results = [item for sublist in all_results for item in sublist]
        else:
            final_results = all_results

        rmses = [r['rmse'] for r in final_results if r.get('rmse') is not None]
        maes = [r['mae'] for r in final_results if r.get('mae') is not None]
        mem_used = sum(1 for r in final_results if r.get('memory_used'))
        avg_retrieval_score = float(np.mean([s for s in retrieval_scores if s is not None])) if retrieval_scores else 0.0

        avg_rmse = np.mean(rmses) if rmses else None
        avg_mae = np.mean(maes) if maes else None
        median_rmse = np.median(rmses) if rmses else None
        median_mae = np.median(maes) if maes else None
        p90_rmse = np.percentile(rmses, 90) if rmses else None
        p90_mae = np.percentile(maes, 90) if maes else None
        count_high_error = sum(1 for v in rmses if v is not None and v >= 100)

        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}_{split}_v2_1.json'
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("------------------------------------------------------")
        logger.info(f"Model Name: {model_name}")
        logger.info("------------------------------------------------------")
        logger.info(f"Finished processing. Total samples: {len(final_results)}")
        logger.info(f"Valid samples: {len(rmses)}")
        if avg_rmse is not None and avg_mae is not None:
            logger.info(f"Average RMSE: {avg_rmse:.4f}")
            logger.info(f"Average MAE: {avg_mae:.4f}")
            logger.info(f"Median RMSE: {median_rmse:.4f}")
            logger.info(f"90th perc. RMSE: {p90_rmse:.4f}")
        logger.info(f"Memory used count: {mem_used}")
        logger.info(f"Avg retrieval score: {avg_retrieval_score:.4f}")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        logger.info("------------------------------------------------------")

        if run_wandb:
            wandb.log({
                "avg_rmse": avg_rmse,
                "avg_mae": avg_mae,
                "median_rmse": median_rmse,
                "median_mae": median_mae,
                "p90_rmse": p90_rmse,
                "p90_mae": p90_mae,
                "count_high_error": count_high_error,
                "memory_used_count": mem_used,
                "avg_retrieval_score": avg_retrieval_score,
                "total_time": total_time,
            })

            try:
                scored = [r for r in final_results if r.get('rmse') is not None]
                worst = sorted(scored, key=lambda x: x['rmse'], reverse=True)[:10]
                best = sorted(scored, key=lambda x: x['rmse'])[:10]

                worst_images = []
                for r in worst:
                    p = r.get('vis_path')
                    if p and os.path.exists(p):
                        caption = f"id:{r.get('question_id')} rmse:{r.get('rmse'):.2f}"
                        try:
                            worst_images.append(wandb.Image(p, caption=caption))
                        except Exception:
                            logger.warning(f"Failed to create wandb.Image for {p}")

                best_images = []
                for r in best:
                    p = r.get('vis_path')
                    if p and os.path.exists(p):
                        caption = f"id:{r.get('question_id')} rmse:{r.get('rmse'):.2f}"
                        try:
                            best_images.append(wandb.Image(p, caption=caption))
                        except Exception:
                            logger.warning(f"Failed to create wandb.Image for {p}")

                if worst_images:
                    wandb.log({"test/top10_worst_images": worst_images})
                if best_images:
                    wandb.log({"test/top10_best_images": best_images})
            except Exception:
                logger.exception("Failed to upload top images to WandB")

            try:
                wandb.finish()
            except Exception:
                logger.warning("wandb.finish() failed")

        global_memory.finalize()


if __name__ == "__main__":
    task_name = "VABench_VisualTrace"
    split = "test"
    disable_visualization = False
    use_flash_attention = False

    model_name = "Embodied-R1-3B"
    model_path = "IffYuan/Embodied-R1-3B-v1"
    dataset_path = "/root/autodl-tmp/Embodied-R1/datasets/FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/test.parquet"
    reasoning_model = True

    vis_dir = f"logs/visualizations/{task_name}_{model_name}"
    os.makedirs(vis_dir, exist_ok=True)

    max_pixels = 1605632
    min_pixels = 256 * 28 * 28
    gen_args = {
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": 2048,
        "repetition_penalty": 1.05,
        "do_sample": False,
    }

    if reasoning_model:
        instruct_following = (
            r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
            r'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. '
            r'The answer consists only of several coordinate points, with the overall format being: '
            r'<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>'
        )
    else:
        instruct_following = "Use 2D points to mark the region mentioned in the task with format <point>[[x1, y1], [x2, y2], ...]</point>."

    current_time = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs/log_v2_1", exist_ok=True)
    log_file_name = f"logs/log_v2_1/inference_{task_name}_{model_name}_{current_time}_v2_1.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - Process %(process)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_name)
        ]
    )
    logger = logging.getLogger(f"{task_name}_{model_name}")

    main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, dataset_path, split, vis_dir, disable_visualization, use_flash_attention)
