from __future__ import annotations

import json
import os

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from disambiguation import InstructionResolver, R1CandidateExtractor
from inference_example import DEFAULT_CKPT_PATH


TEST_CASES = [
    {
        "image": "example_data/roborefit_18992.png",
        "instruction": "bring me the camel model",
        "mode": "REG",
    },
    {
        "image": "example_data/roborefit_18992.png",
        "instruction": "bring me the model",
        "mode": "REG",
    },
    {
        "image": "example_data/put the red block on top of the yellow block.png",
        "instruction": "put the block on top of the block",
        "mode": "VTG",
    },
    {
        "image": "example_data/put the red block on top of the yellow block.png",
        "instruction": "put the red block on top of the yellow block",
        "mode": "VTG",
    },
]


def main():
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_CKPT_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(DEFAULT_CKPT_PATH)
    print("Model loaded.")

    resolver = InstructionResolver(candidate_extractor=R1CandidateExtractor(model, processor))
    results = []

    for case in TEST_CASES:
        image = Image.open(case["image"]).convert("RGB")
        result = resolver.resolve(image=image, instruction=case["instruction"], mode=case["mode"])
        result_dict = {
            "image": case["image"],
            "instruction": case["instruction"],
            "mode": case["mode"],
            "disambiguation": result.to_dict(),
        }
        results.append(result_dict)

        print("=" * 80)
        print("image:", case["image"])
        print("instruction:", case["instruction"])
        print("mode:", case["mode"])
        print("status:", result.status)
        print("needs_clarification:", result.needs_clarification)
        print("resolved_instruction:", result.resolved_instruction)
        print("clarification_question:", result.clarification_question)
        print("candidates:")
        print(json.dumps(result.candidates, ensure_ascii=False, indent=2))
        print("reason:", result.reason)
        if result.raw_candidate_output:
            print("raw candidate output:")
            print(result.raw_candidate_output)

    os.makedirs("output_results", exist_ok=True)
    output_path = "output_results/disambiguation_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
