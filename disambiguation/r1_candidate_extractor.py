from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


CANDIDATE_EXTRACTION_PROMPT = """You are a visual candidate extraction module for a robot.

Given the image and the user instruction, list all visible objects or regions that could satisfy the instruction.

User instruction:
{instruction}

Return only one valid JSON object. Do not include markdown, <think>, <answer>, or explanation.

JSON schema:
{{
  "target_query": "",
  "candidates": [
    {{
      "name": "",
      "visual_attributes": "",
      "location": ""
    }}
  ]
}}

Rules:
- Only list candidates that are visible in the image.
- If exactly one visible object satisfies the instruction, return exactly one candidate.
- If multiple visible objects could satisfy the instruction, return all likely candidates.
- If no visible object satisfies the instruction, return an empty candidates list.
- Do not choose the final target when multiple candidates exist.
- Do not rewrite the instruction as a guessed action.
"""


@dataclass
class CandidateExtraction:
    target_query: str = ""
    candidates: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    parsed: dict[str, Any] | None = None


class R1CandidateExtractor:
    def __init__(self, model: Any, processor: Any, max_new_tokens: int = 512):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def extract_candidates(self, image: Any, instruction: str, mode: str | None = None) -> CandidateExtraction:
        import torch

        prompt = CANDIDATE_EXTRACTION_PROMPT.format(instruction=instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [image]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.02,
            )

        output = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = self._extract_json(output)
        if not isinstance(parsed, dict):
            return CandidateExtraction(raw_output=output, parsed=None)

        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []

        normalized_candidates = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                normalized_candidates.append(
                    {
                        "name": str(candidate.get("name", "")).strip(),
                        "visual_attributes": str(candidate.get("visual_attributes", "")).strip(),
                        "location": str(candidate.get("location", "")).strip(),
                    }
                )

        return CandidateExtraction(
            target_query=str(parsed.get("target_query", "")).strip(),
            candidates=normalized_candidates,
            raw_output=output,
            parsed=parsed,
        )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
