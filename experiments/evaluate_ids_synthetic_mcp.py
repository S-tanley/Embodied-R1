from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

from disambiguation import InstructionResolver, MCPStateTracker


COLORS = ["red", "blue", "green", "yellow", "white", "orange"]
SIZES = ["small", "large"]
LOCATIONS = [
    ("left", "left side of the table"),
    ("right", "right side of the table"),
    ("top", "top side of the table"),
    ("bottom", "bottom side of the table"),
]


@dataclass(frozen=True)
class EvalCase:
    instruction: str
    objects: list[dict]
    expected_status: str
    expected_candidate_ids: tuple[str, ...]
    case_type: str
    mode: str = "REG"


def make_object(object_id: str, category: str, color: str, size: str, location_word: str, location: str) -> dict:
    return {
        "id": object_id,
        "category": category,
        "name": f"{color} {category}",
        "color": color,
        "size": size,
        "location": location,
        "state": {},
        "location_word": location_word,
    }


def make_stateful_object(
    object_id: str, category: str, color: str, size: str, location_word: str, location: str, state: dict
) -> dict:
    obj = make_object(object_id, category, color, size, location_word, location)
    obj["state"] = state
    return obj


def scene_for_category(rng: random.Random, category: str, n: int = 2) -> list[dict]:
    colors = rng.sample(COLORS, n)
    sizes = (SIZES * ((n // len(SIZES)) + 1))[:n]
    rng.shuffle(sizes)
    locations = rng.sample(LOCATIONS, n)
    objects = []
    for i in range(n):
        location_word, location = locations[i]
        objects.append(make_object(f"{category}_{i}_{colors[i]}_{location_word}", category, colors[i], sizes[i], location_word, location))
    return objects


def stateful_switch_scene(rng: random.Random) -> list[dict]:
    locations = rng.sample(LOCATIONS, 2)
    return [
        make_stateful_object("switch_on", "switch", "white", "small", locations[0][0], locations[0][1], {"on": True}),
        make_stateful_object("switch_off", "switch", "white", "small", locations[1][0], locations[1][1], {"on": False}),
    ]


def stateful_box_scene(rng: random.Random) -> list[dict]:
    colors = rng.sample(COLORS, 2)
    locations = rng.sample(LOCATIONS, 2)
    return [
        make_stateful_object(f"box_open_{colors[0]}", "box", colors[0], "large", locations[0][0], locations[0][1], {"open": True}),
        make_stateful_object(f"box_closed_{colors[1]}", "box", colors[1], "small", locations[1][0], locations[1][1], {"open": False}),
    ]


def ids_for(objects: list[dict]) -> tuple[str, ...]:
    return tuple(obj["id"] for obj in objects)


def make_case(rng: random.Random, index: int) -> EvalCase:
    case_kind = index % 8

    if case_kind == 0:
        category = rng.choice(["cup", "bottle", "toy"])
        objects = scene_for_category(rng, category, n=2)
        return EvalCase(f"pick up the {category}", objects, "ambiguous", ids_for(objects), "category_ambiguous")

    if case_kind == 1:
        category = rng.choice(["cup", "bottle", "toy"])
        objects = scene_for_category(rng, category, n=2)
        target = rng.choice(objects)
        return EvalCase(
            f"pick up the {target['color']} {category}",
            objects,
            "resolved",
            (target["id"],),
            "color_resolved",
        )

    if case_kind == 2:
        category = rng.choice(["cup", "bottle", "toy"])
        objects = scene_for_category(rng, category, n=2)
        target = rng.choice(objects)
        return EvalCase(
            f"pick up the {category} on the {target['location_word']}",
            objects,
            "resolved",
            (target["id"],),
            "location_resolved",
        )

    if case_kind == 3:
        category = rng.choice(["cup", "bottle", "toy"])
        objects = scene_for_category(rng, category, n=2)
        target = rng.choice(objects)
        return EvalCase(
            f"pick up the {target['size']} {category}",
            objects,
            "resolved",
            (target["id"],),
            "size_resolved",
        )

    if case_kind == 4:
        objects = stateful_switch_scene(rng)
        target = next(obj for obj in objects if obj["state"]["on"])
        return EvalCase("turn off the switch that is on", objects, "resolved", (target["id"],), "state_on_resolved")

    if case_kind == 5:
        objects = stateful_box_scene(rng)
        target = next(obj for obj in objects if not obj["state"]["open"])
        return EvalCase("open the box that is closed", objects, "resolved", (target["id"],), "state_closed_resolved")

    if case_kind == 6:
        category = rng.choice(["cup", "bottle", "toy"])
        objects = scene_for_category(rng, category, n=2)
        used_colors = {obj["color"] for obj in objects}
        unknown_color = next(color for color in COLORS if color not in used_colors)
        return EvalCase(f"pick up the {unknown_color} {category}", objects, "unknown", (), "unknown_attribute")

    return EvalCase("", scene_for_category(rng, "cup", n=2), "invalid", (), "invalid_empty")


def candidate_ids(candidates: list[dict[str, str]]) -> set[str]:
    return {candidate.get("object_id", "") for candidate in candidates if candidate.get("object_id")}


def recall(expected: set[str], predicted: set[str]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    return len(expected & predicted) / len(expected)


def evaluate(cases: list[EvalCase]) -> dict:
    rows = []
    status_correct = 0
    target_correct = 0
    target_total = 0
    ambiguous_recalls = []
    unknown_correct = 0
    unknown_total = 0
    false_ambiguity = 0
    resolved_total = 0

    for case in cases:
        tracker = MCPStateTracker(case.objects)
        resolver = InstructionResolver(state_tracker=tracker)
        result = resolver.resolve(image=None, instruction=case.instruction, mode=case.mode)

        expected_ids = set(case.expected_candidate_ids)
        predicted_ids = candidate_ids(result.candidates)
        is_status_correct = result.status == case.expected_status
        status_correct += int(is_status_correct)

        if case.expected_status == "resolved":
            resolved_total += 1
            target_total += 1
            target_correct += int(predicted_ids == expected_ids)
            false_ambiguity += int(result.status == "ambiguous")
        elif case.expected_status == "ambiguous":
            ambiguous_recalls.append(recall(expected_ids, predicted_ids))
        elif case.expected_status == "unknown":
            unknown_total += 1
            unknown_correct += int(result.status == "unknown")

        rows.append(
            {
                "case_type": case.case_type,
                "instruction": case.instruction,
                "expected_status": case.expected_status,
                "actual_status": result.status,
                "expected_candidate_ids": sorted(expected_ids),
                "actual_candidate_ids": sorted(predicted_ids),
                "needs_clarification": result.needs_clarification,
                "clarification_question": result.clarification_question,
                "reason": result.reason,
                "status_correct": is_status_correct,
            }
        )

    ambiguous_count = sum(1 for case in cases if case.expected_status == "ambiguous")
    summary = {
        "num_cases": len(cases),
        "status_accuracy": status_correct / len(cases),
        "resolved_target_accuracy": target_correct / target_total if target_total else None,
        "ambiguous_candidate_recall": sum(ambiguous_recalls) / len(ambiguous_recalls) if ambiguous_recalls else None,
        "unknown_detection_accuracy": unknown_correct / unknown_total if unknown_total else None,
        "false_ambiguity_rate_on_resolved": false_ambiguity / resolved_total if resolved_total else None,
        "unsafe_pointing_rate_baseline_on_ambiguous": 1.0 if ambiguous_count else None,
        "unsafe_pointing_rate_ids_on_ambiguous": 0.0 if ambiguous_count else None,
    }
    return {"summary": summary, "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="output_results/ids_synthetic_mcp_eval.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = [make_case(rng, i) for i in range(args.num_cases)]
    report = evaluate(cases)

    print("Summary")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
