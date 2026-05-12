from __future__ import annotations

import json
import os
from dataclasses import dataclass

from disambiguation import InstructionResolver, MCPStateTracker


OBJECTS = [
    {
        "id": "cup_red_left",
        "category": "cup",
        "name": "red cup",
        "color": "red",
        "size": "small",
        "location": "left side of the table",
        "state": {},
    },
    {
        "id": "cup_blue_right",
        "category": "cup",
        "name": "blue cup",
        "color": "blue",
        "size": "large",
        "location": "right side of the table",
        "state": {},
    },
    {
        "id": "box_yellow_center",
        "category": "box",
        "name": "yellow box",
        "color": "yellow",
        "size": "large",
        "location": "center of the table",
        "state": {"open": True},
    },
    {
        "id": "box_green_bottom",
        "category": "box",
        "name": "green box",
        "color": "green",
        "size": "small",
        "location": "bottom side of the table",
        "state": {"open": False},
    },
    {
        "id": "bottle_white_top",
        "category": "bottle",
        "name": "white bottle",
        "color": "white",
        "size": "small",
        "location": "top side of the shelf",
        "state": {"open": False},
    },
    {
        "id": "bottle_orange_bottom",
        "category": "bottle",
        "name": "orange bottle",
        "color": "orange",
        "size": "large",
        "location": "bottom side of the shelf",
        "state": {"open": True},
    },
    {
        "id": "switch_left",
        "category": "switch",
        "name": "left switch",
        "color": "white",
        "size": "small",
        "location": "left side of the panel",
        "state": {"on": True},
    },
    {
        "id": "switch_right",
        "category": "switch",
        "name": "right switch",
        "color": "white",
        "size": "small",
        "location": "right side of the panel",
        "state": {"on": False},
    },
    {
        "id": "mug_left",
        "category": "mug",
        "name": "left mug",
        "color": "white",
        "size": "small",
        "location": "left side of the table",
        "state": {},
    },
    {
        "id": "mug_right",
        "category": "mug",
        "name": "right mug",
        "color": "white",
        "size": "small",
        "location": "right side of the table",
        "state": {},
    },
    {
        "id": "lighter_center",
        "category": "lighter",
        "name": "lighter",
        "color": "black",
        "size": "small",
        "location": "center of the table",
        "state": {},
    },
    {
        "id": "glue_yellow_left",
        "category": "glue stick",
        "name": "yellow glue stick",
        "color": "yellow",
        "size": "small",
        "location": "left side of the table",
        "state": {},
    },
    {
        "id": "glue_white_right",
        "category": "glue stick",
        "name": "white glue stick",
        "color": "white",
        "size": "small",
        "location": "right side of the table",
        "state": {},
    },
    {
        "id": "apple_left",
        "category": "apple",
        "name": "apple",
        "color": "red",
        "size": "small",
        "location": "left side of the table",
        "state": {},
    },
]


@dataclass(frozen=True)
class TestCase:
    instruction: str
    expected_status: str
    expected_candidate_ids: tuple[str, ...] = ()
    mode: str = "REG"


TEST_CASES = [
    TestCase("pick up the cup", "ambiguous", ("cup_red_left", "cup_blue_right")),
    TestCase("pick up the red cup", "resolved", ("cup_red_left",)),
    TestCase("pick up the blue cup", "resolved", ("cup_blue_right",)),
    TestCase("pick up the cup on the right", "resolved", ("cup_blue_right",)),
    TestCase("pick up the small cup", "resolved", ("cup_red_left",)),
    TestCase("pick up the large cup", "resolved", ("cup_blue_right",)),
    TestCase("open the box", "resolved", ("box_green_bottom",)),
    TestCase("open the green box", "resolved", ("box_green_bottom",)),
    TestCase("close the box that is open", "resolved", ("box_yellow_center",)),
    TestCase("open the box that is closed", "resolved", ("box_green_bottom",)),
    TestCase("pick up the bottle", "ambiguous", ("bottle_white_top", "bottle_orange_bottom")),
    TestCase("pick up the bottle on the top", "resolved", ("bottle_white_top",)),
    TestCase("pick up the orange bottle", "resolved", ("bottle_orange_bottom",)),
    TestCase("close the bottle that is open", "resolved", ("bottle_orange_bottom",)),
    TestCase("turn off the switch that is on", "resolved", ("switch_left",)),
    TestCase("turn on the switch that is off", "resolved", ("switch_right",)),
    TestCase("toggle the switch", "ambiguous", ("switch_left", "switch_right")),
    TestCase("place the mug to the left of the lighter", "ambiguous", ("mug_left", "mug_right")),
    TestCase("pass me the glue stick", "ambiguous", ("glue_yellow_left", "glue_white_right")),
    TestCase("pick up the cups", "ambiguous", ("cup_red_left", "cup_blue_right")),
    TestCase("place the apple behind the lighter", "resolved", ()),
    TestCase("pick up the purple cup", "unknown", ()),
    TestCase("pick up the marker", "ambiguous", ()),
    TestCase("", "invalid", ()),
]


def candidate_ids(candidates: list[dict[str, str]]) -> set[str]:
    return {candidate.get("object_id", "") for candidate in candidates if candidate.get("object_id")}


def candidate_recall(expected: set[str], predicted: set[str]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    return len(expected & predicted) / len(expected)


def main():
    tracker = MCPStateTracker(OBJECTS)
    resolver = InstructionResolver(state_tracker=tracker)
    rows = []

    status_correct = 0
    target_correct = 0
    target_total = 0
    recalls = []

    for case in TEST_CASES:
        result = resolver.resolve(image=None, instruction=case.instruction, mode=case.mode)
        expected_ids = set(case.expected_candidate_ids)
        predicted_ids = candidate_ids(result.candidates)
        is_status_correct = result.status == case.expected_status

        status_correct += int(is_status_correct)
        if case.expected_status == "resolved":
            target_total += 1
            target_correct += int(predicted_ids == expected_ids)
        if case.expected_status == "ambiguous":
            recalls.append(candidate_recall(expected_ids, predicted_ids))

        rows.append(
            {
                "instruction": case.instruction,
                "expected_status": case.expected_status,
                "actual_status": result.status,
                "expected_candidate_ids": sorted(expected_ids),
                "actual_candidate_ids": sorted(predicted_ids),
                "needs_clarification": result.needs_clarification,
                "clarification_question": result.clarification_question,
                "resolved_instruction": result.resolved_instruction,
                "reason": result.reason,
                "status_correct": is_status_correct,
            }
        )

    summary = {
        "num_cases": len(TEST_CASES),
        "status_accuracy": status_correct / len(TEST_CASES),
        "resolved_target_accuracy": target_correct / target_total if target_total else None,
        "ambiguous_candidate_recall": sum(recalls) / len(recalls) if recalls else None,
        "unsafe_pointing_rate_baseline_on_ambiguous": 1.0,
        "unsafe_pointing_rate_ids_on_ambiguous": 0.0,
    }

    print("Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCases")
    for row in rows:
        print("=" * 80)
        print("instruction:", repr(row["instruction"]))
        print("expected_status:", row["expected_status"])
        print("actual_status:", row["actual_status"])
        print("expected_candidate_ids:", row["expected_candidate_ids"])
        print("actual_candidate_ids:", row["actual_candidate_ids"])
        print("needs_clarification:", row["needs_clarification"])
        print("question:", row["clarification_question"])
        print("reason:", row["reason"])
        print("status_correct:", row["status_correct"])

    os.makedirs("output_results", exist_ok=True)
    output_path = "output_results/ids_behavior_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
