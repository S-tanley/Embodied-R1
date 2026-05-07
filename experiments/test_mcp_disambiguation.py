from __future__ import annotations

import json

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
]


TEST_CASES = [
    ("REG", "pick up the cup"),
    ("REG", "pick up the red cup"),
    ("REG", "pick up the cup on the right"),
    ("REG", "pick up the large cup"),
    ("REG", "turn off the switch that is on"),
    ("REG", "turn on the switch that is off"),
]


def main():
    tracker = MCPStateTracker(OBJECTS)
    resolver = InstructionResolver(state_tracker=tracker)

    for mode, instruction in TEST_CASES:
        result = resolver.resolve(image=None, instruction=instruction, mode=mode)
        print("=" * 80)
        print("instruction:", instruction)
        print("mode:", mode)
        print("status:", result.status)
        print("needs_clarification:", result.needs_clarification)
        print("resolved_instruction:", result.resolved_instruction)
        print("clarification_question:", result.clarification_question)
        print("reason:", result.reason)
        print("candidates:")
        print(json.dumps(result.candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
