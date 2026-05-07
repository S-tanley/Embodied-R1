from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class MCPObjectState:
    object_id: str
    category: str
    name: str = ""
    color: str = ""
    size: str = ""
    location: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    def to_candidate(self) -> dict[str, str]:
        attributes = ", ".join(value for value in [self.color, self.size] if value)
        state_text = ", ".join(
            self._state_phrase(key, value)
            for key, value in sorted(self.state.items())
            if self._state_phrase(key, value)
        )
        if state_text:
            attributes = f"{attributes}, {state_text}" if attributes else state_text
        return {
            "name": self.name or self.category,
            "visual_attributes": attributes,
            "location": self.location,
            "object_id": self.object_id,
        }

    def _state_phrase(self, key: str, value: Any) -> str:
        if value in (None, ""):
            return ""
        if key == "on" and isinstance(value, bool):
            return "on" if value else "off"
        if key == "open" and isinstance(value, bool):
            return "open" if value else "closed"
        return f"{key}={value}"


class MCPStateTracker:
    """Lightweight structured state tracker for instruction disambiguation."""

    def __init__(self, objects: list[MCPObjectState | dict[str, Any]] | None = None):
        self.objects: list[MCPObjectState] = []
        for obj in objects or []:
            self.add_object(obj)

    def add_object(self, obj: MCPObjectState | dict[str, Any]) -> None:
        if isinstance(obj, MCPObjectState):
            self.objects.append(obj)
            return
        self.objects.append(
            MCPObjectState(
                object_id=str(obj.get("object_id", obj.get("id", ""))).strip(),
                category=str(obj.get("category", "")).strip().lower(),
                name=str(obj.get("name", "")).strip(),
                color=str(obj.get("color", "")).strip().lower(),
                size=str(obj.get("size", "")).strip().lower(),
                location=str(obj.get("location", "")).strip(),
                state=dict(obj.get("state", {})),
            )
        )

    def retrieve_candidates(self, instruction: str, mode: str | None = None) -> list[dict[str, str]]:
        tokens = set(self._tokens(instruction))
        categories = self._mentioned_categories(tokens)
        if not categories:
            return []

        candidates = [obj for obj in self.objects if obj.category in categories or obj.category.rstrip("s") in categories]
        candidates = self._filter_by_text_constraints(candidates, tokens, instruction)
        return [obj.to_candidate() for obj in candidates]

    def _mentioned_categories(self, tokens: set[str]) -> set[str]:
        categories = {obj.category for obj in self.objects if obj.category}
        aliases = {category.rstrip("s"): category for category in categories}
        mentioned = set()
        for token in tokens:
            if token in categories:
                mentioned.add(token)
            elif token in aliases:
                mentioned.add(aliases[token])
            elif token == "object":
                mentioned.update(categories)
            elif token == "thing":
                mentioned.update(categories)
        return mentioned

    def _filter_by_text_constraints(
        self, objects: list[MCPObjectState], tokens: set[str], instruction: str
    ) -> list[MCPObjectState]:
        colors = {obj.color for obj in self.objects if obj.color}
        sizes = {obj.size for obj in self.objects if obj.size}
        locations = {
            word
            for obj in self.objects
            for word in self._tokens(obj.location)
            if word in {"left", "right", "center", "middle", "top", "bottom", "front", "back"}
        }

        color_constraints = tokens & colors
        size_constraints = tokens & sizes
        location_constraints = tokens & locations

        filtered = objects
        if color_constraints:
            filtered = [obj for obj in filtered if obj.color in color_constraints]
        if size_constraints:
            filtered = [obj for obj in filtered if obj.size in size_constraints]
        if location_constraints:
            filtered = [
                obj
                for obj in filtered
                if location_constraints & set(self._tokens(obj.location))
            ]

        state_constraints = self._state_constraints(instruction, tokens)
        for key, expected in state_constraints.items():
            filtered = [obj for obj in filtered if obj.state.get(key) == expected]

        return filtered

    def _state_constraints(self, instruction: str, tokens: set[str]) -> dict[str, Any]:
        lowered = instruction.lower()
        constraints: dict[str, Any] = {}
        if re.search(r"\b(that is|currently|already)\s+open\b", lowered):
            constraints["open"] = True
        elif re.search(r"\b(that is|currently|already)\s+(closed|close)\b", lowered):
            constraints["open"] = False

        if re.search(r"\b(that is|currently|already)\s+(on|enabled)\b", lowered):
            constraints["on"] = True
        elif re.search(r"\b(that is|currently|already)\s+(off|disabled)\b", lowered):
            constraints["on"] = False

        if "open" in tokens and "open" not in constraints:
            constraints["open"] = True
        if ("closed" in tokens or "close" in tokens) and "open" not in constraints:
            constraints["open"] = False
        return constraints

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())
