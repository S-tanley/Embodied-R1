import os
import json
import time
from typing import Any, Dict, List, Optional


class TaskMemoryManager:
    """Minimal task-level episodic memory manager for multi-step visual trace tasks.

    - Stores per-episode `state` and an `event_log` (JSONL) under `output_dir`.
    - Exposes reset/read/write/finalize and a small inject helper.
    """

    def __init__(self, output_dir: str = "logs/results", enabled: bool = True):
        self.enabled = enabled
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # in-memory stores
        self.current_episode: Optional[str] = None
        self.states: Dict[str, Dict[str, Any]] = {}
        # events appended to single jsonl file
        self.events_path = os.path.join(self.output_dir, "exmem_events.jsonl")

    def reset(self, episode_id: str, goal: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self.current_episode = episode_id
        if episode_id not in self.states:
            self.states[episode_id] = {
                "episode_id": episode_id,
                "goal": goal,
                "current_step": 0,
                "completed_subgoals": [],
                "visited_regions": [],
                "failure_signatures": [],
                "last_pred_stats": None,
            }

    def read(self, episode_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return self.states.get(episode_id, {})

    def inject_context(self, base_prompt: str, episode_id: str, top_k: int = 3) -> str:
        """Return a short memory block to prepend to base_prompt."""
        if not self.enabled:
            return base_prompt
        state = self.states.get(episode_id)
        if not state:
            return base_prompt

        bullets: List[str] = []
        # include goal
        if state.get("goal"):
            bullets.append(f"Goal: {state['goal']}")
        # include last_pred_stats
        if state.get("last_pred_stats"):
            s = state["last_pred_stats"]
            bullets.append(f"Last pred: rmse={s.get('rmse')}, mae={s.get('mae')}, points={s.get('num_points')}")
        # include recent failure signatures
        fs = state.get("failure_signatures", [])[-top_k:]
        for f in fs:
            bullets.append(f"Failure: {f.get('reason', 'unknown')}")

        if not bullets:
            return base_prompt

        memory_block = "Relevant memory:\n" + "\n".join(f"- {b}" for b in bullets) + "\n\n"
        return memory_block + base_prompt

    def write_event(self, episode_id: str, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        event = dict(event)
        event.setdefault("episode_id", episode_id)
        event.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # update state quick view
        state = self.states.setdefault(episode_id, {"episode_id": episode_id})
        state["current_step"] = event.get("step_id", state.get("current_step", 0))
        if event.get("event_type") == "step_failure":
            sig = {"step_id": event.get("step_id"), "reason": event.get("failure_reason"), "rmse": event.get("rmse"), "mae": event.get("mae")}
            state.setdefault("failure_signatures", []).append(sig)
        if event.get("event_type") == "subgoal_completed":
            state.setdefault("completed_subgoals", []).append(event.get("subgoal"))
        state["last_pred_stats"] = {"rmse": event.get("rmse"), "mae": event.get("mae"), "num_points": len(event.get("pred_traj", []))}

    def finalize(self, episode_id: str) -> None:
        # placeholder for flush or archiving; keep in-memory state for debugging
        return
