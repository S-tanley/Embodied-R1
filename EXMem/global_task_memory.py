import os
import json
import time
import math
from typing import Any, Dict, List, Optional, Tuple


class GlobalTaskMemoryManager:
    """Cross-episode global memory for visual trace tasks.
    
    Maintains a shared failure library across all episodes to help the model
    identify and avoid problematic image regions, trajectory patterns, etc.
    """

    def __init__(self, output_dir: str = "logs/results", enabled: bool = True, top_k: int = 3):
        self.enabled = enabled
        self.output_dir = output_dir
        self.top_k = top_k
        os.makedirs(self.output_dir, exist_ok=True)

        # Shared event log (all episodes)
        self.events_path = os.path.join(self.output_dir, "global_events.jsonl")
        # Failure library (cached for quick retrieval)
        self.failure_lib_path = os.path.join(self.output_dir, "failure_library.json")

        self.all_events: List[Dict[str, Any]] = []
        self.failure_library: List[Dict[str, Any]] = []

        if self.enabled:
            self._load_existing()

    def write_event(self, episode_id: str, step_id: int, event: Dict[str, Any]) -> None:
        """Write a step result to global event log."""
        if not self.enabled:
            return

        event = dict(event)
        event.setdefault("episode_id", episode_id)
        event.setdefault("step_id", step_id)
        event.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

        # Append to in-memory list
        self.all_events.append(event)

        # Append to JSONL
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Update failure library if this is a high-error step
        if self._is_high_error(event):
            self._add_to_failure_library(event)

    def retrieve_failure_context(self, pred_traj_length: int, image_width: int, image_height: int) -> str:
        """Retrieve relevant failures from library and format as prompt context."""
        if not self.enabled or not self.failure_library:
            return ""

        # Score failures by similarity to current trajectory length and image size
        scored: List[Tuple[float, Dict[str, Any]]] = []
        
        for failure in self.failure_library:
            score = self._score_failure_similarity(
                failure, pred_traj_length, image_width, image_height
            )
            if score > 0:
                scored.append((score, failure))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Format top-k failures as bullet points
        bullets: List[str] = []
        for _, failure in scored[:self.top_k]:
            bullet = self._failure_to_prompt_line(failure)
            if bullet and bullet not in bullets:
                bullets.append(bullet)

        if not bullets:
            return ""

        return "Global failure context:\n" + "\n".join(f"- {b}" for b in bullets) + "\n\n"

    def _is_high_error(self, event: Dict[str, Any]) -> bool:
        """Check if this event represents a high-error step worth remembering."""
        rmse = event.get("rmse")
        mae = event.get("mae")
        if rmse is None or mae is None:
            return False
        # Consider high error if RMSE > 50 or MAE > 25
        return rmse > 50 or mae > 25

    def _add_to_failure_library(self, event: Dict[str, Any]) -> None:
        """Add a high-error event to the failure library."""
        failure_entry = {
            "episode_id": event.get("episode_id"),
            "step_id": event.get("step_id"),
            "rmse": event.get("rmse"),
            "mae": event.get("mae"),
            "pred_traj_length": len(event.get("pred_traj", [])),
            "ans_traj_length": len(event.get("ans_traj", [])),
            "pred_start": event.get("pred_traj", [[]])[0] if event.get("pred_traj") else None,
            "pred_end": event.get("pred_traj", [[]])[-1] if event.get("pred_traj") else None,
            "ans_start": event.get("ans_traj", [[]])[0] if event.get("ans_traj") else None,
            "ans_end": event.get("ans_traj", [[]])[-1] if event.get("ans_traj") else None,
            "timestamp": event.get("timestamp"),
        }
        self.failure_library.append(failure_entry)
        # Keep only recent 500 failures
        if len(self.failure_library) > 500:
            self.failure_library = self.failure_library[-500:]
        self._save_failure_library()

    def _score_failure_similarity(
        self, failure: Dict[str, Any], pred_traj_len: int, img_w: int, img_h: int
    ) -> float:
        """Score how similar a failure is to current trajectory."""
        score = 0.0

        # Trajectory length similarity (prefer similar-length failures)
        failure_traj_len = failure.get("pred_traj_length", 0)
        if failure_traj_len > 0:
            len_similarity = 1.0 / (1.0 + abs(pred_traj_len - failure_traj_len) / max(pred_traj_len, 1))
            score += len_similarity * 2.0

        # Error severity (prefer high-error failures as they're more informative)
        rmse = failure.get("rmse", 0)
        if rmse > 100:
            score += 2.0
        elif rmse > 50:
            score += 1.0

        # Recency (more recent failures get slight boost)
        score += 0.5

        return score

    def _failure_to_prompt_line(self, failure: Dict[str, Any]) -> str:
        """Format a failure as a single prompt line."""
        rmse = failure.get("rmse")
        mae = failure.get("mae")
        pred_len = failure.get("pred_traj_length", 0)
        ans_len = failure.get("ans_traj_length", 0)

        if rmse is None or mae is None:
            return ""

        # Generate a pattern description
        pattern = f"High-error trajectory (RMSE={rmse:.1f}, MAE={mae:.1f})"
        if pred_len != ans_len:
            pattern += f" with length mismatch (pred={pred_len}, ans={ans_len})"
        
        return pattern

    def _load_existing(self) -> None:
        """Load existing global event log and failure library."""
        if os.path.exists(self.events_path):
            with open(self.events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.all_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if os.path.exists(self.failure_lib_path):
            try:
                with open(self.failure_lib_path, "r", encoding="utf-8") as f:
                    self.failure_library = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    def _save_failure_library(self) -> None:
        """Save failure library to disk."""
        with open(self.failure_lib_path, "w", encoding="utf-8") as f:
            json.dump(self.failure_library, f, ensure_ascii=False, indent=2)

    def finalize(self) -> None:
        """Finalize memory operations (placeholder for cleanup)."""
        return
