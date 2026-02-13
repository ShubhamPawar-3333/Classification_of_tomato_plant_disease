"""
Feedback Collector

Saves user feedback (thumbs up/down on diagnosis accuracy) to a
JSONL file for future model improvement and monitoring.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


FEEDBACK_FILE = "feedback.jsonl"


class FeedbackCollector:
    """Collects and stores user feedback on predictions."""

    def __init__(self, filepath: str = FEEDBACK_FILE):
        self.filepath = Path(filepath)

    def save_feedback(
        self,
        image_name: str,
        predicted_class: str,
        confidence: float,
        severity_level: str,
        is_correct: bool,
        user_comment: Optional[str] = None,
    ) -> dict:
        """
        Save user feedback to JSONL file.

        Args:
            image_name: Name of the uploaded image
            predicted_class: Model's predicted class
            confidence: Model confidence score
            severity_level: Predicted severity
            is_correct: Whether user confirms diagnosis is correct
            user_comment: Optional free-text comment

        Returns:
            dict with the saved feedback entry
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_name": image_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "severity_level": severity_level,
            "is_correct": is_correct,
            "user_comment": user_comment or "",
        }

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Feedback] Saved: {predicted_class} — "
              f"{'✓ correct' if is_correct else '✗ incorrect'}")

        return entry

    def get_stats(self) -> dict:
        """Get summary statistics of collected feedback."""
        if not self.filepath.exists():
            return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0}

        total = correct = 0
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total += 1
                    if entry.get("is_correct"):
                        correct += 1

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0,
        }
