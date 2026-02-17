"""
Unit tests for classifier configuration, severity estimation, and feedback.

Run: pytest tests/test_classifier.py -v
"""
import json
import os
import tempfile
from pathlib import Path

import pytest


# ── Config tests ─────────────────────────────────────────────


class TestConfigurationManager:
    """Test that configuration loads correctly."""

    def test_config_yaml_exists(self):
        """config.yaml must exist at expected path."""
        config_path = Path("config/config.yaml")
        assert config_path.exists(), f"Missing {config_path}"

    def test_params_yaml_exists(self):
        """params.yaml must exist at expected path."""
        params_path = Path("params.yaml")
        assert params_path.exists(), f"Missing {params_path}"

    def test_config_manager_instantiates(self):
        """ConfigurationManager should load without errors."""
        from tomato_disease_advisor.config import ConfigurationManager

        cm = ConfigurationManager()
        assert cm is not None

    def test_image_size_is_valid(self):
        """IMAGE_SIZE should be a positive integer."""
        from tomato_disease_advisor.config import ConfigurationManager

        cm = ConfigurationManager()
        size = cm.params.IMAGE_SIZE
        assert isinstance(size, int), f"IMAGE_SIZE should be int, got {type(size)}"
        assert size > 0, f"IMAGE_SIZE should be positive, got {size}"

    def test_class_names_count(self):
        """Should have exactly 10 tomato disease classes."""
        from tomato_disease_advisor.config import ConfigurationManager

        cm = ConfigurationManager()
        class_names = list(cm.config.class_names)
        assert len(class_names) == 10, (
            f"Expected 10 classes, got {len(class_names)}: {class_names}"
        )

    def test_class_names_are_strings(self):
        """All class names should be non-empty strings."""
        from tomato_disease_advisor.config import ConfigurationManager

        cm = ConfigurationManager()
        for name in cm.config.class_names:
            assert isinstance(name, str) and len(name) > 0


# ── Severity tests ───────────────────────────────────────────


class TestSeverityEstimator:
    """Test severity estimation logic (no model needed)."""

    def _make_estimator(self):
        from tomato_disease_advisor.components.severity import SeverityEstimator
        from tomato_disease_advisor.config import ConfigurationManager
        import numpy as np
        cm = ConfigurationManager()
        config = cm.get_severity_config()
        return SeverityEstimator(config), np

    def test_healthy_class(self):
        """Healthy class should always return 'healthy' level."""
        estimator, np = self._make_estimator()
        heatmap = np.random.rand(100, 100).astype(np.float32)
        result = estimator.estimate_severity(heatmap, "Tomato_healthy")
        assert result["severity_level"] == "healthy"
        assert result["affected_area_pct"] == 0.0

    def test_mild_heatmap(self):
        """Low-activation heatmap should be 'mild'."""
        estimator, np = self._make_estimator()
        # Create heatmap with ~5% hot pixels (above 0.5 threshold)
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[:5, :10] = 0.9  # 50 of 10000 = 0.5% hot pixels
        result = estimator.estimate_severity(heatmap, "Tomato_Early_blight")
        assert result["severity_level"] == "mild", (
            f"Expected mild, got {result['severity_level']} "
            f"({result['affected_area_pct']}%)"
        )

    def test_severe_heatmap(self):
        """High-activation heatmap should be 'severe'."""
        estimator, np = self._make_estimator()
        # Create heatmap with 80% hot pixels
        heatmap = np.ones((100, 100), dtype=np.float32) * 0.9
        heatmap[:20, :] = 0.1  # only 20% is cold
        result = estimator.estimate_severity(heatmap, "Tomato_Late_blight")
        assert result["severity_level"] == "severe", (
            f"Expected severe, got {result['severity_level']} "
            f"({result['affected_area_pct']}%)"
        )


# ── Feedback tests ───────────────────────────────────────────


class TestFeedbackCollector:
    """Test feedback collection (uses temp file)."""

    def test_save_feedback_creates_file(self):
        """Saving feedback should create a JSONL file."""
        from tomato_disease_advisor.feedback.collector import FeedbackCollector

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            collector = FeedbackCollector(filepath=tmp_path)
            entry = collector.save_feedback(
                image_name="test.jpg",
                predicted_class="Early_blight",
                confidence=0.95,
                severity_level="moderate",
                is_correct=True,
                user_comment="looks right",
            )
            assert entry["predicted_class"] == "Early_blight"
            assert entry["is_correct"] is True

            # Verify file content
            with open(tmp_path, "r") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["image_name"] == "test.jpg"
        finally:
            os.unlink(tmp_path)

    def test_get_stats_empty(self):
        """Stats on non-existent file should return zeros."""
        from tomato_disease_advisor.feedback.collector import FeedbackCollector

        collector = FeedbackCollector(filepath="nonexistent_test.jsonl")
        stats = collector.get_stats()
        assert stats["total"] == 0
        assert stats["accuracy"] == 0

    def test_get_stats_accuracy(self):
        """Stats should correctly compute accuracy."""
        from tomato_disease_advisor.feedback.collector import FeedbackCollector

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            collector = FeedbackCollector(filepath=tmp_path)
            collector.save_feedback("a.jpg", "X", 0.9, "mild", True)
            collector.save_feedback("b.jpg", "Y", 0.8, "mild", False)
            collector.save_feedback("c.jpg", "Z", 0.7, "mild", True)

            stats = collector.get_stats()
            assert stats["total"] == 3
            assert stats["correct"] == 2
            assert stats["incorrect"] == 1
            assert abs(stats["accuracy"] - 2 / 3) < 1e-6
        finally:
            os.unlink(tmp_path)
