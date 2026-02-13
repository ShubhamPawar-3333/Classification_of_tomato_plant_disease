"""
Severity Estimation Component

Estimates disease severity from GradCAM++ heatmaps by measuring
the proportion of the leaf area that shows signs of disease.

Severity levels (configurable via params.yaml):
  - healthy: predicted class is "Tomato_healthy"
  - mild:     < 10% affected area
  - moderate: 10-30% affected area
  - severe:   > 30% affected area
"""
import numpy as np
from tomato_disease_advisor.entity import SeverityConfig


# Human-readable descriptions for each severity level
SEVERITY_DESCRIPTIONS = {
    "healthy": "No disease detected. The plant appears healthy.",
    "mild": "Early-stage infection detected. Minor spots or discoloration "
            "visible on a small portion of the leaf.",
    "moderate": "Moderate disease spread detected. Significant portions of "
                "the leaf show infection symptoms. Treatment recommended.",
    "severe": "Severe infection detected. Large areas of the leaf are "
              "affected. Immediate treatment is critical to prevent spread.",
}


class SeverityEstimator:
    """
    Estimates disease severity from GradCAM++ heatmaps.

    Uses the proportion of 'hot' pixels (activation > threshold) in the
    heatmap as a proxy for the percentage of affected leaf area.
    """

    def __init__(self, config: SeverityConfig):
        """
        Initialize SeverityEstimator.

        Args:
            config: SeverityConfig with mild_threshold and moderate_threshold
        """
        self.config = config

    def estimate_severity(
        self,
        heatmap: np.ndarray,
        predicted_class: str,
        activation_threshold: float = 0.5,
    ) -> dict:
        """
        Estimate disease severity from a GradCAM++ heatmap.

        Args:
            heatmap: GradCAM++ heatmap, shape (H, W), values in [0, 1]
            predicted_class: Predicted class name (e.g., "Tomato_healthy")
            activation_threshold: Minimum heatmap value to consider as
                                  disease-affected (default: 0.5)

        Returns:
            dict with:
                - affected_area_pct (float): Percentage of affected area
                - severity_level (str): "healthy", "mild", "moderate", "severe"
                - description (str): Human-readable severity description
                - activation_threshold (float): Threshold used
        """
        # Healthy plants have no disease severity
        if "healthy" in predicted_class.lower():
            return {
                "affected_area_pct": 0.0,
                "severity_level": "healthy",
                "description": SEVERITY_DESCRIPTIONS["healthy"],
                "activation_threshold": activation_threshold,
            }

        # Calculate affected area percentage
        total_pixels = heatmap.size
        hot_pixels = np.sum(heatmap >= activation_threshold)
        affected_pct = (hot_pixels / total_pixels) * 100

        # Determine severity level using thresholds from config
        if affected_pct < self.config.mild_threshold:
            level = "mild"
        elif affected_pct < self.config.moderate_threshold:
            level = "moderate"
        else:
            level = "severe"

        return {
            "affected_area_pct": round(float(affected_pct), 2),
            "severity_level": level,
            "description": SEVERITY_DESCRIPTIONS[level],
            "activation_threshold": activation_threshold,
        }
