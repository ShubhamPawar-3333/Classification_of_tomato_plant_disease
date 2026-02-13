"""
Prediction Pipeline

Unified inference pipeline that combines:
  1. Image classification (EfficientNet)
  2. Confidence scoring with abstention
  3. GradCAM++ explainability
  4. Severity estimation

This is the main entry point for making predictions on new images.
Used by the Gradio app and the API.
"""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional
from PIL import Image
from tomato_disease_advisor.entity import PredictionConfig
from tomato_disease_advisor.components.explainer import GradCAMExplainer
from tomato_disease_advisor.components.severity import SeverityEstimator


class PredictionPipeline:
    """
    End-to-end prediction pipeline for tomato disease diagnosis.

    Flow: Image → Preprocess → Classify → Confidence Check →
          GradCAM++ → Severity → Result
    """

    def __init__(self, config: PredictionConfig):
        """
        Initialize PredictionPipeline.

        Args:
            config: PredictionConfig combining model, GradCAM, severity,
                    and confidence configurations
        """
        self.config = config

        # Load model
        print(f"Loading model from: {config.model_path}")
        self.model = tf.keras.models.load_model(str(config.model_path))
        print(f"Model loaded. Output shape: {self.model.output_shape}")

        # Initialize components
        self.explainer = GradCAMExplainer(config.gradcam_config)
        self.severity_estimator = SeverityEstimator(config.severity_config)

        # Class names
        self.class_names = config.class_names
        self.image_size = config.image_size

        # Confidence thresholds
        self.abstention_threshold = config.confidence_config.abstention_threshold
        self.warning_threshold = config.confidence_config.warning_threshold

    def _preprocess_image(self, image_path: str) -> tuple:
        """
        Load and preprocess an image for prediction.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (preprocessed_batch, original_array)
        """
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        # Load original image
        original_img = Image.open(image_path).convert("RGB")
        original_img = original_img.resize(
            (self.image_size, self.image_size)
        )
        original_array = np.array(original_img)

        # Preprocess for model
        input_array = preprocess_input(
            original_array.copy().astype(np.float32)
        )
        input_batch = np.expand_dims(input_array, axis=0)

        return input_batch, original_array

    def _assess_confidence(self, confidence: float) -> dict:
        """
        Assess prediction confidence and determine if we should abstain.

        Args:
            confidence: Maximum softmax probability

        Returns:
            dict with confidence_level, should_abstain, message
        """
        if confidence < self.abstention_threshold:
            return {
                "confidence_level": "abstain",
                "should_abstain": True,
                "message": (
                    f"Model confidence is very low ({confidence:.1%}). "
                    f"The image may not be a tomato leaf or the disease is "
                    f"not in the training data. Please consult an expert."
                ),
            }
        elif confidence < self.warning_threshold:
            return {
                "confidence_level": "warning",
                "should_abstain": False,
                "message": (
                    f"Model confidence is moderate ({confidence:.1%}). "
                    f"The prediction may be less reliable. Consider "
                    f"getting a second opinion."
                ),
            }
        else:
            return {
                "confidence_level": "confident",
                "should_abstain": False,
                "message": None,
            }

    def predict(
        self,
        image_path: str,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the full prediction pipeline on a single image.

        Args:
            image_path: Path to the input image
            save_dir: Directory to save GradCAM outputs. If None, uses
                      config's output_dir.

        Returns:
            dict with complete prediction results:
            {
                "class_name": str,
                "class_index": int,
                "confidence": float,
                "confidence_level": str,
                "should_abstain": bool,
                "message": str or None,
                "severity": {
                    "affected_area_pct": float,
                    "severity_level": str,
                    "description": str
                },
                "heatmap_path": str or None,
                "overlay_path": str or None,
                "all_probabilities": dict
            }
        """
        # Set save directory
        if save_dir is None:
            save_dir = str(
                self.config.report_config.output_dir
            )

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Preprocess
        input_batch, original_array = self._preprocess_image(image_path)

        # Step 2: Classify
        predictions = self.model.predict(input_batch, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        class_name = self.class_names[class_idx]

        # All class probabilities
        all_probs = {
            self.class_names[i]: round(float(predictions[0][i]), 4)
            for i in range(len(self.class_names))
        }

        print(f"\n[Prediction] Class: {class_name}")
        print(f"[Prediction] Confidence: {confidence:.4f}")

        # Step 3: Confidence check
        confidence_result = self._assess_confidence(confidence)

        if confidence_result["should_abstain"]:
            print(f"[Prediction] ABSTAINING: {confidence_result['message']}")

        # Step 4: GradCAM++ (generate even if abstaining, for diagnostics)
        heatmap = self.explainer.generate_heatmap(
            self.model, input_batch, class_idx
        )
        overlay = self.explainer.overlay_heatmap(
            original_array, heatmap
        )

        # Save GradCAM outputs
        heatmap_path = None
        overlay_path = None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as mpl_cm

        heatmap_file = save_path / "gradcam_heatmap.png"
        overlay_file = save_path / "gradcam_overlay.png"

        # Save heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(heatmap, cmap=self.config.gradcam_config.colormap)
        ax.set_title(f"GradCAM++ — {class_name}", fontsize=12)
        ax.axis("off")
        fig.savefig(str(heatmap_file), dpi=150, bbox_inches="tight")
        plt.close(fig)
        heatmap_path = str(heatmap_file)

        # Save overlay
        Image.fromarray(overlay).save(str(overlay_file))
        overlay_path = str(overlay_file)

        print(f"[GradCAM] Saved: {heatmap_path}")
        print(f"[GradCAM] Saved: {overlay_path}")

        # Step 5: Severity estimation
        severity = self.severity_estimator.estimate_severity(
            heatmap=heatmap,
            predicted_class=class_name,
        )
        print(f"[Severity] {severity['severity_level']} "
              f"({severity['affected_area_pct']}%)")

        # Combine results
        result = {
            "class_name": class_name,
            "class_index": class_idx,
            "confidence": round(confidence, 4),
            "confidence_level": confidence_result["confidence_level"],
            "should_abstain": confidence_result["should_abstain"],
            "message": confidence_result["message"],
            "severity": severity,
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "all_probabilities": all_probs,
        }

        return result
