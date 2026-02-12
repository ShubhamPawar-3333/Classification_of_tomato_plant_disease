"""
Model Evaluation Component

Evaluates the trained model on the test set with:
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Per-class metrics
- Saves scores to scores.json for DVC metrics tracking
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tomato_disease_advisor.entity import EvaluationConfig


class ModelEvaluation:
    """
    Evaluates the trained model and generates comprehensive metrics.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize ModelEvaluation.
        
        Args:
            config: EvaluationConfig with model path, data path, and class names
        """
        self.config = config
    
    def _load_model(self) -> tf.keras.Model:
        """Load the trained model."""
        print(f"Loading model from: {self.config.model_path}")
        model = tf.keras.models.load_model(self.config.model_path)
        return model
    
    def _get_test_generator(self) -> tf.keras.preprocessing.image.DirectoryIterator:
        """Create test data generator (no augmentation)."""
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )
        
        # Find dataset directory
        data_dir = self.config.test_data
        subdirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) == 1:
            data_dir = os.path.join(data_dir, subdirs[0])
        
        test_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=(self.config.params_image_size, self.config.params_image_size),
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=42
        )
        
        return test_generator
    
    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict with accuracy, precision, recall, f1
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }
        
        return metrics
    
    def _plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str
    ) -> plt.Figure:
        """
        Generate and save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the figure
            
        Returns:
            plt.Figure: The confusion matrix figure
        """
        # Shorten class names for display
        short_names = [name.replace("Tomato_", "").replace("Tomato__", "")[:20] 
                       for name in class_names]
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_names,
            yticklabels=short_names,
            ax=ax
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix - Tomato Disease Classification", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
        
        return fig
    
    def _generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
    ) -> str:
        """Generate and print sklearn classification report."""
        short_names = [name.replace("Tomato_", "").replace("Tomato__", "")
                       for name in class_names]
        
        report = classification_report(
            y_true, y_pred, target_names=short_names
        )
        
        return report
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation pipeline.
        
        Returns:
            Dict with all evaluation metrics
        """
        # Load model
        model = self._load_model()
        
        # Get test data
        test_gen = self._get_test_generator()
        
        # Get predictions
        print("Generating predictions...")
        predictions = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred)
        
        # Print metrics
        print(f"\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Classification report
        class_names = self.config.class_names
        report = self._generate_classification_report(y_true, y_pred, class_names)
        print(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm_path = os.path.join(self.config.root_dir, "confusion_matrix.png")
        self._plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        
        # Save scores to JSON for DVC
        scores = {
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
        }
        
        with open(self.config.scores_path, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"\nScores saved to: {self.config.scores_path}")
        
        # Store full metrics
        metrics["confusion_matrix_path"] = cm_path
        
        return metrics
    
    def run(self) -> Dict[str, float]:
        """
        Execute the complete evaluation pipeline.
        
        Returns:
            Dict with all evaluation metrics
        """
        print("=" * 50)
        print("Starting Model Evaluation")
        print("=" * 50)
        
        metrics = self.evaluate()
        
        print("=" * 50)
        print("Model Evaluation Complete!")
        print("=" * 50)
        
        return metrics
