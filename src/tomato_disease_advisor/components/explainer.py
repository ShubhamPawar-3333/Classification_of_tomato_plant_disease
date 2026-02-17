"""
GradCAM++ Explainability Component

Generates visual explanations for model predictions using GradCAM++
(Gradient-weighted Class Activation Mapping++).

GradCAM++ improves on GradCAM by using second-order gradients (α weights)
for better localization of multiple disease spots on a leaf.

Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based
Visual Explanations for Deep Convolutional Networks", WACV 2018.
"""
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from tomato_disease_advisor.entity import GradCAMConfig


class GradCAMExplainer:
    """
    Generates GradCAM++ heatmaps to explain model predictions.

    Highlights the regions of a leaf image that contributed most to the
    model's disease classification decision.
    """

    def __init__(self, config: GradCAMConfig):
        """
        Initialize GradCAMExplainer.

        Args:
            config: GradCAMConfig with layer_name and colormap
        """
        self.config = config

    def _find_target_layer(self, model: tf.keras.Model) -> str:
        """
        Find the target convolutional layer for GradCAM.

        Tries the configured layer_name first. If not found, falls back
        to the last Conv2D layer in the model (works for any EfficientNet).

        Args:
            model: The trained Keras model

        Returns:
            str: Name of the target layer
        """
        # First, try the configured layer name
        layer_names = [layer.name for layer in model.layers]
        if self.config.layer_name in layer_names:
            return self.config.layer_name

        # Check if it's inside a nested base model
        for layer in model.layers:
            if hasattr(layer, "layers"):
                nested_names = [l.name for l in layer.layers]
                if self.config.layer_name in nested_names:
                    return self.config.layer_name

        # Fallback: find the last Conv2D layer
        last_conv = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
            # Check nested model layers
            if hasattr(layer, "layers"):
                for nested_layer in layer.layers:
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        last_conv = nested_layer.name

        if last_conv:
            print(f"[GradCAM] Using fallback conv layer: {last_conv}")
            return last_conv

        raise ValueError(
            f"Could not find target layer '{self.config.layer_name}' "
            f"or any Conv2D layer in the model."
        )

    def _get_gradcam_model(
        self, model: tf.keras.Model, layer_name: str
    ) -> tf.keras.Model:
        """
        Create a model that outputs both the conv layer activations
        and the final predictions.

        Args:
            model: The trained model
            layer_name: Name of the target conv layer

        Returns:
            tf.keras.Model: Model with two outputs [conv_output, predictions]
        """
        # Try to find the layer in the top-level model first
        try:
            target_layer = model.get_layer(layer_name)
            grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[target_layer.output, model.output],
            )
            return grad_model
        except ValueError:
            pass

        # Search in nested models (e.g., EfficientNet base)
        for layer in model.layers:
            if hasattr(layer, "layers"):
                try:
                    target_layer = layer.get_layer(layer_name)
                    grad_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=[target_layer.output, model.output],
                    )
                    return grad_model
                except ValueError:
                    continue

        raise ValueError(f"Layer '{layer_name}' not found in model.")

    def generate_heatmap(
        self,
        model: tf.keras.Model,
        image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap for the given image.

        Uses second-order gradients for better multi-region localization.

        Args:
            model: Trained Keras model
            image: Preprocessed image array, shape (1, H, W, 3)
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            np.ndarray: Heatmap array of shape (H, W), values in [0, 1]
        """
        # Find the target layer
        layer_name = self._find_target_layer(model)
        grad_model = self._get_gradcam_model(model, layer_name)

        # Forward pass with gradient tape
        with tf.GradientTape(persistent=True) as tape:
            conv_output, predictions = grad_model(image, training=False)
            # Normalize predictions output (fix for list outputs)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            if class_idx is None:
                class_idx = tf.argmax(predictions[0]).numpy()

            print("TYPE:", type(predictions))

            class_score = tf.gather(predictions, class_idx, axis=1)

        # First-order gradients
        grads = tape.gradient(class_score, conv_output)

        # Second-order gradients for GradCAM++
        grads_2 = tape.gradient(grads, conv_output)

        del tape  # Free persistent tape

        # GradCAM++ alpha weights
        conv_output_val = conv_output[0]
        grads_val = grads[0]

        if grads_2 is not None:
            grads_2_val = grads_2[0]
            # Alpha = grad^2 / (2 * grad^2 + sum(A * grad^3) + 1e-10)
            numerator = grads_val ** 2
            denominator = (
                2.0 * grads_val ** 2
                + tf.reduce_sum(conv_output_val * grads_2_val, axis=(0, 1))
                + 1e-10
            )
            alpha = numerator / denominator

            # Weighted combination using alpha
            weights = tf.reduce_sum(
                alpha * tf.nn.relu(grads_val), axis=(0, 1)
            )
        else:
            # Fallback to standard GradCAM if second-order grads unavailable
            weights = tf.reduce_mean(grads_val, axis=(0, 1))

        # Weighted sum of feature maps
        heatmap = tf.reduce_sum(conv_output_val * weights, axis=-1)

        # Apply ReLU
        heatmap = tf.nn.relu(heatmap)

        # Normalize to [0, 1]
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay heatmap onto the original image.

        Args:
            original_image: Original image array (H, W, 3), uint8 [0, 255]
            heatmap: Heatmap array (H_map, W_map), float [0, 1]
            alpha: Blend factor (0 = only image, 1 = only heatmap)

        Returns:
            np.ndarray: Blended image (H, W, 3), uint8 [0, 255]
        """
        # Resize heatmap to match original image
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            (original_image.shape[0], original_image.shape[1]),
        ).numpy()[:, :, 0]

        # Apply colormap
        colormap_name = getattr(self.config, "colormap", "jet")
        colormap = cm.get_cmap(colormap_name)
        heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Drop alpha
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)

        # Blend
        overlay = (
            (1 - alpha) * original_image.astype(np.float32)
            + alpha * heatmap_colored.astype(np.float32)
        ).astype(np.uint8)

        return overlay

    def explain(
        self,
        model: tf.keras.Model,
        image_path: str,
        image_size: int = 224,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        High-level API: generate explanation for an image.

        Args:
            model: Trained Keras model
            image_path: Path to the input image
            image_size: Target image size
            save_dir: Directory to save heatmap and overlay images

        Returns:
            dict with keys: heatmap, overlay, predicted_class_idx, heatmap_path, overlay_path
        """
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        # Load and preprocess image
        original_img = Image.open(image_path).convert("RGB")
        original_img = original_img.resize((image_size, image_size))
        original_array = np.array(original_img)

        # Preprocess for model
        input_array = preprocess_input(
            original_array.copy().astype(np.float32)
        )
        input_batch = np.expand_dims(input_array, axis=0)

        # Get prediction
        predictions = model.predict(input_batch, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])

        # Generate heatmap
        heatmap = self.generate_heatmap(model, input_batch, class_idx)

        # Create overlay
        overlay = self.overlay_heatmap(original_array, heatmap)

        result = {
            "heatmap": heatmap,
            "overlay": overlay,
            "original_image": original_array,
            "predicted_class_idx": class_idx,
            "confidence": confidence,
            "heatmap_path": None,
            "overlay_path": None,
        }

        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            heatmap_path = save_dir / "gradcam_heatmap.png"
            overlay_path = save_dir / "gradcam_overlay.png"

            # Save heatmap
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(heatmap, cmap=self.config.colormap)
            ax.set_title("GradCAM++ Heatmap", fontsize=14)
            ax.axis("off")
            fig.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Save overlay
            Image.fromarray(overlay).save(str(overlay_path))

            result["heatmap_path"] = str(heatmap_path)
            result["overlay_path"] = str(overlay_path)

            print(f"[GradCAM] Heatmap saved: {heatmap_path}")
            print(f"[GradCAM] Overlay saved: {overlay_path}")

        return result
