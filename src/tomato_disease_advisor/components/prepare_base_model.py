"""
Prepare Base Model Component

Downloads the appropriate EfficientNet model (dynamically selected based on
IMAGE_SIZE) pretrained on ImageNet and prepares it for transfer learning
on the tomato disease classification task.

Backbone selection logic:
  IMAGE_SIZE == 224 → EfficientNetB0
  IMAGE_SIZE == 240 → EfficientNetB1
  IMAGE_SIZE == 260 → EfficientNetB2
  IMAGE_SIZE == 300 → EfficientNetB3
  IMAGE_SIZE >= 380 → EfficientNetB4
"""
import tensorflow as tf
from pathlib import Path
from tomato_disease_advisor.entity import PrepareBaseModelConfig


# Mapping of IMAGE_SIZE to EfficientNet variant
EFFICIENTNET_MAP = {
    224: ("EfficientNetB0", tf.keras.applications.EfficientNetB0),
    240: ("EfficientNetB1", tf.keras.applications.EfficientNetB1),
    260: ("EfficientNetB2", tf.keras.applications.EfficientNetB2),
    300: ("EfficientNetB3", tf.keras.applications.EfficientNetB3),
    380: ("EfficientNetB4", tf.keras.applications.EfficientNetB4),
}


def get_efficientnet_for_size(image_size: int):
    """
    Dynamically select the correct EfficientNet variant for the given image size.

    Args:
        image_size: Input image dimension (e.g., 224, 380)

    Returns:
        Tuple of (model_name: str, model_fn: callable)
    """
    assert image_size in EFFICIENTNET_MAP, (
        f"IMAGE_SIZE={image_size} is not supported. "
        f"Valid sizes: {list(EFFICIENTNET_MAP.keys())}"
    )
    return EFFICIENTNET_MAP[image_size]


class PrepareBaseModel:
    """
    Prepares the base model (EfficientNet) for transfer learning.

    Steps:
        1. Dynamically select EfficientNet variant based on IMAGE_SIZE
        2. Download with ImageNet weights (no top)
        3. Freeze base layers
        4. Add custom classification head
        5. Save the updated model
    """

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize PrepareBaseModel.

        Args:
            config: PrepareBaseModelConfig with model parameters
        """
        self.config = config

    def get_base_model(self) -> tf.keras.Model:
        """
        Download and save the base EfficientNet model.
        Automatically selects the correct variant based on IMAGE_SIZE.

        Returns:
            tf.keras.Model: The base model without classification head
        """
        image_size = self.config.params_image_size
        model_name, model_fn = get_efficientnet_for_size(image_size)

        print(f"[Backbone] Selected: {model_name} (IMAGE_SIZE={image_size})")
        print(f"[Backbone] Preprocessing: tf.keras.applications.efficientnet.preprocess_input")

        self.backbone_name = model_name
        self.model = model_fn(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=tuple(self.config.params_input_shape),
        )

        # Save base model
        self.model.save(self.config.base_model_path)
        print(f"Base model saved to: {self.config.base_model_path}")
        print(f"Base model parameters: {self.model.count_params():,}")

        return self.model

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool = True,
        learning_rate: float = 0.0001,
        backbone_name: str = "EfficientNet",
    ) -> tf.keras.Model:
        """
        Add custom classification head to the base model.

        Args:
            model: Base model (EfficientNet without top)
            classes: Number of output classes
            freeze_all: Whether to freeze all base model layers
            learning_rate: Learning rate for Adam optimizer
            backbone_name: Name of the backbone for the model name

        Returns:
            tf.keras.Model: Complete model with classification head
        """
        # Freeze base model layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
            print(f"Froze {len(model.layers)} base model layers")

        # Add classification head (simple: GAP → Dropout → Dense → Output)
        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_1")(x)
        predictions = tf.keras.layers.Dense(
            classes, activation="softmax", name="predictions"
        )(x)

        full_model = tf.keras.Model(
            inputs=model.input,
            outputs=predictions,
            name=f"{backbone_name}_TomatoDisease",
        )

        # Compile
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"\nModel Summary:")
        print(f"  Total params: {full_model.count_params():,}")
        trainable = sum(
            tf.keras.backend.count_params(w)
            for w in full_model.trainable_weights
        )
        non_trainable = sum(
            tf.keras.backend.count_params(w)
            for w in full_model.non_trainable_weights
        )
        print(f"  Trainable params: {trainable:,}")
        print(f"  Non-trainable params: {non_trainable:,}")

        return full_model

    def update_base_model(self) -> tf.keras.Model:
        """
        Prepare the full model with classification head.

        Returns:
            tf.keras.Model: Complete model ready for training
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            learning_rate=0.0001,
            backbone_name=getattr(self, "backbone_name", "EfficientNet"),
        )

        # Save updated model
        self.full_model.save(self.config.updated_base_model_path)
        print(f"\nUpdated model saved to: {self.config.updated_base_model_path}")

        return self.full_model

    def run(self) -> tf.keras.Model:
        """
        Execute the complete base model preparation pipeline.

        Returns:
            tf.keras.Model: The prepared model
        """
        print("=" * 50)
        print("Preparing Base Model (Dynamic EfficientNet Selection)")
        print("=" * 50)

        # Step 1: Get base model
        self.get_base_model()

        # Step 2: Add classification head
        model = self.update_base_model()

        print("=" * 50)
        print("Base Model Preparation Complete!")
        print("=" * 50)

        return model
