"""
Prepare Base Model Component

Downloads EfficientNet-B4 pretrained on ImageNet and prepares it
for transfer learning on the tomato disease classification task.
"""
import tensorflow as tf
from pathlib import Path
from tomato_disease_advisor.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Prepares the base model (EfficientNet-B4) for transfer learning.
    
    Steps:
        1. Download EfficientNetB4 with ImageNet weights (no top)
        2. Freeze base layers
        3. Add custom classification head
        4. Save the updated model
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
        Download and save the base EfficientNetB4 model.
        
        Returns:
            tf.keras.Model: The base model without classification head
        """
        self.model = tf.keras.applications.EfficientNetB4(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=tuple(self.config.params_input_shape)
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
        learning_rate: float = 0.0001
    ) -> tf.keras.Model:
        """
        Add custom classification head to the base model.
        
        Args:
            model: Base model (EfficientNetB4 without top)
            classes: Number of output classes
            freeze_all: Whether to freeze all base model layers
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            tf.keras.Model: Complete model with classification head
        """
        # Freeze base model layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
            print(f"Froze {len(model.layers)} base model layers")
        
        # Add classification head
        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_dense")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
        predictions = tf.keras.layers.Dense(
            classes, activation="softmax", name="predictions"
        )(x)
        
        full_model = tf.keras.Model(
            inputs=model.input,
            outputs=predictions,
            name="EfficientNetB4_TomatoDisease"
        )
        
        # Compile
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
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
            learning_rate=0.0001
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
        print("Preparing Base Model (EfficientNet-B4)")
        print("=" * 50)
        
        # Step 1: Get base model
        self.get_base_model()
        
        # Step 2: Add classification head
        model = self.update_base_model()
        
        print("=" * 50)
        print("Base Model Preparation Complete!")
        print("=" * 50)
        
        return model
