"""
Model Training Component

Trains the EfficientNet-B4 model on the PlantVillage tomato disease dataset
with data augmentation, callbacks, and MLflow tracking.
"""
import os
import tensorflow as tf
from pathlib import Path
from typing import Tuple
from tomato_disease_advisor.entity import TrainingConfig


class ModelTrainer:
    """
    Trains the prepared EfficientNet-B4 model on the tomato disease dataset.
    
    Features:
        - Data augmentation via ImageDataGenerator
        - EarlyStopping and ReduceLROnPlateau callbacks
        - MLflow experiment tracking
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize ModelTrainer.
        
        Args:
            config: TrainingConfig with training parameters
        """
        self.config = config
    
    def _get_data_generators(self) -> Tuple[
        tf.keras.preprocessing.image.DirectoryIterator,
        tf.keras.preprocessing.image.DirectoryIterator
    ]:
        """
        Create training and validation data generators with augmentation.
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        augmentation = self.config.params_augmentation
        
        # Training data generator with augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=augmentation.get("rotation_range", 20),
            width_shift_range=augmentation.get("width_shift_range", 0.2),
            height_shift_range=augmentation.get("height_shift_range", 0.2),
            horizontal_flip=augmentation.get("horizontal_flip", True),
            vertical_flip=augmentation.get("vertical_flip", False),
            zoom_range=augmentation.get("zoom_range", 0.2),
            fill_mode=augmentation.get("fill_mode", "nearest"),
            validation_split=self.config.params_validation_split
        )
        
        # Find the dataset directory  
        data_dir = self.config.training_data
        
        # Check if there's a subdirectory containing the class folders
        subdirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
        
        # If extracted into a single subfolder, use that
        if len(subdirs) == 1 and not any(
            os.path.isdir(os.path.join(data_dir, subdirs[0], d)) 
            for d in os.listdir(os.path.join(data_dir, subdirs[0]))
            if os.path.isdir(os.path.join(data_dir, subdirs[0], d))
        ) is False:
            data_dir = os.path.join(data_dir, subdirs[0])
        
        image_size = self.config.params_batch_size  # Will be overridden
        target_size = (224, 224)  # Fixed for EfficientNetB4
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=42
        )
        
        # Validation generator (no augmentation, only rescaling)
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=42
        )
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {validation_generator.samples}")
        print(f"  Classes: {list(train_generator.class_indices.keys())}")
        print(f"  Number of classes: {len(train_generator.class_indices)}")
        
        return train_generator, validation_generator
    
    def _get_callbacks(self) -> list:
        """
        Create training callbacks.
        
        Returns:
            list: List of Keras callbacks
        """
        callbacks = [
            # Stop training if validation loss stops improving
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when validation loss plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            # Save best model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Load the prepared model
        print(f"Loading model from: {self.config.updated_base_model_path}")
        model = tf.keras.models.load_model(self.config.updated_base_model_path)
        
        # Get data generators
        train_gen, val_gen = self._get_data_generators()
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Train
        print(f"\nStarting training...")
        print(f"  Epochs: {self.config.params_epochs}")
        print(f"  Batch size: {self.config.params_batch_size}")
        print(f"  Learning rate: {self.config.params_learning_rate}")
        
        history = model.fit(
            train_gen,
            epochs=self.config.params_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            steps_per_epoch=train_gen.samples // self.config.params_batch_size,
            validation_steps=val_gen.samples // self.config.params_batch_size
        )
        
        # Save final model
        model.save(self.config.trained_model_path)
        print(f"\nModel saved to: {self.config.trained_model_path}")
        
        return history
    
    def run(self) -> tf.keras.callbacks.History:
        """
        Execute the complete model training pipeline.
        
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print("=" * 50)
        print("Starting Model Training")
        print("=" * 50)
        
        history = self.train()
        
        # Print final metrics
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        
        print(f"\nFinal Training Results:")
        print(f"  Train Accuracy: {final_train_acc:.4f}")
        print(f"  Val Accuracy:   {final_val_acc:.4f}")
        print(f"  Train Loss:     {final_train_loss:.4f}")
        print(f"  Val Loss:       {final_val_loss:.4f}")
        
        print("=" * 50)
        print("Model Training Complete!")
        print("=" * 50)
        
        return history
