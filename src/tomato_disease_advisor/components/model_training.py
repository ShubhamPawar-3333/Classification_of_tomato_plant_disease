"""
Model Training Component

Trains the EfficientNet-B4 model on the PlantVillage tomato disease dataset
with two-phase transfer learning:
  Phase 1: Train classification head only (high LR)
  Phase 2: Fine-tune top base model layers (low LR)
"""
import os
import tensorflow as tf
from pathlib import Path
from typing import Tuple
from tomato_disease_advisor.entity import TrainingConfig


class ModelTrainer:
    """
    Trains the prepared EfficientNet-B4 model using two-phase transfer learning.
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
        
        # Check for nested directory structure
        subdirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) == 1:
            data_dir = os.path.join(data_dir, subdirs[0])
        
        target_size = (224, 224)
        
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
    
    def train(self) -> tf.keras.callbacks.History:
        """
        Two-phase transfer learning training.
        
        Phase 1: Train classification head only (5 epochs, LR=0.001)
        Phase 2: Unfreeze top 25% layers, fine-tune (remaining epochs, LR=0.0001)
        
        Returns:
            tf.keras.callbacks.History: Combined training history
        """
        # Load the prepared model
        print(f"Loading model from: {self.config.updated_base_model_path}")
        model = tf.keras.models.load_model(self.config.updated_base_model_path)
        
        # Get data generators
        train_gen, val_gen = self._get_data_generators()
        
        # ============================
        # Phase 1: Train head only
        # ============================
        phase1_epochs = min(5, self.config.params_epochs)
        phase1_lr = 0.001
        
        print(f"\n{'='*50}")
        print(f"Phase 1: Training Head Only ({phase1_epochs} epochs, LR={phase1_lr})")
        print(f"{'='*50}")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase1_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        history_phase1 = model.fit(
            train_gen,
            epochs=phase1_epochs,
            validation_data=val_gen,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3,
                    restore_best_weights=True, verbose=1
                )
            ]
        )
        
        print(f"\nPhase 1 Results:")
        print(f"  Train Accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
        print(f"  Val Accuracy:   {history_phase1.history['val_accuracy'][-1]:.4f}")
        
        # ============================
        # Phase 2: Fine-tune top layers
        # ============================
        phase2_epochs = self.config.params_epochs - phase1_epochs
        phase2_lr = self.config.params_learning_rate
        
        # Find the base model and unfreeze top 25%
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                break
        
        if base_model:
            base_model.trainable = True
            unfreeze_from = int(len(base_model.layers) * 0.75)
            for layer in base_model.layers[:unfreeze_from]:
                layer.trainable = False
            unfrozen = sum(1 for l in base_model.layers if l.trainable)
            print(f"\nUnfroze {unfrozen} / {len(base_model.layers)} base layers")
        
        print(f"\n{'='*50}")
        print(f"Phase 2: Fine-Tuning ({phase2_epochs} epochs, LR={phase2_lr})")
        print(f"{'='*50}")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase2_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        history_phase2 = model.fit(
            train_gen,
            epochs=phase2_epochs,
            validation_data=val_gen,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5,
                    restore_best_weights=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=3, min_lr=1e-7, verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.config.trained_model_path),
                    monitor="val_accuracy",
                    save_best_only=True, verbose=1
                )
            ]
        )
        
        # Save final model
        model.save(self.config.trained_model_path)
        print(f"\nModel saved to: {self.config.trained_model_path}")
        
        return history_phase2
    
    def run(self) -> tf.keras.callbacks.History:
        """
        Execute the complete model training pipeline.
        
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print("=" * 50)
        print("Starting Model Training (Two-Phase Transfer Learning)")
        print("=" * 50)
        
        history = self.train()
        
        # Print final metrics
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        best_val_acc = max(history.history["val_accuracy"])
        
        print(f"\nFinal Training Results:")
        print(f"  Train Accuracy: {final_train_acc:.4f}")
        print(f"  Val Accuracy:   {final_val_acc:.4f}")
        print(f"  Best Val Acc:   {best_val_acc:.4f}")
        
        print("=" * 50)
        print("Model Training Complete!")
        print("=" * 50)
        
        return history
