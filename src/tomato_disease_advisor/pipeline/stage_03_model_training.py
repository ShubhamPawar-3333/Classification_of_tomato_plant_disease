"""
Stage 03: Model Training Pipeline

Trains the EfficientNet model with MLflow experiment tracking.
Run: python src/tomato_disease_advisor/pipeline/stage_03_model_training.py
"""
from tomato_disease_advisor.config import ConfigurationManager
from tomato_disease_advisor.components.model_training import ModelTrainer
from tomato_disease_advisor.components.prepare_base_model import get_efficientnet_for_size
from tomato_disease_advisor.utils import (
    MLflowRun,
    log_params,
    log_metrics,
    log_artifact,
    MLflowCallback,
)


STAGE_NAME = "Model Training"


def main():
    """Execute the model training stage with MLflow tracking."""
    print(f"\n{'>'*20} {STAGE_NAME} Stage Started {'<'*20}\n")

    try:
        # Get configuration
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()

        # Determine backbone name dynamically
        image_size = config_manager.params.IMAGE_SIZE
        backbone_name, _ = get_efficientnet_for_size(image_size)

        # Train with MLflow tracking
        with MLflowRun(
            run_name=f"{backbone_name.lower()}_training",
            experiment_name="tomato-disease-classification",
            tags={"model": backbone_name, "stage": "training"},
        ):
            # Log training parameters
            log_params(
                {
                    "model": backbone_name,
                    "image_size": image_size,
                    "epochs": training_config.params_epochs,
                    "batch_size": training_config.params_batch_size,
                    "learning_rate": training_config.params_learning_rate,
                    "phase1_lr": 3e-4,
                    "validation_split": training_config.params_validation_split,
                    "augmentation": training_config.params_augmentation,
                    "preprocessing": "efficientnet.preprocess_input",
                }
            )

            # Run training
            trainer = ModelTrainer(config=training_config)
            history = trainer.run()

            # Log final metrics
            log_metrics(
                {
                    "final_train_accuracy": history.history["accuracy"][-1],
                    "final_val_accuracy": history.history["val_accuracy"][-1],
                    "final_train_loss": history.history["loss"][-1],
                    "final_val_loss": history.history["val_loss"][-1],
                    "best_val_accuracy": max(history.history["val_accuracy"]),
                    "total_epochs_trained": len(history.history["accuracy"]),
                }
            )

            # Log trained model artifact
            log_artifact(str(training_config.trained_model_path))

        print(f"\n{'>'*20} {STAGE_NAME} Stage Completed {'<'*20}\n")

    except Exception as e:
        print(f"Error in {STAGE_NAME}: {e}")
        raise e


if __name__ == "__main__":
    main()
