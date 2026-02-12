"""
Stage 04: Model Evaluation Pipeline

Evaluates the trained model and logs metrics to MLflow.
Run: python src/tomato_disease_advisor/pipeline/stage_04_model_evaluation.py
"""
from tomato_disease_advisor.config import ConfigurationManager
from tomato_disease_advisor.components.model_evaluation import ModelEvaluation
from tomato_disease_advisor.utils import (
    MLflowRun,
    log_params,
    log_metrics,
    log_artifact
)


STAGE_NAME = "Model Evaluation"


def main():
    """Execute the model evaluation stage with MLflow tracking."""
    print(f"\n{'>'*20} {STAGE_NAME} Stage Started {'<'*20}\n")
    
    try:
        # Get configuration
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_evaluation_config()
        
        # Evaluate with MLflow tracking
        with MLflowRun(
            run_name="efficientnetb4_evaluation",
            experiment_name="tomato-disease-classification",
            tags={"model": "EfficientNetB4", "stage": "evaluation"}
        ):
            # Log evaluation parameters
            log_params({
                "model_path": str(eval_config.model_path),
                "image_size": eval_config.params_image_size,
                "batch_size": eval_config.params_batch_size,
                "num_classes": len(eval_config.class_names)
            })
            
            # Run evaluation
            evaluator = ModelEvaluation(config=eval_config)
            metrics = evaluator.run()
            
            # Log metrics to MLflow
            mlflow_metrics = {
                k: v for k, v in metrics.items() 
                if isinstance(v, (int, float))
            }
            log_metrics(mlflow_metrics)
            
            # Log confusion matrix image
            if "confusion_matrix_path" in metrics:
                log_artifact(metrics["confusion_matrix_path"])
            
            # Log scores.json
            log_artifact(str(eval_config.scores_path))
        
        print(f"\n{'>'*20} {STAGE_NAME} Stage Completed {'<'*20}\n")
        
    except Exception as e:
        print(f"Error in {STAGE_NAME}: {e}")
        raise e


if __name__ == "__main__":
    main()
