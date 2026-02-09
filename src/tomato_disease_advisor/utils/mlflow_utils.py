"""
MLflow tracking utilities for the Tomato Disease Advisory System.

Provides functions for experiment tracking, model logging, and metric recording.
"""
import os
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_mlflow(
    experiment_name: str = "tomato-disease-classification",
    tracking_uri: str = "mlruns"
) -> str:
    """
    Set up MLflow tracking.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: URI for MLflow tracking (local directory or server URL)
        
    Returns:
        str: Experiment ID
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> mlflow.ActiveRun:
    """
    Start an MLflow run.
    
    Args:
        run_name: Name for the run (auto-generated if None)
        tags: Optional tags to add to the run
        
    Returns:
        mlflow.ActiveRun: The active run context
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to the current MLflow run.
    
    Args:
        params: Dictionary of parameter names and values
    """
    for key, value in params.items():
        # Handle nested dictionaries
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mlflow.log_param(f"{key}.{sub_key}", sub_value)
        else:
            mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to the current MLflow run.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number for the metrics
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def log_model(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None
) -> None:
    """
    Log a Keras model to the current MLflow run.
    
    Args:
        model: The Keras model to log
        artifact_path: Path within the run's artifact directory
        registered_model_name: Optional name to register the model
    """
    mlflow.keras.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file or directory as an artifact.
    
    Args:
        local_path: Path to the local file or directory
        artifact_path: Optional destination path within artifact directory
    """
    mlflow.log_artifact(local_path, artifact_path=artifact_path)


def log_figure(figure: Any, artifact_file: str) -> None:
    """
    Log a matplotlib figure as an artifact.
    
    Args:
        figure: Matplotlib figure object
        artifact_file: Name of the artifact file (e.g., "confusion_matrix.png")
    """
    mlflow.log_figure(figure, artifact_file)


def log_dict(dictionary: Dict, artifact_file: str) -> None:
    """
    Log a dictionary as a JSON artifact.
    
    Args:
        dictionary: Dictionary to log
        artifact_file: Name of the artifact file (e.g., "scores.json")
    """
    mlflow.log_dict(dictionary, artifact_file)


def end_run(status: str = "FINISHED") -> None:
    """
    End the current MLflow run.
    
    Args:
        status: Status of the run ("FINISHED", "FAILED", "KILLED")
    """
    mlflow.end_run(status=status)


class MLflowCallback:
    """
    Keras callback for logging training metrics to MLflow.
    
    Usage:
        callback = MLflowCallback()
        model.fit(x, y, callbacks=[callback.get_keras_callback()])
    """
    
    def __init__(self):
        """Initialize the MLflow callback."""
        self._keras_callback = None
    
    def get_keras_callback(self):
        """
        Get a Keras callback that logs metrics to MLflow.
        
        Returns:
            tf.keras.callbacks.LambdaCallback: Callback for Keras training
        """
        import tensorflow as tf
        
        def on_epoch_end(epoch, logs):
            if logs:
                log_metrics(logs, step=epoch)
        
        self._keras_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=on_epoch_end
        )
        return self._keras_callback


# Context manager for MLflow runs
class MLflowRun:
    """
    Context manager for MLflow runs.
    
    Usage:
        with MLflowRun("training_run") as run:
            # Training code
            log_params({"epochs": 20})
            log_metrics({"accuracy": 0.95})
    """
    
    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: str = "tomato-disease-classification",
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize MLflowRun context manager.
        
        Args:
            run_name: Name for the run
            experiment_name: Name of the experiment
            tags: Optional tags for the run
        """
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tags = tags
        self.run = None
    
    def __enter__(self):
        """Start the MLflow run."""
        setup_mlflow(self.experiment_name)
        self.run = start_run(self.run_name, self.tags)
        return self.run
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the MLflow run."""
        if exc_type is not None:
            end_run(status="FAILED")
        else:
            end_run(status="FINISHED")
        return False
