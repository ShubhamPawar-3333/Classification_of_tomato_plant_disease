"""
Entity module - Dataclasses for pipeline configuration.

These dataclasses define the structure of configuration objects
used by each stage of the DVC pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage."""
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for base model preparation stage."""
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: int
    params_weights: str
    params_include_top: bool
    params_classes: int
    params_input_shape: List[int]


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training stage."""
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_learning_rate: float
    params_validation_split: float
    params_augmentation: dict


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for model evaluation stage."""
    root_dir: Path
    model_path: Path
    test_data: Path
    scores_path: Path
    params_image_size: int
    params_batch_size: int
    class_names: List[str]


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for vector store building stage."""
    root_dir: Path
    index_path: Path
    knowledge_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class GradCAMConfig:
    """Configuration for GradCAM++ explainability."""
    layer_name: str
    colormap: str


@dataclass(frozen=True)
class SeverityConfig:
    """Configuration for severity estimation."""
    mild_threshold: float
    moderate_threshold: float


@dataclass(frozen=True)
class ConfidenceConfig:
    """Configuration for confidence thresholds."""
    abstention_threshold: float
    warning_threshold: float


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAG pipeline."""
    embedding_model: str
    llm_model: str
    llm_provider: str
    top_k: int
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for report generation."""
    output_dir: Path
    default_language: str
    supported_languages: List[str]


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for prediction pipeline (combines all inference configs)."""
    model_path: Path
    vectorstore_path: Path
    class_names: List[str]
    image_size: int
    gradcam_config: GradCAMConfig
    severity_config: SeverityConfig
    confidence_config: ConfidenceConfig
    rag_config: RAGConfig
    report_config: ReportConfig
