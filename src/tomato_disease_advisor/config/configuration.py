"""
Configuration manager - Creates entity objects from config files.
"""
from pathlib import Path
from tomato_disease_advisor.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from tomato_disease_advisor.utils import read_yaml, create_directories
from tomato_disease_advisor.entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
    VectorStoreConfig,
    GradCAMConfig,
    SeverityConfig,
    ConfidenceConfig,
    RAGConfig,
    ReportConfig,
    PredictionConfig
)


class ConfigurationManager:
    """
    Manages configuration for all pipeline stages.
    
    Reads config.yaml and params.yaml, then creates typed configuration
    objects for each pipeline component.
    """
    
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ):
        """
        Initialize ConfigurationManager.
        
        Args:
            config_filepath: Path to config.yaml
            params_filepath: Path to params.yaml
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Create artifacts root directory
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get configuration for data ingestion stage."""
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Get configuration for base model preparation stage."""
        config = self.config.prepare_base_model
        model_config = self.config.model
        
        create_directories([config.root_dir])
        
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_weights=model_config.weights,
            params_include_top=model_config.include_top,
            params_classes=model_config.classes,
            params_input_shape=list(model_config.input_shape)
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Get configuration for model training stage."""
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        data_ingestion = self.config.data_ingestion
        
        create_directories([training.root_dir])
        
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(data_ingestion.unzip_dir),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_validation_split=self.params.VALIDATION_SPLIT,
            params_augmentation=dict(self.params.AUGMENTATION)
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get configuration for model evaluation stage."""
        evaluation = self.config.evaluation
        training = self.config.training
        data_ingestion = self.config.data_ingestion
        
        create_directories([evaluation.root_dir])
        
        return EvaluationConfig(
            root_dir=Path(evaluation.root_dir),
            model_path=Path(training.trained_model_path),
            test_data=Path(data_ingestion.unzip_dir),
            scores_path=Path(evaluation.scores_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            class_names=list(self.config.class_names)
        )
    
    def get_vectorstore_config(self) -> VectorStoreConfig:
        """Get configuration for vector store building stage."""
        vectorstore = self.config.vectorstore
        rag = self.config.rag
        
        create_directories([vectorstore.root_dir])
        
        return VectorStoreConfig(
            root_dir=Path(vectorstore.root_dir),
            index_path=Path(vectorstore.index_path),
            knowledge_dir=Path(vectorstore.knowledge_dir),
            embedding_model=rag.embedding_model,
            chunk_size=rag.chunk_size,
            chunk_overlap=rag.chunk_overlap
        )
    
    def get_gradcam_config(self) -> GradCAMConfig:
        """Get configuration for GradCAM++ explainability."""
        gradcam = self.params.GRADCAM
        
        return GradCAMConfig(
            layer_name=gradcam.layer_name,
            colormap=gradcam.colormap
        )
    
    def get_severity_config(self) -> SeverityConfig:
        """Get configuration for severity estimation."""
        severity = self.params.SEVERITY
        
        return SeverityConfig(
            mild_threshold=severity.mild_threshold,
            moderate_threshold=severity.moderate_threshold
        )
    
    def get_confidence_config(self) -> ConfidenceConfig:
        """Get configuration for confidence thresholds."""
        confidence = self.params.CONFIDENCE
        
        return ConfidenceConfig(
            abstention_threshold=confidence.abstention_threshold,
            warning_threshold=confidence.warning_threshold
        )
    
    def get_rag_config(self) -> RAGConfig:
        """Get configuration for RAG pipeline."""
        rag_config = self.config.rag
        rag_params = self.params.RAG
        
        return RAGConfig(
            embedding_model=rag_config.embedding_model,
            llm_model=rag_config.llm_model,
            llm_provider=rag_config.llm_provider,
            top_k=rag_config.top_k,
            temperature=rag_params.temperature,
            max_tokens=rag_params.max_tokens
        )
    
    def get_report_config(self) -> ReportConfig:
        """Get configuration for report generation."""
        reports = self.config.reports
        
        create_directories([reports.output_dir])
        
        return ReportConfig(
            output_dir=Path(reports.output_dir),
            default_language=reports.default_language,
            supported_languages=list(reports.supported_languages)
        )
    
    def get_prediction_config(self) -> PredictionConfig:
        """Get combined configuration for prediction pipeline."""
        training = self.config.training
        vectorstore = self.config.vectorstore
        
        return PredictionConfig(
            model_path=Path(training.trained_model_path),
            vectorstore_path=Path(vectorstore.index_path),
            class_names=list(self.config.class_names),
            image_size=self.params.IMAGE_SIZE,
            gradcam_config=self.get_gradcam_config(),
            severity_config=self.get_severity_config(),
            confidence_config=self.get_confidence_config(),
            rag_config=self.get_rag_config(),
            report_config=self.get_report_config()
        )
