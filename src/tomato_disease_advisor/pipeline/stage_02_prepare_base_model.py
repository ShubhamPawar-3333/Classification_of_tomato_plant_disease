"""
Stage 02: Prepare Base Model Pipeline

Downloads EfficientNet-B4 and prepares it for transfer learning.
Run: python src/tomato_disease_advisor/pipeline/stage_02_prepare_base_model.py
"""
from tomato_disease_advisor.config import ConfigurationManager
from tomato_disease_advisor.components.prepare_base_model import PrepareBaseModel


STAGE_NAME = "Prepare Base Model"


def main():
    """Execute the prepare base model stage."""
    print(f"\n{'>'*20} {STAGE_NAME} Stage Started {'<'*20}\n")
    
    try:
        # Get configuration
        config_manager = ConfigurationManager()
        prepare_base_model_config = config_manager.get_prepare_base_model_config()
        
        # Run base model preparation
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.run()
        
        print(f"\n{'>'*20} {STAGE_NAME} Stage Completed {'<'*20}\n")
        
    except Exception as e:
        print(f"Error in {STAGE_NAME}: {e}")
        raise e


if __name__ == "__main__":
    main()
