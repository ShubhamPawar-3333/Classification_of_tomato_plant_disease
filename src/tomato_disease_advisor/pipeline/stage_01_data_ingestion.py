"""
Stage 01: Data Ingestion Pipeline

Downloads and extracts the PlantVillage tomato disease dataset.
Run: python src/tomato_disease_advisor/pipeline/stage_01_data_ingestion.py
"""
from tomato_disease_advisor.config import ConfigurationManager
from tomato_disease_advisor.components.data_ingestion import DataIngestion


STAGE_NAME = "Data Ingestion"


def main():
    """Execute the data ingestion stage."""
    print(f"\n{'>'*20} {STAGE_NAME} Stage Started {'<'*20}\n")
    
    try:
        # Get configuration
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        
        # Run data ingestion
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()
        
        print(f"\n{'>'*20} {STAGE_NAME} Stage Completed {'<'*20}\n")
        
    except Exception as e:
        print(f"Error in {STAGE_NAME}: {e}")
        raise e


if __name__ == "__main__":
    main()
