"""
Tomato Disease Advisory System - Main Pipeline Runner

Runs all DVC pipeline stages sequentially.
Usage: python main.py
"""
from tomato_disease_advisor.pipeline.stage_01_data_ingestion import (
    main as run_data_ingestion,
    STAGE_NAME as STAGE_1_NAME
)
from tomato_disease_advisor.pipeline.stage_02_prepare_base_model import (
    main as run_prepare_base_model,
    STAGE_NAME as STAGE_2_NAME
)
from tomato_disease_advisor.pipeline.stage_03_model_training import (
    main as run_model_training,
    STAGE_NAME as STAGE_3_NAME
)
from tomato_disease_advisor.pipeline.stage_04_model_evaluation import (
    main as run_model_evaluation,
    STAGE_NAME as STAGE_4_NAME
)


STAGES = [
    (STAGE_1_NAME, run_data_ingestion),
    (STAGE_2_NAME, run_prepare_base_model),
    (STAGE_3_NAME, run_model_training),
    (STAGE_4_NAME, run_model_evaluation),
]


def main():
    """Run the complete ML pipeline."""
    print("=" * 60)
    print("Tomato Disease Advisory System - Full Pipeline")
    print("=" * 60)
    
    for stage_name, stage_fn in STAGES:
        try:
            stage_fn()
        except Exception as e:
            print(f"\nFAILED at stage: {stage_name}")
            print(f"Error: {e}")
            raise e
    
    print("\n" + "=" * 60)
    print("All pipeline stages completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
