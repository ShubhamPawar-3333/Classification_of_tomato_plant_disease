"""
Tomato Disease Advisory System - Project Template Generator

Run this script to create the entire project structure.
Usage: python template.py
"""

import os
from pathlib import Path

# Project name
PROJECT_NAME = "tomato-disease-advisor"

# Files and folders to create
list_of_files = [
    # DVC
    ".dvc/.gitignore",
    
    # GitHub Actions
    ".github/workflows/sync-to-hf.yml",
    ".github/workflows/upload-model-to-hf.yml",
    
    # Artifacts (directories only, files created by pipeline)
    "artifacts/.gitkeep",
    
    # Config
    "config/config.yaml",
    
    # Documentation
    "docs/PIPELINE_FLOW.md",
    "docs/PROJECT_FLOW.md",
    
    # Knowledge Base (RAG)
    "knowledge/diseases/bacterial_spot.md",
    "knowledge/diseases/early_blight.md",
    "knowledge/diseases/late_blight.md",
    "knowledge/diseases/leaf_mold.md",
    "knowledge/diseases/septoria_leaf_spot.md",
    "knowledge/diseases/spider_mites.md",
    "knowledge/diseases/target_spot.md",
    "knowledge/diseases/yellow_leaf_curl_virus.md",
    "knowledge/diseases/mosaic_virus.md",
    "knowledge/diseases/healthy.md",
    "knowledge/treatments/chemical.md",
    "knowledge/treatments/organic.md",
    "knowledge/treatments/prevention.md",
    "knowledge/costs.json",
    "knowledge/regions.json",
    
    # Logs
    "logs/.gitkeep",
    
    # MLflow
    "mlruns/.gitkeep",
    
    # Research notebooks
    "research/01_EDA.ipynb",
    "research/02_Training.ipynb",
    "research/03_Evaluation.ipynb",
    "research/04_RAG_Testing.ipynb",
    
    # Scripts
    "scripts/upload_model_to_hf.py",
    
    # Source code - Components
    f"src/{PROJECT_NAME.replace('-', '_')}/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/data_ingestion.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/prepare_base_model.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/model_training.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/model_evaluation.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/explainer.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/components/severity.py",
    
    # Source code - RAG
    f"src/{PROJECT_NAME.replace('-', '_')}/rag/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/rag/store.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/rag/retriever.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/rag/advisor.py",
    
    # Source code - Reports
    f"src/{PROJECT_NAME.replace('-', '_')}/reports/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/reports/schema.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/reports/generator.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/reports/translator.py",
    
    # Source code - Feedback
    f"src/{PROJECT_NAME.replace('-', '_')}/feedback/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/feedback/collector.py",
    
    # Source code - Config
    f"src/{PROJECT_NAME.replace('-', '_')}/config/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/config/configuration.py",
    
    # Source code - Constants
    f"src/{PROJECT_NAME.replace('-', '_')}/constants/__init__.py",
    
    # Source code - Entity
    f"src/{PROJECT_NAME.replace('-', '_')}/entity/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/entity/config_entity.py",
    
    # Source code - Pipeline
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/stage_01_data_ingestion.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/stage_02_prepare_base_model.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/stage_03_model_training.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/stage_04_model_evaluation.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/stage_05_build_vectorstore.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/pipeline/prediction.py",
    
    # Source code - Utils
    f"src/{PROJECT_NAME.replace('-', '_')}/utils/__init__.py",
    f"src/{PROJECT_NAME.replace('-', '_')}/utils/common.py",
    
    # Tests
    "tests/__init__.py",
    "tests/test_classifier.py",
    "tests/test_rag.py",
    "tests/test_reports.py",
    
    # Outputs
    "outputs/.gitkeep",
    
    # Root files
    ".gitignore",
    "app.py",
    "Dockerfile",
    "dvc.yaml",
    "main.py",
    "params.yaml",
    "MODEL_CARD.md",
    "README.md",
    "requirements.txt",
    "setup.py",
]


def create_project_structure():
    """Create all files and directories for the project."""
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir = filepath.parent
        
        # Create directory if needed
        if filedir != Path("."):
            os.makedirs(filedir, exist_ok=True)
            print(f"[DIR] Created directory: {filedir}")
        
        # Create file if it doesn't exist or is empty
        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w") as f:
                # Add minimal content to Python files
                if filepath.suffix == ".py":
                    if filepath.name == "__init__.py":
                        pass  # Leave empty
                    else:
                        f.write(f'"""\n{filepath.stem} module\n"""\n')
                elif filepath.suffix == ".yaml":
                    f.write("# Configuration\n")
                elif filepath.suffix == ".json":
                    f.write("{}\n")
                elif filepath.suffix == ".md":
                    f.write(f"# {filepath.stem.replace('_', ' ').title()}\n")
                elif filepath.name == ".gitkeep":
                    pass  # Leave empty
            print(f"[OK] Created file: {filepath}")
        else:
            print(f"[SKIP] File exists: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tomato Disease Advisory System - Project Generator")
    print("=" * 60)
    create_project_structure()
    print("=" * 60)
    print("[SUCCESS] Project structure created!")
    print("\nNext steps:")
    print("1. cd tomato-disease-advisor")
    print("2. git init")
    print("3. dvc init")
    print("4. pip install -r requirements.txt")
    print("=" * 60)
