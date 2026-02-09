"""
Constants module - defines paths to configuration files.
"""
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Configuration file paths
CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"

# Artifacts directory
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Knowledge base directory
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"

# Output directory for reports
OUTPUTS_DIR = ROOT_DIR / "outputs"
