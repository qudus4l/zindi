#!/usr/bin/env python3
"""Setup script to ensure proper Python path configuration.

This script ensures that the project root is in the Python path,
allowing imports to work correctly from any directory.
"""

import sys
import os
from pathlib import Path


def setup_python_path() -> str:
    """Setup Python path for the project.
    
    Returns:
        str: The project root directory path
    """
    # Get the project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Add to Python path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        print(f"Added {project_root_str} to Python path")
    else:
        print(f"Project root {project_root_str} already in Python path")
    
    return project_root_str


def verify_imports() -> bool:
    """Verify that key modules can be imported.
    
    Returns:
        bool: True if all imports successful, False otherwise
    """
    try:
        from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
        from src.training.trainer import ClinicalTrainer, ClinicalDataset
        from src.utils.config import Config
        print("✓ All key modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


if __name__ == "__main__":
    print("Setting up Python path for Zindi Clinical AI project...")
    project_root = setup_python_path()
    
    print("\nVerifying imports...")
    if verify_imports():
        print("\n✓ Setup complete! You can now run training scripts.")
    else:
        print("\n✗ Setup failed. Please check your environment.")
        sys.exit(1) 