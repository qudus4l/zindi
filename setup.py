#!/usr/bin/env python3
"""Setup script for the Clinical Decision Support Model package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="zindi-clinical-ai",
    version="0.1.0",
    description="Clinical Decision Support Model for Kenyan Healthcare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zindi Clinical AI Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-clinical-model=scripts.train_model:main",
            "evaluate-clinical-model=scripts.evaluate_models:main",
            "clinical-inference=scripts.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="medical ai healthcare clinical decision support kenya",
    project_urls={
        "Bug Reports": "https://github.com/your-org/zindi-clinical-ai/issues",
        "Source": "https://github.com/your-org/zindi-clinical-ai",
    },
) 