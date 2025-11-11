"""
Setup configuration for ML Demo Project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ml-demo-project",
    version="1.0.0",
    author="ML Engineer",
    author_email="ml-engineer@example.com",
    description="Production-ready Machine Learning System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ml-demo-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "tensorflow>=2.15.0",
        "scikit-learn>=1.3.0",
        "pyspark>=3.5.0",
        "fastapi>=0.108.0",
        "uvicorn>=0.25.0",
        "mlflow>=2.9.0",
        "optuna>=3.5.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "loguru>=0.7.0",
        "pydantic>=2.5.0",
        "redis>=5.0.0",
        "prometheus-client>=0.19.0",
        "boto3>=1.34.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.0",
        ],
        "gpu": [
            "torch-cuda>=2.1.0",
            "tensorflow-gpu>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-train=pipelines.training_pipeline:main",
            "ml-serve=src.serving.api:main",
        ],
    },
)
