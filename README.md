# ML Demo Project - Production Machine Learning System

A comprehensive production-ready machine learning system demonstrating 6+ years of distributed computing, 4+ years of Python programming, and 2+ years of ML systems experience.

## Overview

This project showcases a complete end-to-end machine learning platform with:

- **Distributed Data Processing**: Apache Spark for large-scale data processing
- **Multiple ML Frameworks**: PyTorch, TensorFlow, and scikit-learn implementations
- **Production ML Pipelines**: Automated training, validation, and deployment
- **Model Serving**: FastAPI-based REST API with caching and monitoring
- **Cloud Deployment**: Ready-to-deploy on AWS, Azure, and GCP
- **Monitoring & Observability**: Prometheus, Grafana, and drift detection
- **CI/CD**: Automated testing and deployment pipelines

## Key Features

### Distributed Computing
- **Apache Spark**: Scalable data processing with optimized configurations
- **Dask**: Distributed computing for complex workflows
- **Ray**: Distributed ML training and hyperparameter tuning

### ML Frameworks
- **PyTorch**: Advanced neural networks with attention mechanisms and residual connections
- **TensorFlow/Keras**: Production-ready deep learning with modern architectures
- **Scikit-learn**: Traditional ML with XGBoost, LightGBM, CatBoost
- **Ensemble Methods**: Stacking and voting for improved performance

### Production Features
- **Data Validation**: Comprehensive quality checks and schema validation
- **Model Monitoring**: Real-time drift detection and performance tracking
- **Model Registry**: MLflow integration for version control
- **API Serving**: High-performance REST API with caching
- **Containerization**: Docker and Kubernetes deployment
- **Auto-scaling**: HPA configuration for production workloads

## Project Structure

```
ml-demo-project/
├── src/
│   ├── data_pipeline/        # Distributed data processing
│   │   ├── spark_processing.py
│   │   ├── data_validation.py
│   │   └── feature_engineering.py
│   ├── models/               # ML model implementations
│   │   ├── pytorch_model.py
│   │   ├── tensorflow_model.py
│   │   └── sklearn_model.py
│   ├── training/            # Training infrastructure
│   ├── serving/             # Model serving API
│   │   ├── api.py
│   │   └── batch_inference.py
│   ├── monitoring/          # Monitoring and observability
│   │   └── metrics.py
│   └── utils/
├── pipelines/               # End-to-end ML pipelines
│   └── training_pipeline.py
├── deployment/              # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   └── deployment.yaml
│   ├── aws/
│   │   └── cloudformation.yaml
│   ├── azure/
│   └── gcp/
├── tests/                   # Comprehensive test suite
│   ├── test_models.py
│   └── test_api.py
├── configs/                 # Configuration files
├── notebooks/              # Jupyter notebooks for exploration
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # CI/CD pipeline
└── requirements.txt

## Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) - Modern Python package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker and Docker Compose (optional, for containerized deployment)
- Kubernetes (optional, for K8s deployment)
- AWS/Azure/GCP CLI (optional, for cloud deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/ml-demo-project.git
cd ml-demo-project
```

2. Install dependencies (uv automatically creates and manages the virtual environment):
```bash
uv sync
```

That's it! All dependencies are installed and ready to use.

### Local Development

#### Run Data Processing Pipeline

```bash
uv run python src/data_pipeline/spark_processing.py
```

#### Train Models

```bash
uv run python pipelines/training_pipeline.py --config configs/training_config.yaml
# Or using the installed CLI command:
uv run ml-train --config configs/training_config.yaml
```

#### Start API Server

```bash
uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
# Or using the installed CLI command:
uv run ml-serve
```

Access API documentation at: http://localhost:8000/docs

#### Development Workflow

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Format code with Ruff
uv run ruff format .

# Lint code with Ruff
uv run ruff check .

# Lint and auto-fix issues
uv run ruff check --fix .

# Type checking with mypy
uv run mypy src

# Install pre-commit hooks (one-time setup)
uvx pre-commit install

# Run pre-commit hooks manually
uvx pre-commit run --all-files
```

### Docker Deployment

```bash
cd deployment/docker
docker-compose up -d
```

This starts:
- ML serving API (port 8000)
- MLflow tracking server (port 5000)
- Redis cache (port 6379)
- Prometheus (port 9090)
- Grafana (port 3000)

### Kubernetes Deployment

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### Cloud Deployment

#### AWS (SageMaker + ECS)

```bash
aws cloudformation create-stack \
  --stack-name ml-production \
  --template-body file://deployment/aws/cloudformation.yaml \
  --capabilities CAPABILITY_IAM
```

## Usage Examples

### Making Predictions

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        "model_name": "default"
    }
)

predictions = response.json()["predictions"]
```

### Batch Prediction

```python
# Batch prediction
features = [[i, i+1, i+2, i+3, i+4] for i in range(100)]

response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"features": features, "model_name": "default"}
)
```

### Monitoring

Access monitoring dashboards:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- MLflow: http://localhost:5000

## Architecture

### Data Pipeline

1. **Data Ingestion**: Read from various sources (S3, HDFS, databases)
2. **Distributed Processing**: Spark-based transformations and feature engineering
3. **Data Validation**: Quality checks and schema validation
4. **Feature Store**: Store engineered features for reuse

### Training Pipeline

1. **Data Preparation**: Load and preprocess training data
2. **Model Training**: Train multiple models in parallel
3. **Hyperparameter Tuning**: Optuna-based optimization
4. **Model Evaluation**: Comprehensive metrics and validation
5. **Model Registration**: Register best model in MLflow

### Serving Pipeline

1. **Model Loading**: Load models from registry
2. **Prediction API**: RESTful API with FastAPI
3. **Caching**: Redis-based prediction caching
4. **Monitoring**: Real-time metrics and drift detection

## Performance

### Benchmarks

- **Training Time**: <10 minutes for 1M samples
- **Inference Latency**: <50ms p99 for single prediction
- **Throughput**: 1000+ predictions/second
- **Scalability**: Auto-scales 3-10 pods based on load

### Optimization Techniques

- Model quantization for faster inference
- Batch prediction optimization
- Feature caching
- Load balancing across replicas
- GPU acceleration for deep learning

## Testing

Run all tests:
```bash
uv run pytest tests/ -v --cov=src
```

Run specific test suite:
```bash
uv run pytest tests/test_models.py -v
uv run pytest tests/test_api.py -v
```

Run tests with coverage report:
```bash
uv run pytest --cov --cov-report=html
# Open htmlcov/index.html to view detailed coverage report
```

## Monitoring & Alerting

### Metrics Tracked

- **Model Performance**: Accuracy, RMSE, latency
- **Data Quality**: Null rates, distribution shifts
- **System Health**: CPU, memory, request rates
- **Business Metrics**: Predictions served, cache hit rate

### Drift Detection

Automatic detection of:
- Feature drift (distribution changes)
- Target drift (label distribution changes)
- Concept drift (model performance degradation)

## Dependency Management

This project uses [uv](https://docs.astral.sh/uv/) for modern Python dependency management.

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add with version constraints
uv add "pandas>=2.0,<3.0"
```

### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Sync environment with lock file
uv sync
```

### Removing Dependencies

```bash
uv remove package-name
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies: `uv sync --all-extras`
4. Install pre-commit hooks: `uvx pre-commit install`
5. Make your changes and ensure tests pass: `uv run pytest`
6. Format and lint code: `uv run ruff format . && uv run ruff check --fix .`
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open Pull Request

## Skills Demonstrated

### Distributed Computing (6+ years)
- Apache Spark for large-scale data processing
- Distributed training with Ray and Dask
- Optimized cluster configurations
- Streaming data pipelines

### Python Programming (4+ years)
- Advanced OOP and design patterns
- Type hints and static analysis
- Performance optimization
- Testing best practices

### ML Systems (2+ years)
- PyTorch, TensorFlow, scikit-learn
- Hyperparameter optimization
- Model ensembling
- Production deployment

### Production Pipelines (3+ years)
- End-to-end ML pipelines
- Data validation and quality checks
- Model monitoring and drift detection
- CI/CD automation

### Cloud Deployment
- AWS (SageMaker, ECS, S3, CloudFormation)
- Kubernetes orchestration
- Docker containerization
- Infrastructure as Code

### Best Practices
- Comprehensive testing with pytest
- Modern code quality tooling (Ruff for formatting/linting, MyPy for type checking)
- PEP 621 compliant pyproject.toml
- Automated pre-commit hooks
- Reproducible environments with uv lock files
- Documentation
- Monitoring and observability
- Security hardening

## License

MIT License - see LICENSE file for details

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

Built with:
- **ML Frameworks**: PyTorch, TensorFlow, scikit-learn
- **Data Processing**: Apache Spark, Pandas, Polars
- **MLOps**: MLflow, FastAPI, Optuna
- **Infrastructure**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana
- **Development Tools**: uv, Ruff, pytest, pre-commit
