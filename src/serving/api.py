"""
Production Model Serving API
FastAPI-based REST API for model inference with monitoring and caching
Demonstrates production ML deployment best practices
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import mlflow
import redis
import json
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
from functools import lru_cache
import hashlib


# Metrics
PREDICTION_COUNTER = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['model_name', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]] = Field(..., description="Feature matrix for prediction")
    model_name: Optional[str] = Field(default="default", description="Model name to use")

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Features must be a list of lists")
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "model_name": "default"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    model_name: str
    model_version: str
    timestamp: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: List[str]
    version: str


class ModelLoader:
    """
    Singleton model loader with lazy loading and caching.
    """
    _instance = None
    _models = {}
    _model_metadata = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_name: str, model_path: str = None):
        """Load model from disk or MLflow."""
        if model_name in self._models:
            logger.info(f"Model '{model_name}' already loaded")
            return

        try:
            if model_path:
                # Load from disk
                self._models[model_name] = joblib.load(model_path)
                self._model_metadata[model_name] = {
                    'source': 'disk',
                    'path': model_path,
                    'loaded_at': datetime.now().isoformat()
                }
            else:
                # Load from MLflow
                model_uri = f"models:/{model_name}/latest"
                self._models[model_name] = mlflow.pyfunc.load_model(model_uri)
                self._model_metadata[model_name] = {
                    'source': 'mlflow',
                    'uri': model_uri,
                    'loaded_at': datetime.now().isoformat()
                }

            logger.info(f"Successfully loaded model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    def get_model(self, model_name: str):
        """Get loaded model."""
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not loaded")
        return self._models[model_name]

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models."""
        return list(self._models.keys())

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata."""
        return self._model_metadata.get(model_name, {})


class PredictionCache:
    """
    Redis-based prediction cache for repeated queries.
    """

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True
            )
            self.enabled = True
            logger.info("Redis cache enabled")
        except:
            self.enabled = False
            logger.warning("Redis not available, caching disabled")

    def _generate_key(self, features: np.ndarray, model_name: str) -> str:
        """Generate cache key from features."""
        feature_str = features.tobytes()
        hash_obj = hashlib.md5(feature_str)
        return f"prediction:{model_name}:{hash_obj.hexdigest()}"

    def get(self, features: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Get cached prediction."""
        if not self.enabled:
            return None

        try:
            key = self._generate_key(features, model_name)
            cached = self.redis_client.get(key)

            if cached:
                logger.info(f"Cache hit for {model_name}")
                return np.array(json.loads(cached))

        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    def set(self, features: np.ndarray, model_name: str, predictions: np.ndarray, ttl: int = 3600):
        """Cache prediction."""
        if not self.enabled:
            return

        try:
            key = self._generate_key(features, model_name)
            value = json.dumps(predictions.tolist())
            self.redis_client.setex(key, ttl, value)
            logger.info(f"Cached prediction for {model_name}")

        except Exception as e:
            logger.error(f"Cache set error: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready ML model serving with monitoring and caching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_loader = ModelLoader()
prediction_cache = PredictionCache()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting ML serving API...")

    # Load default model
    try:
        model_loader.load_model("default", "models/default_model.joblib")
    except:
        logger.warning("Default model not found, will load on demand")

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=model_loader.get_loaded_models(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Make predictions using loaded model.

    Args:
        request: Prediction request with features
        background_tasks: FastAPI background tasks

    Returns:
        Predictions with metadata
    """
    start_time = datetime.now()
    model_name = request.model_name

    try:
        # Convert features to numpy array
        features = np.array(request.features)

        # Check cache
        cached_predictions = prediction_cache.get(features, model_name)

        if cached_predictions is not None:
            predictions = cached_predictions
            PREDICTION_COUNTER.labels(model_name=model_name, status='cache_hit').inc()

        else:
            # Get model
            model = model_loader.get_model(model_name)

            # Make prediction
            predictions = model.predict(features)

            # Cache predictions in background
            background_tasks.add_task(
                prediction_cache.set,
                features,
                model_name,
                predictions
            )

            PREDICTION_COUNTER.labels(model_name=model_name, status='success').inc()

        # Calculate latency
        latency = (datetime.now() - start_time).total_seconds() * 1000
        PREDICTION_LATENCY.labels(model_name=model_name).observe(latency / 1000)

        # Get model metadata
        metadata = model_loader.get_model_metadata(model_name)

        return PredictionResponse(
            predictions=predictions.tolist(),
            model_name=model_name,
            model_version=metadata.get('version', 'unknown'),
            timestamp=datetime.now().isoformat(),
            latency_ms=latency
        )

    except ValueError as e:
        PREDICTION_COUNTER.labels(model_name=model_name, status='error').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        PREDICTION_COUNTER.labels(model_name=model_name, status='error').inc()
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch_predict")
async def batch_predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch prediction endpoint for large-scale inference.

    Args:
        request: Batch prediction request
        background_tasks: FastAPI background tasks

    Returns:
        Predictions
    """
    # Same as predict but optimized for batches
    return await predict(request, background_tasks)


@app.get("/models")
async def list_models():
    """List all loaded models."""
    models = model_loader.get_loaded_models()
    metadata = {
        name: model_loader.get_model_metadata(name)
        for name in models
    }

    return {
        "models": models,
        "metadata": metadata,
        "count": len(models)
    }


@app.post("/models/{model_name}/load")
async def load_model(model_name: str, model_path: Optional[str] = None):
    """
    Load a model.

    Args:
        model_name: Name of model to load
        model_path: Optional path to model file

    Returns:
        Success message
    """
    try:
        model_loader.load_model(model_name, model_path)
        return {
            "message": f"Model '{model_name}' loaded successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/model_info/{model_name}")
async def model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name of model

    Returns:
        Model information
    """
    try:
        metadata = model_loader.get_model_metadata(model_name)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        return {
            "model_name": model_name,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
