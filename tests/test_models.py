"""
Comprehensive tests for ML models
Demonstrates production testing best practices
"""

import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_regression, make_classification

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pytorch_model import (
    AdvancedNeuralNetwork,
    PyTorchModelTrainer,
    create_model as create_pytorch_model
)
from src.models.sklearn_model import (
    MLModelFactory,
    HyperparameterOptimizer,
    ProductionMLPipeline
)
from src.models.tensorflow_model import create_custom_model


class TestPyTorchModels:
    """Test suite for PyTorch models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        return X, y.reshape(-1, 1)

    @pytest.fixture
    def pytorch_model(self):
        """Create PyTorch model for testing."""
        config = {
            'hidden_dims': [64, 32],
            'dropout': 0.2,
            'use_attention': True,
            'use_residual': False
        }
        return create_pytorch_model(input_dim=10, output_dim=1, config=config)

    def test_model_creation(self, pytorch_model):
        """Test model creation."""
        assert pytorch_model is not None
        assert isinstance(pytorch_model, AdvancedNeuralNetwork)

    def test_model_forward_pass(self, pytorch_model, sample_data):
        """Test model forward pass."""
        X, _ = sample_data
        X_tensor = torch.FloatTensor(X[:10])

        pytorch_model.eval()
        with torch.no_grad():
            output = pytorch_model(X_tensor)

        assert output.shape == (10, 1)
        assert not torch.isnan(output).any()

    def test_model_training(self, pytorch_model, sample_data):
        """Test model training."""
        X, y = sample_data
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        dataloader = DataLoader(dataset, batch_size=10)

        trainer = PyTorchModelTrainer(pytorch_model)
        trainer.setup_training()

        initial_loss = trainer.train_epoch(dataloader)
        assert initial_loss > 0

    def test_model_parameters(self, pytorch_model):
        """Test model has learnable parameters."""
        params = sum(p.numel() for p in pytorch_model.parameters())
        assert params > 0

    def test_model_gradient_flow(self, pytorch_model, sample_data):
        """Test gradients flow through model."""
        X, y = sample_data
        X_tensor = torch.FloatTensor(X[:10])
        y_tensor = torch.FloatTensor(y[:10])

        optimizer = torch.optim.Adam(pytorch_model.parameters())
        criterion = torch.nn.MSELoss()

        optimizer.zero_grad()
        output = pytorch_model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()

        # Check gradients exist
        for param in pytorch_model.parameters():
            assert param.grad is not None


class TestSklearnModels:
    """Test suite for Scikit-learn models."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        return make_regression(n_samples=100, n_features=10, random_state=42)

    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        return make_classification(n_samples=100, n_features=10, random_state=42)

    @pytest.mark.parametrize("model_type", ['rf', 'xgb', 'lgb'])
    def test_regressor_creation(self, model_type):
        """Test creation of different regressors."""
        model = MLModelFactory.create_model(model_type, task='regression')
        assert model is not None

    @pytest.mark.parametrize("model_type", ['rf', 'xgb', 'lgb'])
    def test_classifier_creation(self, model_type):
        """Test creation of different classifiers."""
        model = MLModelFactory.create_model(model_type, task='classification')
        assert model is not None

    def test_model_training(self, regression_data):
        """Test model training."""
        X, y = regression_data
        model = MLModelFactory.create_model('rf', task='regression')

        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])

        assert predictions.shape == (20,)
        assert not np.isnan(predictions).any()

    def test_pipeline_creation(self, regression_data):
        """Test ML pipeline creation."""
        config = {'task': 'regression'}
        pipeline = ProductionMLPipeline(config)

        model = MLModelFactory.create_model('rf', task='regression')
        pipeline.build_pipeline(model, use_scaling=True)

        assert pipeline.pipeline is not None

    def test_pipeline_training(self, regression_data):
        """Test pipeline training and evaluation."""
        X, y = regression_data
        config = {'task': 'regression'}
        pipeline = ProductionMLPipeline(config)

        model = MLModelFactory.create_model('rf', task='regression')
        pipeline.build_pipeline(model)

        metrics = pipeline.train_and_evaluate(
            X[:60], y[:60], X[60:], y[60:], use_mlflow=False
        )

        assert 'rmse' in metrics
        assert metrics['rmse'] >= 0

    def test_hyperparameter_optimization(self, regression_data):
        """Test hyperparameter optimization."""
        X, y = regression_data
        optimizer = HyperparameterOptimizer('rf', task='regression')

        best_params = optimizer.optimize(
            X[:80], y[:80], X[80:], y[80:], n_trials=5
        )

        assert best_params is not None
        assert isinstance(best_params, dict)


class TestTensorFlowModels:
    """Test suite for TensorFlow models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        return make_regression(n_samples=100, n_features=10, random_state=42)

    def test_model_creation(self):
        """Test TensorFlow model creation."""
        config = {
            'model_type': 'residual',
            'input_dim': 10,
            'output_dim': 1,
            'num_blocks': 2,
            'hidden_dim': 64,
            'task': 'regression'
        }

        model = create_custom_model(config)
        assert model is not None

    def test_model_prediction(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        config = {
            'model_type': 'residual',
            'input_dim': 10,
            'output_dim': 1,
            'task': 'regression'
        }

        model = create_custom_model(config)
        model.compile(optimizer='adam', loss='mse')

        # Train briefly
        model.fit(X[:80], y[:80], epochs=2, verbose=0)

        # Predict
        predictions = model.predict(X[80:], verbose=0)
        assert predictions.shape[0] == 20


class TestModelIntegration:
    """Integration tests for complete workflow."""

    @pytest.fixture
    def workflow_data(self):
        """Generate data for workflow testing."""
        X, y = make_regression(n_samples=200, n_features=15, random_state=42)
        split = 160
        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_test': X[split:],
            'y_test': y[split:]
        }

    def test_end_to_end_sklearn(self, workflow_data):
        """Test end-to-end workflow with sklearn."""
        config = {'task': 'regression'}
        pipeline = ProductionMLPipeline(config)

        model = MLModelFactory.create_model('xgb', task='regression')
        pipeline.build_pipeline(model, use_scaling=True)

        metrics = pipeline.train_and_evaluate(
            workflow_data['X_train'],
            workflow_data['y_train'],
            workflow_data['X_test'],
            workflow_data['y_test'],
            use_mlflow=False
        )

        # Verify reasonable performance
        assert metrics['rmse'] < 100  # Adjust threshold as needed
        assert metrics['r2'] > -1

    def test_model_reproducibility(self, workflow_data):
        """Test model training is reproducible."""
        model1 = MLModelFactory.create_model('rf', task='regression', random_state=42)
        model2 = MLModelFactory.create_model('rf', task='regression', random_state=42)

        model1.fit(workflow_data['X_train'], workflow_data['y_train'])
        model2.fit(workflow_data['X_train'], workflow_data['y_train'])

        pred1 = model1.predict(workflow_data['X_test'])
        pred2 = model2.predict(workflow_data['X_test'])

        np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
