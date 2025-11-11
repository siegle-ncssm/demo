"""
Production ML Training Pipeline
End-to-end orchestration of data processing, training, and deployment
Demonstrates 3+ years of production data pipeline experience
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import mlflow
import yaml
from loguru import logger

from src.data_pipeline.spark_processing import SparkDataProcessor
from src.data_pipeline.data_validation import DataValidator
from src.models.pytorch_model import PyTorchModelTrainer, create_model as create_pytorch_model
from src.models.tensorflow_model import create_custom_model as create_tf_model, AdvancedKerasModel
from src.models.sklearn_model import (
    MLModelFactory,
    HyperparameterOptimizer,
    ProductionMLPipeline,
    EnsembleModelBuilder
)


class MLTrainingPipeline:
    """
    Production-ready end-to-end ML training pipeline.
    Orchestrates data processing, validation, training, and model registration.
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.experiment_id = None
        self.run_id = None
        self.metrics = {}
        self.models = {}

        # Setup MLflow
        self._setup_mlflow()

        # Setup logging
        self._setup_logging()

        logger.info("Initialized ML Training Pipeline")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow_config = self.config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
        experiment_name = mlflow_config.get('experiment_name', 'ml-demo-project')

        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger.add(
            log_file,
            rotation="500 MB",
            retention="10 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

    def run(self) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Returns:
            Pipeline results and metrics
        """
        logger.info("=" * 80)
        logger.info("Starting ML Training Pipeline")
        logger.info("=" * 80)

        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            self.run_id = run.info.run_id

            try:
                # Step 1: Data Processing
                logger.info("Step 1: Data Processing")
                processed_data = self._process_data()

                # Step 2: Data Validation
                logger.info("Step 2: Data Validation")
                validation_results = self._validate_data(processed_data)

                if not validation_results['passed']:
                    logger.error("Data validation failed!")
                    raise ValueError("Data quality checks failed")

                # Step 3: Feature Engineering
                logger.info("Step 3: Feature Engineering")
                features = self._engineer_features(processed_data)

                # Step 4: Train Models
                logger.info("Step 4: Model Training")
                self._train_models(features)

                # Step 5: Model Evaluation
                logger.info("Step 5: Model Evaluation")
                evaluation_results = self._evaluate_models(features)

                # Step 6: Model Selection
                logger.info("Step 6: Model Selection")
                best_model = self._select_best_model(evaluation_results)

                # Step 7: Model Registration
                logger.info("Step 7: Model Registration")
                model_version = self._register_model(best_model)

                # Step 8: Generate Report
                logger.info("Step 8: Generate Report")
                report = self._generate_report(evaluation_results)

                logger.info("=" * 80)
                logger.info("Pipeline completed successfully!")
                logger.info(f"Best model: {best_model['name']}")
                logger.info(f"Best score: {best_model['score']:.4f}")
                logger.info(f"Model version: {model_version}")
                logger.info("=" * 80)

                return {
                    'success': True,
                    'run_id': self.run_id,
                    'best_model': best_model,
                    'model_version': model_version,
                    'report': report
                }

            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                mlflow.log_param("status", "failed")
                raise

    def _process_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process data using distributed computing.

        Returns:
            Processed datasets
        """
        data_config = self.config.get('data', {})

        # Option 1: Use Spark for large-scale data
        if data_config.get('use_spark', False):
            processor = SparkDataProcessor()
            try:
                # Read data
                df = processor.read_distributed_data(
                    path=data_config['path'],
                    format=data_config.get('format', 'parquet')
                )

                # Process
                df = processor.process_large_scale_data(df)

                # Feature engineering
                df = processor.feature_engineering_distributed(df)

                # Convert to Pandas for model training
                data = df.toPandas()

            finally:
                processor.close()

        # Option 2: Use Pandas for smaller datasets
        else:
            data = pd.read_csv(data_config['path'])

        # Train/test split
        from sklearn.model_selection import train_test_split

        target_col = data_config['target_column']
        feature_cols = [col for col in data.columns if col != target_col]

        X = data[feature_cols].values
        y = data[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_config.get('test_size', 0.2),
            random_state=42
        )

        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=data_config.get('val_size', 0.2),
            random_state=42
        )

        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        mlflow.log_params({
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': X_train.shape[1]
        })

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols
        }

    def _validate_data(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Validate data quality.

        Args:
            data: Processed datasets

        Returns:
            Validation results
        """
        validator = DataValidator()

        # Convert to DataFrame for validation
        df = pd.DataFrame(data['X_train'])

        # Validation config
        validation_config = {
            'max_null_percentage': 5.0,
            'max_duplicate_percentage': 1.0,
        }

        # Validate
        results = validator.validate_data_quality(df, validation_config)

        # Log to MLflow
        mlflow.log_metrics({
            'data_validation_passed': 1.0 if results['passed'] else 0.0
        })

        return results

    def _engineer_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Engineer additional features.

        Args:
            data: Input data

        Returns:
            Enhanced data with engineered features
        """
        # Feature engineering can be added here
        # For now, return as-is
        return data

    def _train_models(self, data: Dict[str, np.ndarray]):
        """
        Train multiple models with different frameworks.

        Args:
            data: Training data
        """
        model_config = self.config.get('models', {})

        # Train PyTorch model
        if 'pytorch' in model_config:
            logger.info("Training PyTorch model...")
            self._train_pytorch_model(data, model_config['pytorch'])

        # Train TensorFlow model
        if 'tensorflow' in model_config:
            logger.info("Training TensorFlow model...")
            self._train_tensorflow_model(data, model_config['tensorflow'])

        # Train Scikit-learn models
        if 'sklearn' in model_config:
            logger.info("Training Scikit-learn models...")
            self._train_sklearn_models(data, model_config['sklearn'])

    def _train_pytorch_model(self, data: Dict[str, np.ndarray], config: Dict[str, Any]):
        """Train PyTorch model."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(data['X_train']),
            torch.FloatTensor(data['y_train']).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(data['X_val']),
            torch.FloatTensor(data['y_val']).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))

        # Create model
        model = create_pytorch_model(
            input_dim=data['X_train'].shape[1],
            output_dim=1,
            config=config
        )

        # Train
        trainer = PyTorchModelTrainer(model, config=config)
        trainer.setup_training(
            optimizer_name=config.get('optimizer', 'adamw'),
            learning_rate=config.get('learning_rate', 0.001)
        )

        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=config.get('epochs', 50),
            early_stopping_patience=config.get('patience', 10),
            use_mlflow=True
        )

        self.models['pytorch'] = {
            'trainer': trainer,
            'model': model,
            'history': history
        }

    def _train_tensorflow_model(self, data: Dict[str, np.ndarray], config: Dict[str, Any]):
        """Train TensorFlow model."""
        tf_config = {
            'model_type': config.get('model_type', 'residual'),
            'input_dim': data['X_train'].shape[1],
            'output_dim': 1,
            'task': 'regression',
            **config
        }

        model = create_tf_model(tf_config)
        builder = AdvancedKerasModel(tf_config)

        history = builder.compile_and_train(
            model,
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32),
            use_mlflow=True
        )

        self.models['tensorflow'] = {
            'builder': builder,
            'model': model,
            'history': history
        }

    def _train_sklearn_models(self, data: Dict[str, np.ndarray], config: Dict[str, Any]):
        """Train Scikit-learn models."""
        task = config.get('task', 'regression')

        # Train individual models
        for model_type in config.get('model_types', ['xgb', 'lgb', 'rf']):
            logger.info(f"Training {model_type} model...")

            # Hyperparameter optimization
            if config.get('optimize_hyperparams', False):
                optimizer = HyperparameterOptimizer(model_type, task)
                best_params = optimizer.optimize(
                    data['X_train'],
                    data['y_train'],
                    X_val=data['X_val'],
                    y_val=data['y_val'],
                    n_trials=config.get('n_trials', 50)
                )
            else:
                best_params = {}

            # Create model
            model = MLModelFactory.create_model(model_type, task, **best_params)

            # Build pipeline
            pipeline_config = {'task': task}
            pipeline = ProductionMLPipeline(pipeline_config)
            pipeline.build_pipeline(model, use_scaling=True)

            # Train
            metrics = pipeline.train_and_evaluate(
                data['X_train'],
                data['y_train'],
                data['X_val'],
                data['y_val'],
                use_mlflow=True
            )

            self.models[f'sklearn_{model_type}'] = {
                'pipeline': pipeline,
                'model': model,
                'metrics': metrics
            }

        # Create ensemble
        if config.get('use_ensemble', False) and len(self.models) > 1:
            logger.info("Creating ensemble model...")
            self._create_ensemble(data, task)

    def _create_ensemble(self, data: Dict[str, np.ndarray], task: str):
        """Create ensemble of trained models."""
        base_models = []

        for name, model_info in self.models.items():
            if name.startswith('sklearn_'):
                base_models.append((name, model_info['pipeline']))

        if len(base_models) >= 2:
            ensemble_builder = EnsembleModelBuilder(task)
            ensemble = ensemble_builder.create_stacking_ensemble(base_models)

            # Train ensemble
            ensemble.fit(data['X_train'], data['y_train'])

            self.models['ensemble'] = {
                'model': ensemble,
                'builder': ensemble_builder
            }

            logger.info(f"Created ensemble with {len(base_models)} models")

    def _evaluate_models(self, data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.

        Args:
            data: Test data

        Returns:
            Evaluation results for all models
        """
        from sklearn.metrics import mean_squared_error, r2_score

        results = {}

        X_test = data['X_test']
        y_test = data['y_test']

        for name, model_info in self.models.items():
            logger.info(f"Evaluating {name}...")

            try:
                if name == 'pytorch':
                    import torch
                    model = model_info['model']
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test)
                        y_pred = model(X_tensor).cpu().numpy()

                elif name == 'tensorflow':
                    model = model_info['model']
                    y_pred = model.predict(X_test)

                else:  # sklearn models
                    if 'pipeline' in model_info:
                        y_pred = model_info['pipeline'].pipeline.predict(X_test)
                    else:
                        y_pred = model_info['model'].predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                }

                logger.info(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

                # Log to MLflow
                mlflow.log_metrics({
                    f'{name}_mse': mse,
                    f'{name}_rmse': rmse,
                    f'{name}_r2': r2
                })

            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

        return results

    def _select_best_model(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Select best model based on performance.

        Args:
            results: Evaluation results

        Returns:
            Best model info
        """
        # Select based on RMSE (lower is better)
        best_name = min(results.keys(), key=lambda k: results[k]['rmse'])
        best_score = results[best_name]['rmse']

        best_model = {
            'name': best_name,
            'score': best_score,
            'metrics': results[best_name],
            'model_info': self.models[best_name]
        }

        mlflow.log_params({
            'best_model_name': best_name,
            'best_model_score': best_score
        })

        return best_model

    def _register_model(self, best_model: Dict[str, Any]) -> str:
        """
        Register best model in MLflow.

        Args:
            best_model: Best model information

        Returns:
            Model version
        """
        model_name = self.config.get('model_registry_name', 'ml-demo-model')

        # Register model based on framework
        if best_model['name'] == 'pytorch':
            import mlflow.pytorch
            model = best_model['model_info']['model']
            mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)

        elif best_model['name'] == 'tensorflow':
            import mlflow.tensorflow
            model = best_model['model_info']['model']
            mlflow.tensorflow.log_model(model, "model", registered_model_name=model_name)

        else:
            import mlflow.sklearn
            pipeline = best_model['model_info']['pipeline']
            mlflow.sklearn.log_model(
                pipeline.pipeline,
                "model",
                registered_model_name=model_name
            )

        logger.info(f"Registered model: {model_name}")
        return "v1"

    def _generate_report(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate comprehensive training report.

        Args:
            results: Evaluation results

        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'config': self.config,
            'models_trained': list(self.models.keys()),
            'evaluation_results': results,
            'best_model': min(results.keys(), key=lambda k: results[k]['rmse'])
        }

        # Save report
        report_dir = Path(self.config.get('output_dir', 'output'))
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

        with open(report_file, 'w') as f:
            yaml.dump(report, f)

        logger.info(f"Report saved to {report_file}")

        # Log to MLflow
        mlflow.log_artifact(str(report_file))

        return report


def main():
    """Main entry point for training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ML Training Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = MLTrainingPipeline(args.config)
    results = pipeline.run()

    logger.info("Pipeline execution completed!")
    return results


if __name__ == "__main__":
    main()
