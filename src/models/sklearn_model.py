"""
Scikit-learn Model Implementation
Production-ready traditional ML with hyperparameter tuning and ensembles
"""

from typing import Any

import catboost as cb
import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import xgboost as xgb
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom feature selector based on importance."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.important_features = None

    def fit(self, X, y=None):
        # Use tree-based model for feature importance
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)

        importances = model.feature_importances_
        self.important_features = importances >= self.threshold

        logger.info(
            f"Selected {self.important_features.sum()} out of {len(importances)} features"
        )
        return self

    def transform(self, X):
        if self.important_features is None:
            raise ValueError("Fit must be called before transform")
        return X[:, self.important_features]


class MLModelFactory:
    """
    Factory for creating production-ready ML models.
    Supports multiple algorithms with optimized hyperparameters.
    """

    @staticmethod
    def create_model(
        model_type: str, task: str = "regression", **kwargs
    ) -> BaseEstimator:
        """
        Create ML model based on type and task.

        Args:
            model_type: Type of model (rf, xgb, lgb, catboost, etc.)
            task: Task type (regression, classification)
            **kwargs: Additional model parameters

        Returns:
            Scikit-learn compatible estimator
        """
        if task == "regression":
            return MLModelFactory._create_regressor(model_type, **kwargs)
        else:
            return MLModelFactory._create_classifier(model_type, **kwargs)

    @staticmethod
    def _create_regressor(model_type: str, **kwargs) -> BaseEstimator:
        """Create regression model."""
        models = {
            "rf": RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 20),
                min_samples_split=kwargs.get("min_samples_split", 5),
                min_samples_leaf=kwargs.get("min_samples_leaf", 2),
                n_jobs=-1,
                random_state=42,
            ),
            "xgb": xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                n_jobs=-1,
                random_state=42,
            ),
            "lgb": lgb.LGBMRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", -1),
                learning_rate=kwargs.get("learning_rate", 0.1),
                num_leaves=kwargs.get("num_leaves", 31),
                subsample=kwargs.get("subsample", 0.8),
                n_jobs=-1,
                random_state=42,
            ),
            "catboost": cb.CatBoostRegressor(
                iterations=kwargs.get("iterations", 200),
                depth=kwargs.get("depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                verbose=False,
                random_state=42,
            ),
            "gb": GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            ),
            "ridge": Ridge(alpha=kwargs.get("alpha", 1.0)),
            "lasso": Lasso(alpha=kwargs.get("alpha", 1.0)),
            "elasticnet": ElasticNet(
                alpha=kwargs.get("alpha", 1.0), l1_ratio=kwargs.get("l1_ratio", 0.5)
            ),
            "svr": SVR(C=kwargs.get("C", 1.0), kernel=kwargs.get("kernel", "rbf")),
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        return models[model_type]

    @staticmethod
    def _create_classifier(model_type: str, **kwargs) -> BaseEstimator:
        """Create classification model."""
        models = {
            "rf": RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 20),
                min_samples_split=kwargs.get("min_samples_split", 5),
                min_samples_leaf=kwargs.get("min_samples_leaf", 2),
                n_jobs=-1,
                random_state=42,
            ),
            "xgb": xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
            "lgb": lgb.LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", -1),
                learning_rate=kwargs.get("learning_rate", 0.1),
                num_leaves=kwargs.get("num_leaves", 31),
                subsample=kwargs.get("subsample", 0.8),
                n_jobs=-1,
                random_state=42,
            ),
            "catboost": cb.CatBoostClassifier(
                iterations=kwargs.get("iterations", 200),
                depth=kwargs.get("depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                verbose=False,
                random_state=42,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            ),
            "logistic": LogisticRegression(
                C=kwargs.get("C", 1.0), max_iter=1000, n_jobs=-1, random_state=42
            ),
            "svc": SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
                random_state=42,
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5), n_jobs=-1
            ),
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        return models[model_type]


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna.
    """

    def __init__(self, model_type: str, task: str = "regression"):
        self.model_type = model_type
        self.task = task
        self.best_params = None
        self.study = None

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 100,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds

        Returns:
            Best parameters
        """

        def objective(trial):
            params = self._suggest_params(trial)
            model = MLModelFactory.create_model(self.model_type, self.task, **params)

            if X_val is not None and y_val is not None:
                # Use validation set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = self._calculate_score(y_val, y_pred)
            else:
                # Use cross-validation
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=self._get_scoring_metric(),
                    n_jobs=-1,
                )
                score = scores.mean()

            return score

        # Create study
        direction = "maximize" if self.task == "classification" else "minimize"
        self.study = optuna.create_study(direction=direction)

        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value:.4f}")

        return self.best_params

    def _suggest_params(self, trial) -> dict[str, Any]:
        """Suggest hyperparameters for trial."""
        if self.model_type in ["rf", "xgb", "lgb", "catboost", "gb"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
            }

            if self.model_type in ["xgb", "lgb"]:
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                )

            if self.model_type == "lgb":
                params["num_leaves"] = trial.suggest_int("num_leaves", 20, 100)

        elif self.model_type in ["ridge", "lasso", "elasticnet", "logistic"]:
            params = {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}

            if self.model_type == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        elif self.model_type in ["svc", "svr"]:
            params = {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "kernel": trial.suggest_categorical(
                    "kernel", ["rbf", "linear", "poly"]
                ),
            }

        else:
            params = {}

        return params

    def _get_scoring_metric(self) -> str:
        """Get scoring metric for CV."""
        if self.task == "classification":
            return "roc_auc"
        else:
            return "neg_mean_squared_error"

    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate score for optimization."""
        if self.task == "classification":
            return roc_auc_score(y_true, y_pred)
        else:
            return -mean_squared_error(y_true, y_pred)


class EnsembleModelBuilder:
    """
    Build and manage ensemble models (stacking, voting, etc.)
    """

    def __init__(self, task: str = "regression"):
        self.task = task
        self.ensemble = None

    def create_stacking_ensemble(
        self,
        base_models: list[tuple[str, BaseEstimator]],
        meta_model: BaseEstimator | None = None,
    ) -> BaseEstimator:
        """
        Create stacking ensemble.

        Args:
            base_models: List of (name, model) tuples
            meta_model: Meta-learner model

        Returns:
            Stacking ensemble
        """
        if meta_model is None:
            meta_model = Ridge() if self.task == "regression" else LogisticRegression()

        if self.task == "regression":
            self.ensemble = StackingRegressor(
                estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1
            )
        else:
            self.ensemble = StackingClassifier(
                estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1
            )

        logger.info(f"Created stacking ensemble with {len(base_models)} base models")
        return self.ensemble

    def create_voting_ensemble(
        self, models: list[tuple[str, BaseEstimator]], voting: str = "soft"
    ) -> BaseEstimator:
        """
        Create voting ensemble.

        Args:
            models: List of (name, model) tuples
            voting: Voting type ('soft' or 'hard')

        Returns:
            Voting ensemble
        """
        if self.task == "regression":
            self.ensemble = VotingRegressor(estimators=models, n_jobs=-1)
        else:
            self.ensemble = VotingClassifier(
                estimators=models, voting=voting, n_jobs=-1
            )

        logger.info(f"Created {voting} voting ensemble with {len(models)} models")
        return self.ensemble


class ProductionMLPipeline:
    """
    Production-ready ML pipeline with preprocessing, training, and evaluation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.metrics = {}

    def build_pipeline(
        self,
        model: BaseEstimator,
        use_scaling: bool = True,
        use_feature_selection: bool = False,
    ) -> Pipeline:
        """
        Build scikit-learn pipeline.

        Args:
            model: ML model
            use_scaling: Whether to use feature scaling
            use_feature_selection: Whether to use feature selection

        Returns:
            Complete pipeline
        """
        steps = []

        if use_scaling:
            steps.append(("scaler", StandardScaler()))

        if use_feature_selection:
            steps.append(("feature_selector", FeatureSelector()))

        steps.append(("model", model))

        self.pipeline = Pipeline(steps)
        logger.info(f"Built pipeline with {len(steps)} steps")
        return self.pipeline

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_mlflow: bool = True,
    ) -> dict[str, float]:
        """
        Train and evaluate pipeline.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            use_mlflow: Whether to log to MLflow

        Returns:
            Evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built yet")

        if use_mlflow:
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Train
        logger.info("Training pipeline...")
        self.pipeline.fit(X_train, y_train)

        # Predict
        y_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        task = self.config.get("task", "regression")
        if task == "regression":
            self.metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            }
        else:
            self.metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }

        logger.info(f"Evaluation metrics: {self.metrics}")

        if use_mlflow:
            mlflow.log_metrics(self.metrics)
            mlflow.end_run()

        return self.metrics

    def save_pipeline(self, path: str):
        """Save pipeline to disk."""
        joblib.dump(self.pipeline, path)
        logger.info(f"Pipeline saved to {path}")

    def load_pipeline(self, path: str):
        """Load pipeline from disk."""
        self.pipeline = joblib.load(path)
        logger.info(f"Pipeline loaded from {path}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and optimize model
    optimizer = HyperparameterOptimizer(model_type="xgb", task="regression")
    best_params = optimizer.optimize(X_train, y_train, n_trials=10)

    # Create model with best params
    model = MLModelFactory.create_model("xgb", "regression", **best_params)

    # Build and train pipeline
    config = {"task": "regression"}
    pipeline = ProductionMLPipeline(config)
    pipeline.build_pipeline(model, use_scaling=True)
    metrics = pipeline.train_and_evaluate(
        X_train, y_train, X_test, y_test, use_mlflow=False
    )

    logger.info("Scikit-learn pipeline completed successfully")
