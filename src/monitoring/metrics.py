"""
Model Monitoring and Observability
Production monitoring with drift detection and performance tracking
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from datadog import initialize, statsd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prometheus Metrics
MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["model_name", "model_version"],
)

MODEL_LATENCY = Histogram(
    "model_prediction_latency_seconds", "Model prediction latency", ["model_name"]
)

MODEL_ERROR_RATE = Gauge("model_error_rate", "Model error rate", ["model_name"])

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score", "Feature drift score", ["feature_name"]
)


class ModelMonitor:
    """
    Comprehensive model monitoring for production ML systems.
    Tracks performance, drift, and data quality.
    """

    def __init__(
        self,
        model_name: str,
        reference_data: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize model monitor.

        Args:
            model_name: Name of model to monitor
            reference_data: Reference/training data for drift detection
            config: Monitoring configuration
        """
        self.model_name = model_name
        self.reference_data = reference_data
        self.config = config or {}

        # Monitoring history
        self.prediction_history = []
        self.drift_history = []
        self.performance_history = []

        # Setup Datadog if configured
        if self.config.get("datadog_enabled", False):
            self._setup_datadog()

        logger.info(f"Initialized monitor for model: {model_name}")

    def _setup_datadog(self):
        """Setup Datadog monitoring."""
        datadog_config = self.config.get("datadog", {})

        initialize(
            api_key=datadog_config.get("api_key"), app_key=datadog_config.get("app_key")
        )

        logger.info("Datadog monitoring enabled")

    def log_prediction(
        self,
        features: np.ndarray,
        _predictions: np.ndarray,
        model_version: str,
        latency: float,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log prediction for monitoring.

        Args:
            features: Input features
            predictions: Model predictions
            model_version: Version of model used
            latency: Prediction latency
            metadata: Additional metadata
        """
        prediction_log = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_version": model_version,
            "num_samples": len(features),
            "latency": latency,
            "metadata": metadata or {},
        }

        self.prediction_history.append(prediction_log)

        # Update Prometheus metrics
        MODEL_PREDICTIONS.labels(
            model_name=self.model_name, model_version=model_version
        ).inc(len(features))

        MODEL_LATENCY.labels(model_name=self.model_name).observe(latency)

        # Send to Datadog
        if self.config.get("datadog_enabled", False):
            statsd.increment(
                "model.predictions",
                tags=[f"model:{self.model_name}", f"version:{model_version}"],
            )
            statsd.histogram(
                "model.latency", latency, tags=[f"model:{self.model_name}"]
            )

        logger.info(
            f"Logged prediction - Model: {self.model_name}, "
            f"Samples: {len(features)}, Latency: {latency:.3f}s"
        )

    def detect_data_drift(
        self, current_data: pd.DataFrame, threshold: float = 0.05
    ) -> dict[str, Any]:
        """
        Detect data drift using statistical tests.

        Args:
            current_data: Current production data
            threshold: P-value threshold for drift detection

        Returns:
            Drift detection results
        """
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drifted_features": [],
            "feature_scores": {},
        }

        numeric_features = self.reference_data.select_dtypes(include=["number"]).columns

        for feature in numeric_features:
            if feature not in current_data.columns:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[feature].dropna(), current_data[feature].dropna()
            )

            drift_results["feature_scores"][feature] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drifted": p_value < threshold,
            }

            if p_value < threshold:
                drift_results["drift_detected"] = True
                drift_results["drifted_features"].append(feature)

                logger.warning(
                    f"Drift detected in feature '{feature}': "
                    f"KS={ks_stat:.4f}, p={p_value:.4f}"
                )

                # Update Prometheus metric
                FEATURE_DRIFT_SCORE.labels(feature_name=feature).set(ks_stat)

                # Send to Datadog
                if self.config.get("datadog_enabled", False):
                    statsd.gauge(
                        "model.drift.score",
                        ks_stat,
                        tags=[f"model:{self.model_name}", f"feature:{feature}"],
                    )

        self.drift_history.append(drift_results)

        logger.info(
            f"Drift detection complete - "
            f"Drifted features: {len(drift_results['drifted_features'])}/{len(numeric_features)}"
        )

        return drift_results

    def detect_target_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Detect drift in model predictions.

        Args:
            reference_predictions: Historical predictions
            current_predictions: Current predictions
            threshold: P-value threshold

        Returns:
            Target drift results
        """
        ks_stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)

        drift_detected = p_value < threshold

        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected,
            "ks_statistic": ks_stat,
            "p_value": p_value,
        }

        if drift_detected:
            logger.warning(f"Target drift detected: KS={ks_stat:.4f}, p={p_value:.4f}")

        return results

    def calculate_performance_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, task: str = "regression"
    ) -> dict[str, float]:
        """
        Calculate and log performance metrics.

        Args:
            y_true: True labels
            y_pred: Predictions
            task: Task type (regression/classification)

        Returns:
            Performance metrics
        """
        if task == "regression":
            metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
            }

            # Update Prometheus
            MODEL_ERROR_RATE.labels(model_name=self.model_name).set(metrics["rmse"])

        else:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1": f1_score(y_true, y_pred, average="weighted"),
            }

            # Update Prometheus
            MODEL_ERROR_RATE.labels(model_name=self.model_name).set(
                1 - metrics["accuracy"]
            )

        # Log metrics
        performance_log = {"timestamp": datetime.now().isoformat(), "metrics": metrics}

        self.performance_history.append(performance_log)

        # Send to Datadog
        if self.config.get("datadog_enabled", False):
            for metric_name, metric_value in metrics.items():
                statsd.gauge(
                    f"model.performance.{metric_name}",
                    metric_value,
                    tags=[f"model:{self.model_name}"],
                )

        logger.info(f"Performance metrics: {metrics}")

        return metrics

    def generate_drift_report(self, current_data: pd.DataFrame, output_path: str):
        """
        Generate comprehensive drift report using Evidently.

        Args:
            current_data: Current production data
            output_path: Path to save report
        """
        # Create Evidently report
        report = Report(
            metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
            ]
        )

        report.run(reference_data=self.reference_data, current_data=current_data)

        # Save report
        report.save_html(output_path)

        logger.info(f"Drift report saved to {output_path}")

    def generate_test_suite(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """
        Run Evidently test suite for data quality and drift.

        Args:
            current_data: Current production data

        Returns:
            Test results
        """
        test_suite = TestSuite(
            tests=[
                TestNumberOfDriftedColumns(),
                TestShareOfDriftedColumns(),
            ]
        )

        test_suite.run(reference_data=self.reference_data, current_data=current_data)

        results = test_suite.as_dict()

        logger.info(f"Test suite completed: {results}")

        return results

    def check_data_quality(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Check data quality metrics.

        Args:
            data: Data to check

        Returns:
            Quality metrics
        """
        quality_metrics = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(data),
            "num_features": len(data.columns),
            "null_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
            "duplicate_count": data.duplicated().sum(),
            "duplicate_percentage": (data.duplicated().sum() / len(data) * 100),
        }

        # Check for anomalies
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = (
                (data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))
            ).sum()
            quality_metrics[f"{col}_outliers"] = int(outliers)

        logger.info(f"Data quality check: {quality_metrics}")

        return quality_metrics

    def alert_if_threshold_exceeded(
        self,
        metric_name: str,
        metric_value: float,
        threshold: float,
        comparison: str = "greater",
    ):
        """
        Send alert if metric exceeds threshold.

        Args:
            metric_name: Name of metric
            metric_value: Current value
            threshold: Threshold value
            comparison: Comparison type ('greater' or 'less')
        """
        alert_triggered = False

        if (
            comparison == "greater"
            and metric_value > threshold
            or comparison == "less"
            and metric_value < threshold
        ):
            alert_triggered = True

        if alert_triggered:
            alert_message = (
                f"ALERT: {metric_name} = {metric_value:.4f} "
                f"(threshold: {threshold:.4f})"
            )

            logger.warning(alert_message)

            # Send to monitoring systems
            if self.config.get("datadog_enabled", False):
                statsd.event(
                    title=f"Model Alert: {self.model_name}",
                    text=alert_message,
                    alert_type="warning",
                    tags=[f"model:{self.model_name}"],
                )

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for monitoring period.

        Returns:
            Summary statistics
        """
        total_predictions = sum(log["num_samples"] for log in self.prediction_history)

        avg_latency = (
            np.mean([log["latency"] for log in self.prediction_history])
            if self.prediction_history
            else 0
        )

        drift_events = sum(1 for log in self.drift_history if log["drift_detected"])

        summary = {
            "model_name": self.model_name,
            "monitoring_period": {
                "start": self.prediction_history[0]["timestamp"]
                if self.prediction_history
                else None,
                "end": datetime.now().isoformat(),
            },
            "predictions": {"total": total_predictions, "avg_latency": avg_latency},
            "drift": {
                "total_checks": len(self.drift_history),
                "drift_events": drift_events,
            },
            "performance_checks": len(self.performance_history),
        }

        logger.info(f"Summary statistics: {summary}")

        return summary


def main():
    """Example usage of model monitoring."""
    # Create dummy data
    reference_data = pd.DataFrame(
        {
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "feature3": np.random.randn(1000),
        }
    )

    # Initialize monitor
    monitor = ModelMonitor("demo-model", reference_data)

    # Simulate predictions
    features = np.random.randn(100, 3)
    predictions = np.random.randn(100)

    monitor.log_prediction(
        features=features, predictions=predictions, model_version="v1.0", latency=0.05
    )

    # Simulate current data with drift
    current_data = pd.DataFrame(
        {
            "feature1": np.random.randn(1000) + 0.5,  # Shifted distribution
            "feature2": np.random.randn(1000),
            "feature3": np.random.randn(1000),
        }
    )

    # Detect drift
    monitor.detect_data_drift(current_data)

    # Check data quality
    monitor.check_data_quality(current_data)

    # Get summary
    monitor.get_summary_statistics()

    logger.info("Monitoring example completed")


if __name__ == "__main__":
    main()
