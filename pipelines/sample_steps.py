"""
Sample execution step implementations demonstrating the ExecutionStep framework.

This module provides concrete examples of how to create custom execution steps
for ML pipelines using the ExecutionStep base class.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
from pathlib import Path

from pipelines.execution_step import ExecutionStep


class DataLoadingStep(ExecutionStep):
    """
    Step for loading data from various sources.

    This step demonstrates:
    - Configuration usage
    - Validation logic
    - Context updates
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loading step.

        Args:
            config: Must contain 'data_path' key
        """
        super().__init__(
            name="data_loading",
            description="Load dataset from specified path",
            config=config
        )

    def validate(self) -> bool:
        """Validate that data path is provided."""
        if "data_path" not in self.config:
            self.logger.error("data_path not provided in config")
            return False

        data_path = Path(self.config["data_path"])
        if not data_path.exists():
            self.logger.error(f"Data path does not exist: {data_path}")
            return False

        return True

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load data from file.

        Args:
            context: Pipeline context

        Returns:
            Dictionary with loaded dataframe
        """
        data_path = self.config["data_path"]
        file_format = self.config.get("format", "csv")

        self.logger.info(f"Loading data from {data_path}")

        if file_format == "csv":
            df = pd.read_csv(data_path)
        elif file_format == "parquet":
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        self.metadata["num_rows"] = len(df)
        self.metadata["num_columns"] = len(df.columns)
        self.metadata["columns"] = list(df.columns)

        self.logger.info(
            f"Loaded {len(df)} rows and {len(df.columns)} columns"
        )

        return {"dataframe": df}


class DataValidationStep(ExecutionStep):
    """
    Step for validating data quality.

    Checks for:
    - Missing values
    - Duplicates
    - Data type consistency
    - Value ranges
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validation step.

        Args:
            config: Optional validation thresholds
        """
        super().__init__(
            name="data_validation",
            description="Validate data quality and consistency",
            dependencies=["data_loading"],
            config=config
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the dataframe.

        Args:
            context: Must contain 'dataframe' from previous step

        Returns:
            Dictionary with validation results
        """
        df = context["dataframe"]

        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100

        # Check for duplicates
        num_duplicates = df.duplicated().sum()

        # Check numeric ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_stats = df[numeric_cols].describe()

        validation_results = {
            "missing_values": missing_counts.to_dict(),
            "missing_percent": missing_percent.to_dict(),
            "num_duplicates": num_duplicates,
            "numeric_stats": numeric_stats.to_dict(),
            "passed": True
        }

        # Apply validation rules
        max_missing_percent = self.config.get("max_missing_percent", 20)
        if missing_percent.max() > max_missing_percent:
            validation_results["passed"] = False
            self.logger.warning(
                f"Missing values exceed threshold: "
                f"{missing_percent.max():.2f}% > {max_missing_percent}%"
            )

        max_duplicates = self.config.get("max_duplicates", 0)
        if num_duplicates > max_duplicates:
            validation_results["passed"] = False
            self.logger.warning(
                f"Duplicates exceed threshold: "
                f"{num_duplicates} > {max_duplicates}"
            )

        self.metadata["validation_passed"] = validation_results["passed"]
        self.metadata["num_duplicates"] = num_duplicates

        if validation_results["passed"]:
            self.logger.info("Data validation passed all checks")
        else:
            self.logger.warning("Data validation failed some checks")

        return {"validation_results": validation_results}


class FeatureEngineeringStep(ExecutionStep):
    """
    Step for creating and transforming features.

    Demonstrates:
    - Feature creation
    - Feature scaling
    - Feature selection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineering step.

        Args:
            config: Feature engineering configuration
        """
        super().__init__(
            name="feature_engineering",
            description="Engineer features for model training",
            dependencies=["data_loading"],
            config=config
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create engineered features.

        Args:
            context: Must contain 'dataframe'

        Returns:
            Dictionary with engineered dataframe
        """
        df = context["dataframe"].copy()

        # Get numeric columns (excluding target)
        target_col = self.config.get("target_column", "target")
        feature_cols = [col for col in df.columns if col != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        # Create interaction features if requested
        if self.config.get("create_interactions", False):
            n_interactions = min(3, len(numeric_cols))  # Limit interactions
            for i in range(n_interactions - 1):
                for j in range(i + 1, n_interactions):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    new_col = f"{col1}_x_{col2}"
                    df[new_col] = df[col1] * df[col2]
                    self.logger.debug(f"Created interaction feature: {new_col}")

        # Create polynomial features if requested
        if self.config.get("create_polynomials", False):
            poly_degree = self.config.get("polynomial_degree", 2)
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                df[f"{col}_squared"] = df[col] ** 2
                if poly_degree >= 3:
                    df[f"{col}_cubed"] = df[col] ** 3

        # Create aggregation features
        if self.config.get("create_aggregations", True):
            df["feature_mean"] = df[numeric_cols].mean(axis=1)
            df["feature_std"] = df[numeric_cols].std(axis=1)
            df["feature_min"] = df[numeric_cols].min(axis=1)
            df["feature_max"] = df[numeric_cols].max(axis=1)

        original_features = len(feature_cols)
        new_features = len([col for col in df.columns if col != target_col])

        self.metadata["original_features"] = original_features
        self.metadata["engineered_features"] = new_features
        self.metadata["features_added"] = new_features - original_features

        self.logger.info(
            f"Feature engineering complete: "
            f"{original_features} -> {new_features} features "
            f"(+{new_features - original_features})"
        )

        return {"dataframe": df, "feature_columns": [col for col in df.columns if col != target_col]}


class ModelTrainingStep(ExecutionStep):
    """
    Step for training machine learning models.

    This is a simplified example showing how model training
    can be encapsulated as an execution step.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model training step.

        Args:
            config: Model configuration
        """
        super().__init__(
            name="model_training",
            description="Train machine learning model",
            dependencies=["data_loading", "feature_engineering"],
            config=config
        )

    def pre_execute(self, context: Dict[str, Any]) -> None:
        """Set up model training environment."""
        self.logger.info("Setting up model training environment")
        # Could initialize GPU, set random seeds, etc.
        np.random.seed(self.config.get("random_seed", 42))

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            context: Must contain 'dataframe' and optionally 'feature_columns'

        Returns:
            Dictionary with trained model information
        """
        df = context["dataframe"]
        target_col = self.config.get("target_column", "target")

        # Get features
        if "feature_columns" in context:
            feature_cols = context["feature_columns"]
        else:
            feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols]
        y = df[target_col]

        # Simple example: just compute statistics
        # In a real implementation, this would train an actual model
        model_type = self.config.get("model_type", "linear")

        self.logger.info(f"Training {model_type} model with {len(feature_cols)} features")

        # Mock training metrics
        model_info = {
            "model_type": model_type,
            "num_features": len(feature_cols),
            "num_samples": len(X),
            "feature_names": feature_cols,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
        }

        self.metadata.update(model_info)

        self.logger.info(
            f"Model training complete: {len(X)} samples, {len(feature_cols)} features"
        )

        return {"model_info": model_info}

    def post_execute(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Clean up after model training."""
        self.logger.info("Model training post-processing complete")
        # Could save model, clear GPU memory, etc.


# Example pipeline executor using these steps
class StepPipeline:
    """
    Simple pipeline executor for running execution steps in order.

    This demonstrates how to orchestrate multiple ExecutionStep instances.
    """

    def __init__(self, steps: list[ExecutionStep]):
        """
        Initialize pipeline with steps.

        Args:
            steps: List of ExecutionStep instances
        """
        self.steps = steps
        self.context: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """
        Execute all steps in sequence.

        Returns:
            Final pipeline context with all results
        """
        completed_steps = []

        for step in self.steps:
            # Check dependencies
            if not step.can_execute(completed_steps):
                raise RuntimeError(
                    f"Cannot execute step '{step.name}': "
                    f"dependencies not met. Required: {step.dependencies}, "
                    f"Completed: {completed_steps}"
                )

            # Run step
            result = step.run(self.context)

            # Update context with results
            self.context.update(result)

            # Mark as completed
            completed_steps.append(step.name)

        return self.context

    def get_summary(self) -> list[Dict[str, Any]]:
        """
        Get summary of all steps.

        Returns:
            List of step summaries
        """
        return [step.get_summary() for step in self.steps]


# Example usage function
def run_example_pipeline(data_path: str = "data/raw/sample_data.csv"):
    """
    Run an example pipeline with the sample steps.

    Args:
        data_path: Path to the dataset

    Returns:
        Pipeline results
    """
    # Configure steps
    steps = [
        DataLoadingStep(config={
            "data_path": data_path,
            "format": "csv"
        }),
        DataValidationStep(config={
            "max_missing_percent": 5,
            "max_duplicates": 0
        }),
        FeatureEngineeringStep(config={
            "target_column": "target",
            "create_interactions": False,
            "create_polynomials": False,
            "create_aggregations": True
        }),
        ModelTrainingStep(config={
            "model_type": "gradient_boosting",
            "target_column": "target",
            "random_seed": 42
        })
    ]

    # Create and run pipeline
    pipeline = StepPipeline(steps)

    print("=" * 70)
    print("Running Sample Execution Pipeline")
    print("=" * 70)

    results = pipeline.run()

    print("\n" + "=" * 70)
    print("Pipeline Execution Summary")
    print("=" * 70)

    for step_summary in pipeline.get_summary():
        print(f"\n{step_summary['name'].upper()}")
        print("-" * 70)
        print(f"Status: {step_summary['status']}")
        print(f"Execution Time: {step_summary['execution_time']:.3f}s" if step_summary['execution_time'] else "N/A")
        if step_summary['metadata']:
            print("Metadata:")
            for key, value in step_summary['metadata'].items():
                print(f"  - {key}: {value}")

    return results


if __name__ == "__main__":
    # Run the example pipeline
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_example_pipeline()
