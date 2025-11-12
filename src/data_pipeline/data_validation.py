"""
Data Validation and Quality Assurance
Production-ready data validation using Great Expectations and custom rules
"""

import json
from datetime import datetime
from typing import Any

import great_expectations as ge
import pandas as pd
from great_expectations.data_context import BaseDataContext
from loguru import logger


class DataValidator:
    """
    Comprehensive data validation for ML pipelines.
    Ensures data quality before training and inference.
    """

    def __init__(self, context_root_dir: str | None = None):
        """
        Initialize data validator with Great Expectations context.

        Args:
            context_root_dir: Root directory for GE context
        """
        self.context = self._setup_context(context_root_dir)
        self.validation_results = []

    def _setup_context(self, root_dir: str | None) -> BaseDataContext:
        """Setup Great Expectations context."""
        try:
            if root_dir:
                context = ge.data_context.DataContext(root_dir)
            else:
                context = ge.data_context.DataContext()
            return context
        except Exception as e:
            logger.warning(f"Could not setup GE context: {e}")
            return None

    def validate_schema(
        self, df: pd.DataFrame, expected_schema: dict[str, str]
    ) -> dict[str, Any]:
        """
        Validate DataFrame schema against expected schema.

        Args:
            df: Input DataFrame
            expected_schema: Dictionary of {column: dtype}

        Returns:
            Validation results
        """
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Check column existence
        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)

        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            results["passed"] = False
            results["errors"].append(f"Missing columns: {missing_cols}")

        if extra_cols:
            results["warnings"].append(f"Extra columns: {extra_cols}")

        # Check data types
        for col, expected_dtype in expected_schema.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._compatible_dtypes(actual_dtype, expected_dtype):
                    results["passed"] = False
                    results["errors"].append(
                        f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                    )

        self._log_results(results, "Schema Validation")
        return results

    def _compatible_dtypes(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        type_mappings = {
            "int64": ["int64", "int32", "int16", "int"],
            "float64": ["float64", "float32", "float"],
            "object": ["object", "string", "str"],
            "datetime64": ["datetime64", "datetime"],
            "bool": ["bool", "boolean"],
        }

        for key, values in type_mappings.items():
            if actual.startswith(key) and expected in values:
                return True
        return False

    def validate_data_quality(
        self, df: pd.DataFrame, config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Comprehensive data quality validation.

        Args:
            df: Input DataFrame
            config: Validation configuration with rules

        Returns:
            Validation results
        """
        results = {
            "passed": True,
            "checks": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Check for nulls
        if "max_null_percentage" in config:
            null_check = self._check_null_percentage(df, config["max_null_percentage"])
            results["checks"].append(null_check)
            if not null_check["passed"]:
                results["passed"] = False

        # Check for duplicates
        if "max_duplicate_percentage" in config:
            dup_check = self._check_duplicates(df, config["max_duplicate_percentage"])
            results["checks"].append(dup_check)
            if not dup_check["passed"]:
                results["passed"] = False

        # Check value ranges
        if "value_ranges" in config:
            range_check = self._check_value_ranges(df, config["value_ranges"])
            results["checks"].append(range_check)
            if not range_check["passed"]:
                results["passed"] = False

        # Check categorical values
        if "allowed_values" in config:
            cat_check = self._check_categorical_values(df, config["allowed_values"])
            results["checks"].append(cat_check)
            if not cat_check["passed"]:
                results["passed"] = False

        # Check data freshness
        if "timestamp_column" in config:
            freshness_check = self._check_data_freshness(
                df, config["timestamp_column"], config.get("max_age_hours", 24)
            )
            results["checks"].append(freshness_check)
            if not freshness_check["passed"]:
                results["passed"] = False

        self._log_results(results, "Data Quality Validation")
        self.validation_results.append(results)
        return results

    def _check_null_percentage(
        self, df: pd.DataFrame, max_percentage: float
    ) -> dict[str, Any]:
        """Check null percentage per column."""
        check = {"name": "null_percentage", "passed": True, "details": {}}

        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            check["details"][col] = null_pct

            if null_pct > max_percentage:
                check["passed"] = False
                logger.warning(
                    f"Column '{col}' has {null_pct:.2f}% nulls, "
                    f"exceeds threshold of {max_percentage}%"
                )

        return check

    def _check_duplicates(
        self, df: pd.DataFrame, max_percentage: float
    ) -> dict[str, Any]:
        """Check duplicate percentage."""
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df)) * 100

        passed = dup_pct <= max_percentage

        if not passed:
            logger.warning(
                f"Found {dup_count} duplicates ({dup_pct:.2f}%), "
                f"exceeds threshold of {max_percentage}%"
            )

        return {
            "name": "duplicate_percentage",
            "passed": passed,
            "duplicate_count": int(dup_count),
            "duplicate_percentage": dup_pct,
        }

    def _check_value_ranges(
        self, df: pd.DataFrame, ranges: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Check if numeric values are within expected ranges."""
        check = {"name": "value_ranges", "passed": True, "details": {}}

        for col, range_config in ranges.items():
            if col not in df.columns:
                continue

            min_val = range_config.get("min", float("-inf"))
            max_val = range_config.get("max", float("inf"))

            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            out_of_range_pct = (out_of_range / len(df)) * 100

            check["details"][col] = {
                "out_of_range_count": int(out_of_range),
                "out_of_range_percentage": out_of_range_pct,
            }

            if out_of_range > 0:
                check["passed"] = False
                logger.warning(
                    f"Column '{col}' has {out_of_range} values "
                    f"outside range [{min_val}, {max_val}]"
                )

        return check

    def _check_categorical_values(
        self, df: pd.DataFrame, allowed: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Check if categorical values are within allowed set."""
        check = {"name": "categorical_values", "passed": True, "details": {}}

        for col, allowed_vals in allowed.items():
            if col not in df.columns:
                continue

            actual_vals = set(df[col].dropna().unique())
            allowed_set = set(allowed_vals)

            invalid_vals = actual_vals - allowed_set
            invalid_count = df[col].isin(invalid_vals).sum()

            check["details"][col] = {
                "invalid_values": list(invalid_vals),
                "invalid_count": int(invalid_count),
            }

            if invalid_vals:
                check["passed"] = False
                logger.warning(f"Column '{col}' has invalid values: {invalid_vals}")

        return check

    def _check_data_freshness(
        self, df: pd.DataFrame, timestamp_col: str, max_age_hours: int
    ) -> dict[str, Any]:
        """Check if data is fresh enough."""
        if timestamp_col not in df.columns:
            return {
                "name": "data_freshness",
                "passed": False,
                "error": f"Timestamp column '{timestamp_col}' not found",
            }

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        max_timestamp = df[timestamp_col].max()
        age_hours = (datetime.now() - max_timestamp).total_seconds() / 3600

        passed = age_hours <= max_age_hours

        if not passed:
            logger.warning(
                f"Data is {age_hours:.2f} hours old, "
                f"exceeds threshold of {max_age_hours} hours"
            )

        return {
            "name": "data_freshness",
            "passed": passed,
            "max_timestamp": max_timestamp.isoformat(),
            "age_hours": age_hours,
        }

    def validate_feature_distributions(
        self, train_df: pd.DataFrame, inference_df: pd.DataFrame, threshold: float = 0.1
    ) -> dict[str, Any]:
        """
        Detect distribution drift between training and inference data.

        Args:
            train_df: Training data
            inference_df: Inference data
            threshold: KS test p-value threshold

        Returns:
            Drift detection results
        """
        from scipy import stats

        results = {"passed": True, "drifted_features": [], "checks": {}}

        numeric_cols = train_df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if col not in inference_df.columns:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(
                train_df[col].dropna(), inference_df[col].dropna()
            )

            drifted = p_value < threshold

            results["checks"][col] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drifted": drifted,
            }

            if drifted:
                results["passed"] = False
                results["drifted_features"].append(col)
                logger.warning(
                    f"Feature '{col}' has drifted (KS={ks_stat:.4f}, p={p_value:.4f})"
                )

        self._log_results(results, "Distribution Drift Detection")
        return results

    def _log_results(self, results: dict[str, Any], check_name: str):
        """Log validation results."""
        status = "PASSED" if results.get("passed", False) else "FAILED"
        logger.info(f"{check_name}: {status}")

        if not results.get("passed", False):
            logger.error(f"Validation details: {json.dumps(results, indent=2)}")

    def generate_validation_report(self, output_path: str):
        """Generate comprehensive validation report."""
        report = {
            "validation_summary": {
                "total_checks": len(self.validation_results),
                "passed": sum(1 for r in self.validation_results if r["passed"]),
                "failed": sum(1 for r in self.validation_results if not r["passed"]),
            },
            "detailed_results": self.validation_results,
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")
        return report


if __name__ == "__main__":
    # Example usage
    validator = DataValidator()

    # Create sample data
    df = pd.DataFrame(
        {
            "id": range(100),
            "feature1": [1.0] * 100,
            "feature2": ["A"] * 50 + ["B"] * 50,
            "timestamp": pd.date_range("2025-01-01", periods=100),
        }
    )

    # Validate schema
    expected_schema = {
        "id": "int64",
        "feature1": "float64",
        "feature2": "object",
        "timestamp": "datetime64",
    }
    schema_results = validator.validate_schema(df, expected_schema)

    # Validate data quality
    quality_config = {
        "max_null_percentage": 5.0,
        "max_duplicate_percentage": 1.0,
        "value_ranges": {"feature1": {"min": 0, "max": 10}},
        "allowed_values": {"feature2": ["A", "B", "C"]},
    }
    quality_results = validator.validate_data_quality(df, quality_config)

    logger.info("Data validation completed")
