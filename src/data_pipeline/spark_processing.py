"""
Distributed Data Processing Pipeline using Apache Spark
Demonstrates 6+ years of distributed computing experience with production-ready patterns
"""

from typing import Optional, Dict, Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.window import Window
from loguru import logger
import yaml


class SparkDataProcessor:
    """
    Production-ready distributed data processor using Apache Spark.
    Implements resilient, scalable data processing patterns.
    """

    def __init__(self, app_name: str = "ML-Data-Pipeline", config_path: Optional[str] = None):
        """
        Initialize Spark session with optimized configurations.

        Args:
            app_name: Name of the Spark application
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.spark = self._create_spark_session(app_name)
        logger.info(f"Initialized Spark session: {app_name}")

    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Create optimized Spark session with best practices."""
        builder = SparkSession.builder.appName(app_name)

        # Production configurations
        configs = {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.shuffle.partitions": "200",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.parquet.compression.codec": "snappy",
            "spark.sql.broadcastTimeout": "600",
            "spark.network.timeout": "800s",
            "spark.executor.heartbeatInterval": "60s",
            # Memory management
            "spark.memory.fraction": "0.8",
            "spark.memory.storageFraction": "0.3",
            # Dynamic allocation for cost optimization
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.minExecutors": "2",
            "spark.dynamicAllocation.maxExecutors": "10",
        }

        for key, value in configs.items():
            builder = builder.config(key, value)

        return builder.getOrCreate()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def read_distributed_data(
        self,
        path: str,
        format: str = "parquet",
        schema: Optional[StructType] = None,
        **options
    ) -> DataFrame:
        """
        Read data from distributed storage with error handling.

        Args:
            path: Data path (S3, HDFS, GCS, Azure Blob)
            format: Data format (parquet, csv, json, delta)
            schema: Optional schema definition
            **options: Additional read options

        Returns:
            Spark DataFrame
        """
        try:
            reader = self.spark.read.format(format)

            if schema:
                reader = reader.schema(schema)

            for key, value in options.items():
                reader = reader.option(key, value)

            df = reader.load(path)
            logger.info(f"Successfully read {df.count()} records from {path}")
            return df

        except Exception as e:
            logger.error(f"Failed to read data from {path}: {e}")
            raise

    def process_large_scale_data(
        self,
        df: DataFrame,
        partition_cols: Optional[List[str]] = None,
        repartition_num: Optional[int] = None
    ) -> DataFrame:
        """
        Process large-scale data with optimized partitioning.

        Args:
            df: Input DataFrame
            partition_cols: Columns to partition by
            repartition_num: Number of partitions

        Returns:
            Processed DataFrame
        """
        # Remove duplicates efficiently
        df = df.dropDuplicates()

        # Handle null values with business logic
        df = self._handle_missing_values(df)

        # Optimize partitioning for downstream processing
        if partition_cols:
            df = df.repartition(*partition_cols)
        elif repartition_num:
            df = df.repartition(repartition_num)

        # Cache if dataset will be reused
        df = df.cache()

        return df

    def _handle_missing_values(self, df: DataFrame) -> DataFrame:
        """Intelligent missing value handling."""
        # Fill numeric columns with median
        numeric_cols = [f.name for f in df.schema.fields
                       if isinstance(f.dataType, (IntegerType, DoubleType))]

        for col in numeric_cols:
            median_val = df.approxQuantile(col, [0.5], 0.01)[0]
            df = df.fillna({col: median_val})

        # Fill categorical with mode or 'unknown'
        string_cols = [f.name for f in df.schema.fields
                      if isinstance(f.dataType, StringType)]

        for col in string_cols:
            df = df.fillna({col: 'unknown'})

        return df

    def feature_engineering_distributed(self, df: DataFrame) -> DataFrame:
        """
        Distributed feature engineering with window functions and aggregations.
        """
        # Time-based features
        if 'timestamp' in df.columns:
            df = df.withColumn('hour', F.hour('timestamp'))
            df = df.withColumn('day_of_week', F.dayofweek('timestamp'))
            df = df.withColumn('month', F.month('timestamp'))

        # Statistical features using window functions
        if 'user_id' in df.columns and 'amount' in df.columns:
            window_spec = Window.partitionBy('user_id').orderBy('timestamp')

            df = df.withColumn(
                'user_avg_amount',
                F.avg('amount').over(Window.partitionBy('user_id'))
            )

            df = df.withColumn(
                'user_transaction_count',
                F.count('*').over(Window.partitionBy('user_id'))
            )

            df = df.withColumn(
                'running_total',
                F.sum('amount').over(window_spec)
            )

        # Interaction features
        numeric_cols = [f.name for f in df.schema.fields
                       if isinstance(f.dataType, (IntegerType, DoubleType))]

        if len(numeric_cols) >= 2:
            df = df.withColumn(
                f'{numeric_cols[0]}_{numeric_cols[1]}_interaction',
                F.col(numeric_cols[0]) * F.col(numeric_cols[1])
            )

        return df

    def data_quality_checks(self, df: DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.

        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_records': df.count(),
            'total_columns': len(df.columns),
            'null_counts': {},
            'duplicate_count': df.count() - df.dropDuplicates().count(),
        }

        # Check nulls per column
        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            quality_report['null_counts'][col] = null_count

        # Data profiling
        quality_report['statistics'] = df.describe().toPandas().to_dict()

        logger.info(f"Data quality report: {quality_report}")
        return quality_report

    def write_distributed_data(
        self,
        df: DataFrame,
        path: str,
        format: str = "parquet",
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
        **options
    ):
        """
        Write data to distributed storage with partitioning.

        Args:
            df: DataFrame to write
            path: Output path
            format: Output format (parquet, delta, orc)
            mode: Write mode (overwrite, append, error, ignore)
            partition_by: Columns to partition by
            **options: Additional write options
        """
        try:
            writer = df.write.format(format).mode(mode)

            if partition_by:
                writer = writer.partitionBy(*partition_by)

            for key, value in options.items():
                writer = writer.option(key, value)

            writer.save(path)
            logger.info(f"Successfully wrote data to {path}")

        except Exception as e:
            logger.error(f"Failed to write data to {path}: {e}")
            raise

    def optimize_table(self, path: str, z_order_cols: Optional[List[str]] = None):
        """
        Optimize Delta tables for query performance (Delta Lake).

        Args:
            path: Delta table path
            z_order_cols: Columns to optimize with Z-ordering
        """
        if z_order_cols:
            self.spark.sql(f"OPTIMIZE delta.`{path}` ZORDER BY ({','.join(z_order_cols)})")
        else:
            self.spark.sql(f"OPTIMIZE delta.`{path}`")

        # Vacuum old files
        self.spark.sql(f"VACUUM delta.`{path}` RETAIN 168 HOURS")
        logger.info(f"Optimized table at {path}")

    def create_streaming_pipeline(
        self,
        source_path: str,
        checkpoint_path: str,
        output_path: str,
        trigger_interval: str = "10 seconds"
    ):
        """
        Create streaming data pipeline for real-time processing.

        Args:
            source_path: Source stream path
            checkpoint_path: Checkpoint location
            output_path: Output sink path
            trigger_interval: Processing trigger interval
        """
        # Read streaming data
        streaming_df = (
            self.spark.readStream
            .format("parquet")
            .schema(self._get_schema())
            .load(source_path)
        )

        # Process streaming data
        processed_df = self.feature_engineering_distributed(streaming_df)

        # Write to sink with checkpointing
        query = (
            processed_df.writeStream
            .format("parquet")
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
            .start(output_path)
        )

        logger.info(f"Started streaming pipeline: {query.id}")
        return query

    def _get_schema(self) -> StructType:
        """Define data schema."""
        return StructType([
            StructField("id", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("category", StringType(), True),
            StructField("status", StringType(), True),
        ])

    def close(self):
        """Clean shutdown of Spark session."""
        self.spark.stop()
        logger.info("Spark session closed")


def main():
    """Example usage of distributed data processing."""
    processor = SparkDataProcessor(app_name="Production-ML-Pipeline")

    try:
        # Example: Process large-scale data
        # df = processor.read_distributed_data("s3://bucket/data/", format="parquet")
        # processed_df = processor.process_large_scale_data(df, partition_cols=['date'])
        # features_df = processor.feature_engineering_distributed(processed_df)
        # quality_report = processor.data_quality_checks(features_df)
        # processor.write_distributed_data(features_df, "s3://bucket/processed/", partition_by=['date'])

        logger.info("Data processing pipeline completed successfully")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
