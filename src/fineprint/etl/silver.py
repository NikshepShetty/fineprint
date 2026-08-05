from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array_contains, col

# Must match the keys in data_generator/schemas.py CLAUSE_RISK_WEIGHTS.
CLAUSE_TYPES = ["auto_renewal", "unlimited_liability", "vague_termination", "missing_indemnity"]


def build_invoices_silver(invoices_bronze: DataFrame, vendors_bronze: DataFrame) -> DataFrame:
    vendor_cols = vendors_bronze.select(
        col("vendor_id"),
        col("name").alias("vendor_name"),
        col("payment_behavior").alias("vendor_payment_behavior"),
        col("avg_days_late_historical").alias("vendor_avg_days_late_historical"),
        col("industry").alias("vendor_industry"),
    )

    return invoices_bronze.join(vendor_cols, on="vendor_id", how="left")


def build_contracts_silver(contracts_bronze: DataFrame) -> DataFrame:
    df = contracts_bronze
    for clause in CLAUSE_TYPES:
        df = df.withColumn(f"has_{clause}", array_contains(col("clauses_present"), clause))
    return df


def run_silver(spark: SparkSession, bronze_path: Path, silver_path: Path) -> None:
    silver_path.mkdir(parents=True, exist_ok=True)

    vendors_bronze = spark.read.parquet(str(bronze_path / "vendors"))
    invoices_bronze = spark.read.parquet(str(bronze_path / "invoices"))
    contracts_bronze = spark.read.parquet(str(bronze_path / "contracts"))

    invoices_silver = build_invoices_silver(invoices_bronze, vendors_bronze)
    contracts_silver = build_contracts_silver(contracts_bronze)

    invoices_silver.write.mode("overwrite").parquet(str(silver_path / "invoices"))
    contracts_silver.write.mode("overwrite").parquet(str(silver_path / "contracts"))
