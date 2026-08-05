from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, dayofweek

from fineprint.etl.silver import CLAUSE_TYPES


def build_invoice_features_gold(invoices_silver: DataFrame) -> DataFrame:
    return invoices_silver.select(
        col("invoice_id"),
        col("vendor_id"),
        col("amount"),
        col("payment_terms_days"),
        dayofweek(col("issue_date")).alias("issue_day_of_week"),
        col("vendor_avg_days_late_historical"),
        col("vendor_payment_behavior"),
        col("vendor_industry"),
        col("days_late"),
        col("paid_late"),
    )


def build_contract_features_gold(contracts_silver: DataFrame) -> DataFrame:
    flag_columns = [col(f"has_{clause}") for clause in CLAUSE_TYPES]

    df = contracts_silver.select(
        col("contract_id"),
        *flag_columns,
        col("risk_score"),
    )

    num_risky_expr = sum(col(f"has_{clause}").cast("int") for clause in CLAUSE_TYPES)
    return df.withColumn("num_risky_clauses", num_risky_expr)


def run_gold(spark: SparkSession, silver_path: Path, gold_path: Path) -> None:
    gold_path.mkdir(parents=True, exist_ok=True)

    invoices_silver = spark.read.parquet(str(silver_path / "invoices"))
    contracts_silver = spark.read.parquet(str(silver_path / "contracts"))

    invoice_features = build_invoice_features_gold(invoices_silver)
    contract_features = build_contract_features_gold(contracts_silver)

    invoice_features.write.mode("overwrite").parquet(str(gold_path / "invoice_features"))
    contract_features.write.mode("overwrite").parquet(str(gold_path / "contract_features"))
