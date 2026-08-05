from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

REQUIRED_VENDOR_COLUMNS = ["vendor_id", "name", "payment_behavior", "avg_days_late_historical"]
REQUIRED_INVOICE_COLUMNS = ["invoice_id", "vendor_id", "amount", "due_date", "paid_late"]
REQUIRED_CONTRACT_COLUMNS = ["contract_id", "counterparty_name", "clauses_present", "risk_score"]


def _validate_required_columns(df: DataFrame, required: list[str], label: str) -> DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

    for column in required:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            raise ValueError(f"{label}.{column} has {null_count} null values, expected none")

    return df


def load_vendors(spark: SparkSession, raw_path: Path) -> DataFrame:
    # multiLine is required since the generator writes pretty-printed JSON arrays,
    # not one record per line.
    df = spark.read.option("multiLine", "true").json(str(raw_path / "vendors.json"))
    return _validate_required_columns(df, REQUIRED_VENDOR_COLUMNS, "vendors")


def load_invoices(spark: SparkSession, raw_path: Path) -> DataFrame:
    df = spark.read.option("multiLine", "true").json(str(raw_path / "invoices.json"))
    return _validate_required_columns(df, REQUIRED_INVOICE_COLUMNS, "invoices")


def load_contracts(spark: SparkSession, raw_path: Path) -> DataFrame:
    df = spark.read.option("multiLine", "true").json(str(raw_path / "contracts.json"))
    return _validate_required_columns(df, REQUIRED_CONTRACT_COLUMNS, "contracts")


def run_bronze(spark: SparkSession, raw_path: Path, bronze_path: Path) -> None:
    bronze_path.mkdir(parents=True, exist_ok=True)

    vendors = load_vendors(spark, raw_path)
    invoices = load_invoices(spark, raw_path)
    contracts = load_contracts(spark, raw_path)

    vendors.write.mode("overwrite").parquet(str(bronze_path / "vendors"))
    invoices.write.mode("overwrite").parquet(str(bronze_path / "invoices"))
    contracts.write.mode("overwrite").parquet(str(bronze_path / "contracts"))
