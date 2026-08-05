from pathlib import Path

from pyspark.sql import SparkSession

from fineprint.etl.bronze import run_bronze
from fineprint.etl.gold import run_gold
from fineprint.etl.silver import run_silver

DATA_DIR = Path("data")


def main() -> None:
    spark = (
        SparkSession.builder.appName("fineprint-etl-local")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        run_bronze(spark, raw_path=DATA_DIR, bronze_path=DATA_DIR / "bronze")
        run_silver(spark, bronze_path=DATA_DIR / "bronze", silver_path=DATA_DIR / "silver")
        run_gold(spark, silver_path=DATA_DIR / "silver", gold_path=DATA_DIR / "gold")
        print(f"ETL pipeline complete. Bronze, silver, gold written under {DATA_DIR}/")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
