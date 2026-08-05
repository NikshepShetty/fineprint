import pytest
from pyspark.sql import SparkSession

from fineprint.etl.bronze import _validate_required_columns
from fineprint.etl.gold import build_contract_features_gold, build_invoice_features_gold
from fineprint.etl.silver import build_contracts_silver, build_invoices_silver


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("fineprint-etl-tests").master("local[1]").getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()


# --- bronze ---


def test_validate_required_columns_passes_when_present(spark):
    df = spark.createDataFrame([("a", 1)], ["vendor_id", "amount"])
    result = _validate_required_columns(df, ["vendor_id"], "test")
    assert result is df


def test_validate_required_columns_raises_when_missing(spark):
    df = spark.createDataFrame([("a", 1)], ["vendor_id", "amount"])
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_required_columns(df, ["vendor_id", "not_a_column"], "test")


def test_validate_required_columns_raises_on_nulls(spark):
    df = spark.createDataFrame([("a",), (None,)], ["vendor_id"])
    with pytest.raises(ValueError, match="null values"):
        _validate_required_columns(df, ["vendor_id"], "test")


# --- silver ---


def test_build_invoices_silver_joins_vendor_fields(spark):
    vendors = spark.createDataFrame(
        [("VEND-0001", "Acme Ltd", "reliable", 1.5, "retail")],
        ["vendor_id", "name", "payment_behavior", "avg_days_late_historical", "industry"],
    )
    invoices = spark.createDataFrame(
        [("INV-00001", "VEND-0001", 500.0)],
        ["invoice_id", "vendor_id", "amount"],
    )

    result = build_invoices_silver(invoices, vendors).collect()[0]
    assert result["vendor_payment_behavior"] == "reliable"
    assert result["vendor_avg_days_late_historical"] == 1.5
    assert result["vendor_industry"] == "retail"


def test_build_contracts_silver_extracts_clause_flags(spark):
    contracts = spark.createDataFrame(
        [("CTR-0001", ["auto_renewal", "vague_termination"])],
        ["contract_id", "clauses_present"],
    )

    result = build_contracts_silver(contracts).collect()[0]
    assert result["has_auto_renewal"] is True
    assert result["has_vague_termination"] is True
    assert result["has_unlimited_liability"] is False
    assert result["has_missing_indemnity"] is False


# --- gold ---


def test_build_invoice_features_gold_selects_expected_columns(spark):
    invoices_silver = spark.createDataFrame(
        [
            (
                "INV-00001", "VEND-0001", 500.0, 30, "2026-01-05",
                2.0, "reliable", "retail", 3, True,
            )
        ],
        [
            "invoice_id", "vendor_id", "amount", "payment_terms_days", "issue_date",
            "vendor_avg_days_late_historical", "vendor_payment_behavior", "vendor_industry",
            "days_late", "paid_late",
        ],
    )
    invoices_silver = invoices_silver.withColumn(
        "issue_date", invoices_silver["issue_date"].cast("date")
    )

    result = build_invoice_features_gold(invoices_silver)
    expected_columns = {
        "invoice_id", "vendor_id", "amount", "payment_terms_days", "issue_day_of_week",
        "vendor_avg_days_late_historical", "vendor_payment_behavior", "vendor_industry",
        "days_late", "paid_late",
    }
    assert set(result.columns) == expected_columns
    assert result.count() == 1


def test_build_contract_features_gold_computes_num_risky_clauses(spark):
    contracts_silver = spark.createDataFrame(
        [("CTR-0001", True, True, False, False, 50)],
        [
            "contract_id", "has_auto_renewal", "has_unlimited_liability",
            "has_vague_termination", "has_missing_indemnity", "risk_score",
        ],
    )

    result = build_contract_features_gold(contracts_silver).collect()[0]
    assert result["num_risky_clauses"] == 2
    assert result["risk_score"] == 50
