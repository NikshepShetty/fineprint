import pandas as pd
import pytest

from fineprint.models.contract_risk import (
    FEATURE_COLUMNS as CONTRACT_FEATURES,
)
from fineprint.models.contract_risk import (
    train_contract_risk_model,
)
from fineprint.models.explain import explain_prediction
from fineprint.models.invoice_risk import (
    CATEGORICAL_COLUMNS,
    train_invoice_risk_model,
)
from fineprint.models.invoice_risk import (
    FEATURE_COLUMNS as INVOICE_FEATURES,
)


@pytest.fixture
def tiny_invoice_df():
    n = 60
    df = pd.DataFrame(
        {
            "amount": [100.0 + i * 10 for i in range(n)],
            "payment_terms_days": [30] * n,
            "issue_day_of_week": [(i % 7) + 1 for i in range(n)],
            "vendor_avg_days_late_historical": [float(i % 20) for i in range(n)],
            "vendor_payment_behavior": ["reliable", "occasionally_late"] * (n // 2),
            "vendor_industry": ["retail", "construction"] * (n // 2),
            "paid_late": [i % 2 == 0 for i in range(n)],
        }
    )
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


@pytest.fixture
def tiny_contract_df():
    n = 40
    df = pd.DataFrame(
        {
            "has_auto_renewal": [i % 2 == 0 for i in range(n)],
            "has_unlimited_liability": [i % 3 == 0 for i in range(n)],
            "has_vague_termination": [i % 4 == 0 for i in range(n)],
            "has_missing_indemnity": [i % 5 == 0 for i in range(n)],
        }
    )
    df["num_risky_clauses"] = df[
        ["has_auto_renewal", "has_unlimited_liability", "has_vague_termination", "has_missing_indemnity"]
    ].sum(axis=1)
    df["risk_score"] = (
        df["has_auto_renewal"] * 20
        + df["has_unlimited_liability"] * 30
        + df["has_vague_termination"] * 15
        + df["has_missing_indemnity"] * 25
    )
    return df


def test_train_invoice_risk_model_runs_end_to_end(tiny_invoice_df):
    model, metrics = train_invoice_risk_model(tiny_invoice_df)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    sample = tiny_invoice_df[INVOICE_FEATURES].iloc[[0]]
    prediction = model.predict(sample)
    assert prediction[0] in (0, 1)


def test_train_contract_risk_model_runs_end_to_end(tiny_contract_df):
    model, metrics = train_contract_risk_model(tiny_contract_df)
    assert "mae" in metrics
    assert "r2" in metrics

    sample = tiny_contract_df[CONTRACT_FEATURES].iloc[[0]]
    prediction = model.predict(sample)
    assert prediction[0] >= 0


def test_explain_prediction_returns_top_features(tiny_invoice_df):
    model, _ = train_invoice_risk_model(tiny_invoice_df)
    sample = tiny_invoice_df[INVOICE_FEATURES].iloc[[0]]

    explanation = explain_prediction(model, sample, INVOICE_FEATURES, top_n=3)

    assert len(explanation) == 3
    for item in explanation:
        assert item["feature"] in INVOICE_FEATURES
        assert isinstance(item["shap_value"], float)
