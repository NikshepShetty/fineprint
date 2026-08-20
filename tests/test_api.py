import json
import random

import mlflow
import pytest
from fastapi.testclient import TestClient

import fineprint.agent.tools as tools_module
import fineprint.api.routers.documents as documents_module
from fineprint.api.main import app
from fineprint.api.routers.chat import get_ask_fn
from fineprint.models.contract_risk import train_contract_risk_model
from fineprint.models.invoice_risk import train_invoice_risk_model


def _build_tiny_invoice_df():
    import pandas as pd

    rng = random.Random(0)
    rows = []
    for i in range(40):
        late = i % 2 == 0
        rows.append(
            {
                "invoice_id": f"INV-{i:03d}",
                "vendor_id": f"VEND-{i % 5:03d}",
                "amount": rng.uniform(100, 5000),
                "payment_terms_days": rng.choice([15, 30, 60]),
                "issue_day_of_week": rng.randint(1, 7),
                "vendor_avg_days_late_historical": rng.uniform(0, 20),
                "vendor_payment_behavior": rng.choice(["reliable", "chronically_late"]),
                "vendor_industry": rng.choice(["retail", "construction"]),
                "days_late": 5 if late else -2,
                "paid_late": late,
            }
        )
    df = pd.DataFrame(rows)
    for col in ["vendor_payment_behavior", "vendor_industry"]:
        df[col] = df[col].astype("category")
    return df


def _build_tiny_contract_df(contracts: list[dict]):
    import pandas as pd

    clause_keys = [
        "has_auto_renewal",
        "has_unlimited_liability",
        "has_vague_termination",
        "has_missing_indemnity",
    ]
    rows = []
    for c in contracts:
        row = {"contract_id": c["contract_id"]}
        for key in clause_keys:
            row[key] = key.replace("has_", "") in c["clauses_present"]
        row["num_risky_clauses"] = sum(row[k] for k in clause_keys)
        row["risk_score"] = c["risk_score"]
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def seeded_app(tmp_path, monkeypatch):
    contracts = [
        {
            "contract_id": "CTR-000",
            "counterparty_name": "Acme Ltd",
            "contract_text": "Payment terms apply.\n\nAuto renewal clause applies.",
            "risk_score": 20,
            "clauses_present": ["auto_renewal"],
        },
        {
            "contract_id": "CTR-001",
            "counterparty_name": "Widgets Inc",
            "contract_text": "Standard confidentiality clause.",
            "risk_score": 0,
            "clauses_present": [],
        },
        {
            "contract_id": "CTR-002",
            "counterparty_name": "Beta Corp",
            "contract_text": "Unlimited liability clause applies.",
            "risk_score": 30,
            "clauses_present": ["unlimited_liability"],
        },
        {
            "contract_id": "CTR-003",
            "counterparty_name": "Gamma LLC",
            "contract_text": "Vague termination clause applies.",
            "risk_score": 15,
            "clauses_present": ["vague_termination"],
        },
        {
            "contract_id": "CTR-004",
            "counterparty_name": "Delta Co",
            "contract_text": "Auto renewal and unlimited liability clauses apply.",
            "risk_score": 50,
            "clauses_present": ["auto_renewal", "unlimited_liability"],
        },
        {
            "contract_id": "CTR-005",
            "counterparty_name": "Epsilon Group",
            "contract_text": "Missing indemnity clause noted.",
            "risk_score": 25,
            "clauses_present": ["missing_indemnity"],
        },
    ]
    data_dir = tmp_path / "data"
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True)
    with open(data_dir / "contracts.json", "w") as f:
        json.dump(contracts, f)

    invoice_df = _build_tiny_invoice_df()
    contract_df = _build_tiny_contract_df(contracts)
    invoice_df.to_parquet(gold_dir / "invoice_features")
    contract_df.to_parquet(gold_dir / "contract_features")

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run():
        model, _ = train_invoice_risk_model(invoice_df)
        mlflow.xgboost.log_model(model, name="model", registered_model_name="invoice_risk_model")
    with mlflow.start_run():
        model, _ = train_contract_risk_model(contract_df)
        mlflow.xgboost.log_model(model, name="model", registered_model_name="contract_risk_model")

    monkeypatch.setattr(tools_module, "GOLD_PATH", gold_dir)
    monkeypatch.setattr(documents_module, "DATA_DIR", data_dir)

    return TestClient(app)


def test_health(seeded_app):
    response = seeded_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_documents_returns_all_contracts(seeded_app):
    response = seeded_app.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 6
    assert "contract_id" in body[0]


def test_get_document_returns_full_detail(seeded_app):
    response = seeded_app.get("/documents/CTR-000")
    assert response.status_code == 200
    body = response.json()
    assert body["contract_id"] == "CTR-000"
    assert "contract_text" in body
    assert body["clauses_present"] == ["auto_renewal"]


def test_get_document_404_for_unknown_id(seeded_app):
    response = seeded_app.get("/documents/NOT-A-REAL-ID")
    assert response.status_code == 404


def test_predict_invoice_risk_returns_prediction_and_factors(seeded_app):
    response = seeded_app.post("/predict/invoice-risk", json={"invoice_id": "INV-000"})
    assert response.status_code == 200
    body = response.json()
    assert body["invoice_id"] == "INV-000"
    assert isinstance(body["paid_late_prediction"], bool)
    assert len(body["top_factors"]) > 0


def test_predict_invoice_risk_404_for_unknown_id(seeded_app):
    response = seeded_app.post("/predict/invoice-risk", json={"invoice_id": "INV-999"})
    assert response.status_code == 404


def test_predict_contract_risk_returns_prediction_and_factors(seeded_app):
    response = seeded_app.post("/predict/contract-risk", json={"contract_id": "CTR-000"})
    assert response.status_code == 200
    body = response.json()
    assert body["contract_id"] == "CTR-000"
    assert isinstance(body["risk_score_prediction"], float)
    assert len(body["top_factors"]) > 0


def test_predict_contract_risk_404_for_unknown_id(seeded_app):
    response = seeded_app.post("/predict/contract-risk", json={"contract_id": "CTR-999"})
    assert response.status_code == 404


def test_chat_returns_answer_from_injected_ask_fn(seeded_app):
    app.dependency_overrides[get_ask_fn] = lambda: (lambda q: f"fake answer to: {q}")
    try:
        response = seeded_app.post("/chat", json={"question": "Why is CTR-000 risky?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "fake answer to: Why is CTR-000 risky?"
    finally:
        app.dependency_overrides.clear()


def test_chat_returns_503_when_store_not_ingested(seeded_app):
    def failing_ask(question):
        raise RuntimeError("Document store is empty. Run ingestion first.")

    app.dependency_overrides[get_ask_fn] = lambda: failing_ask
    try:
        response = seeded_app.post("/chat", json={"question": "anything"})
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()