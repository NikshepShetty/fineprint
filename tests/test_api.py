import pandas as pd
from fastapi.testclient import TestClient

from fineprint.api.main import app
from fineprint.api.routers.chat import get_ask_fn

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_documents_returns_all_contracts():
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert "contract_id" in body[0]
    assert "risk_score" in body[0]


def test_get_document_returns_full_detail():
    contract_id = client.get("/documents").json()[0]["contract_id"]
    response = client.get(f"/documents/{contract_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["contract_id"] == contract_id
    assert "contract_text" in body
    assert "clauses_present" in body


def test_get_document_404_for_unknown_id():
    response = client.get("/documents/NOT-A-REAL-ID")
    assert response.status_code == 404


def test_predict_invoice_risk_returns_prediction_and_factors():
    invoice_id = pd.read_parquet("data/gold/invoice_features")["invoice_id"].iloc[0]
    response = client.post("/predict/invoice-risk", json={"invoice_id": invoice_id})
    assert response.status_code == 200
    body = response.json()
    assert body["invoice_id"] == invoice_id
    assert isinstance(body["paid_late_prediction"], bool)
    assert len(body["top_factors"]) > 0


def test_predict_invoice_risk_404_for_unknown_id():
    response = client.post("/predict/invoice-risk", json={"invoice_id": "INV-99999"})
    assert response.status_code == 404


def test_predict_contract_risk_returns_prediction_and_factors():
    contract_id = client.get("/documents").json()[0]["contract_id"]
    response = client.post("/predict/contract-risk", json={"contract_id": contract_id})
    assert response.status_code == 200
    body = response.json()
    assert body["contract_id"] == contract_id
    assert isinstance(body["risk_score_prediction"], float)
    assert len(body["top_factors"]) > 0


def test_predict_contract_risk_404_for_unknown_id():
    response = client.post("/predict/contract-risk", json={"contract_id": "CTR-99999"})
    assert response.status_code == 404


def test_chat_returns_answer_from_injected_ask_fn():
    app.dependency_overrides[get_ask_fn] = lambda: (lambda q: f"fake answer to: {q}")
    try:
        response = client.post("/chat", json={"question": "Why is CTR-0000 risky?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "fake answer to: Why is CTR-0000 risky?"
    finally:
        app.dependency_overrides.clear()


def test_chat_returns_503_when_store_not_ingested():
    def failing_ask(question):
        raise RuntimeError("Document store is empty. Run ingestion first.")

    app.dependency_overrides[get_ask_fn] = lambda: failing_ask
    try:
        response = client.post("/chat", json={"question": "anything"})
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()
