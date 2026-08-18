from pathlib import Path

import mlflow
import pandas as pd

from fineprint.models.contract_risk import FEATURE_COLUMNS as CONTRACT_FEATURES
from fineprint.models.explain import explain_prediction
from fineprint.models.invoice_risk import FEATURE_COLUMNS as INVOICE_FEATURES
from fineprint.rag.store import DocumentStore

GOLD_PATH = Path("data/gold")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def search_documents(store: DocumentStore, query: str, k: int = 5) -> list[dict]:
    return store.query(query, k=k)


def predict_invoice_risk(invoice_id: str) -> dict:
    """Predict whether an invoice will be paid late, given its invoice ID."""
    try:
        df = pd.read_parquet(GOLD_PATH / "invoice_features")
        row = df[df["invoice_id"] == invoice_id].copy()
        if row.empty:
            return {"error": f"no invoice found with id {invoice_id}"}

        for col in ["vendor_payment_behavior", "vendor_industry"]:
            row[col] = row[col].astype("category")

        model = mlflow.xgboost.load_model("models:/invoice_risk_model/latest")
        prediction = bool(model.predict(row[INVOICE_FEATURES])[0])
        explanation = explain_prediction(model, row[INVOICE_FEATURES], INVOICE_FEATURES)

        return {"invoice_id": invoice_id, "paid_late_prediction": prediction, "top_factors": explanation}
    except Exception as e:  # noqa: BLE001 - catches any prediction failure
        return {"error": f"invoice risk prediction failed: {e}"}


def predict_contract_risk(contract_id: str) -> dict:
    """Predict a contract's risk score based on its clauses, given its contract ID."""
    try:
        df = pd.read_parquet(GOLD_PATH / "contract_features")
        row = df[df["contract_id"] == contract_id].copy()
        if row.empty:
            return {"error": f"no contract found with id {contract_id}"}

        model = mlflow.xgboost.load_model("models:/contract_risk_model/latest")
        prediction = float(model.predict(row[CONTRACT_FEATURES])[0])
        explanation = explain_prediction(model, row[CONTRACT_FEATURES], CONTRACT_FEATURES)

        return {"contract_id": contract_id, "risk_score_prediction": prediction, "top_factors": explanation}
    except Exception as e:  # noqa: BLE001 - catches any prediction failure
        return {"error": f"contract risk prediction failed: {e}"}
