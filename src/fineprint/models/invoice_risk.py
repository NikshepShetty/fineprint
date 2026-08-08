from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "amount",
    "payment_terms_days",
    "issue_day_of_week",
    "vendor_avg_days_late_historical",
    "vendor_payment_behavior",
    "vendor_industry",
]
LABEL_COLUMN = "paid_late"
CATEGORICAL_COLUMNS = ["vendor_payment_behavior", "vendor_industry"]


def load_training_data(gold_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(gold_path / "invoice_features")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def train_invoice_risk_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return model, metrics


def run(gold_path: Path) -> str:
    df = load_training_data(gold_path)
    model, metrics = train_invoice_risk_model(df)

    with mlflow.start_run(run_name="invoice_risk"):
        mlflow.log_params(
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.xgboost.log_model(
            model, name="model", registered_model_name="invoice_risk_model"
        )
        print(f"invoice_risk_model metrics: {metrics}")
        return model_info.model_uri
