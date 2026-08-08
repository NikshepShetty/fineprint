from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "has_auto_renewal",
    "has_unlimited_liability",
    "has_vague_termination",
    "has_missing_indemnity",
    "num_risky_clauses",
]
LABEL_COLUMN = "risk_score"


def load_training_data(gold_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(gold_path / "contract_features")
    for col in FEATURE_COLUMNS[:4]:
        df[col] = df[col].astype(int)
    return df


def train_contract_risk_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, metrics


def run(gold_path: Path) -> str:
    df = load_training_data(gold_path)
    model, metrics = train_contract_risk_model(df)

    with mlflow.start_run(run_name="contract_risk"):
        mlflow.log_params(
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.xgboost.log_model(
            model, name="model", registered_model_name="contract_risk_model"
        )
        print(f"contract_risk_model metrics: {metrics}")
        return model_info.model_uri
