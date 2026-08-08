from pathlib import Path

import mlflow

from fineprint.models.contract_risk import run as run_contract_risk
from fineprint.models.invoice_risk import run as run_invoice_risk

GOLD_PATH = Path("data/gold")
TRACKING_URI = "sqlite:///mlflow.db"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    invoice_uri = run_invoice_risk(GOLD_PATH)
    contract_uri = run_contract_risk(GOLD_PATH)

    print(f"invoice_risk_model logged at: {invoice_uri}")
    print(f"contract_risk_model logged at: {contract_uri}")


if __name__ == "__main__":
    main()
