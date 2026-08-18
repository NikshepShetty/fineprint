import json
import math
import re
from pathlib import Path
from typing import Any

from fineprint.rag.store import DocumentStore

_embedding_model = None

REQUIRED_CONTRACT_KEYS = [
    "contract_id",
    "counterparty_name",
    "contract_text",
    "risk_score",
]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def chunk_contract_text(text: str) -> list[str]:
    """Splits text into paragraphs. Blank lines with only spaces or tabs count too."""
    normalized = text.replace("\r\n", "\n")
    raw_chunks = re.split(r"\n[ \t]*\n", normalized)
    return [p.strip() for p in raw_chunks if p.strip()]


def default_embedding_fn(texts: list[str]) -> list[list[float]]:
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedding_model.encode(texts).tolist()


def _validate_contract(contract: Any, index: int, contract_ids: set[str]) -> None:
    if not isinstance(contract, dict):
        raise ValueError(f"contract at index {index} must be an object")  # noqa: TRY004 - keep ValueError consistent here

    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(
            f"contract at index {index} is missing required keys: {missing}"
        )

    for key in ("contract_id", "counterparty_name", "contract_text"):
        if not isinstance(contract[key], str) or not contract[key].strip():
            raise ValueError(f"contract at index {index} has an invalid {key}")

    risk_score = contract["risk_score"]
    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, int | float)
        or not math.isfinite(risk_score)
    ):
        raise ValueError(f"contract at index {index} has an invalid risk_score")

    if contract["contract_id"] in contract_ids:
        raise ValueError(
            f"contract at index {index} has a duplicate contract_id: {contract['contract_id']}"
        )
    contract_ids.add(contract["contract_id"])


def load_contract_documents(contracts_path: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Reads and validates contracts.json, and builds the chunks ready to ingest."""
    contracts_file = contracts_path / "contracts.json"

    with open(contracts_file, encoding="utf-8") as f:
        try:
            contracts = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{contracts_file} is not valid JSON: {e}") from e

    if not isinstance(contracts, list):
        raise ValueError(f"{contracts_file} must contain a JSON array of contracts")  # noqa: TRY004 - keep ValueError consistent here

    documents = []
    contract_ids: set[str] = set()
    for i, contract in enumerate(contracts):
        _validate_contract(contract, i, contract_ids)

        chunks = chunk_contract_text(contract["contract_text"])
        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{contract['contract_id']}-chunk-{chunk_index}",
                    "text": chunk,
                    "metadata": {
                        "contract_id": contract["contract_id"],
                        "counterparty_name": contract["counterparty_name"],
                        "risk_score": contract["risk_score"],
                    },
                }
            )

    return documents


def ingest_contracts(contracts_path: Path, store: DocumentStore) -> int:
    documents = load_contract_documents(contracts_path)
    store.add(documents)
    return len(documents)
