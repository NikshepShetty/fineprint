import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/documents")

DATA_DIR = Path("data")


class DocumentSummary(BaseModel):
    contract_id: str
    counterparty_name: str
    risk_score: int


class DocumentDetail(BaseModel):
    contract_id: str
    counterparty_name: str
    risk_score: int
    contract_text: str
    clauses_present: list[str]


def _load_contracts() -> list[dict]:
    contracts_file = DATA_DIR / "contracts.json"
    if not contracts_file.exists():
        raise HTTPException(status_code=503, detail="no contracts have been generated yet")

    with open(contracts_file, encoding="utf-8") as f:
        return json.load(f)


@router.get("", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    contracts = _load_contracts()
    return [DocumentSummary(**c) for c in contracts]


@router.get("/{contract_id}", response_model=DocumentDetail)
def get_document(contract_id: str) -> DocumentDetail:
    contracts = _load_contracts()
    for contract in contracts:
        if contract["contract_id"] == contract_id:
            return DocumentDetail(**contract)
    raise HTTPException(status_code=404, detail=f"no contract found with id {contract_id}")
