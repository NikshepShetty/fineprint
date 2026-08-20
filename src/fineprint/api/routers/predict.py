from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fineprint.agent.tools import predict_contract_risk, predict_invoice_risk

router = APIRouter(prefix="/predict")


class TopFactor(BaseModel):
    feature: str
    shap_value: float


class InvoiceRiskRequest(BaseModel):
    invoice_id: str


class InvoiceRiskResponse(BaseModel):
    invoice_id: str
    paid_late_prediction: bool
    top_factors: list[TopFactor]


class ContractRiskRequest(BaseModel):
    contract_id: str


class ContractRiskResponse(BaseModel):
    contract_id: str
    risk_score_prediction: float
    top_factors: list[TopFactor]


def _raise_for_error(result: dict) -> None:
    error = result.get("error")
    if error is None:
        return
    if error.startswith(("no invoice found", "no contract found")):
        raise HTTPException(status_code=404, detail=error)
    raise HTTPException(status_code=500, detail=error)


@router.post("/invoice-risk", response_model=InvoiceRiskResponse)
def get_invoice_risk(request: InvoiceRiskRequest) -> InvoiceRiskResponse:
    result = predict_invoice_risk(request.invoice_id)
    _raise_for_error(result)
    return InvoiceRiskResponse(**result)


@router.post("/contract-risk", response_model=ContractRiskResponse)
def get_contract_risk(request: ContractRiskRequest) -> ContractRiskResponse:
    result = predict_contract_risk(request.contract_id)
    _raise_for_error(result)
    return ContractRiskResponse(**result)
