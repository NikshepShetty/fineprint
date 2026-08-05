from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Vendor(BaseModel):
    vendor_id: str
    name: str
    payment_behavior: Literal["reliable", "occasionally_late", "chronically_late"]
    avg_days_late_historical: float = Field(ge=0)
    industry: str | None = None


class Invoice(BaseModel):
    invoice_id: str
    vendor_id: str
    amount: float = Field(gt=0)
    currency: str = "GBP"
    issue_date: date
    payment_terms_days: int = Field(ge=0)
    actual_payment_date: date

    due_date: date = None
    days_late: int = None
    paid_late: bool = None

    @model_validator(mode="after")
    def compute_derived_fields(self):
        self.due_date = self.issue_date + timedelta(days=self.payment_terms_days)
        self.days_late = (self.actual_payment_date - self.due_date).days
        self.paid_late = self.days_late > 0
        return self


CLAUSE_RISK_WEIGHTS: dict[str, int] = {
    "auto_renewal": 20,
    "unlimited_liability": 30,
    "vague_termination": 15,
    "missing_indemnity": 25,
}


class Contract(BaseModel):
    contract_id: str
    counterparty_name: str
    contract_text: str
    clauses_present: list[str]

    risk_score: int = None

    @model_validator(mode="after")
    def compute_risk_score(self):
        self.risk_score = sum(
            CLAUSE_RISK_WEIGHTS.get(clause, 0) for clause in self.clauses_present
        )
        return self
