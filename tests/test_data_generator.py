from fineprint.data_generator.clauses import CLAUSE_TEMPLATES
from fineprint.data_generator.contracts import generate_contracts
from fineprint.data_generator.invoices import generate_invoices
from fineprint.data_generator.schemas import CLAUSE_RISK_WEIGHTS
from fineprint.data_generator.vendors import generate_vendors

# --- vendors ---


def test_generate_vendors_count():
    vendors = generate_vendors(n=20, seed=1)
    assert len(vendors) == 20


def test_generate_vendors_valid_behavior():
    vendors = generate_vendors(n=20, seed=1)
    valid = {"reliable", "occasionally_late", "chronically_late"}
    assert all(v.payment_behavior in valid for v in vendors)


def test_generate_vendors_non_negative_lateness():
    vendors = generate_vendors(n=20, seed=1)
    assert all(v.avg_days_late_historical >= 0 for v in vendors)


def test_generate_vendors_reproducible_with_seed():
    a = generate_vendors(n=10, seed=7)
    b = generate_vendors(n=10, seed=7)
    assert [v.model_dump() for v in a] == [v.model_dump() for v in b]


# --- invoices ---


def test_generate_invoices_count():
    vendors = generate_vendors(n=10, seed=1)
    invoices = generate_invoices(vendors, n=50, seed=1)
    assert len(invoices) == 50


def test_generate_invoices_reference_valid_vendors():
    vendors = generate_vendors(n=10, seed=1)
    invoices = generate_invoices(vendors, n=50, seed=1)
    vendor_ids = {v.vendor_id for v in vendors}
    assert all(inv.vendor_id in vendor_ids for inv in invoices)


def test_generate_invoices_derived_fields_consistent():
    vendors = generate_vendors(n=10, seed=1)
    invoices = generate_invoices(vendors, n=50, seed=1)
    for inv in invoices:
        assert (inv.due_date - inv.issue_date).days == inv.payment_terms_days
        assert (inv.actual_payment_date - inv.due_date).days == inv.days_late
        assert inv.paid_late == (inv.days_late > 0)


def test_generate_invoices_reproducible_with_seed():
    vendors = generate_vendors(n=10, seed=1)
    a = generate_invoices(vendors, n=20, seed=5)
    b = generate_invoices(vendors, n=20, seed=5)
    assert [i.model_dump() for i in a] == [i.model_dump() for i in b]


# --- contracts ---


def test_clause_weights_and_templates_in_sync():
    # Keeps schemas.py and clauses.py from drifting apart.
    assert set(CLAUSE_RISK_WEIGHTS.keys()) == set(CLAUSE_TEMPLATES.keys())


def test_generate_contracts_count():
    contracts = generate_contracts(n=30, seed=1)
    assert len(contracts) == 30


def test_generate_contracts_valid_clauses():
    contracts = generate_contracts(n=30, seed=1)
    valid = set(CLAUSE_RISK_WEIGHTS.keys())
    for contract in contracts:
        assert all(clause in valid for clause in contract.clauses_present)


def test_generate_contracts_risk_score_matches_clauses():
    contracts = generate_contracts(n=30, seed=1)
    for contract in contracts:
        expected = sum(CLAUSE_RISK_WEIGHTS[c] for c in contract.clauses_present)
        assert contract.risk_score == expected


def test_generate_contracts_reproducible_with_seed():
    a = generate_contracts(n=15, seed=9)
    b = generate_contracts(n=15, seed=9)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
