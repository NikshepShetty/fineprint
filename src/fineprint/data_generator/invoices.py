import random
from datetime import UTC, datetime, timedelta

from fineprint.data_generator.schemas import Invoice, Vendor

PAYMENT_TERMS_OPTIONS = [15, 30, 45, 60, 90]

LATENESS_STD_DEV = 5.0

ISSUE_DATE_WINDOW_DAYS = 365


def generate_invoices(
    vendors: list[Vendor], n: int = 500, seed: int | None = None
) -> list[Invoice]:
    rng = random.Random(seed)
    today = datetime.now(UTC).date()

    invoices: list[Invoice] = []
    for i in range(n):
        vendor = rng.choice(vendors)

        days_ago = rng.randint(0, ISSUE_DATE_WINDOW_DAYS)
        issue_date = today - timedelta(days=days_ago)

        payment_terms_days = rng.choice(PAYMENT_TERMS_OPTIONS)
        due_date = issue_date + timedelta(days=payment_terms_days)

        # Lateness varies per invoice around the vendor's own average.
        lateness_days = rng.gauss(vendor.avg_days_late_historical, LATENESS_STD_DEV)
        actual_payment_date = due_date + timedelta(days=round(lateness_days))

        amount = round(rng.uniform(100, 50_000), 2)

        invoice = Invoice(
            invoice_id=f"INV-{i:05d}",
            vendor_id=vendor.vendor_id,
            amount=amount,
            currency="GBP",
            issue_date=issue_date,
            payment_terms_days=payment_terms_days,
            actual_payment_date=actual_payment_date,
        )
        invoices.append(invoice)

    return invoices
