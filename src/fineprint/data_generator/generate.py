import json
from pathlib import Path

from fineprint.data_generator.contracts import generate_contracts
from fineprint.data_generator.invoices import generate_invoices
from fineprint.data_generator.vendors import generate_vendors

DATA_DIR = Path("data")


def main(
    n_vendors: int = 50,
    n_invoices: int = 500,
    n_contracts: int = 100,
    seed: int | None = 42,
) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    vendors = generate_vendors(n=n_vendors, seed=seed)
    invoices = generate_invoices(vendors, n=n_invoices, seed=seed)
    contracts = generate_contracts(n=n_contracts, seed=seed)

    _write_json(DATA_DIR / "vendors.json", vendors)
    _write_json(DATA_DIR / "invoices.json", invoices)
    _write_json(DATA_DIR / "contracts.json", contracts)

    print(
        f"Generated {len(vendors)} vendors, {len(invoices)} invoices, "
        f"{len(contracts)} contracts into {DATA_DIR}/"
    )


def _write_json(path: Path, records) -> None:
    with open(path, "w") as f:
        json.dump([r.model_dump(mode="json") for r in records], f, indent=2)


if __name__ == "__main__":
    main()
