import random

from faker import Faker

from fineprint.data_generator.clauses import CLAUSE_TEMPLATES, NEUTRAL_CLAUSES
from fineprint.data_generator.schemas import CLAUSE_RISK_WEIGHTS, Contract


# 0-1 risky clauses is most common and all 4 is rare.
RISKY_CLAUSE_COUNT_WEIGHTS = {0: 0.35, 1: 0.30, 2: 0.20, 3: 0.10, 4: 0.05}


def generate_contracts(n: int = 100, seed: int | None = None) -> list[Contract]:
    rng = random.Random(seed)
    fake = Faker()
    if seed is not None:
        fake.seed_instance(seed)

    all_clause_keys = list(CLAUSE_RISK_WEIGHTS.keys())
    counts = list(RISKY_CLAUSE_COUNT_WEIGHTS.keys())
    count_weights = list(RISKY_CLAUSE_COUNT_WEIGHTS.values())

    contracts: list[Contract] = []
    for i in range(n):
        num_risky = rng.choices(counts, weights=count_weights, k=1)[0]
        risky_clauses = rng.sample(all_clause_keys, k=num_risky)

        neutral_sample = rng.sample(NEUTRAL_CLAUSES, k=min(3, len(NEUTRAL_CLAUSES)))
        paragraphs = neutral_sample + [CLAUSE_TEMPLATES[c] for c in risky_clauses]
        rng.shuffle(paragraphs)

        contract_text = "\n\n".join(paragraphs)

        contract = Contract(
            contract_id=f"CTR-{i:04d}",
            counterparty_name=fake.company(),
            contract_text=contract_text,
            clauses_present=risky_clauses,
        )
        contracts.append(contract)

    return contracts