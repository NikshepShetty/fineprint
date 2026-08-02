import random

from faker import Faker

from fineprint.data_generator.schemas import Vendor

INDUSTRIES = [
    "construction",
    "retail",
    "logistics",
    "healthcare",
    "manufacturing",
    "hospitality",
    "professional services",
    "technology",
    "agriculture",
    "energy",
]


BEHAVIOR_WEIGHTS = {
    "reliable": 0.70,
    "occasionally_late": 0.20,
    "chronically_late": 0.10,
}

BEHAVIOR_RANGES = {
    "reliable": (-2, 3),
    "occasionally_late": (3, 10),
    "chronically_late": (10, 30),
}


def generate_vendors(n: int = 50, seed: int | None = None) -> list[Vendor]:
    rng = random.Random(seed)
    fake = Faker()
    if seed is not None:
        fake.seed_instance(seed)

    behaviors = list(BEHAVIOR_WEIGHTS.keys())
    weights = list(BEHAVIOR_WEIGHTS.values())

    vendors: list[Vendor] = []
    for i in range(n):
        behavior = rng.choices(behaviors, weights=weights, k=1)[0]
        low, high = BEHAVIOR_RANGES[behavior]
        avg_days_late = max(0.0, rng.uniform(low, high))

        vendor = Vendor(
            vendor_id=f"VEND-{i:04d}",
            name=fake.company(),
            payment_behavior=behavior,
            avg_days_late_historical=avg_days_late,
            industry=rng.choice(INDUSTRIES),
        )
        vendors.append(vendor)

    return vendors