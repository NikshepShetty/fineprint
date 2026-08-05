CLAUSE_TEMPLATES: dict[str, str] = {
    "auto_renewal": (
        "This Agreement shall automatically renew for successive one-year terms "
        "unless either party provides written notice of non-renewal at least "
        "thirty (30) days prior to the end of the then-current term."
    ),
    "unlimited_liability": (
        "Each party's liability under this Agreement shall be unlimited and "
        "shall extend to all direct, indirect, incidental, and consequential "
        "damages arising out of or related to this Agreement."
    ),
    "vague_termination": (
        "Either party may terminate this Agreement at any time for any reason "
        "or no reason, with such notice as that party deems appropriate under "
        "the circumstances."
    ),
    "missing_indemnity": (
        "This Agreement contains no provision requiring either party to "
        "indemnify, defend, or hold harmless the other party against claims, "
        "losses, or damages arising from performance of this Agreement."
    ),
}

NEUTRAL_CLAUSES: list[str] = [
    (
        "Payment shall be made within thirty (30) days of the invoice date via "
        "bank transfer to the account specified by the receiving party."
    ),
    (
        "Both parties agree to keep confidential any proprietary information "
        "disclosed during the course of this Agreement, and shall not disclose "
        "such information to third parties without prior written consent."
    ),
    (
        "This Agreement shall be governed by and construed in accordance with "
        "the laws of England and Wales, without regard to conflict of law "
        "principles."
    ),
    (
        "Any notice required under this Agreement shall be in writing and "
        "delivered by email or registered post to the addresses specified by "
        "the parties."
    ),
    (
        "This Agreement constitutes the entire understanding between the "
        "parties and supersedes all prior agreements, whether written or oral, "
        "relating to its subject matter."
    ),
    (
        "Neither party shall be liable for any failure to perform its "
        "obligations where such failure results from circumstances beyond "
        "that party's reasonable control."
    ),
]
