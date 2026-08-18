import hashlib
import json
from pathlib import Path

import pytest

from fineprint.rag.ingest import (
    chunk_contract_text,
    ingest_contracts,
    load_contract_documents,
)
from fineprint.rag.store import DocumentStore


def fake_embedding_fn(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        out.append([b / 255.0 for b in digest[:8]])
    return out


def test_chunk_contract_text_splits_on_blank_lines():
    text = "Paragraph one.\n\nParagraph two.\n\n\nParagraph three."
    chunks = chunk_contract_text(text)
    assert chunks == ["Paragraph one.", "Paragraph two.", "Paragraph three."]


def test_chunk_contract_text_strips_whitespace():
    text = "  Padded paragraph.  \n\nAnother one.  "
    chunks = chunk_contract_text(text)
    assert chunks == ["Padded paragraph.", "Another one."]


def test_chunk_contract_text_handles_crlf_line_endings():
    text = "Paragraph one.\r\n\r\nParagraph two.\r\n\r\n\r\nParagraph three."
    chunks = chunk_contract_text(text)
    assert chunks == ["Paragraph one.", "Paragraph two.", "Paragraph three."]


def test_document_store_add_empty_list_is_a_noop(tmp_path):
    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    store.add([])
    assert store.count() == 0


def test_document_store_clear_removes_all_documents(tmp_path):
    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    store.add(
        [{"id": "a1", "text": "some text", "metadata": {"contract_id": "CTR-0001"}}]
    )
    assert store.count() == 1

    store.clear()
    assert store.count() == 0


def test_document_store_query_rejects_non_positive_k(tmp_path):
    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    store.add(
        [{"id": "a1", "text": "some text", "metadata": {"contract_id": "CTR-0001"}}]
    )

    with pytest.raises(ValueError, match="positive"):
        store.query("some text", k=0)

    with pytest.raises(ValueError, match="positive"):
        store.query("some text", k=True)


def test_chunk_contract_text_handles_whitespace_only_blank_lines():
    text = "Paragraph one.\n \nParagraph two.\n\t\nParagraph three."
    chunks = chunk_contract_text(text)
    assert chunks == ["Paragraph one.", "Paragraph two.", "Paragraph three."]


def test_load_contract_documents_rejects_missing_required_keys(tmp_path):
    contracts_dir = tmp_path / "data"
    contracts_dir.mkdir()
    with open(contracts_dir / "contracts.json", "w") as f:
        json.dump([{"contract_id": "CTR-0001", "contract_text": "text"}], f)

    with pytest.raises(ValueError, match="missing required keys"):
        load_contract_documents(Path(contracts_dir))


def test_load_contract_documents_rejects_malformed_json(tmp_path):
    contracts_dir = tmp_path / "data"
    contracts_dir.mkdir()
    with open(contracts_dir / "contracts.json", "w") as f:
        f.write("not valid json")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_contract_documents(Path(contracts_dir))


@pytest.mark.parametrize(
    "contracts",
    [
        {"contract_id": "CTR-0001"},
        [
            {
                "contract_id": "CTR-0001",
                "counterparty_name": "Acme",
                "contract_text": None,
                "risk_score": 10,
            }
        ],
        [
            {
                "contract_id": "CTR-0001",
                "counterparty_name": "Acme",
                "contract_text": "text",
                "risk_score": True,
            }
        ],
    ],
)
def test_load_contract_documents_rejects_invalid_schema(tmp_path, contracts):
    contracts_dir = tmp_path / "data"
    contracts_dir.mkdir()
    with open(contracts_dir / "contracts.json", "w") as f:
        json.dump(contracts, f)

    with pytest.raises(ValueError):
        load_contract_documents(contracts_dir)


def test_bad_reingest_does_not_wipe_existing_store(tmp_path):
    contracts_dir = tmp_path / "data"
    contracts_dir.mkdir()
    with open(contracts_dir / "contracts.json", "w") as f:
        json.dump(
            [
                {
                    "contract_id": "CTR-0001",
                    "counterparty_name": "Acme",
                    "contract_text": "Good text.",
                    "risk_score": 10,
                }
            ],
            f,
        )

    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    documents = load_contract_documents(Path(contracts_dir))
    store.add(documents)
    assert store.count() > 0

    with open(contracts_dir / "contracts.json", "w") as f:
        f.write("corrupted")

    with pytest.raises(ValueError):
        # mirrors ingest_cli.py's order: validate before clearing
        bad_documents = load_contract_documents(Path(contracts_dir))
        store.clear()
        store.add(bad_documents)

    assert store.count() > 0, "existing index should survive a rejected re-ingest"


def test_failed_embedding_does_not_wipe_existing_store(tmp_path):
    store = DocumentStore(fake_embedding_fn, persist_path=tmp_path / "chroma")
    store.add(
        [{"id": "a1", "text": "existing", "metadata": {"contract_id": "CTR-0001"}}]
    )

    def failing_embedding_fn(texts):
        raise RuntimeError("embedding service unavailable")

    store._embedding_fn = failing_embedding_fn
    with pytest.raises(RuntimeError, match="unavailable"):
        store.replace(
            [
                {
                    "id": "a2",
                    "text": "replacement",
                    "metadata": {"contract_id": "CTR-0002"},
                }
            ]
        )

    assert store.count() == 1


def test_document_store_add_and_query(tmp_path):
    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    store.add(
        [
            {
                "id": "a1",
                "text": "clause one text",
                "metadata": {"contract_id": "CTR-0001"},
            },
            {
                "id": "a2",
                "text": "clause two text",
                "metadata": {"contract_id": "CTR-0001"},
            },
        ]
    )

    results = store.query("clause one text", k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a1"
    assert results[0]["metadata"]["contract_id"] == "CTR-0001"


def test_ingest_contracts_creates_one_chunk_per_paragraph(tmp_path):
    contracts_dir = tmp_path / "data"
    contracts_dir.mkdir()
    contracts = [
        {
            "contract_id": "CTR-0001",
            "counterparty_name": "Acme Ltd",
            "contract_text": "Paragraph one.\n\nParagraph two.",
            "risk_score": 20,
        }
    ]
    with open(contracts_dir / "contracts.json", "w") as f:
        json.dump(contracts, f)

    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    count = ingest_contracts(Path(contracts_dir), store)

    assert count == 2
    results = store.query("Paragraph one.", k=2)
    assert len(results) == 2
    assert all(r["metadata"]["contract_id"] == "CTR-0001" for r in results)
