from collections.abc import Callable
from difflib import get_close_matches
from pathlib import Path
import re
from typing import Any
import unicodedata

import chromadb

DEFAULT_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "chroma"
_REFERENCE_ID_RE = re.compile(r"\b([a-z]{2,})\s*[-_ ]\s*(\d+)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")


def normalize_references(text: str) -> str:
    """Canonicalize user-entered reference IDs, e.g. ``ctr-006`` -> ``CTR-006``."""
    normalized = unicodedata.normalize("NFKC", text)
    return _REFERENCE_ID_RE.sub(
        lambda match: f"{match.group(1).upper()}-{match.group(2)}", normalized
    )


class DocumentStore:
    def __init__(
        self,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        persist_path: str | Path = DEFAULT_PERSIST_PATH,
        collection_name: str = "contracts",
    ):
        self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(collection_name)

    def _prepare_records(
        self, documents: list[dict[str, Any]]
    ) -> tuple[list[str], list[str], list[dict], list[list[float]]]:
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        return ids, texts, metadatas, self._embedding_fn(texts)

    def _upsert(
        self, records: tuple[list[str], list[str], list[dict], list[list[float]]]
    ) -> None:
        ids, texts, metadatas, embeddings = records
        self._collection.upsert(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def add(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return

        self._upsert(self._prepare_records(documents))

    def replace(self, documents: list[dict[str, Any]]) -> None:
        """Rebuilds the store. Embeds first, so a failed embed call never wipes existing data."""
        records = self._prepare_records(documents) if documents else None
        self.clear()
        if records is not None:
            self._upsert(records)

    def query(self, text: str, k: int = 5) -> list[dict]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        query_text = self._expand_query(text)
        query_embedding = self._embedding_fn([query_text])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        semantic_results = [
            {"id": doc_id, "text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                strict=True,
            )
        ]
        exact_matches = self._reference_matches(query_text)
        merged = {record["id"]: record for record in [*exact_matches, *semantic_results]}
        return list(merged.values())[:k]

    def _reference_matches(self, query: str) -> list[dict]:
        requested_ids = {
            normalize_references(match.group(0))
            for match in _REFERENCE_ID_RE.finditer(query)
        }
        if not requested_ids:
            return []

        corpus = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": doc_id, "text": document, "metadata": metadata}
            for doc_id, document, metadata in zip(
                corpus["ids"], corpus["documents"], corpus["metadatas"], strict=True
            )
            if normalize_references(str(metadata.get("contract_id", ""))) in requested_ids
        ]

    def _expand_query(self, text: str) -> str:
        """Normalize reference IDs and repair high-confidence misspellings.

        The embedding model is case-sensitive enough for IDs and can miss a single
        typo in a short query.  The vocabulary is derived from the indexed corpus,
        so corrections are limited to words the assistant can actually retrieve.
        """
        normalized = normalize_references(text)
        corpus = self._collection.get(include=["documents"])
        vocabulary = {
            word.lower()
            for document in corpus.get("documents", [])
            if document
            for word in _WORD_RE.findall(document)
        }
        if not vocabulary:
            return normalized

        def correct(match: re.Match[str]) -> str:
            word = match.group(0)
            lower_word = word.lower()
            if lower_word in vocabulary:
                return word
            matches = get_close_matches(lower_word, vocabulary, n=1, cutoff=0.84)
            return matches[0] if matches else word

        corrected = _WORD_RE.sub(correct, normalized)
        # Keep the original wording too: a correction is only a retrieval aid.
        return normalized if corrected == normalized else f"{normalized}\n{corrected}"

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)
