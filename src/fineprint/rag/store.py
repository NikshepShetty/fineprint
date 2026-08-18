from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb

DEFAULT_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "chroma"


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

        query_embedding = self._embedding_fn([text])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        return [
            {"id": doc_id, "text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                strict=True,
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)
