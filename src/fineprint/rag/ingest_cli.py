from fineprint.rag.ingest import default_embedding_fn, load_contract_documents
from fineprint.rag.store import DocumentStore


def main() -> None:
    """Rebuilds the document store from data/contracts.json."""
    documents = load_contract_documents()

    store = DocumentStore(default_embedding_fn)
    store.replace(documents)

    print(f"Ingested {len(documents)} chunks into the document store.")


if __name__ == "__main__":
    main()
