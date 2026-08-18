import os

from langchain_anthropic import ChatAnthropic

from fineprint.agent.graph import build_graph
from fineprint.rag.ingest import default_embedding_fn
from fineprint.rag.store import DocumentStore


def ask(question: str, store: DocumentStore | None = None, llm=None) -> str:
    if store is None:
        store = DocumentStore(default_embedding_fn)
        if store.count() == 0:
            raise RuntimeError(
                "Document store is empty. Run ingestion first:\n"
                "  uv run python -m fineprint.rag.ingest_cli"
            )
    if llm is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it before asking questions, for example:\n"
                "  export ANTHROPIC_API_KEY='your-api-key'"
            )
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

    graph = build_graph(llm)
    result = graph.invoke(
        {
            "question": question,
            "store": store,
            "retrieved": [],
            "tool_calls_made": [],
            "answer": "",
        }
    )
    return result["answer"]
