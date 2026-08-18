import hashlib
import json
from pathlib import Path

import pytest

from fineprint.agent.graph import build_graph
from fineprint.rag.ingest import ingest_contracts
from fineprint.rag.store import DocumentStore


def fake_embedding_fn(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        out.append([b / 255.0 for b in digest[:8]])
    return out


class FakeResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLMWithTools:
    def __init__(self, tools, tool_calls_to_make):
        self._tools = tools
        self._tool_calls_to_make = tool_calls_to_make

    def invoke(self, prompt):
        return FakeResponse(tool_calls=self._tool_calls_to_make)


class FakeLLM:
    def __init__(self, tool_calls_to_make=None, answer="fake answer"):
        self._tool_calls_to_make = tool_calls_to_make or []
        self._answer = answer

    def bind_tools(self, tools):
        return FakeLLMWithTools(tools, self._tool_calls_to_make)

    def invoke(self, prompt):
        return FakeResponse(content=self._answer)


@pytest.fixture
def store_with_one_contract(tmp_path):
    contracts_dir = tmp_path / "data"
    contracts_dir.mkdir()
    contracts = [
        {
            "contract_id": "CTR-0000",
            "counterparty_name": "Acme Ltd",
            "contract_text": "Payment terms apply.\n\nUnlimited liability clause applies.",
            "risk_score": 30,
        }
    ]
    with open(contracts_dir / "contracts.json", "w") as f:
        json.dump(contracts, f)

    store = DocumentStore(fake_embedding_fn, persist_path=str(tmp_path / "chroma"))
    ingest_contracts(Path(contracts_dir), store)
    return store


def test_graph_retrieves_documents(store_with_one_contract):
    llm = FakeLLM()
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "question": "What clauses does CTR-0000 have?",
            "store": store_with_one_contract,
            "retrieved": [],
            "tool_calls_made": [],
            "answer": "",
        }
    )

    assert len(result["retrieved"]) > 0


def test_graph_calls_tool_when_llm_requests_it(store_with_one_contract):
    llm = FakeLLM(
        tool_calls_to_make=[
            {"name": "predict_contract_risk", "args": {"contract_id": "does-not-exist"}}
        ]
    )
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "question": "Why is CTR-0000 risky?",
            "store": store_with_one_contract,
            "retrieved": [],
            "tool_calls_made": [],
            "answer": "",
        }
    )

    assert len(result["tool_calls_made"]) == 1
    assert result["tool_calls_made"][0]["tool"] == "predict_contract_risk"
    # No gold data in this test, so the tool should return its own error dict
    # rather than the graph raising - confirms tool errors don't break the flow.
    assert "error" in result["tool_calls_made"][0]["result"]


def test_graph_produces_final_answer(store_with_one_contract):
    llm = FakeLLM(answer="This is the synthesized answer.")
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "question": "What clauses does CTR-0000 have?",
            "store": store_with_one_contract,
            "retrieved": [],
            "tool_calls_made": [],
            "answer": "",
        }
    )

    assert result["answer"] == "This is the synthesized answer."


def test_graph_handles_unknown_tool_name_without_crashing(store_with_one_contract):
    llm = FakeLLM(tool_calls_to_make=[{"name": "not_a_real_tool", "args": {}}])
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "question": "Anything",
            "store": store_with_one_contract,
            "retrieved": [],
            "tool_calls_made": [],
            "answer": "",
        }
    )

    assert "error" in result["tool_calls_made"][0]["result"]
    assert "unknown tool" in result["tool_calls_made"][0]["result"]["error"]


def test_ask_raises_clear_error_when_store_not_ingested(tmp_path, monkeypatch):
    import fineprint.agent.run as agent_run

    monkeypatch.setattr(
        agent_run,
        "DocumentStore",
        lambda embedding_fn: DocumentStore(
            embedding_fn, persist_path=tmp_path / "chroma"
        ),
    )

    with pytest.raises(RuntimeError, match="ingestion"):
        agent_run.ask("anything", llm=FakeLLM())


def test_ask_requires_anthropic_api_key(store_with_one_contract, monkeypatch):
    from fineprint.agent.run import ask

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        ask("anything", store=store_with_one_contract)

