from unittest.mock import Mock

import pytest
import requests

from backend import llm
from backend.store import SearchResult


def test_answer_prompt_requires_evidence_and_allows_insufficient_information(
    monkeypatch,
) -> None:
    response = Mock()
    response.json.return_value = {
        "message": {
            "content": (
                "The uploaded agricultural scheme documents do not contain "
                "sufficient information to answer this question."
            )
        }
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(llm.requests, "post", post)
    source = SearchResult(
        text="Excerpt text",
        metadata={"source": "scheme.pdf", "page_number": 4},
        distance=0.3,
    )

    answer = llm.generate_answer("Who is eligible?", [source])

    payload = post.call_args.kwargs["json"]
    assert "Use only the supplied excerpts as evidence" in payload["messages"][0]["content"]
    system_prompt = " ".join(payload["messages"][0]["content"].split())
    assert "do not contain" in system_prompt
    assert "[Source: scheme.pdf, page 4]" in payload["messages"][1]["content"]
    assert "sufficient information" in answer


def test_ollama_connection_failure_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        Mock(side_effect=requests.ConnectionError("connection refused")),
    )

    with pytest.raises(llm.OllamaUnavailableError, match="Cannot connect to Ollama"):
        llm.generate_answer(
            "What is in this document?",
            [
                SearchResult(
                    text="Excerpt",
                    metadata={"source": "scheme.pdf", "page_number": 1},
                    distance=0.1,
                )
            ],
        )
