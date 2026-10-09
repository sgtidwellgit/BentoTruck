from __future__ import annotations

import pytest
import responses

from bentotruck.nigiri import (
    Anthropic,
    Gemini,
    Message,
    Mock,
    ModelResponse,
    Ollama,
    OpenAI,
    ToolCall,
    ToolSpec,
)


def test_mock_returns_fixed_reply():
    provider = Mock(reply="hello")
    response = provider.generate([Message(role="user", content="hi")])
    assert response.content == "hello"
    assert response.tool_calls == []
    assert len(provider.calls) == 1


def test_mock_responder_receives_messages_and_tools():
    seen = {}

    def responder(messages, tools):
        seen["messages"] = messages
        seen["tools"] = tools
        return Mock().generate(messages)

    tool_spec = ToolSpec(name="t", description="d")
    provider = Mock(responder=responder)
    provider.generate([Message(role="user", content="hi")], tools=[tool_spec])

    assert seen["tools"] == [tool_spec]
    assert seen["messages"][0].content == "hi"


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAI(api_key=None)
    with pytest.raises(RuntimeError, match="API key"):
        provider.generate([Message(role="user", content="hi")])


@responses.activate
def test_openai_parses_text_response():
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
        },
        status=200,
    )
    provider = OpenAI(api_key="sk-test")
    response = provider.generate([Message(role="user", content="hi")])
    assert response.content == "hi there"
    assert response.tool_calls == []
    assert response.model == "gpt-4o-mini"


@responses.activate
def test_openai_parses_tool_call():
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}
                        ],
                    }
                }
            ],
        },
        status=200,
    )
    provider = OpenAI(api_key="sk-test")
    response = provider.generate(
        [Message(role="user", content="add 1 and 2")],
        tools=[ToolSpec(name="add", description="adds two numbers")],
    )
    assert response.content == ""
    assert response.tool_calls == [ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2})]


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = Anthropic(model="some-model", api_key=None)
    with pytest.raises(RuntimeError, match="API key"):
        provider.generate([Message(role="user", content="hi")])


@responses.activate
def test_anthropic_parses_text_and_tool_use():
    responses.add(
        responses.POST,
        "https://api.anthropic.com/v1/messages",
        json={
            "model": "some-model",
            "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"query": "weather"}},
            ],
        },
        status=200,
    )
    provider = Anthropic(model="some-model", api_key="key-test")
    response = provider.generate(
        [Message(role="system", content="be terse"), Message(role="user", content="what's the weather?")],
        tools=[ToolSpec(name="lookup", description="looks things up")],
    )
    assert response.content == "let me check"
    assert response.tool_calls == [ToolCall(id="toolu_1", name="lookup", arguments={"query": "weather"})]

    sent_body = responses.calls[0].request.body
    assert b'"system": "be terse"' in sent_body


@responses.activate
def test_gemini_parses_text_and_function_call():
    responses.add(
        responses.POST,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "checking"},
                            {"functionCall": {"name": "lookup", "args": {"query": "weather"}}},
                        ]
                    }
                }
            ]
        },
        status=200,
    )
    provider = Gemini(api_key="key-test")
    response = provider.generate(
        [Message(role="user", content="what's the weather?")],
        tools=[ToolSpec(name="lookup", description="looks things up")],
    )
    assert response.content == "checking"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "weather"}


@responses.activate
def test_ollama_parses_response_no_api_key_required():
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={"model": "llama3", "message": {"role": "assistant", "content": "hi"}},
        status=200,
    )
    provider = Ollama()
    response = provider.generate([Message(role="user", content="hi")])
    assert response.content == "hi"
    assert response.tool_calls == []


def test_mock_replies_queue_repeats_last():
    provider = Mock(replies=["one", ModelResponse.calling("t", x=1), "last"])
    assert provider.generate([]).content == "one"
    second = provider.generate([])
    assert second.tool_calls[0].name == "t" and second.tool_calls[0].arguments == {"x": 1}
    assert second.model == "mock"
    assert [provider.generate([]).content for _ in range(2)] == ["last", "last"]


def test_complete_convenience():
    provider = Mock(reply="hi")
    assert provider.complete("hello", system="be nice") == "hi"
    assert [m.role for m in provider.calls[0]] == ["system", "user"]


@responses.activate
def test_openai_reports_usage():
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    )
    assert OpenAI(api_key="k").generate([Message(role="user", content="hi")]).usage == {
        "input_tokens": 7,
        "output_tokens": 3,
    }


@responses.activate
def test_azure_uses_deployment_url_and_api_key_header():
    from bentotruck.nigiri import Azure

    responses.add(
        responses.POST,
        "https://res.openai.azure.com/openai/deployments/my-dep/chat/completions",
        json={"choices": [{"message": {"content": "azure hi"}}]},
    )
    provider = Azure("my-dep", endpoint="https://res.openai.azure.com", api_key="k")
    assert provider.generate([Message(role="user", content="hi")]).content == "azure hi"
    request = responses.calls[0].request
    assert request.headers["api-key"] == "k"
    assert "api-version=" in request.url


def test_azure_requires_endpoint_and_key(monkeypatch):
    from bentotruck.nigiri import Azure

    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="endpoint"):
        Azure("d").generate([])
    with pytest.raises(RuntimeError, match="API key"):
        Azure("d", endpoint="https://x").generate([])


@responses.activate
def test_vllm_and_lmstudio_need_no_key():
    from bentotruck.nigiri import LMStudio, VLLM

    responses.add(responses.POST, "http://localhost:8000/v1/chat/completions", json={"choices": [{"message": {"content": "v"}}]})
    responses.add(responses.POST, "http://localhost:1234/v1/chat/completions", json={"choices": [{"message": {"content": "l"}}]})
    assert VLLM("m").generate([Message(role="user", content="hi")]).content == "v"
    assert LMStudio("m").generate([Message(role="user", content="hi")]).content == "l"
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_anthropic_gemini_ollama_usage():
    responses.add(responses.POST, "https://api.anthropic.com/v1/messages",
                  json={"content": [{"type": "text", "text": "a"}], "usage": {"input_tokens": 1, "output_tokens": 2}})
    responses.add(responses.POST,
                  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                  json={"candidates": [{"content": {"parts": [{"text": "g"}]}}],
                        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4}})
    responses.add(responses.POST, "http://localhost:11434/api/chat",
                  json={"message": {"content": "o"}, "prompt_eval_count": 5, "eval_count": 6})
    msgs = [Message(role="user", content="hi")]
    assert Anthropic(model="m", api_key="k").generate(msgs).usage == {"input_tokens": 1, "output_tokens": 2}
    assert Gemini(api_key="k").generate(msgs).usage == {"input_tokens": 3, "output_tokens": 4}
    assert Ollama().generate(msgs).usage == {"input_tokens": 5, "output_tokens": 6}
