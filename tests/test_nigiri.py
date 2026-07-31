from __future__ import annotations

import pytest
import responses

from bentotruck.nigiri import (
    Anthropic,
    Gemini,
    Message,
    Mock,
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
