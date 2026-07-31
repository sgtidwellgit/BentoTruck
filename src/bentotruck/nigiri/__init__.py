"""nigiri — models: vendor-neutral providers, all speaking one generate() interface."""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import requests


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ToolSpec:
    """Vendor-neutral description of a callable tool, translated per-provider on the wire."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolCall:
    """A model's request to invoke a tool, normalized across providers."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in a conversation. role is one of 'system' | 'user' | 'assistant' | 'tool'."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ModelResponse:
    """Normalized result of a generate() call — text plus any tool calls the model requested."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class ModelProvider(ABC):
    """Base class every model provider implements: one generate() in, one ModelResponse out."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        raise NotImplementedError


class Mock(ModelProvider):
    """
    Offline, deterministic provider for tests and demos — makes no network calls.

    Pass a fixed `reply`, or a `responder` callable for scripted multi-turn
    behavior: `responder(messages, tools) -> ModelResponse`.
    """

    def __init__(
        self,
        reply: str = "ok",
        *,
        responder: Callable[[list[Message], list[ToolSpec] | None], ModelResponse] | None = None,
    ) -> None:
        self.reply = reply
        self.responder = responder
        self.calls: list[list[Message]] = []

    def generate(self, messages, *, tools=None, **kwargs):
        self.calls.append(list(messages))
        if self.responder is not None:
            return self.responder(messages, tools)
        return ModelResponse(content=self.reply, model="mock")


# --- OpenAI-shaped wire format -------------------------------------------------
# Shared by OpenAI and Ollama, whose chat APIs both mirror this message/tool shape.


def _message_to_openai(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    msg: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in m.tool_calls
        ]
    return msg


def _tool_to_openai(t: ToolSpec) -> dict[str, Any]:
    return {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}


class OpenAI(ModelProvider):
    """Provider for the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, messages, *, tools=None, **kwargs):
        if not self.api_key:
            raise RuntimeError("OpenAI provider requires an API key (pass api_key= or set OPENAI_API_KEY).")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
        }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]

        tool_calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"] or "{}"))
            for tc in choice.get("tool_calls") or []
        ]
        return ModelResponse(content=choice.get("content") or "", tool_calls=tool_calls, raw=data, model=data.get("model"))


class Ollama(ModelProvider):
    """Provider for a local Ollama server — no API key required."""

    def __init__(self, model: str = "llama3", *, base_url: str = "http://localhost:11434", timeout: int = 60) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, messages, *, tools=None, **kwargs):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
        payload.update(kwargs)

        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data["message"]

        tool_calls = [
            ToolCall(id=_new_id(), name=tc["function"]["name"], arguments=tc["function"].get("arguments") or {})
            for tc in message.get("tool_calls") or []
        ]
        return ModelResponse(
            content=message.get("content") or "", tool_calls=tool_calls, raw=data, model=data.get("model", self.model)
        )


# --- Anthropic Messages API -----------------------------------------------------


def _message_to_anthropic(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}],
        }
    if m.tool_calls:
        content: list[dict[str, Any]] = []
        if m.content:
            content.append({"type": "text", "text": m.content})
        content.extend({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m.tool_calls)
        return {"role": m.role, "content": content}
    return {"role": m.role, "content": m.content}


def _tool_to_anthropic(t: ToolSpec) -> dict[str, Any]:
    return {"name": t.name, "description": t.description, "input_schema": t.parameters}


class Anthropic(ModelProvider):
    """
    Provider for the Anthropic Messages API.

    `model` is required with no built-in default — pass whichever model
    identifier your account has access to.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, messages, *, tools=None, **kwargs):
        if not self.api_key:
            raise RuntimeError("Anthropic provider requires an API key (pass api_key= or set ANTHROPIC_API_KEY).")

        system_prompt = "\n".join(m.content for m in messages if m.role == "system") or None
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "messages": [_message_to_anthropic(m) for m in messages if m.role != "system"],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = [_tool_to_anthropic(t) for t in tools]
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {}))

        return ModelResponse(content="".join(text_parts), tool_calls=tool_calls, raw=data, model=data.get("model"))


# --- Gemini generateContent API -------------------------------------------------


def _message_to_gemini(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "function", "parts": [{"functionResponse": {"name": m.name, "response": {"content": m.content}}}]}
    role = "model" if m.role == "assistant" else "user"
    parts: list[dict[str, Any]] = []
    if m.content:
        parts.append({"text": m.content})
    if m.tool_calls:
        parts.extend({"functionCall": {"name": tc.name, "args": tc.arguments}} for tc in m.tool_calls)
    return {"role": role, "parts": parts}


def _tool_to_gemini(t: ToolSpec) -> dict[str, Any]:
    return {"name": t.name, "description": t.description, "parameters": t.parameters}


class Gemini(ModelProvider):
    """Provider for the Gemini generateContent API."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, messages, *, tools=None, **kwargs):
        if not self.api_key:
            raise RuntimeError("Gemini provider requires an API key (pass api_key= or set GEMINI_API_KEY).")

        system_prompt = "\n".join(m.content for m in messages if m.role == "system") or None
        payload: dict[str, Any] = {"contents": [_message_to_gemini(m) for m in messages if m.role != "system"]}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": [_tool_to_gemini(t) for t in tools]}]
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        parts = data["candidates"][0]["content"]["parts"]
        text_parts = [p["text"] for p in parts if "text" in p]
        tool_calls = [
            ToolCall(id=_new_id(), name=p["functionCall"]["name"], arguments=p["functionCall"].get("args") or {})
            for p in parts
            if "functionCall" in p
        ]
        return ModelResponse(content="".join(text_parts), tool_calls=tool_calls, raw=data, model=self.model)


__all__ = [
    "ToolSpec",
    "ToolCall",
    "Message",
    "ModelResponse",
    "ModelProvider",
    "Mock",
    "OpenAI",
    "Anthropic",
    "Gemini",
    "Ollama",
]
