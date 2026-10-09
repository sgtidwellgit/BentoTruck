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
    """
    Normalized result of a generate() call — text plus any tool calls the model requested.

    `usage` holds token counts normalized to `{"input_tokens": ..., "output_tokens": ...}`
    when the vendor reports them, and is empty otherwise.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def calling(cls, name: str, content: str = "", **arguments: Any) -> "ModelResponse":
        """Shorthand for a response that requests one tool call — handy for scripting `Mock` replies."""

        return cls(content=content, tool_calls=[ToolCall(id=_new_id(), name=name, arguments=arguments)])


def _usage(input_tokens: Any, output_tokens: Any) -> dict[str, int]:
    """Normalize a vendor's token counts, dropping anything it didn't report."""

    usage: dict[str, int] = {}
    if input_tokens is not None:
        usage["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        usage["output_tokens"] = int(output_tokens)
    return usage


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

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Convenience: send one user prompt (plus optional system prompt) and return the reply text."""

        messages = [Message(role="system", content=system)] if system else []
        messages.append(Message(role="user", content=prompt))
        return self.generate(messages, **kwargs).content


class Mock(ModelProvider):
    """
    Offline, deterministic provider for tests and demos — makes no network calls.

    Three ways to script it, checked in this order:

    - `responder(messages, tools) -> ModelResponse` for fully dynamic behavior
    - `replies=[...]` — strings or ModelResponses returned one per call; once
      only one is left, it repeats forever
    - `reply="..."` — the same text every call
    """

    def __init__(
        self,
        reply: str = "ok",
        *,
        replies: list[str | ModelResponse] | None = None,
        responder: Callable[[list[Message], list[ToolSpec] | None], ModelResponse] | None = None,
    ) -> None:
        self.reply = reply
        self.replies = list(replies or [])
        self.responder = responder
        self.calls: list[list[Message]] = []

    def generate(self, messages, *, tools=None, **kwargs):
        self.calls.append(list(messages))
        if self.responder is not None:
            return self.responder(messages, tools)
        if self.replies:
            nxt = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(nxt, ModelResponse):
                return nxt if nxt.model else ModelResponse(nxt.content, nxt.tool_calls, nxt.raw, "mock", nxt.usage)
            return ModelResponse(content=nxt, model="mock")
        return ModelResponse(content=self.reply, model="mock")


# --- OpenAI-shaped wire format -------------------------------------------------
# Shared by OpenAI, Azure, vLLM, LM Studio, and Ollama, whose chat APIs all mirror this message/tool shape.


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


class OpenAICompatible(ModelProvider):
    """
    Provider for any server that speaks the OpenAI Chat Completions wire format —
    vLLM, LM Studio, llama.cpp, LiteLLM, and many hosted gateways. The API key is
    optional because most self-hosted servers don't require one.
    """

    def __init__(self, model: str, *, base_url: str, api_key: str | None = None, timeout: int = 60) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _params(self) -> dict[str, str] | None:
        return None

    def generate(self, messages, *, tools=None, **kwargs):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
        }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
        payload.update(kwargs)

        response = requests.post(
            self._url(), headers=self._headers(), params=self._params(), json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]

        tool_calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"] or "{}"))
            for tc in choice.get("tool_calls") or []
        ]
        usage = data.get("usage") or {}
        return ModelResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            raw=data,
            model=data.get("model", self.model),
            usage=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
        )


class OpenAI(OpenAICompatible):
    """Provider for the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 30,
    ) -> None:
        super().__init__(model, base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"), timeout=timeout)

    def generate(self, messages, *, tools=None, **kwargs):
        if not self.api_key:
            raise RuntimeError("OpenAI provider requires an API key (pass api_key= or set OPENAI_API_KEY).")
        return super().generate(messages, tools=tools, **kwargs)


class Azure(OpenAICompatible):
    """
    Provider for Azure OpenAI. `endpoint` is your resource URL
    (https://<resource>.openai.azure.com) and `deployment` is the name of the
    model deployment you created in that resource.
    """

    def __init__(
        self,
        deployment: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-10-21",
        timeout: int = 30,
    ) -> None:
        super().__init__(
            deployment,
            base_url=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "",
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            timeout=timeout,
        )
        self.deployment = deployment
        self.api_version = api_version

    def _url(self) -> str:
        return f"{self.base_url}/openai/deployments/{self.deployment}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "api-key": self.api_key or ""}

    def _params(self) -> dict[str, str]:
        return {"api-version": self.api_version}

    def generate(self, messages, *, tools=None, **kwargs):
        if not self.base_url:
            raise RuntimeError("Azure provider requires an endpoint (pass endpoint= or set AZURE_OPENAI_ENDPOINT).")
        if not self.api_key:
            raise RuntimeError("Azure provider requires an API key (pass api_key= or set AZURE_OPENAI_API_KEY).")
        return super().generate(messages, tools=tools, **kwargs)


class VLLM(OpenAICompatible):
    """Provider for a vLLM server's OpenAI-compatible endpoint (default http://localhost:8000/v1)."""

    def __init__(
        self, model: str, *, base_url: str = "http://localhost:8000/v1", api_key: str | None = None, timeout: int = 60
    ) -> None:
        super().__init__(model, base_url=base_url, api_key=api_key, timeout=timeout)


class LMStudio(OpenAICompatible):
    """Provider for LM Studio's local server (default http://localhost:1234/v1) — no API key required."""

    def __init__(
        self, model: str, *, base_url: str = "http://localhost:1234/v1", api_key: str | None = None, timeout: int = 60
    ) -> None:
        super().__init__(model, base_url=base_url, api_key=api_key, timeout=timeout)


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
            content=message.get("content") or "",
            tool_calls=tool_calls,
            raw=data,
            model=data.get("model", self.model),
            usage=_usage(data.get("prompt_eval_count"), data.get("eval_count")),
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

        usage = data.get("usage") or {}
        return ModelResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=data,
            model=data.get("model"),
            usage=_usage(usage.get("input_tokens"), usage.get("output_tokens")),
        )


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
        usage = data.get("usageMetadata") or {}
        return ModelResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=data,
            model=self.model,
            usage=_usage(usage.get("promptTokenCount"), usage.get("candidatesTokenCount")),
        )


__all__ = [
    "ToolSpec",
    "ToolCall",
    "Message",
    "ModelResponse",
    "ModelProvider",
    "Mock",
    "OpenAICompatible",
    "OpenAI",
    "Azure",
    "VLLM",
    "LMStudio",
    "Anthropic",
    "Gemini",
    "Ollama",
]
