"""gyoza — tools: one calling interface across Python, REST, SQL, files, the web, Docker, and MCP."""

from __future__ import annotations

import ast
import inspect
import ipaddress
import json
import operator
import shutil
import socket
import subprocess
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from bentotruck.nigiri import ToolSpec


class Tool(ABC):
    """Base class every tool implements: a name, a description, a schema, and a run()."""

    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def to_schema(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len}


class PythonTool(Tool):
    """
    Evaluates a single Python expression in a restricted sandbox — arithmetic,
    comparisons, lists, and a small allow-listed function set. No imports, no
    attribute access, no statements: this is a calculator, not a code-exec backdoor.
    """

    name = "python"
    description = "Evaluate a Python arithmetic/comparison expression and return the result."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "A Python expression, e.g. '2 + 2 * 3'."}},
        "required": ["expression"],
    }

    def run(self, *, expression: str) -> Any:
        tree = ast.parse(expression, mode="eval")
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](self._eval(node.left), self._eval(node.comparators[0]))
        if isinstance(node, ast.List):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            args = [self._eval(a) for a in node.args]
            return _FUNCS[node.func.id](*args)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class RESTTool(Tool):
    """Calls a fixed HTTP endpoint. Bind one instance per external API call the agent should be able to make."""

    def __init__(
        self,
        name: str,
        description: str,
        *,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> None:
        self.name = name
        self.description = description
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout = timeout
        self.parameters = {
            "type": "object",
            "properties": {
                "params": {"type": "object", "description": "Query parameters."},
                "json": {"type": "object", "description": "JSON request body."},
            },
        }

    def run(self, *, params: dict[str, Any] | None = None, json: dict[str, Any] | None = None) -> Any:
        response = requests.request(
            self.method, self.url, params=params, json=json, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return response.text


class SQLTool(Tool):
    """
    Runs a parameterized query against a DB-API 2.0 connection (sqlite3, psycopg2, etc).
    Always bind values through `params` — never interpolate them into `query`.
    """

    name = "sql"
    description = "Execute a parameterized read query and return the resulting rows."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL with ? or %s placeholders — never inline values."},
            "params": {"type": "array", "description": "Values bound to the query's placeholders.", "items": {}},
        },
        "required": ["query"],
    }

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def run(self, *, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self.connection.cursor()
        cursor.execute(query, params or [])
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]


class FilesystemTool(Tool):
    """Read/write/list files, sandboxed to a single root directory — path traversal outside it is rejected."""

    name = "filesystem"
    description = "Read, write, or list files within a sandboxed root directory."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write", "list"]},
            "path": {"type": "string", "description": "Path relative to the sandbox root."},
            "content": {"type": "string", "description": "Content to write (action='write' only)."},
        },
        "required": ["action", "path"],
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Path {path!r} escapes the sandbox root.")
        return candidate

    def run(self, *, action: str, path: str, content: str | None = None) -> Any:
        target = self._resolve(path)
        if action == "read":
            return target.read_text(encoding="utf-8")
        if action == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
            return f"Wrote {len(content or '')} chars to {path}"
        if action == "list":
            return sorted(p.name for p in target.iterdir())
        raise ValueError(f"Unsupported action: {action!r}")


class Toolbox:
    """A named collection of tools — hand one to Agent.with_tools(*toolbox) or iterate it directly."""

    def __init__(self, *tools: Tool) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def add(self, tool: Tool) -> "Toolbox":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


# --- Functions as tools -----------------------------------------------------------

_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
# typing.Optional[X] / Union[...] everywhere, plus PEP 604 `X | None` on 3.10+
_UNION_ORIGINS = (typing.Union, types.UnionType) if hasattr(types, "UnionType") else (typing.Union,)


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = typing.get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {}
    base = origin or annotation
    return {"type": _JSON_TYPES[base]} if base in _JSON_TYPES else {}


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable hints just mean an untyped schema
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param.name] = _json_type(hints.get(param.name, Any))
        if param.default is param.empty:
            required.append(param.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class FunctionTool(Tool):
    """
    Turns a plain Python function into a tool. The name defaults to the function's
    name, the description to its docstring, and the JSON schema is inferred from
    its signature and type hints. Still callable as the original function.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        doc = inspect.getdoc(fn) or ""
        self.description = description or doc.split("\n\n")[0].replace("\n", " ") or self.name
        self.parameters = parameters or _schema_from_signature(fn)

    def run(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None) -> Any:
    """
    Decorator form of FunctionTool — use bare (`@gyoza.tool`) or with overrides
    (`@gyoza.tool(name="lookup")`).
    """

    def wrap(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(f, name=name, description=description)

    return wrap(fn) if fn is not None else wrap


# --- Browser ------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._chunks: list[str] = []
        self._skipping = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self._chunks.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._chunks).splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document — scripts, styles, and markup stripped."""

    parser = _TextExtractor()
    parser.feed(html)
    return parser.title.strip(), parser.text()


class BrowserTool(Tool):
    """
    Fetches a web page and returns its readable text (no JavaScript execution).

    Safe by default: only http(s), optional `allowed_domains` allow-list, and
    hosts that resolve to private/loopback/link-local addresses are refused so an
    agent can't be steered into probing your internal network.
    """

    name = "browse"
    description = "Fetch a web page by URL and return its title and readable text."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "An http(s) URL."}},
        "required": ["url"],
    }

    def __init__(
        self,
        *,
        allowed_domains: list[str] | None = None,
        block_private: bool = True,
        max_chars: int = 8000,
        timeout: int = 20,
        user_agent: str = "bentotruck-browser",
    ) -> None:
        self.allowed_domains = [d.lower().lstrip(".") for d in allowed_domains or []]
        self.block_private = block_private
        self.max_chars = max_chars
        self.timeout = timeout
        self.user_agent = user_agent

    def _check(self, url: str) -> None:
        parts = urlparse(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in ("http", "https") or not host:
            raise ValueError(f"Only absolute http(s) URLs are allowed, got {url!r}.")
        if self.allowed_domains and not any(host == d or host.endswith("." + d) for d in self.allowed_domains):
            raise ValueError(f"Host {host!r} is not in the allowed domains.")
        if self.block_private:
            for info in socket.getaddrinfo(host, None):
                address = ipaddress.ip_address(info[4][0])
                if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                    raise ValueError(f"Host {host!r} resolves to a non-public address.")

    def run(self, *, url: str) -> str:
        self._check(url)
        response = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
        response.raise_for_status()
        if "html" in response.headers.get("Content-Type", "html"):
            title, text = html_to_text(response.text)
        else:
            title, text = "", response.text
        body = text[: self.max_chars]
        if len(text) > self.max_chars:
            body += "\n[truncated]"
        return f"{title}\n\n{body}" if title else body


# --- Docker -------------------------------------------------------------------------


class DockerTool(Tool):
    """
    Runs a shell command inside a throwaway Docker container — the place for
    arbitrary code an agent writes, rather than your host.

    Locked down by default: no network, capped memory/CPU/PIDs, read-only root
    filesystem with a writable /tmp, all capabilities dropped, and removed on exit.
    """

    name = "docker"
    description = "Run a shell command in an isolated, network-less Docker container and return its output."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Shell command to run, e.g. 'python -c \"print(1)\"'."}},
        "required": ["command"],
    }

    def __init__(
        self,
        image: str = "python:3.12-slim",
        *,
        network: bool = False,
        memory: str = "256m",
        cpus: str = "1",
        pids_limit: int = 128,
        timeout: int = 60,
        max_output: int = 8000,
        docker: str = "docker",
    ) -> None:
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout = timeout
        self.max_output = max_output
        self.docker = docker

    def command_line(self, command: str) -> list[str]:
        args = [self.docker, "run", "--rm", "-i", "--memory", self.memory, "--cpus", self.cpus]
        args += ["--pids-limit", str(self.pids_limit), "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL"]
        args += ["--security-opt", "no-new-privileges"]
        if not self.network:
            args += ["--network", "none"]
        return args + [self.image, "sh", "-c", command]

    def run(self, *, command: str) -> dict[str, Any]:
        if shutil.which(self.docker) is None:
            raise RuntimeError(f"{self.docker!r} was not found on PATH — DockerTool needs Docker installed.")
        try:
            proc = subprocess.run(
                self.command_line(command), capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "stdout": "", "stderr": f"Timed out after {self.timeout}s."}
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-self.max_output :],
            "stderr": proc.stderr[-self.max_output :],
        }


# --- MCP (Model Context Protocol) ---------------------------------------------------


class MCPClient:
    """
    Minimal Model Context Protocol client over stdio: launches an MCP server
    process, performs the initialize handshake, and lists/calls its tools.

    >>> with MCPClient(["python", "my_server.py"]) as client:   # doctest: +SKIP
    ...     agent.with_tools(*client.tools())
    """

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(self, command: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None) -> None:
        self.command = list(command)
        self.env = env
        self.cwd = cwd
        self.server_info: dict[str, Any] = {}
        self._proc: subprocess.Popen | None = None
        self._next_id = 0

    def start(self) -> "MCPClient":
        if self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=self.env,
            cwd=self.cwd,
        )
        result = self.request(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "bentotruck", "version": "1"},
            },
        )
        self.server_info = result.get("serverInfo", {})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            if self._proc.stdout:
                self._proc.stdout.close()
            self._proc = None

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._proc is None:
            self.start()
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            reply = json.loads(line)
            if reply.get("id") != request_id or "method" in reply:
                continue  # server notifications / requests we don't handle
            if "error" in reply:
                raise RuntimeError(f"MCP error from {method}: {reply['error'].get('message', reply['error'])}")
            return reply.get("result", {})
        raise RuntimeError(f"MCP server exited before answering {method!r}.")

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def tools(self) -> list["MCPTool"]:
        """Every tool the server exposes, wrapped as a gyoza Tool ready for `Agent.with_tools(...)`."""

        return [
            MCPTool(self, t["name"], t.get("description", ""), t.get("inputSchema") or {"type": "object", "properties": {}})
            for t in self.list_tools()
        ]


class MCPTool(Tool):
    """One tool hosted on an MCP server, callable like any other gyoza Tool."""

    def __init__(self, client: MCPClient, name: str, description: str, parameters: dict[str, Any]) -> None:
        self.client = client
        self.name = name
        self.description = description
        self.parameters = parameters

    def run(self, **kwargs: Any) -> str:
        result = self.client.call_tool(self.name, kwargs)
        parts = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block))
        if not parts and "structuredContent" in result:
            parts.append(json.dumps(result["structuredContent"]))
        text = "\n".join(parts)
        if result.get("isError"):
            raise RuntimeError(text or f"MCP tool {self.name!r} reported an error.")
        return text


__all__ = [
    "Tool",
    "PythonTool",
    "RESTTool",
    "SQLTool",
    "FilesystemTool",
    "FunctionTool",
    "tool",
    "BrowserTool",
    "html_to_text",
    "DockerTool",
    "MCPClient",
    "MCPTool",
    "Toolbox",
]
