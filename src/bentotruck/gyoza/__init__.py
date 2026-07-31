"""gyoza — tools: one calling interface across Python, REST, SQL, and the filesystem."""

from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


__all__ = ["Tool", "PythonTool", "RESTTool", "SQLTool", "FilesystemTool", "Toolbox"]
