from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest
import responses

from bentotruck.gyoza import BrowserTool, DockerTool, FunctionTool, MCPClient, html_to_text, tool
from bentotruck.nigiri import Mock, ModelResponse
from bentotruck.rice import Agent

FAKE_SERVER = str(Path(__file__).with_name("fake_mcp_server.py"))


# --- FunctionTool -------------------------------------------------------------


def test_tool_decorator_infers_schema_from_signature():
    @tool
    def lookup_price(item: str, quantity: int = 1, rush: Optional[bool] = None) -> float:
        """Look up the price of a menu item.

        Longer details that should not be in the description.
        """
        return 6.0 * quantity

    assert isinstance(lookup_price, FunctionTool)
    assert lookup_price.name == "lookup_price"
    assert lookup_price.description == "Look up the price of a menu item."
    assert lookup_price.parameters == {
        "type": "object",
        "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}, "rush": {"type": "boolean"}},
        "required": ["item"],
    }
    assert lookup_price("gyoza", 2) == 12.0
    assert lookup_price.run(item="gyoza", quantity=3) == 18.0


def test_tool_decorator_with_overrides_and_agent_use():
    @tool(name="double", description="Double it.")
    def twice(x: float):
        return x * 2

    assert (twice.name, twice.description) == ("double", "Double it.")
    agent = Agent("a").using(Mock(replies=[ModelResponse.calling("double", x=21), "42"])).with_tools(twice)
    assert agent.run("double 21") == "42"


# --- BrowserTool --------------------------------------------------------------


def test_html_to_text_strips_scripts_and_markup():
    title, text = html_to_text(
        "<html><head><title> Menu </title><style>p{}</style></head>"
        "<body><h1>Gyoza</h1><script>alert(1)</script><p>Six   dollars</p></body></html>"
    )
    assert title == "Menu"
    assert text == "Gyoza\nSix dollars"


@responses.activate
def test_browser_fetches_and_truncates():
    responses.add(responses.GET, "https://example.com/menu", body="<title>Menu</title><p>" + "x" * 50 + "</p>",
                  content_type="text/html")
    out = BrowserTool(block_private=False, max_chars=10).run(url="https://example.com/menu")
    assert out == "Menu\n\nxxxxxxxxxx\n[truncated]"


def test_browser_safety_checks():
    browser = BrowserTool(allowed_domains=["example.com"], block_private=False)
    with pytest.raises(ValueError, match="http"):
        browser.run(url="file:///etc/passwd")
    with pytest.raises(ValueError, match="allowed"):
        browser.run(url="https://evil.test/")
    with pytest.raises(ValueError, match="non-public"):
        BrowserTool().run(url="http://127.0.0.1:8080/admin")


# --- DockerTool ---------------------------------------------------------------


def test_docker_command_line_is_locked_down():
    args = DockerTool("alpine").command_line("echo hi")
    for flag in ["--rm", "--read-only", "--cap-drop", "no-new-privileges"]:
        assert flag in args
    assert args[args.index("--network") + 1] == "none"
    assert args[-4:] == ["alpine", "sh", "-c", "echo hi"]
    assert "--network" not in DockerTool(network=True).command_line("x")


def test_docker_run_captures_output(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="hi\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert DockerTool().run(command="echo hi") == {"exit_code": 0, "stdout": "hi\n", "stderr": ""}
    assert seen["args"][0] == "docker"


def test_docker_timeout_and_missing_binary(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    def slow(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(subprocess, "run", slow)
    assert DockerTool(timeout=1).run(command="sleep 9")["stderr"] == "Timed out after 1s."
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="not found"):
        DockerTool().run(command="x")


# --- MCP ----------------------------------------------------------------------


def test_mcp_client_lists_and_calls_tools():
    with MCPClient([sys.executable, FAKE_SERVER]) as client:
        assert client.server_info == {"name": "fake"}
        tools = {t.name: t for t in client.tools()}
        assert set(tools) == {"add", "fail"}
        assert tools["add"].parameters["required"] == ["a", "b"]
        assert tools["fail"].parameters == {"type": "object", "properties": {}}
        assert tools["add"].run(a=2, b=3) == "5"
        with pytest.raises(RuntimeError, match="nope"):
            tools["fail"].run()
        with pytest.raises(RuntimeError, match="unknown method"):
            client.request("bogus/method")


def test_mcp_tools_work_inside_an_agent():
    with MCPClient([sys.executable, FAKE_SERVER]) as client:
        provider = Mock(replies=[ModelResponse.calling("add", a=40, b=2), "It's 42."])
        agent = Agent("a").using(provider).with_tools(*client.tools())
        assert agent.run("add 40 and 2") == "It's 42."
        assert [m.content for m in provider.calls[1] if m.role == "tool"] == ["42"]
