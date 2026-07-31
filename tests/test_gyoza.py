from __future__ import annotations

import sqlite3

import pytest
import responses

from bentotruck.gyoza import FilesystemTool, PythonTool, RESTTool, SQLTool, Toolbox


class TestPythonTool:
    def setup_method(self):
        self.tool = PythonTool()

    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 + 2", 4),
            ("2 + 2 * 3", 8),
            ("(2 + 3) * 4", 20),
            ("10 / 4", 2.5),
            ("-5 + 1", -4),
            ("3 > 2", True),
            ("max(1, 2, 3)", 3),
            ("sum([1, 2, 3])", 6),
            ("len([1, 2, 3])", 3),
            ("round(3.456, 2)", 3.46),
        ],
    )
    def test_evaluates_safe_expressions(self, expression, expected):
        assert self.tool.run(expression=expression) == expected

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo hi')",
            "().__class__.__bases__[0].__subclasses__()",
            "open('/etc/passwd')",
            "[x for x in range(10)]",
            "exec('1+1')",
            "a = 1",
            "os.system('echo hi')",
        ],
    )
    def test_rejects_unsafe_expressions(self, expression):
        with pytest.raises((ValueError, SyntaxError)):
            self.tool.run(expression=expression)

    def test_to_schema(self):
        schema = self.tool.to_schema()
        assert schema.name == "python"
        assert "expression" in schema.parameters["properties"]


class TestRESTTool:
    @responses.activate
    def test_run_get(self):
        responses.add(responses.GET, "https://example.com/api", json={"ok": True}, status=200)
        tool = RESTTool("api", "calls example api", url="https://example.com/api")
        assert tool.run() == {"ok": True}

    @responses.activate
    def test_run_post_with_json_body(self):
        responses.add(responses.POST, "https://example.com/api", json={"created": True}, status=201)
        tool = RESTTool("api", "creates a thing", url="https://example.com/api", method="post")
        result = tool.run(json={"name": "widget"})
        assert result == {"created": True}
        assert responses.calls[0].request.body == b'{"name": "widget"}'


class TestSQLTool:
    def setup_method(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        self.conn.execute("INSERT INTO users VALUES (1, 'alice'), (2, 'bob')")
        self.conn.commit()
        self.tool = SQLTool(self.conn)

    def test_run_returns_rows_as_dicts(self):
        rows = self.tool.run(query="SELECT * FROM users ORDER BY id")
        assert rows == [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]

    def test_run_binds_params_safely(self):
        malicious_name = "bob' OR '1'='1"
        rows = self.tool.run(query="SELECT * FROM users WHERE name = ?", params=[malicious_name])
        assert rows == []

    def test_run_with_matching_param(self):
        rows = self.tool.run(query="SELECT * FROM users WHERE name = ?", params=["alice"])
        assert rows == [{"id": 1, "name": "alice"}]


class TestFilesystemTool:
    def setup_method(self, tmp_path=None):
        pass

    def test_write_read_list_roundtrip(self, tmp_path):
        tool = FilesystemTool(tmp_path)
        tool.run(action="write", path="notes.txt", content="hello")
        assert tool.run(action="read", path="notes.txt") == "hello"
        assert tool.run(action="list", path=".") == ["notes.txt"]

    def test_rejects_path_traversal(self, tmp_path):
        tool = FilesystemTool(tmp_path / "sandbox")
        with pytest.raises(ValueError, match="escapes"):
            tool.run(action="read", path="../../secret.txt")

    def test_write_creates_nested_dirs(self, tmp_path):
        tool = FilesystemTool(tmp_path)
        tool.run(action="write", path="a/b/c.txt", content="deep")
        assert tool.run(action="read", path="a/b/c.txt") == "deep"


class TestToolbox:
    def test_add_get_iter_len(self):
        box = Toolbox(PythonTool())
        assert len(box) == 1
        assert box.get("python") is not None
        assert box.get("missing") is None

        box.add(RESTTool("api", "d", url="https://example.com"))
        assert len(box) == 2
        names = {tool.name for tool in box}
        assert names == {"python", "api"}
