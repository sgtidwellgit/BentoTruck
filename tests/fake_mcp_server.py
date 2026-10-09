"""A tiny stdio MCP server used by the gyoza MCP tests: one 'add' tool and one 'fail' tool."""

import json
import sys


def reply(message_id, result=None, error=None):
    payload = {"jsonrpc": "2.0", "id": message_id}
    payload.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method, message_id = message.get("method"), message.get("id")
    if message_id is None:
        continue  # notification
    if method == "initialize":
        # an unsolicited notification first, to prove the client skips it
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {}}) + "\n")
        reply(message_id, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}})
    elif method == "tools/list":
        reply(message_id, {"tools": [
            {"name": "add", "description": "Add two numbers.", "inputSchema": {
                "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}},
            {"name": "fail", "description": "Always errors."},
        ]})
    elif method == "tools/call":
        params = message["params"]
        if params["name"] == "add":
            total = params["arguments"]["a"] + params["arguments"]["b"]
            reply(message_id, {"content": [{"type": "text", "text": str(total)}]})
        else:
            reply(message_id, {"content": [{"type": "text", "text": "nope"}], "isError": True})
    else:
        reply(message_id, error={"code": -32601, "message": f"unknown method {method}"})
