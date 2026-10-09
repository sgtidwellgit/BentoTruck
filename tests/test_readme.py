"""Every ```python block in README.md must run as written — the examples are part of the contract.

Blocks whose first line is a `# requires: ...` comment need keys, network, or Docker, and are
only compiled (syntax-checked), not run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)


def _id(block: str) -> str:
    first_code_line = next((line for line in block.splitlines() if line and not line.startswith(("#", "import", "from"))), "")
    return first_code_line[:40]


@pytest.mark.parametrize("block", BLOCKS, ids=[_id(b) for b in BLOCKS])
def test_readme_example_runs(block, tmp_path, monkeypatch):
    code = compile(block, "README.md", "exec")
    if block.lstrip().startswith("# requires:"):
        pytest.skip("needs external services; syntax-checked only")
    monkeypatch.chdir(tmp_path)
    exec(code, {"__name__": "readme_example"})


def test_readme_has_examples_for_every_compartment():
    text = README.read_text(encoding="utf-8")
    import bentotruck

    for module in [m for m in bentotruck.__all__ if m.islower()]:
        assert f"## {module} " in text, f"README is missing a section for {module}"
