
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "rag_core"

FORBIDDEN = {"numpy", "scipy", "sklearn", "faiss", "rank_bm25", "torch", "gensim", "pandas"}

ALLOWED_STDLIB = {
    "__future__",
    "math",
    "heapq",
    "re",
    "unicodedata",
    "dataclasses",
    "collections",
    "typing",
    "json",
    "pathlib",
    "functools",
    "itertools",
    "statistics",
    "enum",
}

PROJECT_PACKAGES = {"app", "services", "ingest", "embed", "store"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], 0
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            yield (node.module or "").split(".")[0], node.level


@pytest.mark.parametrize("path", sorted(ROOT.glob("*.py")), ids=lambda p: p.name)
def test_scoring_path_is_stdlib_only(path):
    for module, _level in _imports(path):
        assert module not in FORBIDDEN, f"{path.name} imports forbidden library {module}"
        assert module not in PROJECT_PACKAGES, f"{path.name} imports project package {module}"
        assert module in ALLOWED_STDLIB, f"{path.name} imports unlisted module {module}"


def test_guard_covers_every_core_module():
    modules = {path.name for path in ROOT.glob("*.py")}
    assert "tokenize.py" in modules
    assert "lexical.py" in modules
    assert "pipeline.py" in modules
