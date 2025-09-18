from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DECISION_TREE_DIR = STATIC_DIR / "decision_trees"


for directory in (STATIC_DIR, DECISION_TREE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def to_static_url(path: Path) -> str:
    """Convert an absolute path under STATIC_DIR to a `/static` URL."""
    relative = path.relative_to(STATIC_DIR)
    return f"/static/{relative.as_posix()}"

