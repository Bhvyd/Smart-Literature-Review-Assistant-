
from __future__ import annotations

import hashlib
from pathlib import Path


class Cache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.root / digest[:2] / f"{digest[2:]}.bin"

    def get(self, url: str) -> bytes | None:
        path = self.path_for(url)
        if not path.is_file():
            return None
        return path.read_bytes()

    def put(self, url: str, body: bytes) -> None:
        path = self.path_for(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def has(self, url: str) -> bool:
        return self.path_for(url).is_file()
