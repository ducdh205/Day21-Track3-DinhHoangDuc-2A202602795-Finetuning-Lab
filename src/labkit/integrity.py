"""Content hashes shared by baseline freezing and submission verification."""
from __future__ import annotations

import hashlib
from pathlib import Path


def corpus_sha256(path: Path) -> str:
    """Hash corpus bytes with CRLF normalized to LF, matching the seed generator.

    Git's Windows checkout changes line endings, not dataset content. Every other
    byte remains significant; changing a label, row order or prompt still fails.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def eval_checksums(root: Path) -> dict[str, str]:
    return {name: corpus_sha256(root / "data" / name)
            for name in ("eval_target.jsonl", "eval_regression.jsonl")}
