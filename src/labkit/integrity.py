"""Content hashes shared by baseline freezing and submission verification."""
from __future__ import annotations

import hashlib
import json
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


def require_frozen_baselines(root: Path, model_id: str) -> dict:
    """Stop before model loading if NB2's full, frozen experiment is missing/stale."""
    from .config import OPTIMIZED_PROMPT

    path = root / "results/baselines_frozen.json"
    if not path.exists():
        raise RuntimeError("Run NB2 and freeze full baselines before training.")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if (frozen.get("model") != model_id or frozen.get("smoke_mode")
            or not frozen.get("measured_before_training")
            or frozen["baseline_b"]["target"] <= frozen["baseline_a"]["target"]):
        raise RuntimeError("NB2 must use this model, full eval, and baseline (b) > (a).")
    sha = hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16]
    if frozen.get("optimized_prompt_sha") != sha or frozen.get("eval_checksums_sha256") != eval_checksums(root):
        raise RuntimeError("Frozen prompt or eval changed — do not train on this comparison.")
    return frozen
