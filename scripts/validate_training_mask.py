"""Check the real pre-tokenized training data against NB1, without model weights."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, report, train
from labkit.config import get_tier


def main() -> None:
    from transformers import AutoTokenizer

    tier = get_tier()
    tok = AutoTokenizer.from_pretrained(tier.model_id, trust_remote_code=True)
    records = [json.loads(line) for line in (ROOT / "data/split/train.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    old_rows = data.to_training_dataset(tok, records, max_length=tier.max_length)
    tier, recipe = train.nb1_training_recipe(ROOT, tier, tok)
    rows = data.to_training_dataset(tok, records, **recipe)
    assert len(rows) == len(records), "Training silently dropped a sample"
    stats = json.loads((ROOT / "results/token_stats.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    supervised = total = 0
    for record, row in zip(records, rows):
        messages = data.to_messages(record)
        full = tok.apply_chat_template(messages, tokenize=False,
                                       enable_thinking=recipe["enable_thinking"])
        prefix = tok.apply_chat_template(messages[:-1], tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=recipe["enable_thinking"])
        text = tok.decode([t for t, label in zip(row["input_ids"], row["labels"])
                           if label != data.IGNORE_INDEX], skip_special_tokens=False)
        assert full.startswith(prefix)
        assert text == full[len(prefix):], "Training loss differs from answer tail"
        assert messages[-1]["content"] in text
        assert messages[-2]["content"] not in text
        assert all(label in (data.IGNORE_INDEX, token)
                   for token, label in zip(row["input_ids"], row["labels"]))
        supervised += sum(label != data.IGNORE_INDEX for label in row["labels"])
        total += len(row["labels"])
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    assert 0 < supervised / total < 0.95
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "trl", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    proof = {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
             "model_id": tier.model_id, "recipe": recipe, "p95": stats["p95"],
             "n_checked": len(rows), "all_passed": True,
             "old_default_rows_differ": sum(a != b for a, b in zip(old_rows, rows)),
             "total_tokens": total, "supervised_tokens": supervised,
             "supervised_fraction": supervised / total,
             "training_dataset_sha256": digest.hexdigest(), "packages": packages,
             "sft_trainer_executed": False}
    report.write_json(proof, "training_mask_validation.json", results_dir=ROOT / "results")
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
