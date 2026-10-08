"""Extract received Colab console evidence without replacing original run artifacts."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit.config import OPTIMIZED_PROMPT
from labkit.integrity import eval_checksums


def main() -> None:
    source = ROOT / "results/colab_console_received.log"
    raw = source.read_bytes()
    console = raw.decode("utf-8-sig").replace("\r\n", "\n")
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"(?m)^\{", console):
        try:
            value, _ = decoder.raw_decode(console[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)

    def find(key: str) -> dict:
        found = [obj for obj in objects if key in obj]
        assert len(found) == 1, (key, len(found))
        return found[0]

    frozen = find("baseline_a")
    prompt_sha = hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()
    assert frozen["optimized_prompt_sha256"] == prompt_sha
    assert frozen["optimized_prompt_sha"] == prompt_sha[:16]
    assert frozen["eval_checksums_sha256"] == eval_checksums(ROOT)
    assert frozen["measured_before_training"] and not frozen["smoke_mode"]
    assert frozen["eval_limit"] is None
    assert frozen["baseline_b"]["target"] > frozen["baseline_a"]["target"]

    fine_tune = ast.literal_eval(re.search(r"(?m)^fine-tune: (.+)$", console)[1])
    train_rows, eval_rows = {}, {}
    names = ("correct", "attn_only", "wrong_lr", "qlora")
    for line in console.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|") or cells[0] not in names:
            continue
        if len(cells) == 8:
            train_rows[cells[0]] = dict(zip(
                ("label", "r", "trainable_params", "learning_rate", "final_loss", "train_seconds", "peak_vram_gb"),
                [cells[1], int(cells[2]), int(cells[3]), *map(float, cells[4:])]))
        elif len(cells) == 5:
            eval_rows[cells[0]] = dict(zip(
                ("target", "format", "latency_ms_rounded", "n"),
                [*map(float, cells[1:4]), int(cells[4])]))
    assert set(train_rows) == set(eval_rows) == set(names)
    correct = find("actual_steps")
    assert correct["actual_steps"] == correct["max_steps"] == 30
    assert correct["max_length"] == find("selected_max_length")["selected_max_length"] == 256
    assert correct["enable_thinking"] is False
    gap = abs(train_rows["attn_only"]["trainable_params"] - correct["trainable_params"]) / correct["trainable_params"]
    assert gap < 0.05
    regression_drop = frozen["baseline_b"]["regression"] - fine_tune["regression"]
    assert "FAILED\n - general capability regressed" in console and regression_drop > 0.02

    stages = [{"script": m[0], "exit_code": int(m[1]), "seconds": float(m[2])}
              for m in re.findall(r"(?m)^((?:notebooks|scripts)/[\w]+\.py): exit=(\d+), seconds=([\d.]+)", console)]
    summary = {
        "evidence_kind": "received_console_log",
        "original_colab_zip_imported": (ROOT / "results/colab_import_manifest.json").exists(),
        "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "token_stats_printed": find("selected_max_length"), "mask_proof_printed": find("answer_is_supervised"),
        "training_mask_validation_printed": find("training_dataset_sha256"),
        "baselines_frozen_printed": frozen, "architecture_printed": find("layer_types"),
        "training_args_printed": find("output_dir"), "correct_run_printed": correct,
        "runs_table_printed": train_rows, "fine_tune_metrics_printed": fine_tune,
        "autopsy_table_printed": eval_rows, "stages": stages,
        "cross_checks": {"current_eval_checksums_match": True, "current_optimized_prompt_sha_matches": True,
                         "full_evaluation": True, "baseline_b_beats_a": True,
                         "attention_budget_relative_difference": gap},
        "derived_comparisons": {
            "target_gain_over_b": fine_tune["target"] - frozen["baseline_b"]["target"],
            "regression_drop_from_b": regression_drop,
            "qlora_peak_vram_reduction_fraction": 1 - train_rows["qlora"]["peak_vram_gb"] / correct["peak_vram_gb"],
            "target_ranking": sorted(names, key=lambda name: eval_rows[name]["target"], reverse=True),
        },
        "model_verdict_from_console": "FAIL", "regression_tolerance_from_console": 0.02,
        "gatekeeper_exit_from_user_screenshot": 0,
        "limitations": ["Contrast latency is printed rounded to 0.1 ms.",
                        "Qualitative console output is truncated and has no baseline predictions.",
                        "Gatekeeper exit 0 comes from the user screenshot, not this text log."],
    }
    target = ROOT / "results/colab_log_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(target.relative_to(ROOT)), "model_verdict": "FAIL",
                      "eval_and_prompt_match": True, "ranking": summary["derived_comparisons"]["target_ranking"],
                      "attention_budget_difference_pct": 100 * gap}, ensure_ascii=False))


if __name__ == "__main__":
    main()
