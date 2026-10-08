"""Re-score saved model outputs and assemble evidence for the student report (CPU)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import evaluate as ev
from labkit.integrity import require_frozen_baselines


def read(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> None:
    model = "unsloth/Qwen3.5-4B"
    frozen = require_frozen_baselines(ROOT, model)
    bases = read("results/baseline_predictions.json")
    adapters = read("results/adapter_predictions.json")
    verdict = read("results/verdict.json")
    autopsy = {r["run"]: r for r in read("results/autopsy.json")}
    target = [json.loads(s) for s in (ROOT / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    regression = [json.loads(s) for s in (ROOT / "data/eval_regression.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(target) == 50 and len(regression) == 15
    assert hashlib.sha256((ROOT / "results/baseline_predictions.json").read_bytes()).hexdigest() == frozen["predictions_sha256"]
    metrics = {}
    for name in ("baseline_a", "baseline_b", "correct", "attn_only", "wrong_lr", "qlora"):
        entries = bases["target"] if name.startswith("baseline_") else adapters[name]["target"]
        assert len(entries) == len(target)
        predictions = []
        for saved, row in zip(entries, target):
            assert saved["input"] == row["input"] and saved["label"] == row["label"]
            predictions.append(saved[name] if name.startswith("baseline_") else saved["prediction"])
        score = sum(ev.triage_field_accuracy(p, row["label"]) for p, row in zip(predictions, target)) / len(target)
        fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in predictions) / len(target)
        expected = frozen[name] if name.startswith("baseline_") else autopsy[name]
        assert math.isclose(score, expected["target"], abs_tol=1e-12), name
        assert math.isclose(fmt, expected["format"], abs_tol=1e-12), name
        metrics[name] = {"target": score, "format": fmt, "n": len(target), "per_field_accuracy": {
            key: sum(ev.triage_field_accuracy(p, row["label"], keys=[key]) for p, row in zip(predictions, target)) / len(target)
            for key in ev.TRIAGE_KEYS}}
        if name not in ("baseline_a", "baseline_b", "correct"):
            continue
        entries = bases["regression"] if name.startswith("baseline_") else adapters["correct"]["regression"]
        assert len(entries) == len(regression)
        recalls = []
        for saved, row in zip(entries, regression):
            assert saved["instruction"] == row["instruction"] and saved["keywords"] == row["keywords"]
            p = saved[name] if name.startswith("baseline_") else saved["prediction"]
            recalls.append(ev.keyword_recall(p, row["keywords"]))
        metrics[name]["regression"] = sum(recalls) / len(recalls)
        if name.startswith("baseline_"):
            assert math.isclose(metrics[name]["regression"], frozen[name]["regression"], abs_tol=1e-12)
        else:
            assert round(metrics[name]["regression"], 4) == verdict["comparison"][2]["regression"]

    new_verdict = ev.regression_gate(ev.GroupScores(**{k: metrics["correct"][k] for k in ("target", "regression", "format")}),
                                     ev.GroupScores(**{k: metrics["baseline_b"][k] for k in ("target", "regression", "format")}))
    assert new_verdict.passed == verdict["verdict"]["passed"]
    assert math.isclose(new_verdict.regression_delta, verdict["verdict"]["regression_delta"], abs_tol=1e-12)
    q = read("results/qualitative.json")
    assert len(q) == len(target) and {r["i"] for r in q} == set(range(len(target)))
    for r in q:
        i = r["i"]
        assert r["ft_prediction"] == adapters["correct"]["target"][i]["prediction"]
        assert r["baseline_b_prediction"] == bases["target"][i]["baseline_b"]
        assert r["input"] == target[i]["input"] and r["label"] == target[i]["label"]
        ft = ev.triage_field_accuracy(r["ft_prediction"], r["label"])
        b = ev.triage_field_accuracy(r["baseline_b_prediction"], r["label"])
        assert ft == r["ft_score"] and b == r["baseline_b_score"] and ft - b == r["delta_vs_b"]
        assert r["outcome_vs_b"] == ("loss" if ft < b else "win" if ft > b else "tie")

    regression_examples = []
    for i, (b, ft) in enumerate(zip(bases["regression"], adapters["correct"]["regression"])):
        bs = ev.keyword_recall(b["baseline_b"], b["keywords"])
        fs = ev.keyword_recall(ft["prediction"], b["keywords"])
        regression_examples.append({"i": i, "group": "regression", "question": b["instruction"],
            "keywords": b["keywords"], "baseline_b_prediction": b["baseline_b"], "ft_prediction": ft["prediction"],
            "baseline_b_score": bs, "ft_score": fs, "delta_vs_b": fs - bs,
            "outcome_vs_b": "loss" if fs < bs else "win" if fs > bs else "tie"})
    selected_target = [next(r for r in q if r["i"] == i) for i in (3, 5, 12, 25, 49)]
    selected_regression = [regression_examples[i] for i in (9, 14)]
    assert len(selected_target + selected_regression) >= 5
    assert sum(r["outcome_vs_b"] == "loss" for r in selected_regression) >= 2

    with (ROOT / "results/runs.csv").open(encoding="utf-8", newline="") as f:
        runs = list(csv.DictReader(f))
    assert {r["run"] for r in runs} == set(autopsy)
    assert all(int(r["actual_steps"]) == int(r["max_steps"]) == 30 for r in runs)
    for r in runs:
        assert int(r["max_length"]) == 256 and r["enable_thinking"] == "False"
        batch = read(f"results/training_batch_{r['run']}.json")
        assert batch["all_dataset_labels_equal_nb1"] and batch["collator_labels_equal_nb1"] and batch["n_checked"] == 225
        assert all(label == -100 or label == token for token, label in zip(batch["input_ids"], batch["labels"]))

    with (ROOT / "adapters/correct/adapter_model.safetensors").open("rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    tensor_parameters = sum(math.prod(t["shape"]) for name, t in header.items() if name != "__metadata__")
    assert tensor_parameters == int(next(r for r in runs if r["run"] == "correct")["trainable_params"])
    import_manifest = read("results/colab_import_manifest.json")
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
               for name, sha in import_manifest["imported_files_sha256"].items())
    source_manifest = read("colab_code_manifest.json")
    experiment_files = {name: sha for name, sha in source_manifest["files_sha256"].items()
                        if name.startswith(("src/", "tests/", "notebooks/", "data/")) or name == "scripts/verify.py"}
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha for name, sha in experiment_files.items())
    evidence = {"scorer": "unchanged src/labkit/evaluate.py; saved outputs re-scored on CPU",
        "metrics_recomputed": metrics, "verdict_recomputed": new_verdict.as_dict(),
        "target_outcome_counts": dict(Counter(r["outcome_vs_b"] for r in q)),
        "regression_outcome_counts": dict(Counter(r["outcome_vs_b"] for r in regression_examples)),
        "selected_target_examples": selected_target, "selected_regression_examples": selected_regression,
        "all_regression_examples": regression_examples,
        "all_four_runs_actual_steps": 30, "all_four_batch_proofs_passed": True,
        "adapter_tensor_parameter_count": tensor_parameters, "original_imported_bytes_unchanged": True,
        "experiment_source_files_match_colab": len(experiment_files)}
    (ROOT / "results/report_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "PASS", "model_verdict": "FAIL", "target_outcomes": evidence["target_outcome_counts"],
                      "regression_outcomes": evidence["regression_outcome_counts"],
                      "selected_examples": len(selected_target + selected_regression), "actual_steps": 30}, ensure_ascii=True))


if __name__ == "__main__":
    main()
