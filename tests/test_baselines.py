"""Baseline freezing must reject weak prompts, overwritten results and eval drift."""
from __future__ import annotations

import json
import pathlib
import re
import runpy
from types import SimpleNamespace

import pytest

from labkit import data, device, generate
from labkit.config import OPTIMIZED_PROMPT
from labkit.integrity import corpus_sha256

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_corpus_hash_accepts_windows_line_endings_but_rejects_content_changes(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_bytes(b'{"label":1}\n')
    expected = corpus_sha256(path)
    path.write_bytes(b'{"label":1}\r\n')
    assert corpus_sha256(path) == expected
    path.write_bytes(b'{"label":2}\r\n')
    assert corpus_sha256(path) != expected


def test_added_few_shots_are_train_examples_and_do_not_leak_eval():
    examples = re.findall(r'Ticket: "([^"]+)"\nJSON: (\{[^\n]+\})', OPTIMIZED_PROMPT)
    records = list(map(json.loads,
                       (ROOT / "data/train_seed.jsonl").read_text(encoding="utf-8").splitlines()))
    training, _ = data.split(records, seed=42)
    train = {r["input"]: r["label"] for r in training}
    eval_inputs = {r["input"] for r in
                   map(json.loads, (ROOT / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(examples) == 6
    assert {json.loads(label)["intent"] for _, label in examples} == {
        "doi_tra", "van_chuyen", "hoan_tien", "san_pham_loi", "hoi_thong_tin"}
    for ticket, label in examples[1:]:
        assert train[ticket] == json.loads(label)
    assert not ({ticket for ticket, _ in examples} & eval_inputs)


@pytest.fixture
def baseline_runtime(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "results").mkdir()
    label = {"intent": "doi_tra", "urgency": "cao", "product": "áo", "sentiment": "trung_tinh"}
    (data_dir / "eval_target.jsonl").write_text(
        json.dumps({"input": "ticket", "label": label}) + "\n", encoding="utf-8")
    (data_dir / "eval_regression.jsonl").write_text(
        json.dumps({"instruction": "question", "keywords": ["answer"]}) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVAL_LIMIT", raising=False)
    monkeypatch.delenv("BASE_MODEL", raising=False)
    monkeypatch.setattr(device, "describe", lambda: {
        "device": "cuda", "name": "test GPU", "bf16": False, "fp16": True,
        "capability": "7.5", "vram_gb": 16})
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision"), eval=lambda: None)
    monkeypatch.setattr(generate, "load_base", lambda tier: (model, None))
    monkeypatch.setattr(generate, "free_memory", lambda: None)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-version")

    def generate_mock(model, tok, prompts, system=None, **kwargs):
        prediction = "answer" if system is None else (
            json.dumps(label) if system == OPTIMIZED_PROMPT else "unstructured answer")
        return [prediction] * len(prompts), 1.0

    monkeypatch.setattr(generate, "generate_batch", generate_mock)
    return tmp_path


def run_nb2():
    return runpy.run_path(str(ROOT / "notebooks/02_baselines.py"), run_name="__main__")


def test_freezes_only_measured_winning_full_baselines_and_keeps_predictions(baseline_runtime):
    run_nb2()
    results = baseline_runtime / "results"
    frozen = json.loads((results / "baselines_frozen.json").read_text(encoding="utf-8"))
    assert frozen["baseline_b"]["target"] > frozen["baseline_a"]["target"]
    assert frozen["measured_before_training"] and not frozen["smoke_mode"]
    assert frozen["optimized_prompt"] == OPTIMIZED_PROMPT
    assert len(frozen["eval_checksums_sha256"]) == 2
    assert (results / "baseline_predictions.json").is_file()
    before = (results / "baselines_frozen.json").read_bytes()
    with pytest.raises(RuntimeError, match="không ghi đè"):
        run_nb2()
    assert (results / "baselines_frozen.json").read_bytes() == before


def test_does_not_freeze_a_prompt_that_does_not_beat_naive(baseline_runtime, monkeypatch):
    monkeypatch.setattr(generate, "generate_batch",
                        lambda model, tok, prompts, **kwargs: (["bad"] * len(prompts), 1.0))
    with pytest.raises(RuntimeError, match="Chưa đóng băng"):
        run_nb2()
    assert not (baseline_runtime / "results/baselines_frozen.json").exists()
    assert (baseline_runtime / "results/baselines_candidate.json").exists()


def test_does_not_freeze_smoke_run(baseline_runtime, monkeypatch):
    monkeypatch.setenv("EVAL_LIMIT", "1")
    with pytest.raises(RuntimeError, match="Chưa đóng băng"):
        run_nb2()
    assert not (baseline_runtime / "results/baselines_frozen.json").exists()


def test_refuses_baseline_measurement_after_training(baseline_runtime):
    (baseline_runtime / "results/runs.csv").write_text("run\ncorrect\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="trước khi train"):
        run_nb2()


def test_does_not_freeze_when_eval_changes_during_generation(baseline_runtime, monkeypatch):
    original = generate.generate_batch
    def changed_eval(*args, **kwargs):
        result = original(*args, **kwargs)
        path = baseline_runtime / "data/eval_target.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result
    monkeypatch.setattr(generate, "generate_batch", changed_eval)
    with pytest.raises(AssertionError, match="Tập eval thay đổi"):
        run_nb2()
    assert not (baseline_runtime / "results/baselines_frozen.json").exists()
