"""Package the completed report, real results, correct adapter and runnable sources."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "submission/Lab21_2A202602795_SUBMISSION.zip"


def main() -> None:
    required = ["submission/REPORT.md", "results/mask_proof.json", "results/token_stats.json",
                "results/template_check.json", "results/baselines_frozen.json", "results/runs.csv",
                "results/verdict.json", "results/autopsy.json", "results/qualitative.json",
                "results/baseline_predictions.json", "results/adapter_predictions.json",
                "results/report_evidence.json", "results/verify_final.log",
                "adapters/correct/adapter_config.json", "adapters/correct/adapter_model.safetensors"]
    assert all((ROOT / name).is_file() for name in required)
    assert "0 failures" in (ROOT / "results/verify_final.log").read_text(encoding="utf-8")
    files = {ROOT / "submission/REPORT.md", ROOT / "submission/lab21_colab_code.zip"}
    for folder in ("results", "adapters/correct", "notebooks", "colab", "src", "scripts", "data", "tests", "docs"):
        files.update(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.update(p for p in ROOT.iterdir() if p.is_file() and
                 (p.suffix in (".md", ".toml", ".txt") or p.name in ("Makefile", "LICENSE", "colab_code_manifest.json")))
    manifest = {}
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(files):
            name = p.relative_to(ROOT).as_posix()
            bundle.write(p, name)
            manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with zipfile.ZipFile(DEST) as bundle:
        assert bundle.testzip() is None
        assert all(name in bundle.namelist() for name in required)
        assert bundle.read("submission/REPORT.md") == (ROOT / "submission/REPORT.md").read_bytes()
        assert bundle.read("results/verdict.json") == (ROOT / "results/verdict.json").read_bytes()
    receipt = {"archive": DEST.relative_to(ROOT).as_posix(), "size_bytes": DEST.stat().st_size,
               "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(), "crc_passed": True,
               "files_sha256": manifest, "model_verdict": "FAIL", "verifier_exit_code": 0}
    (ROOT / "submission/final_package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("archive", "size_bytes", "sha256", "crc_passed")}))


if __name__ == "__main__":
    main()
