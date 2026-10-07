"""Build the Colab runner that uses this student's main branch and runs only NB1/NB2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# Lab 21 — NB1 → NB2, trước khi train

Runtime → Change runtime type → **T4 GPU**, rồi chạy các ô từ trên xuống.
Notebook lấy code từ nhánh **main của ducdh205**, dùng `unsloth/Qwen3.5-4B`.
Chạy đầy đủ 50 ticket target + 15 câu regression, không đặt `EVAL_LIMIT`.
Chỉ đo baseline (a), (b); baseline (c) sẽ được đo sau khi fine-tune.
Nếu (b) chưa thắng (a), script lưu candidate và dừng để cải thiện prompt trước train.
Kết quả đã đóng băng sẽ không bị ghi đè khi chạy lại. Xem `docs/RUN-NB2.md`.
"""),
    ("code", """# 1. Clone đúng repo + cài môi trường GPU
import os, pathlib, subprocess, sys

REPO = "https://github.com/ducdh205/Day21-Track3-DinhHoangDuc-2A202602795-Finetuning-Lab.git"
ROOT = pathlib.Path("/content/Day21-Track3-DinhHoangDuc-2A202602795-Finetuning-Lab")
if not ROOT.exists():
    subprocess.run(["git", "clone", "--branch", "main", "--single-branch", REPO, str(ROOT)], check=True)
os.chdir(ROOT)
subprocess.run(["git", "pull", "--ff-only", "origin", "main"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
os.environ["COMPUTE_TIER"] = "T4"
os.environ["BASE_MODEL"] = "unsloth/Qwen3.5-4B"
os.environ.pop("EVAL_LIMIT", None)

import torch
assert torch.cuda.is_available(), "Chọn T4 GPU trong Runtime rồi chạy lại."
print("GPU:", torch.cuda.get_device_name(0))
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
"""),
    ("code", """# 2. Chạy NB1 rồi NB2 — không train
with open("results/nb2_console.log", "w", encoding="utf-8") as log:
    for stage in ("notebooks/01_data_and_mask.py", "notebooks/02_baselines.py"):
        print("Running", stage, flush=True)
        process = subprocess.Popen([sys.executable, "-u", stage], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        if code:
            raise RuntimeError(f"{stage} exit={code}. Xem log; chưa được train.")

import json
frozen = json.loads(pathlib.Path("results/baselines_frozen.json").read_text())
assert frozen["baseline_b"]["target"] > frozen["baseline_a"]["target"]
print(json.dumps(frozen, ensure_ascii=False, indent=2))
"""),
    ("code", """# 3. Tải bằng chứng về máy — có thể chạy ô này cả khi ô 2 lỗi
import zipfile
from google.colab import files

archive = pathlib.Path("/content/lab21_nb1_nb2_results.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for folder in ("results", "data/split"):
        for path in pathlib.Path(folder).glob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                bundle.write(path, str(path))
files.download(str(archive))
"""),
]


def main() -> None:
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
                "id": hashlib.sha1(f"{index}\0{source}".encode()).hexdigest()[:8]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    path = ROOT / "colab/Lab21_NB2_RUN.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
