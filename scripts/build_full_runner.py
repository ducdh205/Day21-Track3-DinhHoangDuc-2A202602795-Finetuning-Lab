"""Build a full Colab runner and a source ZIP including local, unpublished fixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# Lab21 — NB1 → NB5, đầy đủ để nộp

Mở bằng Chrome/Edge thường đã đăng nhập Google. Chọn **Runtime → Change runtime type → T4 GPU**.
Chạy ô 1–4 lần lượt. Ô 1 yêu cầu chọn `lab21_colab_code.zip` đi kèm notebook này:
gói chứa đúng code local đã sửa, không cần push GitHub. EVAL_LIMIT=0 và EPOCHS=2.
Ô 3 có thể mất khoảng 100–130 phút theo hướng dẫn repo; đây chưa phải số đo của lần chạy này.
Nếu một bước lỗi, vẫn chạy ô 4 để tải kết quả/log đã có. Kết quả FAIL cũng phải giữ nguyên.
Sau khi lấy ZIP kết quả, cập nhật report từ số đo mới trước khi nộp.
"""),
    ("code", """# @title 1. Nạp code local và cài môi trường
import hashlib, json, os, pathlib, subprocess, sys, zipfile
from google.colab import files

ROOT = pathlib.Path('/content/lab21_student')
ROOT.mkdir(exist_ok=True)
if not (ROOT / 'requirements.txt').exists():
    print('Chọn lab21_colab_code.zip từ máy của bạn.')
    uploaded = files.upload()
    assert 'lab21_colab_code.zip' in uploaded, 'Cần gói code đi kèm notebook này.'
    archive = pathlib.Path('/content/lab21_colab_code.zip')
    archive.write_bytes(uploaded['lab21_colab_code.zip'])
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            assert (ROOT / item.filename).resolve().is_relative_to(ROOT.resolve()), 'Unsafe ZIP path'
        bundle.extractall(ROOT)
os.chdir(ROOT)
manifest = json.loads((ROOT / 'colab_code_manifest.json').read_text())
for name, sha in manifest['files_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, f'Code checksum changed: {name}'
print('Source checksum verified, including original tests and eval.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['COMPUTE_TIER'] = 'T4'
os.environ['BASE_MODEL'] = 'unsloth/Qwen3.5-4B'
os.environ['EPOCHS'] = '2'
os.environ['MASK_MODE'] = 'assistant-only'
os.environ.pop('EVAL_LIMIT', None)
sys.path.insert(0, str(ROOT / 'src'))
(ROOT / 'results').mkdir(exist_ok=True)
import torch
assert torch.cuda.is_available(), 'Chọn T4 GPU rồi chạy lại ô này.'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)
"""),
    ("code", """# @title 2. Smoke — tests nguyên bản
subprocess.run([sys.executable, '-u', 'scripts/verify.py', '--smoke'], check=True)
"""),
    ("code", """# @title 3. Core pipeline — NB1 → NB5, full eval
import time
STAGES = 'nb1 nb2 nb3 nb4 nb5'  # @param {type:'string'}
ONLY = ''  # @param {type:'string'}
os.environ.pop('EVAL_LIMIT', None)
os.environ['EPOCHS'] = '2'
os.environ['ONLY'] = ONLY

def run_logged(script, log_name, allow_nonzero=False):
    started = time.perf_counter()
    with (ROOT / 'results' / log_name).open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', script], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    print(f'{script}: exit={code}, seconds={time.perf_counter() - started:.1f}', flush=True)
    if code and not allow_nonzero:
        raise RuntimeError(f'{script} failed. Giữ log và chạy ô 4 để tải bằng chứng.')
    return code

stages = {'nb1': '01_data_and_mask', 'nb2': '02_baselines', 'nb3': '03_train_correct',
          'nb4': '04_misconfig_autopsy', 'nb5': '05_evaluate_and_verdict'}
for stage in STAGES.split():
    assert stage in stages, f'Unknown stage: {stage}'
    run_logged(f'notebooks/{stages[stage]}.py', f'{stage}_console.log')
    if stage == 'nb1':
        code = run_logged('scripts/check_mask_agreement.py', 'mask_agreement.log', allow_nonzero=True)
        (ROOT / 'results/mask_agreement_exit.json').write_text(json.dumps({'exit_code': code}))
        run_logged('scripts/validate_training_mask.py', 'training_mask_validation.log')
"""),
    ("code", """# @title 4. Gatekeeper + tải kết quả (chạy cả khi ô 3 lỗi)
code = subprocess.run([sys.executable, '-u', 'scripts/verify.py']).returncode
print('Verifier exit:', code, '— report cần cập nhật theo kết quả GPU trước khi nộp.')
archive = pathlib.Path('/content/lab21_2A202602795_results.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for folder in ('results', 'data/split', 'adapters/correct', 'notebooks'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                bundle.write(path, path.relative_to(ROOT).as_posix())
    for name in ('submission/REPORT.md', 'colab_code_manifest.json'):
        path = ROOT / name
        if path.exists():
            bundle.write(path, name)
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
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    notebook = ROOT / "colab/Lab21_RUN_ALL.ipynb"
    notebook.write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")
    paths = []
    for folder in ("src", "scripts", "tests", "notebooks", "colab", "docs"):
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.ipynb', '.md') and '__pycache__' not in p.parts)
    paths.extend((ROOT / "data").glob('*.jsonl'))
    paths.append(ROOT / "data/checksums.json")
    paths.extend(ROOT.glob('*.md'))
    paths.extend(ROOT / name for name in ('requirements.txt', 'requirements-cpu.txt',
                                         'pyproject.toml', 'Makefile', 'submission/REPORT.md'))
    manifest = {"source": "local workspace, unpublished changes included",
                "model_id": "unsloth/Qwen3.5-4B", "epochs": 2, "eval_limit": None,
                "files_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(set(paths))}}
    archive = ROOT / "submission/lab21_colab_code.zip"
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(paths)):
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr('colab_code_manifest.json', json.dumps(manifest, indent=2))
    print(notebook)
    print(archive)


if __name__ == '__main__':
    main()
