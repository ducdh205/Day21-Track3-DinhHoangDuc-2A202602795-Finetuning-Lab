# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NB1 — Dữ liệu, chat template & mask
#
# **Chạy được trên CPU. Không cần GPU.** Đây là notebook duy nhất như vậy — và cũng là
# notebook quyết định kết quả của cả lab.
#
# > Deck §17.2: *che loss và chat template quyết định kết quả nhiều hơn mọi biến thể
# > LoRA cộng lại.* Notebook này không dạy bạn tin điều đó — nó bắt bạn **nhìn thấy** nó.
#
# Cuối notebook bạn sẽ có 4 artefact bắt buộc nộp:
# 1. `results/mask_proof.json` — bằng chứng mask đúng
# 2. `results/template_check.json` — template có nuốt khối `<think>` không
# 3. `results/token_stats.json` — p95 → `max_length`
# 4. `data/split/{train,val}.jsonl` — split cố định seed=42

# %%
import json
import os
import pathlib
import sys
import hashlib
import importlib.metadata
import platform
import time
from datetime import datetime, timezone

STARTED = time.perf_counter()

sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))

from labkit import data, report
from labkit.config import get_tier

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
ENABLE_THINKING = False  # Corpus là JSON thuần; cùng chế độ render với evaluation.
print(f"tier={TIER.name}  model={TIER.model_id}  max_length={TIER.max_length}")

# %% [markdown]
# ## 1. Nạp corpus
#
# Corpus đi kèm lab: ticket CSKH tiếng Việt → JSON triage 4 trường. Bạn **được khuyến
# khích** thay bằng dữ liệu miền của mình (xem §"Đổi dataset" trong README) — nhưng hãy
# chạy hết notebook này một lượt với corpus mặc định trước, để có mốc so sánh.

# %%
def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


train_raw = load_jsonl(ROOT / "data" / "train_seed.jsonl")
print(f"{len(train_raw)} mẫu huấn luyện")
print(json.dumps(train_raw[0], ensure_ascii=False, indent=2)[:400])

# %% [markdown]
# ## 2. Tokenizer + kiểm tra template
#
# Chỉ tải **file tokenizer** (vài MB), không tải trọng số. Chạy được trên máy không GPU.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TIER.model_id, trust_remote_code=True)
print("eos_token:", tok.eos_token)

# %% [markdown]
# ### Kiểm tra bắt buộc #1 — template có giữ khối suy luận không?
#
# Deck §22: **một số chat template xoá nội dung `<think>` ngay trong
# `apply_chat_template`.** Khi đó reasoning traces trong dataset của bạn *không bao giờ*
# tới được hàm loss — và không có gì báo lỗi cả. Kiểm tra một lần cho mỗi base model.

# %%
check = data.thinking_survives(tok)
print("VERDICT:", check["verdict"])
print("\n--- chuỗi đã render ---")
print(check["rendered"])
report.write_json(check, "template_check.json", results_dir=ROOT / "results")

# %% [markdown]
# ## Đo độ dài trước khi xây mask
#
# Đếm toàn bộ token sau template, không truncation. Dùng p95 để chọn độ dài
# thực tế cho NB1; cấu hình tier chỉ là giá trị mặc định để đối chiếu.

# %%
lengths = [len(tok(tok.apply_chat_template(data.to_messages(r), tokenize=False,
                                         enable_thinking=ENABLE_THINKING),
                   add_special_tokens=False)["input_ids"]) for r in train_raw]
stats = data.token_stats(lengths)
MAX_LENGTH = stats["suggested_max_length"]
stats.update({"model_id": TIER.model_id, "selected_max_length": MAX_LENGTH,
              "enable_thinking": ENABLE_THINKING,
              "tier_max_length": TIER.max_length,
              "n_would_truncate": sum(n > MAX_LENGTH for n in lengths)})
print(json.dumps(stats, ensure_ascii=False, indent=2))
report.write_json(stats, "token_stats.json", results_dir=ROOT / "results")

# %% [markdown]
# ## 3. Xây mask — và ĐỌC nó
#
# Bốn chế độ, và bạn phải hiểu khác biệt trước khi train:
#
# | mode | Loss tính trên | Dùng khi |
# |---|---|---|
# | `assistant-only` | toàn bộ lượt assistant | mặc định SFT |
# | `masked-think` | lượt assistant **trừ** khối suy luận | base có chế độ thinking (§17.5) |
# | `response-only` | chỉ phần sau `</think>` | nghiêm ngặt nhất |
# | `everything` | **cả prompt** | ✗ đây là bug kinh điển — để bạn nhìn thấy nó |

# %%
sample = data.to_messages(train_raw[0])
rendered = tok.apply_chat_template(sample, tokenize=False, enable_thinking=ENABLE_THINKING)
print("--- CHUỖI SAU apply_chat_template ---")
print(rendered)

for mode in ("assistant-only", "everything"):
    ex = data.build_example(tok, sample, max_length=MAX_LENGTH, mask_mode=mode,
                            enable_thinking=ENABLE_THINKING)
    print("=" * 70)
    print(f"mode = {mode}   supervised {ex.n_supervised}/{ex.n_total} "
          f"({ex.supervised_fraction:.0%})")
    print("--- LOSS TÍNH TRÊN ĐOẠN NÀY ---")
    print(data.decode_supervised(tok, ex))

# %% [markdown]
# **Dừng lại và đọc kỹ output ở trên.**
#
# Với `everything`, câu hỏi của bạn nằm trong phần được tính loss → model sẽ học cách
# *viết lại câu hỏi*. Đó chính xác là triệu chứng ở deck §22 (“Model viết tiếp câu hỏi
# của bạn”). Rất nhiều người chỉ phát hiện ra sau khi train xong 3 tiếng.

# %% [markdown]
# ### Kiểm tra bắt buộc #2 — mask proof
#
# Đây là artefact nộp bài. Nó khẳng định: phần được tính loss **chứa** câu trả lời và
# **không chứa** câu hỏi.

# %%
ex = data.build_example(tok, sample, max_length=MAX_LENGTH, mask_mode="assistant-only",
                        enable_thinking=ENABLE_THINKING)
supervised = data.decode_supervised(tok, ex)
masked = data.decode_masked(tok, ex)

answer = sample[-1]["content"]
question_fragment = sample[-2]["content"]
prefix = tok.apply_chat_template(sample[:-1], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=ENABLE_THINKING)
expected_tail = rendered[len(prefix):]

proof = {
    "mask_mode": "assistant-only",
    "model_id": TIER.model_id,
    "max_length": MAX_LENGTH,
    "enable_thinking": ENABLE_THINKING,
    "n_supervised": ex.n_supervised,
    "n_total": ex.n_total,
    "supervised_fraction": round(ex.supervised_fraction, 4),
    "answer_is_supervised": answer in supervised,
    "question_is_masked": question_fragment not in supervised,
    "supervised_equals_rendered_answer_tail": supervised == expected_tail,
    "rendered": rendered,
    "answer": answer,
    "question": question_fragment,
    "supervised_text": supervised,
    "masked_text": masked,
    "input_ids": ex.input_ids,
    "labels": ex.labels,
    "supervised_preview": supervised[:300],
    "masked_preview": masked[:300],
}
assert proof["answer_is_supervised"], "câu trả lời KHÔNG nằm trong loss — mask sai"
assert proof["question_is_masked"], "câu hỏi ĐANG nằm trong loss — mask sai"
assert proof["supervised_equals_rendered_answer_tail"], "loss không khớp đuôi assistant"
assert 0 < ex.supervised_fraction < 0.95, "loss rỗng hoặc tính cả prompt"
print(json.dumps({k: v for k, v in proof.items() if not k.endswith("preview")},
                 ensure_ascii=False, indent=2))
report.write_json(proof, "mask_proof.json", results_dir=ROOT / "results")

# %% [markdown]
# ## 4. Độ dài token → `max_length`
#
# Deck §17: `max_length` là **số đo**, không phải con số đoán. Đặt theo p95 rồi làm tròn
# lên luỹ thừa 2. Đặt quá lớn = trả tiền cho padding; quá nhỏ = cắt mất câu trả lời.

# %%
print(f"max_length thực tế={MAX_LENGTH}, p95={stats['p95']}, "
      f"tier mặc định={TIER.max_length}, số mẫu bị cắt={stats['n_would_truncate']}")

# Kiểm tra toàn corpus để tránh chỉ chứng minh đúng một mẫu.
audits = []
for index, record in enumerate(train_raw):
    messages = data.to_messages(record)
    row = data.build_example(tok, messages, max_length=MAX_LENGTH,
                             enable_thinking=ENABLE_THINKING)
    text = data.decode_supervised(tok, row)
    full = tok.apply_chat_template(messages, tokenize=False, enable_thinking=ENABLE_THINKING)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=ENABLE_THINKING)
    audits.append({"index": index, "n_total": row.n_total,
                   "n_supervised": row.n_supervised,
                   "supervised_fraction": row.supervised_fraction,
                   "answer_is_supervised": messages[-1]["content"] in text,
                   "question_is_masked": messages[-2]["content"] not in text,
                   "exact_tail": text == full[len(prompt):]})
assert all(r["answer_is_supervised"] and r["question_is_masked"] and r["exact_tail"]
           and 0 < r["supervised_fraction"] < 0.95 for r in audits)
report.write_json({"model_id": TIER.model_id, "max_length": MAX_LENGTH,
                   "enable_thinking": ENABLE_THINKING,
                   "n_checked": len(audits), "all_passed": True,
                   "rows": audits}, "corpus_mask_check.json", results_dir=ROOT / "results")

# %% [markdown]
# ## 5. Split cố định
#
# `seed=42` ở mọi notebook. Hai lần chạy khác seed thì **không so sánh được với nhau** —
# và cả lab này là về việc so sánh.

# %%
train, val = data.split(train_raw, train_frac=0.9, seed=42)
split_dir = ROOT / "data" / "split"
split_dir.mkdir(exist_ok=True)
for name, rows in (("train", train), ("val", val)):
    with (split_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"train={len(train)}  val={len(val)}  -> {split_dir}")

report.write_json({
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": round(time.perf_counter() - STARTED, 3),
    "platform": platform.platform(), "python": platform.python_version(),
    "model_id": TIER.model_id, "tier": TIER.name, "device": "CPU (tokenizer only)",
    "max_length": MAX_LENGTH, "enable_thinking": ENABLE_THINKING,
    "seed": 42, "n_train": len(train), "n_val": len(val),
    "packages": {p: importlib.metadata.version(p)
                 for p in ("transformers", "tokenizers", "jinja2")},
    "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
    "files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "data/train_seed.jsonl", split_dir / "train.jsonl",
                               split_dir / "val.jsonl")},
}, "nb1_run.json", results_dir=ROOT / "results")

# %% [markdown]
# ## ✅ Checkpoint NB1
#
# - [ ] `results/template_check.json` — biết template có giữ `<think>` không
# - [ ] `results/mask_proof.json` — hai assert đều xanh
# - [ ] `results/token_stats.json` — có p95
# - [ ] `data/split/{train,val}.jsonl` — seed 42
#
# → Sang **NB2**: đóng băng tập eval và đo ba baseline **trước khi** train.
