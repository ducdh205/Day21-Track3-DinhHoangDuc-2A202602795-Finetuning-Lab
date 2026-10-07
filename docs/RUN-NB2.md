# Chạy NB2 trên máy khác — trước khi train

Base model giữ nguyên `unsloth/Qwen3.5-4B`. Dùng NVIDIA T4 hoặc GPU đủ bộ nhớ.
Máy local hiện tại chỉ có Intel Iris Xe; chưa đo baseline NB2 tại máy này.

## Colab T4

Mở [notebook NB1 → NB2](https://colab.research.google.com/github/ducdh205/Day21-Track3-DinhHoangDuc-2A202602795-Finetuning-Lab/blob/main/colab/Lab21_NB2_RUN.ipynb).
Chọn **Runtime → Change runtime type → T4 GPU**, rồi chạy ô 1, 2, 3.
Ô 1 clone đúng repo của Đinh Hoàng Đức và nhánh `main`; ô 2 chỉ chạy NB1 và NB2;
ô 3 tải `lab21_nb1_nb2_results.zip` chứa JSON, dự đoán, log và split.

Nếu ô 2 báo lỗi, vẫn chạy ô 3 để lấy log/candidate. Nếu đã có
`results/baselines_frozen.json`, script sẽ từ chối ghi đè; đọc kết quả đã có.
Không dùng notebook RUN ALL để làm riêng bước này vì nó có thể chạy cả train.

## Máy NVIDIA khác

```bash
git clone --branch main https://github.com/ducdh205/Day21-Track3-DinhHoangDuc-2A202602795-Finetuning-Lab.git
cd Day21-Track3-DinhHoangDuc-2A202602795-Finetuning-Lab
python -m venv .venv
source .venv/bin/activate
# Cài torch đúng CUDA của máy nếu chưa có, rồi:
python -m pip install -r requirements.txt
export COMPUTE_TIER=T4
export BASE_MODEL=unsloth/Qwen3.5-4B
unset EVAL_LIMIT
python notebooks/01_data_and_mask.py
python -u notebooks/02_baselines.py
```

Windows PowerShell: dùng `.venv\Scripts\Activate.ps1`, đặt
`$env:COMPUTE_TIER="T4"`, `$env:BASE_MODEL="unsloth/Qwen3.5-4B"` và bỏ `EVAL_LIMIT`
bằng `Remove-Item Env:EVAL_LIMIT -ErrorAction SilentlyContinue`.

## Đọc kết quả

NB2 chạy đầy đủ **50 mẫu target + 15 mẫu regression cho mỗi baseline**. Generation
greedy, `enable_thinking=False`, cùng base model và cùng scorer cho (a), (b).
Baseline (c) chưa đo ở bước này.

- `results/baseline_predictions.json`: output từng mẫu của (a), (b), kèm nhãn để chấm lại.
- `results/baselines_frozen.json`: chỉ tạo khi full eval và **target(b) > target(a)**;
  chứa điểm bốn nhóm, SHA prompt, checksum eval, thời điểm, phần cứng và cấu hình generation.
- `results/baselines_candidate.json`: nếu smoke hoặc (b) chưa thắng (a), không đóng băng;
  cải thiện prompt (b) trước khi train, không sửa hay làm yếu (a), không sửa eval.

Prompt (b) giữ nguyên schema và ví dụ ban đầu, bổ sung năm ví dụ từ **tập train**
bao phủ năm intent, cùng hướng dẫn tách urgency khỏi sentiment. Chưa có số đo
chứng minh hiệu quả của bổ sung này; yêu cầu `(b) > (a)` được kiểm tra khi chạy GPU.
Không lấy ví dụ từ eval hoặc holdout. Sau đóng băng, giữ nguyên prompt (b) và eval.

`python scripts/verify.py` (hoặc `make verify`) kiểm tra SHA prompt và checksum
so với baseline đóng băng; thay đổi nội dung bị FAIL. Checksum corpus chuẩn hóa
CRLF thành LF để clone trên Windows và Linux cho cùng kết quả. Chỉ khác xuống dòng
được chấp nhận; thay nhãn, nội dung hoặc thứ tự mẫu vẫn bị phát hiện.
Full verifier còn yêu cầu artefact NB3–NB5, nên chưa thể báo ready-to-submit chỉ sau NB2.

Bằng chứng NB1 ngày 07/10/2026 được giữ riêng ở `submission/evidence/nb1` và được
report liên kết tới. Kết quả mới trên máy GPU ghi vào `results/` để không ghi đè
snapshot đó. Khi có file ZIP kết quả mới, dùng số đo trong đó để bổ sung report NB2;
không điền điểm baseline theo ước lượng.
