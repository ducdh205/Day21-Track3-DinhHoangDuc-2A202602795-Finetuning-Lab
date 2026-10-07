# NB1 — Dữ liệu, chat template và loss mask

**Họ tên:** Đinh Hoàng Đức · **MSSV:** 2A202602795

**Ngày thực nghiệm:** 07/10/2026 · **Phạm vi:** NB1, chạy thật trên CPU.

## 1. Thiết lập và nguồn bằng chứng

Thực nghiệm dùng tokenizer thật của `unsloth/Qwen3.5-4B`, giữ nguyên base model mặc định của tier T4 để bằng chứng áp dụng cho model dự kiến dùng trong lab. T4 ở đây là tên cấu hình; phần cứng thực thi NB1 là CPU trong WSL2, không tải trọng số model và không chạy huấn luyện. Corpus là bộ ticket CSKH tiếng Việt đi kèm repo, với đầu ra JSON gồm `intent`, `urgency`, `product`, `sentiment`. Đây là corpus seed của lab, không phải dữ liệu khách hàng thu thập thực tế; các số đo trong report là kết quả thực thi trên corpus này.

| Thuộc tính | Kết quả thực tế | Bằng chứng |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` | [nb1_run.json](evidence/nb1/nb1_run.json) |
| Corpus | 250 mẫu | [token_stats.json](evidence/nb1/token_stats.json) |
| Train / validation | 225 / 25, seed 42 | [nb1_run.json](evidence/nb1/nb1_run.json) |
| Python | 3.12.3 | [nb1_run.json](evidence/nb1/nb1_run.json) |
| Transformers / Tokenizers / Jinja2 | 5.19.0 / 0.23.2 / 3.1.6 | [nb1_run.json](evidence/nb1/nb1_run.json) |
| Loss mask | `assistant-only` | [mask_proof.json](evidence/nb1/mask_proof.json) |
| Render dùng cho mask | `enable_thinking=False` | [mask_proof.json](evidence/nb1/mask_proof.json) |
| `max_length` thực dùng trong NB1 | 256 | [token_stats.json](evidence/nb1/token_stats.json) |
| Thời gian NB1 | 33,696 giây | [nb1_run.json](evidence/nb1/nb1_run.json) |

Lần chạy hoàn tất lúc **18:27:17 ngày 07/10/2026, giờ Việt Nam**. Thời gian trên đo từ đầu script NB1 đến lúc ghi metadata, gồm nạp tokenizer, tính thống kê, kiểm tra mask toàn corpus và tạo split; không gồm cài dependencies, lần thử trước đó hay script so TRL. Report ghi thời gian đo được, không lấy mốc khoảng 25 giây trong đề bài làm kết quả.

Tokenizer được tải từ snapshot `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`. SHA-256 chat template: `8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8`. SHA-256 corpus và hai file split được lưu trong `nb1_run.json` để đối chiếu lần chạy.

## 2. Chuỗi sau template và phần thực sự được tính loss

Mẫu đầu tiên sau `apply_chat_template`, với `enable_thinking=False`:

```text
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Giải mã ngược đúng các vị trí `labels != -100`:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Chuỗi loss có một ký tự xuống dòng sau `<|im_end|>`; bản đầy đủ nằm trong trường `supervised_text`. Nội dung được giám sát khớp toàn bộ JSON trả lời, cộng token kết thúc lượt và xuống dòng của template. Token kết thúc được giữ để model học dừng. Các thẻ role, system prompt, câu hỏi và khối thinking rỗng ở generation prompt đều bị mask. “Đúng bằng câu trả lời” ở đây nghĩa là nội dung câu trả lời cùng phần kết thúc lượt, không phải chỉ một đoạn đầu của JSON.

| Kiểm tra trên mẫu đầu | Kết quả |
|---|---|
| Tổng token | 94 |
| Token được giám sát | 37 |
| Token bị mask | 57 |
| `supervised_fraction` | **0.3936**, thấp hơn 0.95 |
| `answer_is_supervised` | **true** |
| `question_is_masked` | **true** |
| `supervised_equals_rendered_answer_tail` | **true** |

Hai assert bắt buộc đều xanh. Kiểm tra câu trả lời dùng **toàn bộ output**, không chỉ 40 ký tự đầu. Các vị trí được giám sát là chỉ số **57–93**, tính từ 0; `input_ids` và `labels` được lưu trong [mask_proof.json](evidence/nb1/mask_proof.json), nên có thể kiểm tra trực tiếp phép giải mã.

Đã kiểm tra **250/250 mẫu** với cùng cấu hình. Mỗi mẫu đều chứa toàn bộ câu trả lời trong loss, không chứa câu hỏi, khớp chính xác đuôi assistant sau generation prompt, và có `0 < supervised_fraction < 0.95`. Tỷ lệ giám sát từng mẫu nằm trong khoảng **0.376344–0.437500**. Toàn corpus có **23.266 token**, trong đó **9.506 token** được giám sát. Bằng chứng từng mẫu nằm trong [corpus_mask_check.json](evidence/nb1/corpus_mask_check.json).

Đối chứng `everything` trên cùng mẫu giám sát **94/94 token, tức 100%**, và đoạn giải mã chứa cả câu hỏi. Đối chứng này cho thấy ngưỡng 0.95 phát hiện lỗi tính loss trên prompt. Chế độ được chọn cho NB1 là `assistant-only`.

## 3. Template có giữ `<think>` không?

**Có, trong ca kiểm tra của NB1.** [template_check.json](evidence/nb1/template_check.json) ghi `ok=true`, `open_tag_present=true`, `body_present=true`. Nội dung kiểm tra vẫn xuất hiện sau render:

```text
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
<think>
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

Phép kiểm tra này dùng chế độ thinking mặc định của template và một lượt assistant chứa trace. Kết quả chứng minh trace được giữ trong ca đó; không suy rộng thành bảo đảm cho mọi hội thoại nhiều lượt. Corpus ticket có câu trả lời JSON thuần, nên phần chứng minh mask dùng `enable_thinking=False`. Template vẫn render khối thinking rỗng, nhưng đặt nó trong generation prompt và mask toàn bộ khối này.

Lần thử ban đầu với thinking mặc định cho thấy `</think>` còn nằm trong đoạn loss. Bằng chứng được giữ ở [nb1_initial_attempt.log](evidence/nb1/nb1_initial_attempt.log). Đây là lý do chọn cấu hình render tường minh cho lần chạy cuối, thay vì kết luận mask đúng chỉ vì JSON có mặt trong loss.

## 4. Đặt `max_length` từ số đo

Độ dài được đo trên cả 250 hội thoại **sau chat template**, gồm system, user, assistant và special tokens, với cùng `enable_thinking=False` như phần chứng minh mask. Phép đo lấy số phần tử trong `input_ids`, không truncation trước khi tính percentile.

| Thống kê | Token |
|---|---|
| Trung bình | 93.1 |
| p50 | 93 |
| p95 | **98** |
| p99 | 100 |
| Lớn nhất | 101 |
| `suggested_max_length` / `selected_max_length` | **256 / 256** |
| Mẫu có độ dài vượt 256 | **0** |

Nguồn: [token_stats.json](evidence/nb1/token_stats.json). Percentile dùng nearest-rank: với 250 mẫu, p95 là phần tử thứ 238 trong danh sách độ dài đã sắp xếp. Hàm của lab làm tròn lên lũy thừa 2 và có mức tối thiểu 256; do đó p95=98 cho kết quả 256, thay vì phép làm tròn trực tiếp lên 128.

Tier T4 mặc định đặt 1024, nhưng NB1 đã **thực sự dùng 256** để xây mask và kiểm tra corpus. Vì độ dài lớn nhất chỉ là 101, lựa chọn này giữ toàn bộ câu trả lời và token kết thúc ở tất cả mẫu. Thực nghiệm không đo hiệu năng train hay VRAM nên không quy đổi việc giảm trần độ dài thành mức tiết kiệm bộ nhớ thực tế.

## 5. So mask với hai đường TRL

Đã chạy `python scripts/check_mask_agreement.py` với **TRL 1.14.2**, trước mọi huấn luyện. Output đầy đủ ở [mask_agreement.log](evidence/nb1/mask_agreement.log). Script dùng cặp hỏi–đáp ngắn riêng và thinking mặc định; số token của nó không phải số token mẫu ticket trong phần 2.

| Đường kiểm tra | Kết quả thực tế |
|---|---|
| Mask lab trên mẫu của script | 11/31 token, gồm `</think>`, JSON và kết thúc lượt |
| Tokenizer với `return_assistant_tokens_mask=True` | **0/31 token**; cảnh báo thiếu `{% generation %}` |
| Hàm `get_training_chat_template` mà `SFTTrainer` dùng | **ValueError**: template không tương thích huấn luyện |
| Exit code của script | **1 — FAIL** |

Template Unsloth hiện tại không có generation markers, nên lấy trực tiếp `assistant_masks` sẽ tạo batch không có token được giám sát. Đường trainer báo lỗi thay vì âm thầm sử dụng mask đó. Phép kiểm tra trainer ở đây thực thi hàm xử lý template của TRL; không khởi tạo model hay thực hiện một bước `SFTTrainer`.

Không thể khẳng định mask TRL tương đương mask lab. Hướng dùng khi train là dữ liệu đã tokenize bằng `labkit.data.to_training_dataset`, giữ nguyên `input_ids` và `labels` đã chứng minh, với `max_length=256`, `mask_mode="assistant-only"`, **`enable_thinking=False`**, và không packing. NB3/NB4 hiện vẫn gọi helper với độ dài tier và thinking mặc định; cần đồng bộ hai tham số này trước khi train để dùng đúng cấu hình đã chứng minh. Report NB1 chưa xác nhận một batch train của NB3/NB4. Nếu đổi base model, phải chạy lại NB1 và script so mask cho model mới.

## 6. Tái chạy và kiểm tra

Từ thư mục repo, trong môi trường Python đã cài dependencies CPU:

```bash
export COMPUTE_TIER=T4
unset BASE_MODEL
.venv/bin/python notebooks/01_data_and_mask.py
.venv/bin/python scripts/check_mask_agreement.py
.venv/bin/python -m pytest tests
```

Để kiểm tra thêm hàm template của TRL trong môi trường CPU, cần cài `trl` và `datasets`; lần đo này dùng lần lượt 1.14.2 và 5.1.0. Script so mask hiện dự kiến trả mã 1 với template trên, và mã này phải được đọc cùng output thay vì đổi thành PASS.

Log NB1 thành công: [nb1_console.log](evidence/nb1/nb1_console.log). Split cố định: [train.jsonl](evidence/nb1/train.jsonl), [val.jsonl](evidence/nb1/val.jsonl). Bộ test hiện có chạy bằng Python Windows và trả **116 passed, 3 skipped**; đây là kiểm tra phần mềm bổ sung, tách biệt với lần NB1 chạy tokenizer thật trong WSL. Đối chiếu labels, checksum và đường dẫn report đều pass, được ghi trong [nb1_validation.json](evidence/nb1/nb1_validation.json). Notebook Colab NB1 đã được sinh lại từ nguồn Python.

## 7. Kết luận và điều học được

NB1 hoàn thành mục tiêu chứng minh vùng loss trên corpus đã chọn: nội dung được giám sát khớp đầy đủ câu trả lời JSON, giữ token kết thúc lượt, và loại toàn bộ prompt. Kết luận dựa trên phép giải mã labels, hai assert bắt buộc và kiểm tra từng mẫu trong toàn corpus. Tỷ lệ giám sát 0.3936 ở mẫu đầu, cùng khoảng tỷ lệ của 250 mẫu, cho thấy đây không phải chế độ tính loss trên mọi token. Tuy nhiên, chỉ nhìn tỷ lệ thấp chưa đủ: lần thử ban đầu vẫn giám sát phần đóng thinking dù câu trả lời có mặt. Vì vậy phép so khớp chính xác đuôi assistant và khai báo `enable_thinking` là phần cần thiết của bằng chứng.

Qua thực nghiệm này, tôi rút ra rằng tên cờ `assistant-only` không tự bảo đảm vùng loss đúng với ý định huấn luyện. Chat template quyết định generation prompt kết thúc ở đâu, và cùng một model có thể cho vùng loss khác khi thay chế độ thinking. Tôi cũng thấy cần kiểm tra dạng dữ liệu trả về khi đo token: độ dài dictionary không phải độ dài chuỗi token. Số đo p95 chỉ có ý nghĩa khi lấy từ đúng hội thoại đã render và đúng cấu hình tạo labels. Cuối cùng, mask lab pass nhưng kiểm tra tương thích TRL fail là hai kết quả có thể cùng đúng. Bước tiếp theo cần dùng labels đã chứng minh và kiểm tra batch thực tế trước khi train; chưa có cơ sở kết luận fine-tuning cải thiện chất lượng, latency hay khả năng triển khai.

**Trạng thái:** NB1 PASS với cấu hình trong report; kiểm tra tương thích TRL FAIL như log. NB2–NB5 chưa chạy, nên report không có số liệu baseline, training loss, VRAM, cổng hồi quy hay phán quyết fine-tune thắng/thua.
