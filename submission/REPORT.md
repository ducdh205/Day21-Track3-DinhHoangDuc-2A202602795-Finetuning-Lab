# Lab21 — Fine-tuning ticket CSKH: tăng target nhưng không vượt cổng regression

**Họ tên:** Đinh Hoàng Đức · **MSSV:** 2A202602795  
**Ngày thực nghiệm:** 08/10/2026 · **Phạm vi:** NB1–NB5, full eval, EPOCHS=2.

**Phán quyết model: FAIL.** `correct` đạt target **0.97**, vượt baseline tối ưu **0.85**, nhưng regression giảm **0.046666666666666745**, vượt tolerance **0.02**. Gatekeeper kiểm tra hồ sơ trả mã 0; điều này không đổi phán quyết model thành PASS. Kết quả Colab được nhập nguyên byte từ ZIP người chạy cung cấp, sau đó chấm lại dự đoán bằng scorer nguyên bản trên CPU. Không sửa test, eval, prompt tối ưu hoặc ngưỡng chấm sau khi thấy kết quả.

## 1. Model, dataset và lý do chọn

Tôi chọn **`unsloth/Qwen3.5-4B`**, base mặc định của tier T4 trong repo, để giữ cùng model xuyên suốt bằng chứng mask, baseline và bốn adapter. Quy mô này phù hợp với cấu hình T4 của lab; phép đo thực tế cho thấy `correct` dùng đỉnh VRAM **8.78 GB**. Tôi không đổi sang model nhỏ hơn sau khi xem điểm. Model có text decoder kết hợp linear attention và full attention, nên bài toán vị trí LoRA có ý nghĩa trực tiếp trên kiến trúc được nạp thật.

Dataset là **250 ticket CSKH tiếng Việt thuộc corpus seed của repo**, với bốn nhãn `intent`, `urgency`, `product`, `sentiment`. Tôi chọn bộ này vì có nhãn từng trường, đầu ra JSON rõ ràng và có thể đánh giá tự động. Đây là corpus tổng hợp của lab, không phải dữ liệu khách hàng thực tế; kết quả chưa chứng minh khả năng xử lý ticket ngoài phân phối. Split seed **42** tạo **225 mẫu train / 25 validation**. Eval độc lập gồm **50 ticket target** và **15 câu hỏi regression**, giữ nguyên nội dung và thứ tự.

Máy local có Intel i5-1135G7, RAM khoảng 16 GB, Intel Iris Xe và không có NVIDIA CUDA; đủ chạy tokenizer NB1 và scorer CPU, không đáp ứng đường train GPU của project. Vì vậy phần model chạy trên **Colab Tesla T4**, capability **7.5**, VRAM báo **14.6 GB**, base precision **fp16**. T4 không hỗ trợ bf16. Colab dùng Python **3.13.15**, Transformers **5.18.0**, Tokenizers **0.23.2**, PyTorch **2.11.0+cu130**, TRL **1.14.2**, Datasets **5.1.0**. Nguồn: [local_hardware.json](../results/local_hardware.json), [nb1_run.json](../results/nb1_run.json), [baselines_frozen.json](../results/baselines_frozen.json), [training_mask_validation.json](../results/training_mask_validation.json).

## 2. NB1 — chứng minh vùng loss và chọn độ dài

Chuỗi mẫu đầu sau `apply_chat_template`, với `enable_thinking=False`:

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

Giải mã chỉ các vị trí `labels != -100`:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Chuỗi được giám sát có thêm xuống dòng sau token kết thúc lượt. Tôi giữ token kết thúc để model học dừng; toàn bộ nội dung JSON được giám sát, còn system, câu hỏi, role marker và khối thinking rỗng trong generation prompt đều bị mask. Bằng chứng gồm token IDs, labels và chuỗi giải mã đầy đủ trong [mask_proof.json](../results/mask_proof.json).

| Kiểm tra mẫu đầu | Kết quả |
|---|---:|
| Tổng token / token được giám sát | 94 / 37 |
| Vị trí được giám sát, chỉ số từ 0 | 57–93 |
| `supervised_fraction` | **0.3936 < 0.95** |
| `answer_is_supervised` | true |
| `question_is_masked` | true |
| `supervised_equals_rendered_answer_tail` | true |

Hai assert bắt buộc đều xanh. Kiểm tra exact-tail giúp tránh trường hợp câu trả lời có mặt nhưng một phần prompt vẫn bị tính loss. Đối chứng `everything` tính loss **94/94 token**, chứa câu hỏi, nên không được dùng để train.

Toàn corpus **250/250 mẫu** qua kiểm tra mask; có **23,266 token**, **9,506 token** được giám sát. Tỷ lệ từng mẫu nằm trong khoảng **0.376344–0.437500**. Dataset train có **20,951 token**, **8,564 token** được giám sát, tỷ lệ **0.40876330485418355**. Nguồn: [corpus_mask_check.json](../results/corpus_mask_check.json), [training_mask_validation.json](../results/training_mask_validation.json).

| Thống kê sau template, không truncation trước khi đo | Token |
|---|---:|
| Mean / p50 | 93.1 / 93 |
| p95 / p99 / max | **98** / 100 / 101 |
| `suggested_max_length` / `selected_max_length` | **256 / 256** |
| Số mẫu sẽ bị truncate | 0 |

`max_length=256` được suy ra từ p95 đo thật và quy tắc làm tròn lên lũy thừa hai với sàn 256 của lab. Giá trị **1024** ở đầu console là mặc định tier trước phép đo; recipe train thực tế và cả bốn dòng `runs.csv` đều dùng **256**. NB1 trên Colab mất **18.904 giây** theo metadata, không lấy mốc thời gian ước lượng trong đề bài làm kết quả. Nguồn: [token_stats.json](../results/token_stats.json), [nb1_run.json](../results/nb1_run.json).

Template **giữ `<think>` và thân trace** trong ca kiểm tra một lượt của NB1: `ok`, `open_tag_present`, `body_present` đều true. Điều này chưa bảo đảm mọi hội thoại nhiều lượt đều giữ trace. Corpus ticket không có trace và thực nghiệm dùng `enable_thinking=False`; không suy diễn `valid_trace_rate=0` thành mất năng lực reasoning trong thí nghiệm này. Nguồn: [template_check.json](../results/template_check.json).

Script `check_mask_agreement.py` đã chạy trước train và trả **exit 1**: đường tokenizer của TRL trả mask rỗng vì thiếu `{% generation %}`, còn đường xử lý template của SFTTrainer báo `ValueError`. Đây là kết quả kiểm tra tương thích thất bại được giữ nguyên, không phải một run train âm thầm bỏ loss. NB3/NB4 dùng dữ liệu **pre-tokenize với labels của lab**, không dựa vào cờ `assistant_only_loss`.

Trong lần chạy thật, mỗi trainer kiểm tra **225 mẫu** và batch collator trước `.train()`: `all_dataset_labels_equal_nb1=true`, `collator_labels_equal_nb1=true`. Cả bốn batch proof đều đạt. Nguồn: [mask_agreement.log](../results/mask_agreement.log), [training_batch_correct.json](../results/training_batch_correct.json), [training_batch_attn_only.json](../results/training_batch_attn_only.json), [training_batch_wrong_lr.json](../results/training_batch_wrong_lr.json), [training_batch_qlora.json](../results/training_batch_qlora.json). Nếu đổi base model phải chứng minh lại NB1 và chạy lại script so mask.

## 3. NB2 — baseline đóng băng trước train

Baseline (a) dùng prompt ngắn **“Phân loại ticket sau.”** Baseline (b) dùng schema, danh sách giá trị hợp lệ, quy tắc phân biệt ý định, quy tắc urgency độc lập sentiment và few-shot; các ví dụ bổ sung lấy từ train, không lấy từ eval. Prompt (b) được giữ nguyên sau khi đo. Bản fine-tune (c) được chấm sau train với prompt ngắn của (a), để kiểm tra hành vi đã học trong adapter.

NB2 bắt đầu **03:47:15 UTC**, kết thúc **03:58:58 UTC**, mất **703.157 giây** theo metadata. Mốc đóng băng: **2026-10-08T03:58:58.506695+00:00**, trước NB3; `measured_before_training=true`, `eval_limit=null`, `smoke_mode=false`. Điểm target **(b)=0.85 > (a)=0.0**, đạt điều kiện baseline. Nguồn: [baselines_frozen.json](../results/baselines_frozen.json).

| Nhóm | (a) Base + prompt đơn giản | (b) Base + prompt tối ưu | (c) `correct` fine-tune |
|---|---:|---:|---:|
| Target, accuracy từng trường | 0.0000 | **0.8500** | **0.9700** |
| Regression, keyword recall | 0.7911 | **0.7911** | **0.7444** |
| Format, JSON có đủ khóa | 0.0000 | **1.0000** | **1.0000** |
| Latency target, ms/mẫu | 3181.5 | **1412.7** | **1399.5** |

Nguồn bảng: [verdict.json](../results/verdict.json), baseline chính xác chưa làm tròn trong [baselines_frozen.json](../results/baselines_frozen.json). Target/format dùng đủ 50 ticket, regression dùng đủ 15 câu hỏi. Greedy decode, `do_sample=False`, `enable_thinking=False`, batch generation **4**, tối đa **160 token target / 96 token regression**. Latency là thời gian generate của batch chia số mẫu, không phải latency đơn lẻ end-to-end và không bao gồm tải model. Không kết luận hơn kém ổn định từ chênh lệch latency nhỏ giữa (b) và (c) trong một lần đo.

Prompt (a) không cung cấp schema nên điểm 0 phản ánh thất bại với hợp đồng đầu ra của tác vụ, không chứng minh model không hiểu tiếng Việt. Đối thủ cần vượt là (b), đã đạt format hoàn chỉnh và target cao. Regression của (a)/(b) giống nhau vì đều hỏi không có system prompt ticket ở nhóm này.

SHA-256 prompt (b): `4f9ca3c2c36b3c108235a971e99971644c563cc34db57460324fa3e703503a38`.

| File eval | SHA-256, chuẩn hóa CRLF thành LF |
|---|---|
| `eval_target.jsonl` | `2991050fefb848c8bf3fdbc021badea9500f758d90717d7a94ae2f9b09b8915e` |
| `eval_regression.jsonl` | `9c6ba90ea86e28fbc7102bf4cc4cb1e23c4668cf95bfcae55c3c114d34dc133b` |

Đã đối chiếu hai checksum, SHA prompt và SHA file baseline predictions với bản đóng băng; đều khớp. Không làm yếu (b) hoặc thay eval để tăng mức thắng của fine-tune.

## 4. NB3/NB4/NB5 — cấu hình, công bằng và xếp hạng

Kiến trúc nạp thật có **32 lớp**, gồm **24 linear attention / 8 full attention**, `full_attention_interval=4`, `linear_num_key_heads=16`. Nguồn: [architecture.json](../results/architecture.json). `correct` gắn LoRA vào toàn bộ linear của text decoder qua các suffix:

```text
down_proj, gate_proj, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z,
k_proj, o_proj, out_proj, q_proj, up_proj, v_proj
```

Đây là **12 loại module**, không phải chỉ 12 lớp linear. Bộ chọn target loại phần vision; không gắn adapter vào vision tower của model đa phương thức. `correct` dùng **r=16, alpha=32=2r, LR=1e-4**, bằng **10×** mốc full-FT **1e-5** của lab. Batch mỗi device **1**, accumulation **16**, batch hiệu dụng **16 < 32**. **EPOCHS=2** quy đổi thành **30 optimizer step**, cùng budget và actual_steps cho cả bốn run; giữ dataset, mask, seed, max_length và thinking mode.

| Run | Biến đối chứng | r / alpha | Tham số trainable | LR | Loss train | Train, giây | Peak VRAM, GB |
|---|---|---:|---:|---:|---:|---:|---:|
| `correct` | Text decoder linear, base fp16 | 16 / 32 | 32,464,896 | 1e-4 | 0.6599 | 390.1 | 8.78 |
| `attn_only` | Vị trí chỉ q,v, rank khớp ngân sách | 283 / 566 | 32,456,704 | 1e-4 | 0.5658 | 274.0 | 8.79 |
| `wrong_lr` | Chỉ LR xuống thang full-FT | 16 / 32 | 32,464,896 | 1e-5 | 1.7252 | 408.5 | 8.78 |
| `qlora` | Base lượng tử hóa 4-bit NF4 | 16 / 32 | 32,464,896 | 1e-4 | 0.7428 | 463.0 | 3.86 |

Nguồn: [runs.csv](../results/runs.csv). Cột `final_loss` của lab lấy `training_loss` trả về sau train, tức loss huấn luyện trung bình, không phải điểm đánh giá hay riêng loss ở step cuối. Peak VRAM là bộ nhớ CUDA allocated cao nhất do PyTorch báo, không phải toàn bộ bộ nhớ reserved của runtime.

`attn_only` dùng `matched_rank()` trên chính model: chênh lệch ngân sách **8,192 tham số**, tương đương **0.025233409033560434%**, thấp hơn 5%. Alpha cũng tăng để giữ **alpha/r=2**. Nếu giữ q,v ở r=16, ngân sách chỉ **1,835,008**, không còn là đối chứng công bằng về vị trí. Nguồn số tham số này: [nb3_console.log](../results/nb3_console.log); chênh lệch được tính lại trong [colab_log_summary.json](../results/colab_log_summary.json).

| Hạng theo target NB5 | Run | Target | Format | Latency, ms/mẫu | Regression |
|---|---|---:|---:|---:|---|
| 1 | `correct` | **0.970** | 1.0 | 1399.5 | 0.7444 |
| 2 | `attn_only` | **0.965** | 1.0 | 899.4 | Chưa đo |
| 3 | `qlora` | **0.940** | 1.0 | 1785.1 | Chưa đo |
| 4 | `wrong_lr` | **0.000** | 0.0 | 5163.9 | Chưa đo |

Nguồn: [autopsy.json](../results/autopsy.json), [verdict.json](../results/verdict.json). NB5 mặc định chỉ đo regression cho `correct`; không điền 0 hoặc suy đoán cho ba contrast. Cổng đủ bốn nhóm áp dụng cho `correct` so với hai baseline.

Thứ tự target là **correct > attn_only > qlora > wrong_lr**. Nếu chọn bằng loss, `attn_only` sẽ đứng đầu; thí nghiệm này cho thấy loss thấp nhất không đồng nghĩa target cao nhất. Tuy nhiên correct chỉ hơn attention **0.005**, tương đương một trường đúng trên tổng các nhãn của eval; một seed và tập nhỏ chưa đủ kết luận text-linear luôn ưu việt. Attention còn nhanh hơn ở phép đo latency này.

QLoRA giảm peak VRAM từ **8.78 xuống 3.86 GB**, tương đương khoảng **56.04%**, nhưng target giảm từ **0.97 xuống 0.94** và latency tăng trong runtime này. Đây là đánh đổi đo được, không chứng minh mọi model đều nên tránh QLoRA. `wrong_lr` không học được hợp đồng JSON trong cùng số step: format/target đều 0. Kết quả chỉ áp dụng cho budget hiện tại; không chứng minh LR nhỏ không thể hội tụ nếu train lâu hơn. Runtime thiếu các kernel tối ưu cho linear attention và dùng triển khai PyTorch dự phòng; các phép đo thời gian gắn với môi trường đó.

Chấm lại từng trường trên saved predictions cho kết quả:

| Run | intent | urgency | product | sentiment |
|---|---:|---:|---:|---:|
| Baseline (b) | 0.88 | 0.64 | 1.00 | 0.88 |
| `correct` | 1.00 | 0.88 | 1.00 | 1.00 |
| `attn_only` | 1.00 | 0.86 | 1.00 | 1.00 |
| `wrong_lr` | 0.00 | 0.00 | 0.00 | 0.00 |
| `qlora` | 0.92 | 0.86 | 1.00 | 0.98 |

Nguồn: [report_evidence.json](../results/report_evidence.json), tạo bằng scorer nguyên bản trên dự đoán gốc. Sai sót target còn lại của correct tập trung ở urgency, nên tăng rank không phải chẩn đoán đầu tiên hợp lý.

## 5. Định tính — cả thắng, hòa và thua

Trên 50 target, `correct` có **22 thắng / 28 hòa / 0 thua** so với (b). Trên 15 regression có **1 thắng / 11 hòa / 3 thua theo scorer**. Vì không có ca thua target, tôi đưa hai ca thua **regression** vào report, ghi rõ nhóm và tiêu chí; không gọi một ticket FT sai nhưng baseline cũng sai là “FT thua”. Nguồn: [qualitative.json](../results/qualitative.json), [baseline_predictions.json](../results/baseline_predictions.json), [adapter_predictions.json](../results/adapter_predictions.json), bản đối chiếu [report_evidence.json](../results/report_evidence.json). Chỉ số mẫu dưới đây bắt đầu từ 0; trích đoạn chỉ lược các trường không liên quan, output đầy đủ giữ trong file.

| Nhóm / i | Điểm (b) → FT | Kết quả | Quan sát |
|---|---:|---|---|
| Target / 3 | 0.50 → 0.75 | Thắng, còn sai | Sửa intent hoàn tiền; urgency vẫn sai |
| Target / 5 | 0.75 → 0.75 | Hòa, cùng sai | “Khi nào tiện” bị gán trung_binh |
| Target / 12 | 0.75 → 0.75 | Hòa, cùng sai | Sản phẩm lỗi không đồng nghĩa cần xử lý gấp |
| Target / 25 | 0.50 → 1.00 | Thắng | Sửa cả intent đổi/trả và urgency |
| Target / 49 | 0.50 → 1.00 | Thắng | Giữ đúng urgency và sentiment theo nhãn |
| Regression / 9 | 1.00 → 0.00 | **Thua** | Câu hỏi số tháng bị trả thành ticket JSON |
| Regression / 14 | 1.00 → 0.50 | **Thua theo keyword recall** | Giải thích vẫn hợp lý nhưng thiếu từ “cây” |

**Target 3 — lỗi urgency dù đã sửa intent.** Ticket: “Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.” Nhãn là `hoan_tien / thap / bình giữ nhiệt / tich_cuc`. Baseline xuất `hoi_thong_tin / trung_binh / bình giữ nhiệt / tich_cuc`; FT xuất `hoan_tien / trung_binh / bình giữ nhiệt / tich_cuc`. FT nhận ra yêu cầu hoàn tiền nhưng bỏ qua “Khi nào tiện”; đây là thắng một trường, chưa hoàn hảo.

**Target 5 — cùng sai, không phải FT thua.** Ticket: “Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.” Nhãn urgency là `thap`; cả (b) và FT đều xuất `trung_binh`, ba trường còn lại đúng. Đây là một trong ba ca FT có điểm thấp nhất được in ở NB5, nhưng delta so với baseline bằng 0.

**Target 12 — lỗi tương tự trên sản phẩm khác.** Ticket: “Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.” Nhãn là `san_pham_loi / thap / áo khoác gió / tich_cuc`. Hai model đều xuất đúng intent/product/sentiment nhưng urgency `trung_binh`. Các ca này gợi ý model chưa áp dụng nhất quán tín hiệu thời gian; chưa đủ bằng chứng khẳng định nó suy urgency từ intent.

**Target 25 — thắng rõ hai trường.** Ticket: “Cho mình hỏi, mình đặt tai nghe bluetooth mã đơn OD851441. Cho tôi trả lại. Hỏi cho biết thôi. Shop hỗ trợ tốt.” Nhãn là `doi_tra / thap / tai nghe bluetooth / tich_cuc`. Baseline xuất `hoi_thong_tin / trung_binh`; FT sửa thành `doi_tra / thap`, vẫn giữ đúng product và sentiment. Lời mở đầu “Cho mình hỏi” không làm FT bỏ qua yêu cầu trả hàng phía sau.

**Target 49 — giảm suy diễn quá mức.** Ticket: “Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé. Shop xem giúp.” Nhãn là `san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh`. Baseline gán `cao / tieu_cuc`; FT gán `trung_binh / trung_tinh`, khớp cả bốn trường. Tôi diễn giải theo nhãn của corpus, không khẳng định mọi khách ngoài thực tế dùng “Sớm nhé” đều có cùng urgency.

**Regression 9 — thua về nội dung.** Câu hỏi: “Một năm có bao nhiêu tháng?”, keyword `12`. Baseline trả “Một năm bình thường có **12 tháng**”; FT trả toàn bộ:

```json
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```

FT áp schema ticket ngay cả khi nhóm regression không có system prompt ticket, và bỏ câu trả lời phổ thông. Đây là bằng chứng cụ thể của hành vi chuyên biệt lấn sang tác vụ khác, không chỉ lỗi trình bày.

**Regression 14 — thua theo thước đo, cần giới hạn diễn giải.** Câu hỏi: “Giải thích ngắn gọn quang hợp là gì.” Keywords là `ánh sáng`, `cây`. Baseline có cả hai; FT có “năng lượng ánh sáng mặt trời” và giải thích thực vật chuyển CO₂/nước thành chất hữu cơ, nhưng dùng “thực vật” thay cho từ “cây”, nên recall giảm từ 1 xuống 0.5. Nội dung này vẫn giải thích hợp lý. Tôi giữ điểm gốc và phán quyết FAIL, đồng thời không coi ca này là bằng chứng model mất kiến thức quang hợp. Regression 11 cũng cho thấy giới hạn scorer: chuẩn hóa dấu khiến keyword “dứa” có thể khớp từ “đưa” trong câu trả lời baseline dù ví dụ trái cây của baseline không đúng. Không sửa scorer hoặc keywords để thay kết quả đã đóng băng.

## 6. Phán quyết, diễn giải và điều học được

**Giữ nguyên FAIL trong `verdict.json`.** Mục tiêu fine-tuning trong lab yêu cầu đồng thời cải thiện target so với prompt tối ưu và giữ năng lực phổ thông trong tolerance. Run correct thỏa điều kiện đầu nhưng không thỏa điều kiện sau: target tăng **0.12**, còn regression delta là **−0.046666666666666745**, thấp hơn ngưỡng cho phép **−0.02**. Format đã đạt tối đa từ baseline (b), nên lợi ích bổ sung của fine-tuning chủ yếu nằm ở phân loại nhãn với prompt ngắn. Nó không tạo thêm lợi ích format để bù regression. Latency gần baseline tối ưu trong phép đo hiện tại; không có bằng chứng đủ mạnh về lợi ích tốc độ ổn định.

Các dự đoán regression cho thấy adapter có xu hướng áp kiểu trả lời JSON ra ngoài tác vụ ticket. Câu hỏi số tháng là ví dụ trực tiếp: model không trả lời kiến thức đơn giản mà phân loại câu hỏi như một ticket. Việc train chỉ trên corpus ticket khiến hành vi đó trở nên phù hợp với mục tiêu tối ưu của dữ liệu, nhưng không phù hợp với yêu cầu giữ năng lực tổng quát. Đây là giả thuyết cơ chế được hỗ trợ bởi output quan sát, chưa phải chứng minh nhân quả tuyệt đối; chưa có đối chứng replay trong thí nghiệm này. Tôi cũng phân biệt lỗi nội dung với giảm keyword recall do cách diễn đạt, như ca quang hợp. Tập regression nhỏ và scorer dựa từ khóa có giới hạn, nhưng các giới hạn ấy không cho phép tôi tự nới gate sau khi xem điểm.

Vì vậy tôi chưa chọn adapter này cho một trợ lý phải trả lời cả ticket và câu hỏi phổ thông. Nếu tiếp tục nghiên cứu, tôi sẽ giữ nguyên bộ eval, thêm một lượng nhỏ dữ liệu replay phổ thông như hướng dẫn deck, đo lại cả bốn nhóm và xem có giảm hành vi ép JSON ngoài tác vụ hay không. Đó là đề xuất cho thí nghiệm tiếp theo, **chưa được chạy và không có kết quả replay trong bài nộp**. Với tác vụ ticket biệt lập, target của correct rất tốt trên corpus hiện tại, nhưng cần tập dữ liệu thực tế rộng hơn trước khi khẳng định khả năng triển khai.

Những điều cụ thể tôi học được:

- **Mask cần chứng minh ở batch thực tế.** Tokenizer proof xanh chưa đủ nếu trainer dùng thinking mode, độ dài hoặc collator khác. Lỗi mặc định thinking làm **225/225** mẫu khác recipe; sửa bằng cách truyền recipe NB1 và kiểm tra labels trước train.
- **Ngân sách LoRA quyết định tính công bằng.** Attention r=16 không thể so trực tiếp với text-linear r=16 để kết luận về vị trí. Rank 283 khớp ngân sách, và kết quả gần correct hơn nhiều so với một tuyên bố “attention-only luôn tệ”.
- **Loss là chỉ số thay thế.** Attention có loss thấp nhất nhưng correct có target cao nhất; tôi phải chờ NB5 mới xếp hạng.
- **QLoRA cần đánh giá đánh đổi thực đo.** Nó tiết kiệm VRAM rõ trong lần chạy này nhưng đánh đổi target và thời gian; không dùng nhãn “cấu hình sai” để đoán trước kết quả.
- **Ca sai chưa chắc là ca thua baseline.** Ba ticket tệ nhất có một ca thắng và hai ca hòa; hai ca thua trong report lấy từ regression với điểm đối chiếu thật.
- **Metric không bao quát toàn bộ chất lượng.** Keyword recall có thể phạt từ đồng nghĩa hoặc thưởng một substring sai nghĩa. Tôi ghi hạn chế, giữ nguyên phép chấm, và phân tích output đầy đủ.

## 7. Kiểm tra và hồ sơ nộp

Đã nhập các file kết quả và `adapters/correct/` nguyên byte từ `submission/lab21_2A202602795_results.zip`; CRC và SHA của từng file được ghi trong [colab_import_manifest.json](../results/colab_import_manifest.json). Header tensor của adapter xác nhận **32,464,896 tham số**, khớp `runs.csv`. Dự đoán gốc đủ cho cả bốn adapter; contrast adapters được lưu khi chạy Colab nhưng gói tải xuống chỉ mang adapter correct theo hình thức ZIP gọn. NB6 merge/hot-swap không thực hiện và không xin điểm thưởng cho bước đó.

Đối chiếu source thí nghiệm với manifest Colab cho thấy source Python, tests và data liên quan khớp bản đã chạy. Report được viết lại sau khi có kết quả GPU; checksum report cũ trong manifest code mô tả bản được upload trước train, không phải report cuối. Các file `pre_gpu_check`, `verify_local`, `nb1_validation` trong results là bằng chứng checkpoint CPU trước Colab; kết quả thí nghiệm cuối lấy từ NB1–NB5 vừa nhập, `runs.csv`, baseline đóng băng và verdict.

Tái kiểm tra không cần GPU:

```bash
python -X utf8 scripts/review_colab_log.py
python -X utf8 scripts/audit_results.py
python -X utf8 scripts/verify.py
```

Lệnh cuối dùng verifier nguyên bản, tương đương target `make verify`; `-X utf8` chỉ cho phép console Windows in tiếng Việt và ký tự Δ. Không thay đổi logic chấm. Log kiểm tra cuối nằm trong [verify_final.log](../results/verify_final.log). Bài nộp giữ cả verdict FAIL và lý do; không thay metric để biến thành PASS.
