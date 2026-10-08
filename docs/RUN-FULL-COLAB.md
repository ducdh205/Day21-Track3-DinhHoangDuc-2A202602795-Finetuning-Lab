# Chạy đầy đủ NB1–NB5 bằng code local

Máy đã kiểm tra có Intel Iris Xe, không có GPU NVIDIA CUDA. NB1 và tests chạy local;
phần train dùng T4 trên Colab. Google đã từ chối đăng nhập từ trình duyệt MCP tự động.
Dùng Chrome/Edge thường đã đăng nhập Google để chạy notebook; không gửi mật khẩu vào chat.

1. Mở https://colab.research.google.com bằng trình duyệt thường.
2. **File → Upload notebook**: chọn `colab/Lab21_RUN_ALL.ipynb` từ repo local.
3. **Runtime → Change runtime type → T4 GPU**.
4. Chạy ô 1, chọn `submission/lab21_colab_code.zip` khi Colab yêu cầu upload.
5. Chạy ô 2, 3, 4 lần lượt. Giữ `STAGES=nb1 nb2 nb3 nb4 nb5`, `ONLY` rỗng.
6. Ô 4 tải `lab21_2A202602795_results.zip`; giữ ZIP này để cập nhật report bằng kết quả thật.

Gói code local chứa cả các thay đổi chưa push. Nó không chứa môi trường .venv,
cache, dữ liệu đăng nhập hay kết quả GPU giả. Manifest SHA-256 trong ZIP kiểm tra
từng file, gồm tests nguyên bản và eval. Không chạy notebook trên GitHub cũ vì
setup cũ clone repo upstream và đặt mặc định EVAL_LIMIT=8.

Mặc định của notebook mới: full eval, EPOCHS=2, base `unsloth/Qwen3.5-4B`.
NB3/NB4 lấy max_length và thinking từ NB1; kiểm tra dataset và collator của TRL
trước train. NB2 phải đóng băng b > a trước khi NB3/NB4 được phép tải model.
NB5 lưu cả dự đoán đầy đủ và thắng/thua trực tiếp so với baseline (b).

Nếu lỗi, vẫn chạy ô 4 để lấy log và artefact đã hoàn thành. Khi runtime còn dữ liệu:

- NB4 dừng ở qlora: ô 3 đặt `STAGES=nb4 nb5`, `ONLY=qlora`.
- NB2 đã đóng băng, train chưa bắt đầu: `STAGES=nb3 nb4 nb5`, `ONLY` rỗng.
- NB5 lỗi: `STAGES=nb5`, `ONLY` rỗng.

Không chạy lại NB2 nếu đã có baseline đóng băng; script từ chối ghi đè có chủ ý.
Runtime bị xoá thì gói kết quả cuối chỉ chứa adapter correct; cần chạy lại những
adapter đối chứng bị mất. Đừng đánh dấu thí nghiệm hoàn thành chỉ vì tests pass.

`scripts/verify.py` nguyên bản chạy trong ô 4, tương đương `make verify`. Report
hiện ghi kết quả local và điểm chặn GPU. Sau khi có ZIP, cập nhật phần NB2–NB5,
phán quyết và ít nhất năm ví dụ (hai ca thua có thật), rồi chạy verifier lại.

Sinh lại gói sau mỗi thay đổi code/report: `python scripts/build_full_runner.py`.
