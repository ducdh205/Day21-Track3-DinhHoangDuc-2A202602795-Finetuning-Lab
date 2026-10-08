# Lab21: thí nghiệm thật và report

Yêu cầu đã được người dùng giao: đọc hướng dẫn, chạy thật NB1–NB5 với mặc định,
không sửa tests/eval, viết report từ results. Dùng model và corpus hiện tại.

1. Kiểm tra phần cứng và truy cập Colab; xác nhận CUDA hoặc ghi rõ điểm chặn.
2. Chứng minh lại NB1 trên tokenizer thật. Đồng bộ NB3/NB4 với p95 và thinking của NB1.
3. Đo và đóng băng NB2 trước train; chỉ tiếp tục khi b > a, checksum/SHA đúng.
4. Train bốn run cùng step, khớp ngân sách attention; lưu adapter và log ngay.
5. Đánh giá NB5, xếp theo target, lấy ít nhất năm ví dụ và hai ca thua có thật.
6. Viết report đối chiếu results, chạy verifier nguyên bản, đóng gói khi đủ gate.

Phụ thuộc: 1 → 2 → 3 → 4 → 5 → 6. NB1/tests có thể hoàn thành khi chờ đăng nhập.
Rủi ro: local chỉ có Iris Xe; Colab cần người dùng đăng nhập và được cấp T4.
Không đưa số đo của instructor/simulation vào report như kết quả cá nhân.
Không tạo verdict hay metrics cho bước chưa chạy. Không publish/push.

Verification: tokenizer thật + kiểm tra labels từng mẫu, tests nguyên bản,
python scripts/verify.py (tương đương target make verify trên Windows thiếu make).
