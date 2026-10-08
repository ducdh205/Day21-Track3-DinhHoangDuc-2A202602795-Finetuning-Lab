# Lab21 — tiến độ

- [x] Đọc hướng dẫn và kiểm tra hardware: Iris Xe, không CUDA; Colab yêu cầu login.
- [x] NB1 chạy lại thật và NB3/NB4 dùng chính recipe đã chứng minh.
- [x] Checkpoint CPU: 123 tests pass, 3 skip; labels corpus và 225 mẫu train pass.
- [x] NB2 full eval: b > a, đóng băng checksum và prompt SHA trước train.
- [x] NB3 correct: adapter lưu, loss/VRAM ghi runs.csv.
- [x] NB4: ba adapter cùng step; attention budget lệch dưới 5%.
- [x] Checkpoint GPU: bốn run hoàn tất với logs và artifacts.
- [x] NB5: đủ bốn nhóm, autopsy theo target, bảy ví dụ có hai ca thua regression.
- [x] Report khớp results, kết luận ≥150 từ, phản tư cụ thể.
- [x] Final verifier exit 0 và gói nộp đủ artifacts, ZIP CRC đạt.

Người dùng chạy Colab qua trình duyệt thường và cung cấp ZIP trong submission.
Đã nhập kết quả nguyên byte, chấm lại toàn bộ outputs, đối chiếu source/checksum.
Target: correct 0.970 > attn_only 0.965 > qlora 0.940 > wrong_lr 0.000.
Verdict model FAIL do regression giảm; giữ nguyên kết quả. Target không có ca thua,
report dùng hai ca thua regression và ghi rõ giới hạn keyword recall.
Verifier nguyên bản (UTF-8): 27 pass, 1 warning (verdict FAIL), 0 fail.
Tests: 123 passed, 3 skipped. Tests/eval/scorer/config/verifier khớp HEAD.
Gói cuối: submission/Lab21_2A202602795_SUBMISSION.zip (124541228 byte).
Report trong ZIP khớp bản đã cập nhật; verdict và kết quả gốc giữ nguyên.
