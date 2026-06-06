# Chess AI — Checklist trạng thái (24 task)

Cập nhật cuối: đã chạy thật (parse 20k → train 10 epoch → self-play 5 vòng post-fix) **và** hoàn thiện code (adjudication + ELO + batch inference + opening book + plots + báo cáo).

Ký hiệu: `[x]` = xong + verify · `[~]` = code xong, cần môi trường ngoài để chạy · `[ ]` = chưa làm

---

## Tuần 1 — Nền tảng & Môi trường  (8/8)

- [x] 1. Cài đặt môi trường
- [x] 2. Hiểu API python-chess
- [x] 3. Thiết kế hàm encode board (12×8×8 + flip) — có test
- [x] 4. Download & parse PGN — 1.768.985 thế cờ
- [x] 5. Dataset & DataLoader (split 90/10)
- [x] 6. Action space 4096 — có test
- [x] 7. Model v1 (PolicyValueNet ≈ 1,43M params)
- [x] 8. Training loop & eval — 10 epoch, val top-1 33,5% / top-5 66%

## Tuần 2 — Self-Play & MCTS  (8/8)

- [x] 9. Minimax + Alpha-Beta (verify mate-in-1)
- [x] 10. Policy Net move ordering
- [x] 11. MCTS PUCT — có batch inference (`num_parallel`)
- [x] 12. Value Head (đã train; MAE ≈ 0,63)
- [x] 13. Self-play data generation
- [x] 14. Training từ self-play data
- [x] 15. Pit model mới vs cũ (ngưỡng 55%)
- [x] 16. Debug & ổn định pipeline + Log ELO estimate (`elo.py`, có test)

## Tuần 3 — Cải thiện & Demo  (8/8)

- [x] **17. Chạy nhiều vòng self-play** — đã chạy 5 vòng post-fix: **0 ván hòa** (adjudication OK), ELO tracking chạy đúng. Kết luận: self-play chưa cải thiện model (phân tích ở REPORT mục 6).
- [x] **18. Tối ưu tốc độ inference** — batch inference cho MCTS (`--num-parallel`). (ONNX optional, bỏ.)
- [x] **19. Opening book** — `src/search/opening_book.py` (~15 khai cuộc) + `play_cli --opening-book`, có test 4/4 pass.
- [x] **20. Visualize training curves** — đã xuất `reports/supervised.png` + `reports/selfplay.png` (gồm đường ELO).
- [x] 21. Chơi thử với người (CLI) — `scripts/play_cli.py` (UCI, undo/fen/quit, minimax|mcts, opening book).
- [x] **22. Benchmark Stockfish** — đã chạy 10 ván vs Stockfish depth-2: bot **3,0/10 (30%)** — hòa 6, thua 4, thắng 0; ELO ước lượng thô ≈ 1353.
- [x] **23. Báo cáo** — `reports/REPORT.md` (thay slide): kiến trúc, kết quả, Minimax-vs-MCTS, self-play, bài học, hạn chế.
- [x] 24. Demo live — **2 cách**: web app Flask (`app.py`, chơi trong trình duyệt, đã test endpoint) + CLI (`play_cli.py`).

---

## Tiến độ tổng

| Trạng thái | Số task |
|---|---|
| `[x]` Xong + verify | **24** |
| `[~]` Còn dang dở | 0 |
| `[ ]` Chưa làm | 0 |

**Hoàn thành 24/24.** 🎉

## Kiểm thử (không cần torch — chạy được tại chỗ)

`tests/test_adjudication_elo.py` (12) + `tests/test_opening_book.py` (4) + `tests/test_action_space.py` (5) = **21/21 pass**.
Các test cần torch (`test_network`, `test_pipeline`, `test_search`, `test_encode_board`) chạy trên máy bạn.

## Kết quả benchmark cuối (task 22)

Bot (MCTS 200 sim, `best.pt`) vs Stockfish depth-2, 10 ván: **3,0/10 (30%)** — thắng 0, hòa 6, thua 4. ELO thô ≈ 1353. Bot chơi chắc (hòa nhiều) nhưng chưa biết thắng — value head là điểm cần cải thiện tiếp theo.
