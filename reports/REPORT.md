# Báo cáo dự án — Chess AI (AlphaZero-lite)

Môn AIT2004 — Cơ sở AI. Bot cờ vua kết hợp mạng neural (policy + value), search cổ điển (Minimax/MCTS) và vòng lặp self-play kiểu AlphaZero thu nhỏ.

---

## 1. Tóm tắt

Dự án xây dựng một engine cờ vua học từ dữ liệu chuyên gia rồi tự cải thiện qua self-play. Pipeline gồm 5 khối: parse PGN → encode bàn cờ → mạng CNN hai đầu (policy + value) → search (Minimax alpha-beta hoặc MCTS) → vòng lặp self-play (sinh data → retrain → pit → giữ model mạnh hơn).

Kết quả chính: sau khi train supervised trên **1,77 triệu thế cờ** từ 20.000 ván Lichess Elite, model đạt **val top-1 = 33,5%** và **top-5 = 66,0%** trong việc dự đoán nước đi của chuyên gia — đúng mức kỳ vọng cho một mạng ~1,4M tham số. Pipeline self-play chạy trọn vẹn; lần chạy đầu gặp vấn đề "toàn hòa", đã được vá bằng adjudication theo material (kèm ELO tracking và batch inference). Sau khi vá, pit luôn phân thắng bại (0 ván hòa) nhưng self-play vẫn **chưa cải thiện** được model — với lượng self-play nhỏ (20 ván/vòng) và learning rate cao, việc retrain làm hỏng trọng số đã pretrain (phân tích ở Mục 6). Benchmark với Stockfish depth-2: bot đạt **3,0/10 (30%)** — hòa 6, thua 4, thắng 0 — ELO ước lượng thô ≈ 1353 (Mục 7).

---

## 2. Kiến trúc hệ thống

```
PGN games  ->  Encode 12x8x8  ->  CNN (Policy + Value heads)
                                         |
                              Minimax/Alpha-Beta  hoặc  MCTS (PUCT)
                                         |
                          Self-play -> Retrain -> Pit  (lặp)
```

**Encode bàn cờ.** Mỗi thế cờ → tensor `12×8×8` one-hot (6 loại quân × 2 màu). Khi đến lượt Đen, bàn cờ được xoay 180° và đảo màu để mạng luôn "nhìn" mình là Trắng — khai thác tính đối xứng, giảm nhu cầu dữ liệu.

**Action space.** 4096 = 64×64 (ô đi → ô đến). Đơn giản, phủ hầu hết nước đi; chấp nhận mất thông tin phong cấp dưới hậu (rất hiếm).

**Mạng PolicyValueNet.** Thân chung: 1 conv stem (12→128) + 3 residual block (128 kênh). Hai đầu ra:
- *Policy head*: conv 1×1 (2 kênh) → FC → 4096 logits.
- *Value head*: conv 1×1 (1 kênh) → FC 64 → FC 1 → tanh, cho giá trị [−1, 1].

Tổng tham số ≈ **1,43 triệu** (chạy được trên CPU). Đây là con số tính lại chính xác từ kiến trúc hiện tại (128 kênh / 3 block), nhỉnh hơn ước lượng "~1M" ghi trong README.

**Search.**
- *Minimax + Alpha-Beta* (negamax). Eval tĩnh = material + bảng vị trí quân (PST). Sắp xếp nước đi theo policy net để cắt nhánh tốt hơn. Không cần value head.
- *MCTS (PUCT)* kiểu AlphaZero. Mỗi leaf được mạng đánh giá (policy + value); value được lan ngược lên cây. Có Dirichlet noise tại root khi self-play. Hỗ trợ batch inference (`num_parallel`) để tăng tốc trên GPU.

**Opening book.** `src/search/opening_book.py` hardcode ~15 khai cuộc phổ biến (Ruy Lopez, Sicilian, QGD, King's Indian...). Khi bật (`play_cli.py --opening-book`), bot đi theo sách lúc đầu ván rồi mới chuyển sang search — giúp khai cuộc chắc tay hơn.

### 2.1. Cấu trúc mã nguồn (module-by-module)

| File | Vai trò |
|------|---------|
| `src/data/encode_board.py` | Board → tensor 12×8×8 one-hot + perspective flip cho Đen-to-move. |
| `src/data/action_space.py` | Move ↔ index (4096 = from×64+to); mask nước hợp lệ. |
| `src/data/pgn_parser.py` | Đọc PGN/.pgn.zst → mẫu `(fen, move, result)` lưu `.pt`. |
| `src/data/dataset.py` | PyTorch `Dataset` + DataLoader, split train/val 90/10. |
| `src/model/network.py` | `PolicyValueNet` (trunk + policy head + value head) và `PolicyNet`. |
| `src/search/evaluation.py` | Eval tĩnh material+PST; `adjudicate_result` (chấm material khi cắt ván). |
| `src/search/minimax.py` | Negamax + alpha-beta; move ordering bằng policy net. |
| `src/search/mcts.py` | MCTS PUCT; batch inference (`num_parallel`) + virtual loss. |
| `src/search/opening_book.py` | ~15 khai cuộc hardcode; `book_move(board)`. |
| `src/search/training/supervised.py` | Train supervised trên PGN (CE policy + MSE value). |
| `src/search/training/self_play.py` | Sinh self-play games bằng MCTS. |
| `src/search/training/pit.py` | Đấu 2 model, chấm thắng/hòa (có adjudication). |
| `src/search/training/self_play_loop.py` | Vòng lặp generate→retrain→pit→giữ; log ELO. |
| `src/search/training/elo.py` | Tính ELO tương đối từ kết quả pit. |
| `app.py` | Web app Flask (chơi với bot trong trình duyệt). |
| `scripts/*.py` | Entry-point: `parse_pgn`, `train_supervised`, `run_self_play`, `play_cli`, `benchmark_stockfish`, `plot_history`. |
| `tests/*.py` | Unit test (pytest): encode, action space, network, search, pipeline, adjudication+ELO, opening book. |

---

## 3. Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Lichess Elite 2024-01 (298k ván, rating ≥ ~2400) |
| Lọc | min_rating 1800, min_moves 10 |
| Số ván dùng | 20.000 |
| Số thế cờ (mẫu) | 1.768.985 |
| Cân bằng nhãn | Trắng thắng 37,5% · Đen thắng 37,1% · Hòa 25,4% |

Nhãn rất cân bằng → không lệch về một phía, tốt cho value head. Kiểm tra encode: mọi thế đầu ván có tổng = 32 quân, tensor chỉ chứa 0/1 (one-hot sạch); 100% chỉ số nước đi nằm trong [0, 4095].

---

## 4. Kết quả train supervised

Train 10 epoch, AdamW lr 1e-3 + cosine, batch 256, loss = CE(policy) + 1·MSE(value).

| Epoch | train top-1 | val top-1 | val top-5 | val policy loss | val value MAE |
|---|---|---|---|---|---|
| 1 | 20,2% | 24,9% | 54,4% | 3,169 | 0,750 |
| 3 | 30,7% | 30,0% | 61,9% | 2,799 | 0,720 |
| 5 | 35,0% | 32,2% | 64,8% | 2,673 | 0,692 |
| 8 | 40,4% | 33,3% | 66,1% | 2,634 | 0,642 |
| 10 | 42,4% | 33,5% | 66,0% | 2,642 | 0,633 |

**Nhận xét.** Policy học tốt: top-1 33,5% nghĩa là cứ 3 thế cờ thì model đoán đúng 1 nước y hệt chuyên gia; top-5 66% nghĩa là nước đúng nằm trong 5 gợi ý đầu 2/3 số lần. Val loss chững lại từ epoch 8 trong khi train top-1 vẫn tăng (42,4%) → bắt đầu overfit nhẹ; epoch 9 là điểm tốt nhất và đã được lưu tự động vào `best.pt`.

**Ảnh hưởng kích thước dữ liệu:**

| | 2.000 ván (5 epoch) | 20.000 ván (10 epoch) |
|---|---|---|
| Số thế cờ | 178.362 | 1.768.985 |
| val top-1 | 20,8% | **33,5%** |
| val top-5 | 44,4% | **66,0%** |

Tăng dữ liệu ~10× đẩy top-1 từ 20,8% lên 33,5% và lùi điểm overfit ra xa hơn — minh chứng rõ ràng dữ liệu là yếu tố quyết định.

**Điểm yếu: value head.** Val value MAE chỉ giảm tới ~0,63 (trên thang [−1, 1]). Dự đoán kết quả ván từ một thế cờ đơn lẻ vốn khó; đây là phần yếu nhất của model và là lý do MCTS (vốn dựa vào value) chưa mạnh vượt trội so với Minimax.

---

## 5. Minimax vs MCTS

| | Minimax + Alpha-Beta | MCTS (PUCT) |
|---|---|---|
| Dùng value head? | Không (eval tĩnh material+PST) | Có (mạng đánh giá leaf) |
| Quyết định | Duyệt tới depth cố định | Phân bổ simulation cho nhánh hứa hẹn |
| Tốc độ | Nhanh, không cần GPU | Chậm hơn (mỗi sim 1 lần gọi mạng) |
| Phụ thuộc chất lượng mạng | Chỉ policy (để sắp xếp) | Cả policy lẫn value |
| Kiểm chứng | Tìm đúng chiếu hết 1 nước (test) | Chạy hợp lệ trong self-play/pit |

Thực tế ở quy mô dự án: Minimax depth 3 cho nước chơi ổn định, độc lập với chất lượng value head. MCTS chỉ thực sự vượt trội khi value head đủ tốt — điều mà mức MAE 0,63 hiện tại chưa đảm bảo. Khuyến nghị demo bằng Minimax, còn MCTS dùng để chạy self-play.

---

## 6. Self-play loop — vấn đề, cách khắc phục, và kết quả

### 6.1. Vấn đề ở lần chạy đầu

Lần chạy đầu (5 iteration) chạy đúng kỹ thuật nhưng **gần như mọi ván đều hòa** (10/10 hòa ở 4/5 vòng) → pit không phân biệt được model mới/cũ, quyết định accept/reject gần như ngẫu nhiên. Nguyên nhân: ván chạm giới hạn `max_moves = 200` mà chưa chiếu hết bị mặc định tính **hòa**; với model chơi tất định, ván dễ kéo dài rồi bị cắt → toàn hòa.

### 6.2. Các bản vá đã triển khai

1. **Adjudication theo material** (`evaluation.adjudicate_result`). Khi ván chạm `max_moves`, thay vì mặc định hòa, chấm theo material+PST: bên hơn rõ (|score| > margin, mặc định 100cp ≈ 1 tốt) được tính thắng. Hòa tự nhiên (stalemate, lặp 3 lần, thiếu quân, 50 nước) vẫn là hòa. Áp dụng cho cả `self_play.py` (nhãn value) và `pit.py` (kết quả pit).

2. **ELO tracking** (`elo.py`). Mỗi vòng pit suy ra chênh lệch ELO của candidate từ tỉ số thắng, cộng dồn (chỉ khi accept) → đường ELO tương đối qua các iteration (model ban đầu = 0). Lưu vào mỗi `iter_*.pt`, vẽ bằng `plot_history.py`. *Đây là ELO tương đối so với chính mình, không phải so với Stockfish.*

3. **Batch inference cho MCTS** (`mcts.py`, tham số `num_parallel`). Gom nhiều leaf rồi gọi mạng một lần (virtual loss để các leaf không trùng nhánh) → nhanh hơn nhiều trên GPU. Mặc định `num_parallel=1` giữ nguyên hành vi cũ.

Adjudication và ELO đã được **kiểm thử đơn vị** (`tests/test_adjudication_elo.py`, 12/12 pass) bằng python-chess thuần.

### 6.3. Kết quả sau khi vá (chạy lại 5 iteration)

```
iter1: candidate 0-10 current, 0 hòa -> reject   (ELO kept 0)
iter2: candidate 0-10 current, 0 hòa -> reject   (ELO kept 0)
iter3: candidate 0-10 current, 0 hòa -> reject   (ELO kept 0)
iter4: candidate 5-5  current, 0 hòa -> reject   (ELO kept 0)
iter5: candidate 5-5  current, 0 hòa -> reject   (ELO kept 0)
```

**Hai kết luận:**

1. **Bản vá adjudication hoạt động** — **0 ván hòa** ở mọi vòng (trước là 10/10 hòa). Pit giờ luôn phân thắng bại; ELO tracking chạy đúng (giữ ở 0 vì không candidate nào vượt ngưỡng).

2. **Self-play KHÔNG cải thiện được model.** Candidate hoặc tệ hơn hẳn (0-10 ở vòng 1-3) hoặc ngang ngửa (5-5 ở vòng 4-5, kết quả do bên đi trước quyết định chứ không phải do trình độ), không bao giờ đạt ≥55% → không vòng nào được accept, ELO phẳng ở 0.

**Vì sao self-play làm model tệ đi?** `current` được train supervised trên **1,77 triệu** thế cờ chuyên gia. Mỗi vòng lại deepcopy nó rồi train 2 epoch trên chỉ **~2.400** mẫu self-play ở **lr 1e-3** — lượng dữ liệu nhỏ và nhiễu này (sinh bởi MCTS yếu, value head MAE 0,63) ở learning rate cao **ghi đè** lên trọng số tốt, khiến candidate đi nước kém và thua adjudication. Đây là kết quả hợp lý và là bài học quan trọng: **20 ván × 100 sim mỗi vòng là quá ít để cải thiện một model đã được pretrain mạnh** — muốn ELO leo thật cần lr thấp hơn (~1e-4), replay buffer tích lũy nhiều vòng, nhiều ván và nhiều simulation hơn (nhưng rất chậm trên CPU).

Biểu đồ kết quả: `reports/supervised.png` (đường cong train) và `reports/selfplay.png` (win-rate + ELO theo iteration).

---

## 7. Benchmark vs Stockfish

Đấu 10 ván với Stockfish giới hạn **depth-2** (yếu hết mức), bot dùng MCTS 200 simulation trên `best.pt` (model supervised). Mỗi bên cầm Trắng/Đen phân nửa số ván. Thời gian ~258s, trung bình ~95 nước/ván.

| Kết quả | Số ván |
|---|---|
| Bot thắng | 0 |
| Hòa | 6 |
| Stockfish thắng | 4 |
| **Điểm bot** | **3,0 / 10 (30%)** |

ELO ước lượng (thô): Stockfish depth-2 ≈ 1500 → **bot ≈ 1353**. Đây là con số rất thô (ánh xạ depth→ELO không chuẩn), chỉ tham khảo tương đối.

**Đọc kết quả.** Bot hòa 6/10 nhưng không thắng ván nào — và nhiều ván hòa rất dài (79–127 nước, một ván chạm 200 nước → cắt). Nghĩa là bot chơi đủ chắc để không thua nhanh, nhưng **không biết chuyển hóa lợi thế thành thắng** — đúng với phát hiện ở Mục 4: value head yếu (MAE 0,63) nên không định hướng được tàn cuộc. Với một model chỉ học supervised (chưa có self-play cải thiện), hòa được Stockfish dù ở depth thấp là kết quả chấp nhận được.

---

## 8. Bài học rút ra

- **Dữ liệu quan trọng hơn epoch.** Tăng từ 2k lên 20k ván cải thiện accuracy nhiều hơn hẳn việc train thêm epoch; train lâu trên data nhỏ chỉ dẫn tới overfit.
- **Value head là nút thắt.** Học policy (bắt chước nước đi) dễ hơn nhiều so với học value (dự đoán kết quả). Chất lượng MCTS bị giới hạn bởi value head.
- **Self-play cần kết quả "phân thắng bại".** Nếu phần lớn ván hòa, vòng lặp không có gradient cải thiện. Adjudication theo material là cách rẻ để tạo tín hiệu.
- **Tốc độ MCTS là rào cản thực tế.** MCTS thuần Python, mỗi simulation một lần gọi mạng → rất chậm; batch inference là tối ưu cần thiết để self-play khả thi ở quy mô lớn.

---

## 9. Hạn chế & hướng phát triển

**Hạn chế.** Value head còn yếu (MAE ~0,63) → bot không chuyển hóa được lợi thế (thắng 0 ván vs Stockfish, chỉ hòa); self-play (sau khi vá) đã chạy phân thắng bại nhưng **chưa cải thiện** model do dữ liệu self-play quá nhỏ + lr quá cao làm hỏng trọng số pretrain; action space mất thông tin phong cấp dưới hậu; input 12 kênh bỏ qua trạng thái nhập thành/bắt tốt qua đường/lặp nước.

**Hướng phát triển.** (1) Để self-play thực sự cải thiện: hạ learning rate retrain xuống ~1e-4, thêm replay buffer tích lũy nhiều vòng, tăng số ván/simulation mỗi vòng; (2) tăng dữ liệu/độ sâu mạng để nâng value head — đây là chìa khóa để bot biết thắng chứ không chỉ hòa; (3) bật batch inference (`--num-parallel`) cho self-play nhanh hơn trên GPU; (4) đấu Stockfish ở depth cao hơn / nhiều ván hơn để có ELO ổn định hơn.

---

## 10. Phụ lục — cách tái lập

```bash
# 1. Parse dữ liệu
python scripts/parse_pgn.py --input data/raw/lichess_elite_2024-01.pgn \
    --output data/processed/train.pt --max-games 20000 --min-rating 1800 --min-moves 10

# 2. Train supervised
python scripts/train_supervised.py --data data/processed/train.pt \
    --epochs 10 --batch-size 256 --channels 128 --n-res 3 --output-dir models

# 3. Self-play (đã vá: adjudication + ELO; bật batch inference trên GPU)
python scripts/run_self_play.py --initial-model models/best.pt \
    --iterations 5 --games 20 --simulations 100 --train-epochs 2 \
    --pit-games 10 --adjudication-margin 100 --num-parallel 8 \
    --output-dir models/selfplay

# 4. Vẽ biểu đồ (gồm đường ELO)
python scripts/plot_history.py --ckpt models/best.pt --output reports/supervised.png
python scripts/plot_history.py --selfplay-dir models/selfplay --output reports/selfplay.png

# 5. Benchmark vs Stockfish (cần binary Stockfish)
python scripts/benchmark_stockfish.py --model models/best.pt --stockfish stockfish.exe \
    --games 10 --stockfish-depth 2 --search mcts --simulations 200
```

### Hai cách demo chơi với bot

**Web app (Flask):**

```bash
pip install flask
python app.py --model models/best.pt          # rồi mở http://localhost:5000
```
Options: `--search {minimax,mcts}`, `--depth`, `--simulations`, `--no-book`, `--host`, `--port`.
Không có model / không có torch → tự fallback minimax + eval tĩnh.

**Terminal (CLI):**

```bash
python scripts/play_cli.py --model models/best.pt --search minimax --depth 3
python scripts/play_cli.py --model models/best.pt --color black --opening-book
python scripts/play_cli.py --model models/best.pt --search mcts --simulations 200
```
Trong game nhập nước đi UCI (`e2e4`, `g1f3`, `e7e8q`); lệnh `undo` / `fen` / `quit`.
