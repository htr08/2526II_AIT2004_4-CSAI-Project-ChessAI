# Chess AI 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

>**AIT2004#_4 — Cơ sở Trí tuệ nhân tạo | Nhóm 9 | Học kỳ 2526II**

Dự án xây dựng một tác nhân chơi cờ vua hoàn chỉnh theo phong cách AlphaZero: từ học có giám sát trên dữ liệu kỳ thủ người, đến tự chơi (self-play) với Monte Carlo Tree Search hướng dẫn bởi mạng nơ-ron chính sách, kèm giao diện web tương tác xây dựng bằng Flask.

---

## Mục lục

1. [Mục tiêu và thách thức](#1-mục-tiêu-và-thách-thức)
2. [Kiến trúc tổng quan](#2-kiến-trúc-tổng-quan)
3. [Biểu diễn trạng thái và không gian hành động](#3-biểu-diễn-trạng-thái-và-không-gian-hành-động)
4. [Mạng nơ-ron](#4-mạng-nơ-ron)
5. [Thuật toán tìm kiếm](#5-thuật-toán-tìm-kiếm)
6. [Vòng lặp tự chơi](#6-vòng-lặp-tự-chơi)
7. [Cài đặt môi trường](#7-cài-đặt-môi-trường)
8. [Sử dụng nhanh](#8-sử-dụng-nhanh)
9. [Pipeline huấn luyện](#9-pipeline-huấn-luyện)
10. [Giao diện Web (Flask)](#10-giao-diện-web-flask)
11. [Đánh giá vị trí (FEN Eval)](#11-đánh-giá-vị-trí-fen-eval)
12. [Kết quả thực nghiệm](#12-kết-quả-thực-nghiệm)
13. [Cấu trúc dự án](#13-cấu-trúc-dự-án)
14. [Kiểm thử](#14-kiểm-thử)
15. [Kết luận](#15-kết-luận)
16. [Tài liệu tham khảo](#16-tài-liệu-tham-khảo)

---

## 1. Mục tiêu và thách thức

### 1.1 Mục tiêu

- Xây dựng **PolicyNet** học bắt chước nước đi người (học có giám sát, Lichess ≥ 1800 ELO).
- Xây dựng **DualNet** (hai đầu policy + value) và cải tiến bằng vòng lặp tự chơi AlphaZero.
- Cài đặt **PUCT-MCTS** để policy head thực sự hướng dẫn tìm kiếm (thay vì UCB1 bỏ qua policy).
- Triển khai **giao diện web** (Flask + chessboard.js) để chơi trực tiếp trên trình duyệt.
- Đánh giá khách quan thông qua đấu với Stockfish, ước tính ELO, và FEN evaluation.

### 1.2 Thách thức kỹ thuật

| Vấn đề | Giải pháp |
|---|---|
| Không gian trạng thái ~$10^{43}$ vị trí | Mạng nơ-ron đánh giá vị trí thay vì vét cạn |
| Không gian hành động 4,544 nước | Từ điển UCI cố định, policy head phủ toàn bộ |
| MCTS cũ (UCB1) bỏ qua policy head | Thay bằng PUCT: prior từ policy net hướng dẫn exploration |
| Self-play collapse (mọi ván giống nhau) | Dirichlet noise tại root + temperature sampling |
| Value head yếu ở tàn cuộc | Fine-tuning riêng value head trên dữ liệu endgame |
| Chi phí tính toán lớn | Batched MCTS (gom leaf nodes, 1 forward pass / batch) |

### 1.3 Vấn đề cốt lõi đã giải quyết

**Vấn đề gốc:** DualNet có policy head được huấn luyện, nhưng vòng MCTS cũ (UCB1 + select/expand từng nước một) chỉ dùng value head để đánh giá leaf — policy net hoàn toàn không tham gia hướng dẫn tìm kiếm. Self-play chọn nước bằng `argmax(visits)` thuần túy khiến mọi ván cờ gần như giống hệt nhau và mô hình suy sụp (collapse) từ iteration 4–5.

**Giải pháp:** Thay toàn bộ selection formula bằng **PUCT**, mở rộng bằng `expand_with_policy()` (gán prior từ policy softmax, không random rollout), thêm **Dirichlet noise** tại root và **temperature sampling** cho diversity.

---

## 2. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE ALPHAZERO                         │
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │   Self-Play    │──▶│  Huấn luyện    │──▶│    Arena       │  │
│  │  PUCT-MCTS     │   │   DualNet      │   │  (Pit test)    │  │
│  │  + DualNet     │   │  L = Lp + Lv   │   │  win > 55%  →  │  │
│  │  + Noise       │   │                │   │  promote       │  │
│  └────────────────┘   └────────────────┘   └────────────────┘  │
│          ▲                                          │            │
│          └──────────────────────────────────────────┘            │
│                       Lặp K iterations                           │
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐                        │
│  │ Fine-tune      │   │  Flask Web UI  │                        │
│  │ value head     │   │  (chơi thực)   │                        │
│  │ (endgame data) │   │                │                        │
│  └────────────────┘   └────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

**Tác nhân hiện có:**

| Tác nhân | Tìm kiếm | Đánh giá vị trí |
|---|---|---|
| `MinimaxAgent` | Alpha-beta minimax | Material + Piece-Square Tables (PST) |
| `DualNetMinimaxAgent` | Alpha-beta minimax | Value head DualNet (học được) |
| PUCT-MCTS Agent | Cây PUCT với DualNet | Policy prior + value head |

---

## 3. Biểu diễn trạng thái và không gian hành động

### 3.1 Mã hóa bàn cờ — tensor `(17, 8, 8)`

| Kênh | Nội dung |
|---|---|
| 0–5 | Quân Trắng: Tốt, Mã, Tượng, Xe, Hậu, Vua |
| 6–11 | Quân Đen: Tốt, Mã, Tượng, Xe, Hậu, Vua |
| 12 | Lượt đi (1.0 = Trắng, 0.0 = Đen) |
| 13 | Trắng nhập thành cánh vua |
| 14 | Trắng nhập thành cánh hậu |
| 15 | Đen nhập thành cánh vua |
| 16 | Đen nhập thành cánh hậu |

Mỗi kênh piece là mảng 8×8 nhị phân: `1.0` nếu có quân tại ô đó, `0.0` nếu không.

### 3.2 Không gian hành động — 4,544 nước đi

Từ điển cố định UCI, lưu tại `data/processed/move2idx.json`:

| Loại | Số lượng | Công thức |
|---|---|---|
| Nước thường | 4,032 | $64 \times 63$ (bỏ tự di chuyển đến chính ô đó) |
| Nước phong cấp | 512 | $4 \text{ quân} \times 2 \text{ hàng} \times 64 \text{ ô}$ |
| **Tổng** | **4,544** | |

---

## 4. Mạng nơ-ron

### 4.1 PolicyNet — Baseline học có giám sát

```
Input (B, 17, 8, 8)
  → Conv2d(17→64,  3×3, pad=1) → BN → ReLU  ×4
  → Flatten
  → Linear(8192 → 1024) → ReLU → Dropout(0.3)
  → Linear(1024 → 4544)
Output: (B, 4544) logits
```

### 4.2 DualNet — AlphaZero style (~1.2M tham số)

```
Input (B, 17, 8, 8)
  → Conv2d(17→128, 3×3, pad=1) → BN → ReLU          [stem]
  → ResBlock × 4                                      [backbone]
       ResBlock: Conv→BN→ReLU→Conv→BN + skip→ReLU

      ├── Policy Head
      │     Conv2d(128→2, 1×1) → BN → ReLU → Flatten
      │     Linear(128 → 4544)
      │     Output: (B, 4544) policy logits
      │
      └── Value Head
            Conv2d(128→1, 1×1) → BN → ReLU → Flatten
            Linear(64 → 64) → ReLU → Linear(64 → 1) → Tanh
            Output: (B, 1) ∈ [-1, 1]
```

### 4.3 Hàm mất mát

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$$

$$\mathcal{L}_{\text{policy}} = -\sum_{a} \pi_{\text{MCTS}}(a \mid s)\, \log P_\theta(a \mid s)$$

$$\mathcal{L}_{\text{value}} = \bigl(z - v_\theta(s)\bigr)^2$$

Trong đó $\pi_{\text{MCTS}}(a \mid s)$ là phân phối visit counts chuẩn hóa (soft target), $z \in \{-1, 0, +1\}$ là kết quả trận, $v_\theta(s) \in [-1,1]$ là đầu ra value head.

### 4.4 Tóm tắt hàm mất mát theo giai đoạn

| Giai đoạn | Mục tiêu | Loss Policy | Loss Value | API PyTorch |
|---|---|---|---|---|
| **SL — PolicyNet** | Bắt chước nước đi người | $\mathcal{L}_p = \text{CE}(P_\theta, y_{\text{hard}})$ | — | `F.cross_entropy` |
| **SL — DualNet** | Bắt chước + đánh giá outcome | $\mathcal{L}_p = \text{CE}(P_\theta, y_{\text{hard}})$ | $\mathcal{L}_v = (z - v_\theta)^2$ | `F.cross_entropy` + `F.mse_loss` |
| **RL — Self-play** | Cải thiện qua tự chơi | $\mathcal{L}_p = -\pi_{\text{MCTS}} \cdot \log P_\theta$ | $\mathcal{L}_v = (z - v_\theta)^2$ | `F.cross_entropy(soft)` + `F.mse_loss` |
| **Fine-tune value** | Tinh chỉnh value ở tàn cuộc | — (frozen) | $\mathcal{L}_v = (z - v_\theta)^2$ | `F.mse_loss` (value head only) |

---

## 5. Thuật toán tìm kiếm

### 5.1 UCB1-MCTS (baseline)

Vòng lặp 4 bước: **Select → Expand → Simulate → Backpropagate**.

$$\text{UCB1}(v) = -\frac{W_v}{N_v} + c \cdot \sqrt{\frac{\ln N_{\text{parent}}}{N_v}}$$

Dấu âm ở số hạng khai thác là do **negamax**: giá trị tốt cho con (đối thủ) đồng nghĩa xấu cho cha, nên cha chọn con có $-W/N$ lớn nhất.

**Giới hạn:** Selection không dùng policy prior → policy head bị lãng phí hoàn toàn.

### 5.2 PUCT-MCTS — AlphaZero (phiên bản hiện tại)

$$\text{PUCT}(s, a) = Q(s, a) + U(s, a)$$

$$Q(s, a) = -\frac{W(s,a)}{N(s,a)}$$

$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

Trong đó:
- $P(s, a)$ — prior từ policy head, chuẩn hóa trên nước hợp lệ: $P(s,a) = \dfrac{p_a}{\sum_{a'} p_{a'}}$
- $N(s) = \sum_{a} N(s, a)$ — tổng số lần thăm nút cha
- $c_{\text{puct}} = 1.5$ — hệ số cân bằng exploration / exploitation

Khi $N(s,a) = 0$: $Q = 0$, $U > 0$ nhờ prior $P$ → policy net hướng dẫn expansion ngay lập tức.

**So sánh UCB1 vs PUCT:**

| | UCB1-MCTS | PUCT-MCTS |
|---|---|---|
| Selection | Chỉ dựa vào $Q$ và $N$ | $Q$ + prior $P$ từ policy net |
| Expansion | Mở rộng từng nước một | Mở rộng tất cả nước hợp lệ cùng lúc |
| Simulation | Value head (1 forward pass) | Value head (trong `expand_with_policy`) |
| Node chưa thăm | UCB1 = $\infty$ | $U > 0$ nhờ prior → policy hướng dẫn |

### 5.3 Dirichlet noise tại root (self-play)

$$\tilde{P}(s, a) = (1 - \varepsilon)\,P(s, a) + \varepsilon\,\eta_a, \qquad \eta \sim \text{Dir}(\alpha)$$

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| $c_{\text{puct}}$ | 1.5 | Cân bằng exploration / exploitation |
| $n_{\text{sims}}$ | 50–200 | Số simulation mỗi nước đi |
| $\alpha$ | 0.3 | Độ tập trung của Dirichlet (nhỏ = phân tán hơn) |
| $\varepsilon$ | 0.25 | Tỉ lệ noise trộn vào prior gốc |

### 5.4 Temperature sampling

$$\pi(a \mid s) = \frac{N(s,a)^{1/T}}{\displaystyle\sum_{a'} N(s,a')^{1/T}}$$

- $t < 30$: lấy mẫu $a \sim \pi(\cdot)$ với $T = 1.0$ — khám phá đa dạng khai cuộc.
- $t \geq 30$: $a = \arg\max_a N(s, a)$ — khai thác, tương đương $T \to 0$.

### 5.5 Batched MCTS

Gom `batch_size` leaf nodes, thực hiện một forward pass GPU duy nhất:

```
n_rounds = n_sims // batch_size
Mỗi round:
  - Thu thập batch_size leaf qua select + expand
  - Stack tensors → 1 GPU forward pass → (policies, values)
  - Backpropagate từng value
Tăng tốc: 5–20× so với evaluation tuần tự
```

### 5.6 Alpha-Beta Minimax

```
minimax(node, depth, α, β, maximizing):
  if depth == 0 or terminal:
      return evaluate(node)          # PST hoặc value head
  if maximizing:
      v = -∞
      for move in order_moves(node): # captures > promotions > quiet
          v = max(v, minimax(child, depth-1, α, β, False))
          α = max(α, v)
          if α ≥ β: break            # beta cutoff
      return v
```

`DualNetMinimaxAgent` dùng value head thay PST để đánh giá leaf, và policy head để sắp xếp nước đi (move ordering).

---

## 6. Vòng lặp tự chơi

### 6.1 Sinh dữ liệu một ván

```python
while not board.is_game_over():
    # 1. Tra opening book (nếu có) → bỏ qua MCTS
    # 2. PUCT-MCTS với Dirichlet noise tại root
    root = run_puct_search(board, model, move2idx, n_sims=50, add_noise=True)

    # 3. Policy target = phân phối visit counts chuẩn hóa
    # π(a|s) = N(s,a) / Σ N(s,a')
    policy[idx] = child.visits / total_visits

    # 4. Chọn nước theo temperature
    move = select_move_with_temperature(root, temperature=1.0, move_count=t)
    board.push(move)

# 5. Gán value target hồi tố (retroactive assignment)
```

Policy target tại mỗi vị trí:

$$\pi(a \mid s) = \frac{N(s, a)}{\displaystyle\sum_{a'} N(s, a')}$$

Sau khi ván kết thúc, với $z \in \{+1, -1, 0\}$ là kết quả cuối (góc nhìn Trắng):

$$z_t = z \cdot (-1)^{t}, \quad t = 0, 1, 2, \ldots$$

Nước lượt chẵn (Trắng đi) nhận $z_t = z$; lượt lẻ (Đen đi) nhận $z_t = -z$.

### 6.2 Replay buffer

- Giữ dữ liệu của $W$ iteration gần nhất (sliding window)
- Trộn 50% supervised + 50% self-play trong mỗi batch
- Tránh catastrophic forgetting

### 6.3 Arena (Pit)

Chơi $N_{\text{games}}$ ván (luân phiên màu quân):

$$\text{win\_rate} = \frac{W + 0.5 \cdot D}{N_{\text{games}}}$$

Nếu $\text{win\_rate} > 0.55$ thì candidate được promote thành `best_dual.pt`.

---

## 7. Cài đặt môi trường

### 7.1 Yêu cầu

- Python 3.10+
- GPU CUDA khuyến nghị cho self-play (CPU vẫn hoạt động, chậm hơn)
- Stockfish binary tại `bin/stockfish.exe` (Windows) hoặc `bin/stockfish` (Linux)

### 7.2 Tạo và kích hoạt môi trường ảo

**Linux / macOS / WSL:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Nếu PowerShell báo lỗi execution policy, chạy trước: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 7.3 Kiểm tra cài đặt

```bash
python -c "import torch, chess, flask; print('OK')"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 8. Sử dụng nhanh

### Chơi qua terminal (ASCII board)

```bash
# Người (Trắng) vs AI (Đen), minimax depth 4
python play.py

# Người (Đen) vs AI (Trắng)
python play.py --side black

# AI tự đánh nhau
python play.py --ai-vs-ai

# Chỉ định checkpoint và độ sâu tìm kiếm
python play.py --model checkpoints/best_dual.pt --depth 4
```

Nhập nước đi theo ký hiệu UCI: `e2e4`, `g1f3`, `e7e8q` (phong hậu).

### Khởi động giao diện Web

```bash
# Mặc định: cổng 5000, checkpoint best_model.pt
python web/app.py

# Tuỳ chỉnh qua biến môi trường
CHECKPOINT=checkpoints/best_dual.pt AGENT_DEPTH=3 python web/app.py
```

Mở trình duyệt tại `http://localhost:5000`.

---

## 9. Pipeline huấn luyện

### Bước 1 — Xây từ điển nước đi

```bash
python -m scripts.build_vocab
# Output: data/processed/move2idx.json, idx2move.json  (4544 entries)
```

### Bước 2 — Parse dữ liệu PGN

```bash
python -m scripts.parse_pgn \
  --pgn  data/raw/lichess_elite_2022-03.pgn \  # file PGN đầu vào
  --out  data/processed/train.pt \              # dataset PyTorch đầu ra
  --min_elo  1800 \                             # lọc ván dưới ELO này
  --max_positions  200000                       # giới hạn số vị trí
```

### Bước 3 — Train PolicyNet (có giám sát)

```bash
python -m src.train \
  --mode supervised \                  # chế độ: supervised | supervised_dual | selfplay | finetune_value
  --config configs/default.yml
# Output: checkpoints/best_policy.pt
# Log:    logs/supervised_loss.csv
```

### Bước 4 — Train DualNet (supervised warmup)

```bash
python -m src.train \
  --mode supervised_dual \
  --config configs/default.yml
# Output: checkpoints/best_dual.pt
# Log:    logs/dual_supervised_loss.csv
```

### Bước 5 — Fine-tune value head (endgame)

```bash
python -m src.train \
  --mode finetune_value \              # chỉ train value head, freeze backbone + policy
  --data data/processed/endgame.pt \  # dataset tàn cuộc có nhãn kết quả
  --epochs 50
# Output: checkpoints/best_dual.pt (cập nhật value head)
# Trainable params: ~4,356 / 10,521,892 (value head only)
```

### Bước 6 — Sinh dữ liệu self-play

```bash
python -m scripts.self_play \
  --n_games     200 \                  # số ván cần sinh
  --n_sims      200 \                  # số MCTS simulation mỗi nước
  --checkpoint  checkpoints/best_dual.pt \
  --device      cuda \                 # cpu | cuda
  --book        data/opening_books/human.bin  # (tuỳ chọn) opening book
# Output: data/selfplay/selfplay_200games_200sims.pt
```

### Bước 7 — Train DualNet trên self-play data

```bash
python -m src.train \
  --mode selfplay \
  --config configs/default.yml
# Log: logs/selfplay_loss.csv
```

### Bước 8 — Arena (candidate vs best)

```bash
python -m scripts.pit \
  --candidate  checkpoints/candidate.pt \
  --best       checkpoints/best_dual.pt \
  --n_games    40
# Candidate promoted nếu win_rate > 55%
```

### Bước 9 — Benchmark vs Stockfish

```bash
python -m scripts.benchmark \
  --model           checkpoints/best_dual.pt \
  --stockfish       bin/stockfish.exe \
  --n_games         10 \
  --stockfish_level 1              # level 1–20
# Log: logs/elo_dual.csv  →  (timestamp, iteration, win_rate, est_elo)
```

### Vòng lặp tự động

```bash
bash scripts/run_iterations.sh
# Chạy Bước 6–9 lặp đi lặp lại với sliding replay buffer
```

---

## 10. Giao diện Web (Flask)

Ứng dụng Flask cho phép chơi cờ trực tiếp trên trình duyệt với bàn cờ đồ họa (ảnh PNG quân cờ thực).

### Kiến trúc

```
web/
├── app.py              # Flask backend: route / và POST /move
├── templates/
│   └── index.html      # Frontend: bàn cờ, logic giao diện
└── static/
    └── pieces/         # Ảnh PNG quân cờ (wP, wR, ..., bK)
```

### API

| Endpoint | Method | Mô tả |
|---|---|---|
| `/` | GET | Trang chủ — render bàn cờ |
| `/move` | POST | Nhận FEN, trả về nước đi của AI |

**Request `/move`:**
```json
{ "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" }
```

**Response `/move`:**
```json
{
  "move": "e7e5",
  "fen":  "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
  "over": false,
  "result": null
}
```

### Khởi động

```bash
# Dùng DualNet checkpoint (nếu có), fallback về MinimaxAgent
CHECKPOINT=checkpoints/best_dual.pt \
AGENT_DEPTH=3 \
python web/app.py
```

```powershell
# Windows PowerShell
$env:CHECKPOINT="checkpoints/best_dual.pt"; $env:AGENT_DEPTH="3"
python web/app.py
```

Mở trình duyệt tại `http://localhost:5000`.

---

## 11. Đánh giá vị trí (FEN Eval)

Script `scripts/fen_eval.py` kiểm tra chất lượng model trên các vị trí cờ kinh điển — hiển thị điểm value và top-5 nước được ưu tiên.

```bash
# Đánh giá DualNet (mặc định)
python scripts/fen_eval.py

# Chỉ định checkpoint và loại model
python scripts/fen_eval.py \
  --checkpoint checkpoints/best_dual.pt \
  --model dual               # dual | policy
```

**Ví dụ output sau khi fine-tune value head:**

```
Starting position
  FEN  : rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
  Note : Expected: near 0.0 (balanced)
  Score: -0.1040  [Black]   Top-5: e2e4(64.8%)  d2d4(29.5%)  g1f3(4.5%)

Ruy Lopez (after 1.e4 e5 2.Nf3 Nc6 3.Bb5)
  Score: +0.3017  [White]   Top-5: a7a6(81.0%)  g8f6(8.5%)  d7d6(5.7%)

White winning — Queen + Rook vs King
  Note : Expected: ~+1.0
  Score: +0.2915  [White]   (model chưa đạt ~+1.0, cần thêm endgame data)

Stalemate-like — King and pawn endgame
  Score: -0.5572  [Black]   Top-5: a3b3(46.6%)  a3b4(19.0%)  a3a4(4.5%)
```

**Giải thích điểm:**
- `+1.0` — model đánh giá Trắng thắng chắc chắn
- ` 0.0` — vị trí cân bằng
- `-1.0` — model đánh giá Đen thắng chắc chắn

---

## 12. Kết quả thực nghiệm

### PolicyNet — Học có giám sát (5 epoch)

![Supervised Loss](reports/figures/supervised_loss.png)

| Epoch | Train Loss | Val Loss | Train Top-5 | Val Top-5 |
|---|---|---|---|---|
| 0 | 5.920 | 5.211 | 15.3% | 24.5% |
| 1 | 4.756 | 4.120 | 30.3% | 39.7% |
| 2 | 4.064 | 3.752 | 39.6% | 44.1% |
| 3 | 3.733 | 3.597 | 44.6% | 46.1% |
| 4 | 3.523 | 3.507 | 48.2% | 48.9% |

Top-5 accuracy ~49%: nước đúng nằm trong top-5 dự đoán khoảng một nửa số trường hợp.

### DualNet — Supervised warmup (10 epoch)

![Dual-Net Loss](reports/figures/dual_loss.png)

| Epoch | Train Policy | Train Value | Val Policy | Val Value |
|---|---|---|---|---|
| 0 | 4.624 | 0.883 | 3.430 | 0.861 |
| 2 | 2.054 | 0.747 | 3.325 | 0.730 |
| 5 | 0.798 | 0.385 | 4.309 | 0.486 |
| 9 | 0.327 | 0.212 | 5.586 | 0.394 |

Val policy loss tăng sau epoch 2 (overfit) — điểm dừng tối ưu là epoch 1–2.

### Fine-tuning value head (50 epoch, 365 vị trí tàn cuộc)

Chỉ 4,356 / 10,521,892 tham số được huấn luyện (backbone và policy head đóng băng):

| Epoch | Train Value | Val Value |
|---|---|---|
| 0 | 0.4875 | 0.4969 |
| 10 | 0.4230 | 0.4381 |
| 20 | 0.2925 | 0.3837 |
| 30 | 0.2695 | 0.3460 |
| 40 | 0.2357 | 0.3144 |
| 49 | 0.2095 | 0.2918 |

Val value loss giảm đều đặn từ 0.497 → 0.292 sau 50 epoch — không có dấu hiệu overfit nhờ dataset nhỏ và số tham số rất ít.

### Đánh giá ELO vs Stockfish Level 1

$$\text{ELO}_{\text{agent}} = \text{ELO}_{\text{opponent}} + 400 \cdot \log_{10}\!\left(\frac{p}{1 - p}\right)$$

Trong đó $p$ là win rate (hòa tính 0.5 điểm).

| Mô hình | Win Rate | Est. ELO |
|---|---|---|
| PolicyNet + Minimax depth 3 | 95% | ~1312 |
| PolicyNet + Minimax depth 3 | 15% | ~699 |
| DualNet + Minimax depth 3 | 70% | ~947 |
| DualNet + Minimax depth 3 | 40% | ~930 |
| DualNet + Minimax depth 5 | 35% | ~892 |
| Self-Play DualNet (5 iter, UCB1 cũ) | 0% | ~202 |

![ELO Progress](reports/figures/elo_progress.png)

> **Lưu ý về self-play:** Kết quả 0% là từ phiên bản cũ dùng UCB1 — policy head không tham gia tìm kiếm, gây collapse từ iteration 4–5. Phiên bản nâng cấp PUCT đã khắc phục vấn đề nhưng chưa có đủ tài nguyên GPU để chạy lại đủ iteration.

---

## 13. Cấu trúc dự án

```
.
├── src/
│   ├── model.py          # PolicyNet và DualNet (ResBlock, dual heads)
│   ├── board.py          # Mã hóa bàn cờ → tensor (17, 8, 8)
│   ├── vocab.py          # Xây và load từ điển 4544 nước UCI
│   ├── mcts.py           # UCB1-MCTS, PUCT-MCTS, Batched MCTS; MCTSNode
│   ├── agent.py          # MinimaxAgent, DualNetMinimaxAgent
│   ├── search.py         # Alpha-beta minimax, PST, move ordering
│   ├── dataset.py        # ChessDataset, SelfPlayDataset, mixed dataloader
│   ├── train.py          # supervised / supervised_dual / selfplay / finetune_value
│   ├── config.py         # Load YAML config
│   └── opening_book.py   # Polyglot .bin handler
│
├── scripts/
│   ├── self_play.py      # Sinh dữ liệu self-play (PUCT + temperature)
│   ├── pit.py            # Arena: so sánh hai checkpoint
│   ├── benchmark.py      # Đánh giá vs Stockfish, ước tính ELO
│   ├── fen_eval.py       # Đánh giá model trên FEN position cụ thể
│   ├── parse_pgn.py      # Parse Lichess PGN → PyTorch .pt dataset
│   ├── build_vocab.py    # Xây từ điển move2idx.json / idx2move.json
│   ├── plot_results.py   # Vẽ training curves và ELO history
│   └── run_iterations.sh # Vòng lặp tự động toàn bộ pipeline
│
├── web/
│   ├── app.py            # Flask server: GET / và POST /move
│   ├── templates/
│   │   └── index.html    # Frontend: bàn cờ đồ họa, nhận/gửi nước đi
│   └── static/
│       └── pieces/       # Ảnh PNG quân cờ (wP.png, bK.png, ...)
│
├── tests/                # Unit tests (pytest)
│
├── configs/
│   └── default.yml       # Hyperparameters và đường dẫn dữ liệu
│
├── data/
│   ├── raw/              # PGN thô từ Lichess (không commit lên git)
│   ├── processed/        # train.pt, endgame.pt, move2idx.json
│   ├── selfplay/         # iter_k.pt — dữ liệu tự chơi mỗi iteration
│   └── opening_books/    # File Polyglot .bin
│
├── checkpoints/          # best_policy.pt, best_dual.pt, candidate_k.pt
├── logs/                 # CSV metrics + TensorBoard event files
├── reports/              # Báo cáo (report.md) và biểu đồ
├── bin/                  # Stockfish binary
├── play.py               # Terminal UI (Human vs AI / AI vs AI)
├── requirements.txt
└── pytest.ini
```

---

## 14. Kiểm thử

```bash
# Chạy tất cả tests
pytest

# Chạy một module cụ thể với output chi tiết
pytest tests/test_mcts.py -v

# Kèm báo cáo coverage
pytest --cov=src tests/
```

| File test | Nội dung kiểm tra |
|---|---|
| `test_board.py` | Kích thước và giá trị tensor mã hóa bàn cờ |
| `test_mcts.py` | UCB1, PUCT selection, Dirichlet noise, nước hợp lệ |
| `test_agent.py` | MinimaxAgent và DualNet agent chọn nước hợp lệ |
| `test_dual_net.py` | Forward pass DualNet: shape và value $\in [-1, 1]$ |
| `test_dataset.py` | Load dataset, train/val split, batching |
| `test_train.py` | Tính loss, một bước gradient descent |
| `test_search.py` | Alpha-beta pruning, PST evaluation, move ordering |
| `test_self_play.py` | Sinh ván, gán value target hồi tố |
| `test_pit.py` | Tính win rate trong arena |
| `test_pipeline.py` | Pipeline end-to-end từ data đến checkpoint |

---

## 15. Kết luận

### Những gì đã xây dựng

- Hệ thống cờ vua AI hoàn chỉnh: biểu diễn bàn cờ → mạng nơ-ron → MCTS → self-play → ELO benchmark.
- Khắc phục vấn đề policy head không tham gia MCTS bằng PUCT và `expand_with_policy()`.
- Khắc phục self-play collapse bằng Dirichlet noise và temperature sampling.
- Fine-tuning chuyên biệt value head trên dữ liệu tàn cuộc (val_v giảm từ 0.497 → 0.292).
- Giao diện web Flask cho phép chơi cờ trực tiếp trên trình duyệt.
- DualNet + Minimax đạt ~930–947 ELO ước tính khi đối đầu Stockfish Level 1.

### Giới hạn và hướng phát triển

**Giới hạn hiện tại:**
- Self-play PUCT chưa chạy đủ iteration do chi phí tính toán (cần GPU và nhiều giờ).
- Value head vẫn chưa đạt ~±1.0 ở các vị trí tàn cuộc quyết định — cần thêm dữ liệu endgame.
- Số MCTS simulations thấp (50) so với AlphaZero gốc (800+).

**Hướng phát triển:**
- Tăng `n_sims` lên 200–400 và chạy 20+ iteration self-play với GPU.
- Thêm transposition table để tái sử dụng kết quả MCTS giữa các nước.
- Mở rộng dataset endgame (7-piece tablebase) để fine-tune value head chính xác hơn.
- Thử kiến trúc Transformer thay CNN (tham khảo Leela Chess Zero — lc0).
- Export model sang ONNX để deploy đa nền tảng (mobile, web).

---

## 16. Tài liệu tham khảo

1. Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.* arXiv:1712.01815.
2. Silver, D. et al. (2016). *Mastering the Game of Go with Deep Neural Networks and Tree Search.* Nature, 529, 484–489.
3. Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.* IEEE TCIAIG, 4(1), 1–43.
4. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
5. python-chess: https://python-chess.readthedocs.io/
6. Lichess open database: https://database.lichess.org/
7. Stockfish chess engine: https://stockfishchess.org/
8. Flask documentation: https://flask.palletsprojects.com/
