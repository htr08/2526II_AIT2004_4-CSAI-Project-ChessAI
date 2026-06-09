# ♟️ ChessNet — Huấn luyện mô hình cờ vua

Dự án cuối kỳ môn **Cơ sở Trí tuệ Nhân tạo** — Trường Đại học Công nghệ, ĐHQGHN.

Xây một AI cờ vua từ đầu theo tinh thần AlphaZero: đầu tiên cho nó học từ 250k ván của các kỳ thủ chuyên nghiệp, sau đó thả ra tự chơi với chính mình để tiếp tục cải thiện. Kèm theo web app Flask để ai cũng có thể vào thử sức.

> 📄 **Báo cáo đầy đủ:** [Chess_report.pdf](./Chess_report.pdf)

---

## Mục lục
- [Pipeline tổng quan](#pipeline-tổng-quan)
- [Kiến trúc mạng ChessNet](#kiến-trúc-mạng-chessnet)
- [Self-Play & Học tăng cường](#self-play--học-tăng-cường)
- [Kết quả](#kết-quả)
- [Chạy trên Kaggle](#chạy-trên-kaggle)
- [Web App](#web-app)
- [Thành viên](#thành-viên)

---

## Pipeline tổng quan

```
[250k ván PGN — Lichess Elite]
           │
           ▼
   Pretrain — 9 epoch
   AdamW + Warmup + CosineAnnealing
   Policy: Cross-Entropy
   Value:  Cross-Entropy WDL
           │
           ▼
       Best Model ◄──────────────────────┐
           │                             │ Win Rate ≥ 55% → update
           ▼                             │
   Self-play: 128 ván × 400 sims         │
   Batched MCTS → ~15.000 positions      │
           │                             │
           ▼                             │
   RL Training: 3 epoch                  │
   Replay buffer 4 iter + weighted       │
           │                             │
           ▼                             │
   Eval: 40 ván, khai cuộc ngẫu nhiên ───┘
   (lặp 15 iterations)
```

---

## Kiến trúc mạng ChessNet

### Mã hóa đầu vào — tensor (15, 8, 8)

Bàn cờ không được đưa vào dưới dạng ký hiệu chữ mà được mã hóa thành 15 lớp ảnh 8×8 xếp chồng nhau:

| Kênh | Nội dung |
|------|----------|
| 1–6  | Vị trí 6 loại quân Trắng (Tốt, Mã, Tượng, Xe, Hậu, Vua) |
| 7–12 | Vị trí 6 loại quân Đen |
| 13   | Tất cả = 1 (flag bên đang đến lượt) |
| 14–15| Ô xuất phát và ô đích của nước đi liền trước |

**Canonical Flip:** khi đến lượt Đen, bàn cờ lật dọc và màu quân hoán đổi. Mạng chỉ cần học một chiến lược chung thay vì học riêng cho cả hai màu — giảm một nửa độ phức tạp mà không mất thông tin gì.

**Lọc dữ liệu PGN:** chỉ giữ ván ≥ 40 lượt, bỏ 8–10 lượt đầu (khai cuộc thư viện) và 8–10 lượt cuối (tàn cuộc hỗn loạn). Lấy 1 vị trí mỗi 2 lượt với offset ngẫu nhiên để mỗi ván chỉ chứa toàn Trắng hoặc toàn Đen, đảm bảo cân bằng tỉ lệ sau Canonical Flip trên toàn dataset.

### Sơ đồ kiến trúc

```
Input (15, 8, 8)
      │
      ▼
Conv 3×3 (15→64) → BN → ReLU
      │
      ▼
┌─────────────────────────────────────┐
│  ResBlock 1                         │
│  ResBlock 2                         │
│  ResBlock 3                         │
│     + BoardAttention ①              │  ← Self-Attention 4 heads sau block 3
│  ResBlock 4                         │
│  ResBlock 5                         │
│  ResBlock 6                         │
│     + BoardAttention ②              │  ← Self-Attention 4 heads sau block 6
└──────────────────┬──────────────────┘
                   │
           Feature map (64×8×8)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Policy Head            Value Head
  Conv1×1 (64→32)        Conv1×1 (64→8)
  BN → ReLU              BN → ReLU
  Flatten (2048)         Flatten (512)
  FC → 4096 logits       FC (512→64) → ReLU
  Legal Mask (−∞)        FC (64→3)
  Softmax                Softmax
        │                     │
  P(nước đi)       [P(Win), P(Draw), P(Loss)]
                   v = P(Win) − P(Loss) ∈ [−1, 1]
```

Mỗi ResBlock có cấu trúc:
```
output = ReLU( Conv2( Conv1(x) ) × w_SE + x )
```
`w_SE` là trọng số từ SE gate, `+ x` là skip connection.

| Thông số | Giá trị |
|----------|---------|
| Channels | 64 |
| ResBlocks | 6 |
| Attention heads | 4 |
| Attention sau mỗi | 3 blocks |
| Tổng tham số | ~1.2M |

### SE Gate (Squeeze-Excite)

SE gate hoạt động như một bộ lọc thông minh: tự học xem kênh nào quan trọng trong từng tình huống và điều chỉnh trọng số tương ứng. Kênh Mã sẽ tự bị giảm về gần 0 khi tàn cuộc không còn Mã trên bàn.

```
Squeeze : GlobalAvgPool(64×8×8) → z ∈ ℝ⁶⁴
Excite  : FC(64→16) → ReLU → FC(16→64) → Sigmoid → w ∈ (0,1)⁶⁴
Scale   : x_out[c] = x[c] × w[c]
```

Chỉ tốn 2.048 tham số nhưng hiệu quả rõ rệt.

### BoardAttention

Tích chập chỉ nhìn được vùng 3×3 xung quanh mỗi ô. Self-Attention 4 heads cho phép mỗi ô "hỏi thăm" tất cả 63 ô còn lại cùng lúc — bắt được những mối quan hệ xa như Hậu kiểm soát đường chéo dài hay Xe đang phong tỏa cột.

### Hàm mất mát

```
L_total = L_policy + λ × L_value     (λ = 1.0)

L_policy : Cross-Entropy
           so sánh nước dự đoán với nước GM thực tế đã đi

L_value  : Cross-Entropy WDL
           nhãn ∈ {0=Win, 1=Draw, 2=Loss} từ góc nhìn bên đang đi
```

Value Head dùng WDL thay vì 1 scalar như AlphaZero gốc vì cờ vua có 3 kết quả rời rạc — phân biệt rõ hơn giữa "thắng chắc" và "đang dẫn nhưng dễ hòa".

### Kỹ thuật huấn luyện

| Kỹ thuật | Chi tiết |
|----------|----------|
| AdamW | weight decay 1e-4, xử lý L2 chuẩn hơn Adam thuần cho mô hình có Attention |
| Mixed Precision (AMP) | float16 cho forward/backward, tăng ~2× tốc độ, tiết kiệm VRAM |
| GradScaler | tránh gradient underflow khi dùng float16 |
| Gradient Clipping | L2-norm ≤ 1.0, ngăn exploding gradient |
| Warmup + Cosine LR | warmup 1000 bước → cosine decay; LR: 1e-3 → 1e-5 |
| Chunk DataLoader | chia 9M positions thành chunk ~30k, tránh tràn RAM |

---

## Self-Play & Học tăng cường

### MCTS — Monte Carlo Tree Search

Mạng không chọn ngay nước từ policy head mà "suy nghĩ trước" qua 400 simulations. Mỗi simulation gồm 4 bước:

**① Selection** — đi xuống cây theo PUCT score:
```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N_parent / (1 + N(s,a))

Q(s,a)   : chất lượng trung bình từ các sim trước (khai thác)
P(s,a)   : prior từ policy head (định hướng ban đầu)
N(s,a)   : số lần đã thăm — node ít thăm được ưu tiên (khám phá)
c_puct   : 2.0
```

**② Expand** — tạo node con từ các nước hợp lệ, gán prior từ policy head

**③ Evaluate** — gọi mạng tại leaf node → `v = P(Win) − P(Loss)`

**④ Backprop** — cập nhật Q ngược lên root, đảo dấu tại mỗi nút cha vì mỗi lượt là của đối thủ

Sau đủ simulations, policy target được rút từ visit counts:
```
π(a|s) = N(s,a) / Σ N(s,b)
```
Phân phối này tốt hơn raw policy vì đã được "lọc" qua hàng trăm lần thử nghiệm thực tế.

### Dirichlet Noise — buộc mạng phải khám phá

Nếu không có nhiễu, mọi ván self-play sẽ đi theo cùng một đường — mạng chỉ ôn lại những gì đã biết và không tiến thêm được. Giải pháp là thêm nhiễu ngẫu nhiên vào prior của **root node** trước mỗi lượt tìm kiếm:

```
P'(a) = (1 − ε) × P(a) + ε × η(a),   η ~ Dir(α)

ε = 0.25   (25% nhiễu, 75% prior gốc)
α = 0.3    (nhiễu phân tán đều, buộc khám phá nhánh ít quen)
```

Nhiễu chỉ thêm vào root — các node sâu hơn là mô phỏng giả định, thêm nhiễu ở đó làm giảm chất lượng đánh giá mà không có lợi gì.

### Batched MCTS — speedup 8–12×

Vấn đề: MCTS thông thường gọi GPU với batch=1 mỗi simulation → 128 ván × 400 sims = 51.200 lần gọi đơn lẻ → GPU gần như idle.

Giải pháp: chạy 128 ván đồng thời, gom tất cả leaf nodes thành 1 batch mỗi step:

```
128 ván chạy song song
      │
      ▼
Selection × 128          (CPU — traverse xuống leaf)
      │
      ▼
Gom 128 leaves → tensor (128, 15, 8, 8)
      │
      ▼
1 forward pass GPU       ← thay vì 51.200 lần riêng lẻ
      │
      ▼
Expand + Backprop × 128  (CPU)
```

Ba tối ưu thêm giúp đạt speedup thực tế:
- Build legal mask trên NumPy (CPU) → push GPU 1 lần, tránh sync storm
- Dùng `push/pop` trực tiếp thay vì `board.copy()` tốn kém
- Tái sử dụng engine object, chỉ reinit root sau mỗi nước

### Vòng lặp RL (15 iterations)

```
① Self-play
   128 ván × 400 sims 
   30 nước đầu : sample theo π (temperature=1) → đa dạng khai cuộc
   Từ nước 30  : greedy (visit count cao nhất) → đảm bảo chất lượng ván

② Train
   Copy best model → 3 epoch AdamW
   Replay buffer: giữ 4 iter gần nhất
   WeightedRandomSampler: cũ=0.25 → mới=1.0  (replacement=True)
   Dynamic warmup = max(1, total_steps // 10)

   Policy loss : L = −Σ π_mcts(a|s) × log π_net(a|s)
   Value loss  : Cross-Entropy WDL (nhãn từ kết quả ván thực)

③ Eval
   40 ván (20 Trắng + 20 Đen), khai cuộc ngẫu nhiên từ 20 lines
   add_noise=True để phá tính tất định
   Win Rate = (W + 0.5×D) / N

④ Update
   Win Rate ≥ 55% → thay best model
   Ngưỡng 55% thay vì 50% vì 40 ván vẫn có variance đáng kể
```

---

## Kết quả

| Chỉ số | Giá trị |
|--------|---------|
| Top-1 Accuracy (pretrain, epoch 9) | 46% |
| Top-5 Accuracy (pretrain, epoch 9) | 86% |
| Winrate vs Stockfish ELO 1400 | 54% / 50 ván |

Top-5 86% có nghĩa là trong phần lớn tình huống, nước đúng nằm trong top 5 mạng đề xuất — nền tảng chiến thuật đủ tốt để RL tiếp tục cải thiện.

---

## Chạy trên Kaggle

### Yêu cầu
Tài khoản Kaggle free — GPU T4 × 2, ~30h/tuần.

### Các bước

**1. Upload notebook**

Vào [kaggle.com/code](https://kaggle.com/code) → **New Notebook** → **File** → **Import Notebook** → chọn `chesskaggle.ipynb`.

**2. Bật GPU T4 × 2**

Trong notebook: **Settings** (góc phải) → **Accelerator** → **GPU T4 × 2**.

**3. Upload dataset PGN**

Vào [kaggle.com/datasets](https://kaggle.com/datasets) → **New Dataset** → upload 3 file:
```
lichess_elite_2024-07.pgn
lichess_elite_2025-03.pgn
lichess_elite_2025-09.pgn
```
Sau đó trong notebook: **+ Add Data** → tìm dataset vừa tạo.

**4. Chỉnh đường dẫn trong CONFIG**

```python
PGN_FILES = [
    "/kaggle/input/<tên-dataset>/lichess_elite_2024-07.pgn",
    "/kaggle/input/<tên-dataset>/lichess_elite_2025-03.pgn",
    "/kaggle/input/<tên-dataset>/lichess_elite_2025-09.pgn",
]
DRIVE_DIR     = "/kaggle/working"
PRETRAIN_CKPT = "/kaggle/working/pretrain.pt"
```

**5. Chạy**

**Run All** — pretrain ~4–6h, mỗi RL iteration ~25–35 phút.

Model lưu tại `/kaggle/working/rl/best_model.pt` → tab **Output** để download.

**6. Resume sau khi hết giờ**

```python
RESUME_EPOCH = 5      # tiếp pretrain từ epoch đã lưu
run_train()

run_rl_loop(start_iteration=3)   # tiếp RL từ iteration đã lưu
```

---

## Web App

Flask server + UI bàn cờ, chơi trực tiếp qua trình duyệt.

### Chạy local

```bash
git clone https://github.com/<your-username>/chessbot.git
cd chessbot
```

Tạo môi trường ảo:
```bash
# Linux / macOS
python3 -m venv venv && source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat
```

```bash
pip install -r requirements.txt
# đặt best_model.pt vào thư mục gốc
python app.py
# → http://localhost:5000
```

Thoát: `deactivate`

### Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `MODEL_PATH` | `best_model.pt` | Đường dẫn đến file model |
| `MCTS_SIMS` | `200` | Số simulations mỗi nước |
| `C_PUCT` | `2.0` | Exploration constant |

### API

```
POST /api/move         { fen, last_move, sims }  →  { move, eval }
POST /api/legal_moves  { fen }                   →  { moves, is_game_over }
GET  /api/health                                 →  { status, device, sims }
```
### Giao diện

![Giao diện Flask](pics\485ed0d9-76e2-445b-9273-7037478875f9.jpg)

## Thành viên



| Họ và tên | 
| :--- | 
| Trương Ái Linh |
| Nguyễn Quốc Khánh | 
| Cao Huy Hòa | 
| Nguyễn Thị Hiền Trang | 

---

> *Lấy cảm hứng từ [AlphaZero — DeepMind (2017)](https://arxiv.org/abs/1712.01815), triển khai lại trên GPU free-tier (Kaggle T4 × 2).*