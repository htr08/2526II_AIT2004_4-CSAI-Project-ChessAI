# Chess AI - AlphaZero-lite (AIT2004 Co so AI)

Project xay dung mot bot co vua ket hop **Neural Network (Policy + Value)** voi **Minimax/MCTS search** va **self-play training** - mot phien ban rut gon cua AlphaZero.

## Kien truc tong quan

```
PGN games  ->  Encode 12x8x8  ->  CNN (Policy + Value heads)
                                         |
                              Minimax/MCTS search
                                         |
                          Self-play -> Retrain -> Pit test
                                         (lap)
```

5 buoc lon:
1. **Du lieu** - Lichess PGN (Elite 2024-01 da pre-filter rating ~2400+), parse thanh (vi tri, nuoc di, ket qua).
2. **Encode board** - Tensor `12x8x8` one-hot (6 loai quan x 2 mau x 64 o). Co flip theo perspective cua side-to-move.
3. **Neural network** - CNN voi residual blocks, 2 dau ra: policy head (4096 moves) + value head (tanh [-1, 1]).
4. **Search** - Minimax+Alpha-Beta (voi move ordering bang policy net) hoac MCTS (PUCT, Dirichlet noise).
5. **Self-play loop** - Bot tu dau sinh data -> retrain candidate -> pit test -> giu phien ban manh hon.

## Cau truc thu muc

```
.
+-- src/
|   +-- data/
|   |   +-- encode_board.py      # Board -> tensor 12x8x8 + perspective flip
|   |   +-- action_space.py      # Move <-> index (4096 actions)
|   |   +-- pgn_parser.py        # PGN -> (fen, move, result) samples
|   |   +-- dataset.py           # PyTorch Dataset + DataLoader
|   +-- model/
|   |   +-- network.py           # PolicyValueNet (CNN + residual blocks)
|   +-- search/
|   |   +-- evaluation.py        # Material + PST static eval
|   |   +-- minimax.py           # Negamax + Alpha-Beta + policy ordering
|   |   +-- mcts.py              # MCTS voi PUCT, neural guide
|   +-- training/
|   |   +-- supervised.py        # Supervised pretrain tren PGN data
|   |   +-- self_play.py         # Sinh self-play games
|   |   +-- pit.py               # Doi dau hai phien ban model
|   |   +-- self_play_loop.py    # Vong lap self-play hoan chinh
|   +-- utils/                   # (helper modules, hien trong)
+-- tests/                       # Unit tests (pytest)
+-- scripts/                     # Entry-point scripts
|   +-- parse_pgn.py
|   +-- train_supervised.py
|   +-- run_self_play.py
|   +-- play_cli.py
|   +-- benchmark_stockfish.py
|   +-- plot_history.py
+-- data/{raw,processed}/        # PGN + .pt files (gitignored)
+-- models/                      # Checkpoints (gitignored)
+-- notebooks/                   # Exploration
+-- reports/                     # Plots + bao cao
+-- requirements.txt
+-- .gitignore
```

## Setup

```bash
# 1. Tao virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Sanity test
python -c "import chess, torch; print('chess', chess.__version__, '| torch', torch.__version__)"

# 4. Chay unit tests
pytest tests/ -v
```

## Quick start - pipeline hoan chinh

### Buoc 1: Chuan bi PGN data

Dat file PGN vao `data/raw/`. Project nay duoc test voi **Lichess Elite 2024-01** (~235MB, 298k games da pre-filter rating ~2400+).

### Buoc 2: Parse PGN -> tensor data

```bash
python scripts/parse_pgn.py \
    --input data/raw/lichess_elite_2024-01.pgn \
    --output data/processed/train.pt \
    --max-games 50000 \
    --min-rating 2400 \
    --min-moves 10
```

### Buoc 3: Train supervised

```bash
python scripts/train_supervised.py \
    --data data/processed/train.pt \
    --epochs 5 \
    --batch-size 256 \
    --channels 128 \
    --n-res 3 \
    --output-dir models
# -> tao models/best.pt va models/latest.pt
```

Ket qua: model hoc tu data expert. Top-1 accuracy thuong dat ~30-40%, top-5 ~70-80% sau vai epochs.

### Buoc 4: Self-play loop (phan "AI" thuc su)

```bash
python scripts/run_self_play.py \
    --initial-model models/best.pt \
    --iterations 5 \
    --games 20 \
    --simulations 100 \
    --train-epochs 2 \
    --pit-games 10 \
    --output-dir models/selfplay
```

Moi iteration: 20 self-play games -> retrain candidate -> pit vs current 10 games -> giu candidate neu thang >=55% (loai tru hoa).

### Buoc 5: Choi voi bot

```bash
# Minimax (nhanh, khong can GPU)
python scripts/play_cli.py --model models/best.pt --search minimax --depth 3

# MCTS (manh hon, cham hon)
python scripts/play_cli.py --model models/selfplay/latest.pt --search mcts --simulations 200
```

Trong game: nhap nuoc di UCI (vd `e2e4`, `g1f3`, `e7e8q` de phong hau). Lenh khac: `undo`, `fen`, `quit`.

### Buoc 6: Benchmark vs Stockfish (optional)

```bash
python scripts/benchmark_stockfish.py \
    --model models/selfplay/latest.pt \
    --stockfish path/to/stockfish.exe \
    --games 10 \
    --stockfish-depth 2 \
    --search mcts --simulations 200
```

### Buoc 7: Plot training curves

```bash
# Supervised curves
python scripts/plot_history.py --ckpt models/best.pt --output reports/supervised.png

# Self-play loop curves
python scripts/plot_history.py --selfplay-dir models/selfplay --output reports/selfplay.png
```

## Decisions thiet ke

| Decision | Ly do |
|----------|-------|
| 12-channel input (khong them state) | Don gian, du phu thong tin board. Castling/en passant it anh huong accuracy o scale project. |
| Perspective flip khi black-to-move | Model luon "thay" minh la trang -> hoc duoc symmetry, it data hon van on. |
| 4096 actions (from x to) | Lossy voi underpromotion (rat hiem) nhung don gian, du cho project. AlphaZero goc dung 4672. |
| PolicyValueNet 2 heads | AlphaZero-style: chia se trunk, moi head 1 conv 1x1 + FC. ~1M params. |
| Negamax + Alpha-Beta cho Minimax | Cleaner code, tan dung symmetry cua zero-sum game. |
| Dirichlet noise tai MCTS root | Tang exploration khi self-play, tranh overfit vao nuoc di dau. |
| Pit threshold 55% (decisive games) | Theo AlphaZero paper. Khong tinh draws de tranh nhieu. |

## Trang thai 3 tuan

| Tuan | Muc tieu | Trang thai |
|------|----------|------------|
| 1 | Data pipeline, encode board, Policy Net supervised | DONE - Code xong, da verify |
| 2 | Value Head, Minimax/MCTS, self-play loop | DONE - Code xong, minimax verified |
| 3 | Polish, benchmark, demo, bao cao | DONE - Scripts san sang, cho chay thuc te |

Code da duoc unit-test (`tests/`):
- test_encode_board.py - 12x8x8 encoding + perspective flip
- test_action_space.py - move<->index roundtrip, promotion, legal mask
- test_network.py - forward shapes, backward pass
- test_search.py - eval correctness, minimax mate-in-1, MCTS legal
- test_pipeline.py - end-to-end smoke (encode -> predict -> search)

Minimax verified thuc te (khong can torch):
- Starting position eval = 0 OK
- Mat queen -> eval ~= -895 OK
- Mate-in-1: tim dung Qxf7# voi score 99992 OK
- Stalemate detection OK

PGN parser verified thuc te tren Lichess Elite 2024-01:
- Parse 20 games -> 1629 samples trong 0.1s OK
- Trung binh ~81 samples/game OK

## Thu tu uu tien neu het thoi gian

1. **Self-play loop** (`run_self_play.py`) - phan "AI" thuc su
2. **Minimax + Policy Net supervised** - van choi duoc, du cho bao cao
3. **Supervised training** mot minh - model hoc tu expert, khong tu cai thien
4. **Demo CLI** - de show off cho slide

Benchmark Stockfish va visualization la nice-to-have.

## Hyperparameters mac dinh

| Param | Value | Ghi chu |
|-------|-------|--------|
| Model channels | 128 | Can bang accuracy/toc do |
| Residual blocks | 3 | Du de hoc pattern co ban |
| Total params | ~1M | Chay duoc tren CPU |
| Supervised LR | 1e-3 | AdamW + cosine schedule |
| Value loss weight | 1.0 | Can bang voi policy loss |
| Batch size | 256 | Tuy GPU/CPU memory |
| MCTS c_puct | 1.5 | Tieu chuan |
| MCTS simulations | 100-200 | Trade-off chat luong/toc do |
| Pit accept threshold | 55% (decisive) | AlphaZero paper |
| Dirichlet alpha | 0.3 | Chess noise level |

## Tham khao

- [AlphaZero paper (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [python-chess docs](https://python-chess.readthedocs.io/)
- [Lichess open database](https://database.lichess.org/)
- [Lichess Elite database (pre-filtered)](https://database.nikonoel.fr/)
- [Chess Programming Wiki - Negamax, PST](https://www.chessprogramming.org/)
