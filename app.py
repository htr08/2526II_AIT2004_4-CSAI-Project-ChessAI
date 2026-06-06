"""
app.py
------
Flask web UI để chơi cờ với bot trong trình duyệt.

Chạy:
    python app.py --model models/best.pt --search minimax --depth 3
    # rồi mở http://localhost:5000

Nếu không có model (hoặc chưa cài torch), app vẫn chạy với Minimax + eval
tĩnh (material + PST), không cần GPU. Search "mcts" yêu cầu có model.

Bàn cờ render bằng Unicode (không cần ảnh quân / thư viện ngoài). Người chơi
click ô nguồn → ô đích để đi; phong cấp tự động thành Hậu. Bot tự trả lời.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import chess
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.search.minimax import search_best_move
from src.search.opening_book import book_move

app = Flask(__name__)

# ---- Trạng thái game toàn cục (demo 1 người chơi) ----
board = chess.Board()
CONFIG = {
    "model": None,         # model đã load (hoặc None)
    "search": "minimax",
    "depth": 3,
    "simulations": 200,
    "device": "cpu",
    "human_white": True,
    "use_book": True,
    "last_move": None,     # uci nước vừa đi (để highlight)
}

UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def load_model(path: str, device: str):
    """Lazy load — chỉ import torch khi thực sự cần model."""
    import torch
    from src.model.network import PolicyValueNet

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = PolicyValueNet(
        channels=cfg.get("channels", 128),
        n_res_blocks=cfg.get("n_res_blocks", 3),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_bot_move(b: chess.Board):
    """Chọn nước cho bot theo config hiện tại."""
    if CONFIG["use_book"]:
        bm = book_move(b)
        if bm is not None:
            return bm, "opening book"
    if CONFIG["search"] == "mcts":
        if CONFIG["model"] is None:
            # fallback: không có model thì dùng minimax
            mv, _ = search_best_move(b, depth=CONFIG["depth"])
            return mv, "minimax (no model)"
        from src.search.mcts import search_best_move_mcts
        mv, _ = search_best_move_mcts(
            b, CONFIG["model"], num_simulations=CONFIG["simulations"], device=CONFIG["device"]
        )
        return mv, f"mcts ({CONFIG['simulations']} sims)"
    mv, score = search_best_move(b, depth=CONFIG["depth"], model=CONFIG["model"], device=CONFIG["device"])
    return mv, f"minimax depth {CONFIG['depth']} (eval {score})"


def board_squares():
    """Trả mảng 64 ô (a8..h1, hàng trên xuống) cho frontend."""
    out = []
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            out.append({
                "name": chess.square_name(sq),
                "glyph": UNICODE.get(piece.symbol(), "") if piece else "",
                "light": (file + rank) % 2 == 1,
            })
    return out


def game_status():
    if board.is_checkmate():
        winner = "Đen" if board.turn == chess.WHITE else "Trắng"
        return f"Chiếu hết — {winner} thắng!", True
    if board.is_stalemate():
        return "Hòa — hết nước (stalemate).", True
    if board.is_insufficient_material():
        return "Hòa — không đủ quân chiếu hết.", True
    if board.can_claim_threefold_repetition():
        return "Hòa — lặp 3 lần.", True
    if board.can_claim_fifty_moves():
        return "Hòa — luật 50 nước.", True
    if board.is_check():
        return ("Tới lượt bạn — đang bị chiếu!" if board.turn == (chess.WHITE if CONFIG["human_white"] else chess.BLACK)
                else "Bot đang bị chiếu."), False
    turn_txt = "Trắng" if board.turn == chess.WHITE else "Đen"
    return f"Tới lượt: {turn_txt}", False


def state_payload(bot_move_uci=None, bot_desc=None):
    status, over = game_status()
    human_turn = board.turn == (chess.WHITE if CONFIG["human_white"] else chess.BLACK)
    # legal moves của người (để highlight): map from_square -> [to_square...]
    legal = {}
    if human_turn and not over:
        for m in board.legal_moves:
            legal.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
    return jsonify({
        "squares": board_squares(),
        "fen": board.fen(),
        "status": status,
        "over": over,
        "human_white": CONFIG["human_white"],
        "human_turn": human_turn,
        "legal": legal,
        "last_move": CONFIG["last_move"],
        "bot_move": bot_move_uci,
        "bot_desc": bot_desc,
        "search": CONFIG["search"],
    })


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/state")
def state():
    return state_payload()


@app.route("/new", methods=["POST"])
def new_game():
    global board
    data = request.get_json(silent=True) or {}
    CONFIG["human_white"] = bool(data.get("human_white", True))
    board = chess.Board()
    CONFIG["last_move"] = None
    # Nếu người chơi Đen, bot (Trắng) đi trước
    bot_uci, bot_desc = None, None
    if not CONFIG["human_white"]:
        mv, desc = compute_bot_move(board)
        if mv:
            board.push(mv)
            CONFIG["last_move"] = mv.uci()
            bot_uci, bot_desc = mv.uci(), desc
    return state_payload(bot_uci, bot_desc)


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    frm, to = data.get("from"), data.get("to")
    try:
        f_sq, t_sq = chess.parse_square(frm), chess.parse_square(to)
    except (ValueError, TypeError):
        return jsonify({"error": "ô không hợp lệ"}), 400

    mv = chess.Move(f_sq, t_sq)
    if mv not in board.legal_moves:
        mv = chess.Move(f_sq, t_sq, promotion=chess.QUEEN)  # tự phong Hậu
    if mv not in board.legal_moves:
        return jsonify({"error": "nước đi không hợp lệ"}), 400

    board.push(mv)
    CONFIG["last_move"] = mv.uci()

    # Bot trả lời nếu game chưa kết thúc
    bot_uci, bot_desc = None, None
    if not board.is_game_over(claim_draw=True):
        bmv, desc = compute_bot_move(board)
        if bmv is not None:
            board.push(bmv)
            CONFIG["last_move"] = bmv.uci()
            bot_uci, bot_desc = bmv.uci(), desc
    return state_payload(bot_uci, bot_desc)


PAGE = """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess AI — chơi với bot</title>
<style>
  :root { --light:#eeeed2; --dark:#769656; --sel:#f6f669; --hint:rgba(20,80,20,.35); --last:#bbcb44; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background:#312e2b; color:#eee; margin:0; padding:24px;
         display:flex; flex-direction:column; align-items:center; gap:16px; }
  h1 { font-size:20px; margin:0; font-weight:600; }
  #board { display:grid; grid-template-columns:repeat(8,64px); grid-template-rows:repeat(8,64px);
           border:4px solid #232120; border-radius:4px; box-shadow:0 8px 30px rgba(0,0,0,.4); }
  .sq { width:64px; height:64px; display:flex; align-items:center; justify-content:center;
        font-size:46px; cursor:pointer; position:relative; user-select:none; line-height:1; }
  .sq.light { background:var(--light); } .sq.dark { background:var(--dark); }
  .sq.sel { background:var(--sel) !important; }
  .sq.last { box-shadow: inset 0 0 0 4px var(--last); }
  .piece-w { color:#fff; text-shadow:0 1px 2px #000, 0 0 1px #000; }
  .piece-b { color:#111; text-shadow:0 1px 1px rgba(255,255,255,.25); }
  .hint::after { content:""; position:absolute; width:22px; height:22px; border-radius:50%;
                 background:var(--hint); }
  .hint.cap::after { width:58px; height:58px; border-radius:50%; background:transparent;
                     border:5px solid var(--hint); }
  #status { font-size:16px; min-height:24px; font-weight:500; }
  #botinfo { font-size:13px; color:#9fbf8f; min-height:18px; }
  .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:center; }
  button, select { font-size:14px; padding:7px 12px; border-radius:6px; border:none; cursor:pointer;
                    background:#4a4845; color:#eee; }
  button.primary { background:#769656; } button:hover { filter:brightness(1.1); }
</style>
</head>
<body>
  <h1>♟ Chess AI — chơi với bot</h1>
  <div id="status">Đang tải…</div>
  <div id="board"></div>
  <div id="botinfo"></div>
  <div class="bar">
    <label>Bạn chơi:
      <select id="color">
        <option value="white">Trắng</option>
        <option value="black">Đen</option>
      </select>
    </label>
    <button class="primary" id="new">Ván mới</button>
    <span id="searchinfo"></span>
  </div>

<script>
let S = null, sel = null, busy = false;
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const botEl = document.getElementById('botinfo');

function render() {
  boardEl.innerHTML = '';
  const last = S.last_move ? [S.last_move.slice(0,2), S.last_move.slice(2,4)] : [];
  for (const sq of S.squares) {
    const d = document.createElement('div');
    d.className = 'sq ' + (sq.light ? 'light' : 'dark');
    d.dataset.name = sq.name;
    if (last.includes(sq.name)) d.classList.add('last');
    if (sel === sq.name) d.classList.add('sel');
    if (sq.glyph) {
      const isWhite = '♔♕♖♗♘♙'.includes(sq.glyph);
      const span = document.createElement('span');
      span.textContent = sq.glyph;
      span.className = isWhite ? 'piece-w' : 'piece-b';
      d.appendChild(span);
    }
    if (sel && S.legal[sel] && S.legal[sel].includes(sq.name)) {
      d.classList.add('hint');
      if (sq.glyph) d.classList.add('cap');
    }
    d.onclick = () => onClick(sq.name);
    boardEl.appendChild(d);
  }
  statusEl.textContent = S.status;
  document.getElementById('searchinfo').textContent = 'search: ' + S.search;
  if (S.bot_move) botEl.textContent = 'Bot: ' + S.bot_move + (S.bot_desc ? '  (' + S.bot_desc + ')' : '');
}

async function onClick(name) {
  if (busy || S.over || !S.human_turn) return;
  if (sel === null) {
    if (S.legal[name]) { sel = name; render(); }
    return;
  }
  if (name === sel) { sel = null; render(); return; }
  if (S.legal[sel] && S.legal[sel].includes(name)) {
    const from = sel; sel = null; busy = true;
    botEl.textContent = 'Bot đang suy nghĩ…';
    try {
      const r = await fetch('/move', {method:'POST', headers:{'Content-Type':'application/json'},
                 body: JSON.stringify({from, to: name})});
      S = await r.json();
      if (S.error) { statusEl.textContent = 'Lỗi: ' + S.error; }
      else render();
    } finally { busy = false; }
  } else if (S.legal[name]) { sel = name; render(); }
  else { sel = null; render(); }
}

async function newGame() {
  busy = true; sel = null;
  const human_white = document.getElementById('color').value === 'white';
  botEl.textContent = '';
  const r = await fetch('/new', {method:'POST', headers:{'Content-Type':'application/json'},
             body: JSON.stringify({human_white})});
  S = await r.json(); busy = false; render();
}

document.getElementById('new').onclick = newGame;
(async () => { S = await (await fetch('/state')).json(); render(); })();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Flask web UI chơi cờ với bot")
    p.add_argument("--model", default="models/best.pt", help="Path .pt (bỏ qua nếu không có)")
    p.add_argument("--search", choices=["minimax", "mcts"], default="minimax")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--no-book", action="store_true", help="Tắt opening book")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    CONFIG["search"] = args.search
    CONFIG["depth"] = args.depth
    CONFIG["simulations"] = args.simulations
    CONFIG["use_book"] = not args.no_book

    # Device + model (lazy: chỉ load nếu file tồn tại)
    try:
        import torch
        CONFIG["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        CONFIG["device"] = "cpu"

    model_path = pathlib.Path(args.model)
    if model_path.exists():
        try:
            CONFIG["model"] = load_model(str(model_path), CONFIG["device"])
            print(f"[app] loaded model: {model_path}  (device={CONFIG['device']})")
        except Exception as e:
            print(f"[app] WARN: không load được model ({e}); dùng minimax + eval tĩnh.")
    else:
        print(f"[app] không thấy {model_path}; dùng minimax + eval tĩnh (không cần torch).")
        if args.search == "mcts":
            print("[app] WARN: --search mcts cần model; sẽ fallback minimax.")

    print(f"[app] mở http://{args.host}:{args.port}  (search={CONFIG['search']})")
    app.run(host=args.host, port=args.port, debug=False)
