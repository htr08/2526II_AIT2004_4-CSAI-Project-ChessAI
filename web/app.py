import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chess
from flask import Flask, jsonify, render_template, request

from src.agent import DualNetMinimaxAgent, MinimaxAgent

app = Flask(__name__)

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/best_model.pt")
DEPTH = int(os.environ.get("AGENT_DEPTH", "3"))
BOOK = os.environ.get("OPENING_BOOK", None)

if os.path.exists(CHECKPOINT):
    agent = DualNetMinimaxAgent(depth=DEPTH, model_path=CHECKPOINT, book_path=BOOK)
    print(f"Loaded DualNet agent from {CHECKPOINT}, depth={DEPTH}")
else:
    agent = MinimaxAgent(depth=DEPTH, book_path=BOOK)
    print(f"No checkpoint found — using heuristic MinimaxAgent, depth={DEPTH}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/move", methods=["POST"])
def get_move():
    body = request.get_json(silent=True)
    if not body or "fen" not in body:
        return jsonify({"error": "missing fen"}), 400
    try:
        board = chess.Board(body["fen"])
    except ValueError as e:
        return jsonify({"error": f"invalid fen: {e}"}), 400

    if board.is_game_over():
        return jsonify({"error": "game over", "result": board.result()})

    move = agent.select_move(board)
    board.push(move)
    over = board.is_game_over()
    return jsonify({
        "move": move.uci(),
        "fen": board.fen(),
        "over": over,
        "result": board.result() if over else None,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
