#!/usr/bin/env bash
# Self-play training loop: generate data → train → pit → benchmark
# Usage: bash scripts/run_iterations.sh [N_ITER] [N_GAMES] [N_SIMS] [EPOCHS] [WINDOW] [PIT_GAMES]
set -euo pipefail

N_ITER=${1:-5}
N_GAMES=${2:-200}
N_SIMS=${3:-50}
TRAIN_EPOCHS=${4:-5}
WINDOW=${5:-5}       # replay-buffer window: train on last N iters of self-play data
PIT_GAMES=${6:-20}   # arena games per pit (≥20 to reduce noise)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/venv/Scripts/python.exe"
export PYTHONPATH="${PROJECT_ROOT};${PYTHONPATH:-}"
SUP_DATA="data/processed/train.pt"   # supervised data for 50/50 mixing
BEST_MODEL="checkpoints/best_dual.pt"
DEVICE=$("$VENV_PYTHON" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")

mkdir -p data/selfplay checkpoints logs

echo "Self-play loop: ${N_ITER} iters | ${N_GAMES} games/iter | ${N_SIMS} sims/move | ${TRAIN_EPOCHS} epochs | window=${WINDOW} | pit=${PIT_GAMES} games"
echo "Supervised mix: ${SUP_DATA}"
echo ""

for i in $(seq 1 "${N_ITER}"); do
    echo "=============================="
    echo "=== ITER ${i} / ${N_ITER}   ==="
    echo "=============================="

    SP_OUT="data/selfplay/iter_${i}.pt"
    CANDIDATE="checkpoints/candidate_${i}.pt"

    # ── 1. Generate self-play data ────────────────────────────────────────────
    CKPT_ARGS=()
    if [ -f "${BEST_MODEL}" ]; then
        CKPT_ARGS=(--checkpoint "${BEST_MODEL}")
    else
        echo "[bootstrap] No best model yet — self-play with random weights"
    fi

    "$VENV_PYTHON" scripts/self_play.py \
        --n_games  "${N_GAMES}" \
        --n_sims   "${N_SIMS}"  \
        --out      "${SP_OUT}"  \
        --device   "${DEVICE}"  \
        "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"

    # ── 2. Train DualNet (mixed supervised + self-play) ───────────────────────
    MIX_ARGS=()
    if [ -f "${SUP_DATA}" ]; then
        MIX_ARGS=(--sup_data "${SUP_DATA}")
    fi

    "$VENV_PYTHON" src/train.py \
        --mode     selfplay       \
        --data     "${SP_OUT}"    \
        --window   "${WINDOW}"    \
        --epochs   "${TRAIN_EPOCHS}" \
        --out      "${CANDIDATE}" \
        "${MIX_ARGS[@]+"${MIX_ARGS[@]}"}"

    # ── 3. Pit candidate vs current best (promotes if win_rate > 55%) ─────────
    # ≥20 games to cut the statistical noise of a 10-game arena.
    "$VENV_PYTHON" scripts/pit.py \
        --new     "${CANDIDATE}"  \
        --old     "${BEST_MODEL}" \
        --n_games "${PIT_GAMES}"  \
        --device  "${DEVICE}"

    # ── 4. Benchmark best model and append ELO to log ─────────────────────────
    "$VENV_PYTHON" scripts/benchmark.py \
        --model   "${BEST_MODEL}" \
        --skill   2               \
        --n_games 10              \
        --device  "${DEVICE}"     \
        --log     logs/elo_history.csv \
        --iter    "${i}"

    echo "--- iter ${i} done ---"
    echo ""
done

echo "=============================="
echo "=== All ${N_ITER} iterations complete ==="
echo "=============================="
echo "ELO history → logs/elo_history.csv"
