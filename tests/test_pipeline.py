"""End-to-end smoke test cho toàn bộ pipeline."""
import pathlib
import tempfile

import chess
import torch

from src.data.encode_board import board_to_tensor_perspective
from src.data.action_space import uci_to_index, NUM_ACTIONS
from src.model.network import PolicyValueNet
from src.search.minimax import search_best_move
from src.search.mcts import MCTS


def test_full_pipeline_smoke():
    """
    Pipeline tối giản:
    1. Encode board
    2. Forward qua model
    3. MCTS chọn move
    4. Push move
    5. Re-encode → repeat 5 nước
    """
    model = PolicyValueNet(channels=32, n_res_blocks=1)
    model.eval()

    board = chess.Board()
    for ply in range(5):
        # Encode
        x = board_to_tensor_perspective(board).unsqueeze(0)

        # Forward
        with torch.no_grad():
            policy, value = model(x)
        assert policy.shape == (1, NUM_ACTIONS)
        assert value.shape == (1,)

        # MCTS choose
        mcts = MCTS(model, num_simulations=10, add_noise=False)
        move, probs, moves = mcts.get_action_probs(board, temperature=0.0)
        assert move in board.legal_moves
        board.push(move)

    assert len(board.move_stack) == 5


def test_minimax_no_model_works():
    """Minimax không cần model — chạy được chỉ với evaluation function."""
    board = chess.Board()
    move, _ = search_best_move(board, depth=2)
    assert move in board.legal_moves


def test_save_load_checkpoint():
    """Test save/load model checkpoint."""
    model = PolicyValueNet(channels=32, n_res_blocks=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "test.pt"
        torch.save({"model_state": model.state_dict(),
                    "config": {"channels": 32, "n_res_blocks": 1}}, path)

        ckpt = torch.load(path, weights_only=False)
        model2 = PolicyValueNet(
            channels=ckpt["config"]["channels"],
            n_res_blocks=ckpt["config"]["n_res_blocks"],
        )
        model2.load_state_dict(ckpt["model_state"])

        # Check outputs khớp
        x = torch.randn(1, 12, 8, 8)
        with torch.no_grad():
            p1, v1 = model(x)
            p2, v2 = model2(x)
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)
