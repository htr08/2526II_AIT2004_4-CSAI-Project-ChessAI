"""Wrapper: chạy self-play loop hoàn chỉnh."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    import argparse
    import torch
    from src.search.training.self_play import SelfPlayConfig
    from src.search.training.self_play_loop import LoopConfig, run_self_play_loop

    p = argparse.ArgumentParser()
    p.add_argument("--initial-model", default=None,
                   help="Path tới checkpoint (vd models/best.pt từ supervised). Optional.")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--train-epochs", type=int, default=2)
    p.add_argument("--pit-games", type=int, default=10)
    p.add_argument("--pit-simulations", type=int, default=100)
    p.add_argument("--output-dir", default="models/selfplay")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sp = SelfPlayConfig(
        num_games=args.games,
        num_simulations=args.simulations,
        device=device,
    )
    cfg = LoopConfig(
        num_iterations=args.iterations,
        self_play=sp,
        train_epochs=args.train_epochs,
        pit_games=args.pit_games,
        pit_simulations=args.pit_simulations,
        output_dir=args.output_dir,
        device=device,
    )
    run_self_play_loop(args.initial_model, cfg)
