"""Wrapper script cho src.training.supervised."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from src.search.training.supervised import train, TrainConfig
    import argparse
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/train.pt")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--n-res", type=int, default=3)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        channels=args.channels,
        n_res_blocks=args.n_res,
        value_weight=args.value_weight,
        num_workers=args.workers,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    train(cfg)
