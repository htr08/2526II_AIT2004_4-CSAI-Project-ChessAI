"""Vòng huấn luyện PolicyNet (supervised) và DualNet (supervised, self-play, fine-tune value)."""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from pathlib import Path
from src.model import DualNet, PolicyNet, load_pretrained_backbone
from src.dataset import get_dataloaders, get_selfplay_dataloaders, get_mixed_dataloaders, get_dual_supervised_dataloaders
from src.vocab import load_or_build_move2idx

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_policy.pt"
LOG_DIR = Path("logs")


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    topk = logits.topk(k, dim=1).indices
    return topk.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()


def train(
    pt_path: str = "data/processed/train.pt",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    print(f"Training on {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    move2idx = load_or_build_move2idx()
    num_moves = max(move2idx.values()) + 1

    # Fail fast if the .pt file was built with a different vocab
    raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    y_max = raw["y"].max().item()
    if y_max >= num_moves:
        raise ValueError(
            f"train.pt has label {y_max} but vocab only covers 0–{num_moves - 1}. "
            "Re-run scripts/parse_pgn.py to regenerate train.pt with the current vocab."
        )
    del raw

    train_dl, val_dl = get_dataloaders(pt_path, batch_size=batch_size, pin_memory=use_cuda)
    model = PolicyNet(num_moves=num_moves).to(device)
    print(f"Vocab size: {num_moves}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(LOG_DIR / "supervised")) if _TB_AVAILABLE else None
    csv_path = LOG_DIR / "supervised_loss.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "train_top5", "val_top5"])

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = train_acc5 = 0.0
        for step, (X, y) in enumerate(train_dl):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc5 += topk_accuracy(logits, y, k=5)
            if step % 100 == 0:
                print(f"  E{epoch} step {step:>4d}  loss={loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dl)
        avg_train_acc5 = train_acc5 / len(train_dl)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = val_acc5 = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item()
                val_acc5 += topk_accuracy(logits, y, k=5)

        avg_val_loss = val_loss / len(val_dl)
        avg_val_acc5 = val_acc5 / len(val_dl)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch:>2d} | "
            f"train_loss={avg_train_loss:.4f}  train_top5={avg_train_acc5:.3f} | "
            f"val_loss={avg_val_loss:.4f}  val_top5={avg_val_acc5:.3f}"
        )

        csv_writer.writerow([epoch, avg_train_loss, avg_val_loss, avg_train_acc5, avg_val_acc5])
        csv_file.flush()
        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val",   avg_val_loss,   epoch)
            writer.add_scalar("Top5/train", avg_train_acc5, epoch)
            writer.add_scalar("Top5/val",   avg_val_acc5,   epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved new best (val_loss={best_val_loss:.4f})")

    csv_file.close()
    if writer:
        writer.close()
    print(f"Loss log saved → {csv_path}")

def train_dual_supervised(
    pt_path: str = "data/processed/train_with_value.pt",
    policy_checkpoint: str = "checkpoints/best_policy.pt",
    out_checkpoint: str = "checkpoints/best_dual.pt",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    value_weight: float = 1.0,
) -> None:
    """Huấn luyện DualNet supervised với cả policy head và value head (cần train.pt --with_value)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    print(f"Training DualNet (supervised) on {device}  value_weight={value_weight}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = DualNet(in_ch=17, num_moves=4544).to(device)
    if Path(policy_checkpoint).exists():
        load_pretrained_backbone(model, policy_checkpoint)
    else:
        print(f"  [warn] {policy_checkpoint} not found — training from scratch")

    train_dl, val_dl = get_dual_supervised_dataloaders(
        pt_path, batch_size=batch_size, pin_memory=use_cuda
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(LOG_DIR / "dual_supervised")) if _TB_AVAILABLE else None
    csv_path = LOG_DIR / "dual_supervised_loss.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_p_loss", "train_v_loss", "val_p_loss", "val_v_loss"])

    for epoch in range(epochs):
        model.train()
        train_p, train_v = 0.0, 0.0
        for X, y, v in train_dl:
            X, y, v = X.to(device), y.to(device), v.to(device)
            p_logits, v_pred = model(X)
            # one-hot policy target
            p_target = torch.zeros(len(y), 4544, device=device)
            p_target.scatter_(1, y.unsqueeze(1), 1.0)
            p_loss = -(p_target * F.log_softmax(p_logits, dim=-1)).sum(dim=-1).mean()
            v_loss = F.mse_loss(v_pred, v)
            loss = p_loss + value_weight * v_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_p += p_loss.item()
            train_v += v_loss.item()

        model.eval()
        val_p, val_v = 0.0, 0.0
        with torch.no_grad():
            for X, y, v in val_dl:
                X, y, v = X.to(device), y.to(device), v.to(device)
                p_logits, v_pred = model(X)
                p_target = torch.zeros(len(y), 4544, device=device)
                p_target.scatter_(1, y.unsqueeze(1), 1.0)
                val_p += -(p_target * F.log_softmax(p_logits, dim=-1)).sum(dim=-1).mean().item()
                val_v += F.mse_loss(v_pred, v).item()

        n, m = len(train_dl), len(val_dl)
        avg_tp, avg_tv = train_p / n, train_v / n
        avg_vp, avg_vv = val_p / m, val_v / m
        avg_val_total = avg_vp + value_weight * avg_vv
        scheduler.step(avg_val_total)

        print(
            f"Epoch {epoch:>2d} | "
            f"train p={avg_tp:.4f} v={avg_tv:.4f} | "
            f"val   p={avg_vp:.4f} v={avg_vv:.4f}"
        )
        csv_writer.writerow([epoch, avg_tp, avg_tv, avg_vp, avg_vv])
        csv_file.flush()
        if writer:
            writer.add_scalar("Loss/policy_train", avg_tp, epoch)
            writer.add_scalar("Loss/value_train",  avg_tv, epoch)
            writer.add_scalar("Loss/policy_val",   avg_vp, epoch)
            writer.add_scalar("Loss/value_val",    avg_vv, epoch)

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            torch.save(model.state_dict(), out_checkpoint)
            print(f"  -> Saved best → {out_checkpoint}")

    csv_file.close()
    if writer:
        writer.close()
    print(f"Loss log saved → {csv_path}")


def resolve_replay_window(data_path: str, window: int = 1) -> list[str]:
    """Trả về danh sách `window` file iter_*.pt gần nhất để làm sliding-window replay buffer."""
    if window <= 1:
        return [data_path]
    files = sorted(glob(str(Path(data_path).parent / "iter_*.pt")))
    if not files:
        return [data_path]
    return files[-window:]


def train_selfplay(
    sp_path,
    policy_checkpoint: str = "checkpoints/best_policy.pt",
    out_checkpoint: str = "checkpoints/best_dual.pt",
    sup_path: str = None,
    sp_ratio: float = 0.5,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 256,
    value_weight: float = 1.0,
) -> None:
    """Huấn luyện DualNet trên self-play data. Nếu có sup_path thì trộn supervised để tránh catastrophic forgetting."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    print(f"Training DualNet on {device}  value_weight={value_weight}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = DualNet(in_ch=17, num_moves=4544).to(device)
    if Path(policy_checkpoint).exists():
        load_pretrained_backbone(model, policy_checkpoint)
    else:
        print(f"  [warn] {policy_checkpoint} not found — training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    if sup_path and Path(sup_path).exists():
        print(f"Mixed training: sp_ratio={sp_ratio}  sup={sup_path}")
        train_dl, val_dl = get_mixed_dataloaders(
            sup_path, sp_path, sp_ratio=sp_ratio,
            batch_size=batch_size, pin_memory=use_cuda,
        )
    else:
        train_dl, val_dl = get_selfplay_dataloaders(
            sp_path, batch_size=batch_size, pin_memory=use_cuda
        )

    best_val_loss = float("inf")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(LOG_DIR / "selfplay")) if _TB_AVAILABLE else None
    csv_path = LOG_DIR / "selfplay_loss.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_policy_loss", "train_value_loss", "val_policy_loss", "val_value_loss"])

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_p_loss = train_v_loss = train_loss = 0.0

        for X, p_target, v_target in train_dl:
            X        = X.to(device)
            p_target = p_target.to(device)   # (B, 4544) soft targets
            v_target = v_target.to(device)   # (B,)

            p_logits, v_pred = model(X)

            # Soft cross-entropy: −Σ p_target * log_softmax(p_logits)
            p_loss = -(p_target * F.log_softmax(p_logits, dim=-1)).sum(dim=-1).mean()
            v_loss = F.mse_loss(v_pred, v_target)
            loss   = p_loss + value_weight * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_p_loss += p_loss.item()
            train_v_loss += v_loss.item()
            train_loss   += loss.item()

        n = len(train_dl)
        avg_p = train_p_loss / n
        avg_v = train_v_loss / n
        avg_t = train_loss   / n

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_p_loss = val_v_loss = val_loss = 0.0

        with torch.no_grad():
            for X, p_target, v_target in val_dl:
                X        = X.to(device)
                p_target = p_target.to(device)
                v_target = v_target.to(device)
                p_logits, v_pred = model(X)
                vp = -(p_target * F.log_softmax(p_logits, dim=-1)).sum(dim=-1).mean()
                vv = F.mse_loss(v_pred, v_target)
                val_p_loss += vp.item()
                val_v_loss += vv.item()
                val_loss   += (vp + value_weight * vv).item()

        m = len(val_dl)
        avg_val_p = val_p_loss / m
        avg_val_v = val_v_loss / m
        avg_val   = val_loss   / m
        scheduler.step(avg_val)

        print(
            f"Epoch {epoch:>2d} | "
            f"train  p={avg_p:.4f}  v={avg_v:.4f}  total={avg_t:.4f} | "
            f"val    p={avg_val_p:.4f}  v={avg_val_v:.4f}  total={avg_val:.4f}"
        )

        csv_writer.writerow([epoch, avg_p, avg_v, avg_val_p, avg_val_v])
        csv_file.flush()
        if writer:
            writer.add_scalar("Loss/policy_train", avg_p,     epoch)
            writer.add_scalar("Loss/value_train",  avg_v,     epoch)
            writer.add_scalar("Loss/policy_val",   avg_val_p, epoch)
            writer.add_scalar("Loss/value_val",    avg_val_v, epoch)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), out_checkpoint)
            print(f"  -> Saved best (val_loss={best_val_loss:.4f}) → {out_checkpoint}")

    csv_file.close()
    if writer:
        writer.close()
    print(f"Loss log saved → {csv_path}")


def finetune_value_head(
    pt_path: str,
    checkpoint: str = "checkpoints/best_dual.pt",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> None:
    """Fine-tune chỉ value head của DualNet; backbone và policy head bị đóng băng."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning value head on {device}  (backbone+policy frozen)")

    model = DualNet(in_ch=17, num_moves=4544).to(device)
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)

    # Freeze everything except value_head
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("value_head")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} (value head only)")

    train_dl, val_dl = get_dual_supervised_dataloaders(
        pt_path, batch_size=batch_size, pin_memory=(device == "cuda")
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val_loss = float("inf")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(LOG_DIR / "finetune_value")) if _TB_AVAILABLE else None

    for epoch in range(epochs):
        model.train()
        train_v = 0.0
        for X, _, v in train_dl:
            X, v = X.to(device), v.to(device)
            _, v_pred = model(X)
            loss = F.mse_loss(v_pred, v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_v += loss.item()

        model.eval()
        val_v = 0.0
        with torch.no_grad():
            for X, _, v in val_dl:
                X, v = X.to(device), v.to(device)
                _, v_pred = model(X)
                val_v += F.mse_loss(v_pred, v).item()

        avg_t = train_v / len(train_dl)
        avg_v = val_v   / len(val_dl)
        print(f"Epoch {epoch:>2d} | train_v={avg_t:.4f}  val_v={avg_v:.4f}")

        if writer:
            writer.add_scalar("ValueFinetune/train", avg_t, epoch)
            writer.add_scalar("ValueFinetune/val",   avg_v, epoch)

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), checkpoint)
            print(f"  -> Saved best (val_v={best_val_loss:.4f}) → {checkpoint}")

    if writer:
        writer.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["supervised", "selfplay", "dual_supervised", "finetune_value"], default="supervised")
    p.add_argument("--data", default="data/processed/train.pt")
    p.add_argument("--sup_data", default=None, help="Supervised .pt for 50/50 mixing (selfplay mode)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", default=None, help="Output checkpoint path (selfplay mode)")
    p.add_argument("--sp_ratio", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--window", type=int, default=1,
                   help="Replay-buffer window: train on the last N iter_*.pt files (selfplay mode)")
    p.add_argument("--value_weight", type=float, default=1.0,
                   help="Weight for value loss (dual_supervised mode)")
    args = p.parse_args()

    if args.mode == "supervised":
        train(pt_path=args.data, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "dual_supervised":
        train_dual_supervised(
            pt_path=args.data,
            out_checkpoint=args.out or "checkpoints/best_dual.pt",
            epochs=args.epochs,
            batch_size=args.batch_size,
            value_weight=args.value_weight,
        )
    elif args.mode == "finetune_value":
        finetune_value_head(
            pt_path=args.data,
            checkpoint=args.out or "checkpoints/best_dual.pt",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-4,
        )
    else:
        sp_data = resolve_replay_window(args.data, window=args.window)
        print(f"Replay buffer ({len(sp_data)} file(s)): {sp_data}")
        train_selfplay(
            sp_path=sp_data,
            epochs=args.epochs,
            out_checkpoint=args.out or "checkpoints/best_dual.pt",
            sup_path=args.sup_data,
            sp_ratio=args.sp_ratio,
            batch_size=args.batch_size,
        )
