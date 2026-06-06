"""Vẽ đồ thị kết quả training từ file log (ELO history, loss CSV).

Usage:
    # Sinh tất cả chart (mặc định)
    python scripts/plot_results.py

    # Chỉ sinh chart cụ thể
    python scripts/plot_results.py --mode supervised
    python scripts/plot_results.py --mode dual
    python scripts/plot_results.py --mode elo
    python scripts/plot_results.py --mode all   (default)

Output:
    reports/figures/supervised_loss.png
    reports/figures/dual_loss.png
    reports/figures/elo_progress.png
    reports/figures/training_progress.png  (combined, legacy)
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BLUE   = "#185FA5"
RED    = "#D4537E"
GREEN  = "#0F6E56"
ORANGE = "#EF9F27"
GRAY   = "#888888"

FIG_DIR = Path("reports/figures")


def _try_load_csv(path: str, **read_kwargs) -> "pd.DataFrame | None":
    """Đọc CSV, trả None nếu không tồn tại hoặc rỗng."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, **read_kwargs)
        return df if len(df) > 0 else None
    except Exception:
        return None


def _load_elo(path: str) -> "pd.DataFrame | None":
    # elo_history.csv không có header: timestamp,iter,win_rate,est_elo
    df = _try_load_csv(path, header=None, names=["timestamp", "iter", "win_rate", "est_elo"])
    if df is None:
        return None
    if not pd.api.types.is_numeric_dtype(df["iter"]):
        df = _try_load_csv(path)
        if df is None or "est_elo" not in df.columns:
            return None
    return df


def _no_data_ax(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color=GRAY, style="italic", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linestyle("--"); spine.set_alpha(0.4)


def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


# ── chart riêng lẻ ──────────────────────────────────────────────────────────

def plot_supervised_loss(loss_csv: str = "logs/supervised_loss.csv") -> None:
    """Loss curve của supervised training (PolicyNet)."""
    df = _try_load_csv(loss_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Supervised Training Loss (PolicyNet)", fontsize=13, fontweight="bold")
    if df is not None:
        ax.plot(df["epoch"], df["train_loss"], label="Train loss", color=BLUE, lw=2, marker="o", ms=5)
        ax.plot(df["epoch"], df["val_loss"],   label="Val loss",   color=RED,  lw=2, marker="s", ms=5, ls="--")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    else:
        _no_data_ax(ax, f"Không có dữ liệu: {loss_csv}")
    plt.tight_layout()
    _save(fig, FIG_DIR / "supervised_loss.png")
    plt.close(fig)


def plot_dual_loss(loss_csv: str = "logs/dual_supervised_loss.csv") -> None:
    """Loss curve của dual-net (policy + value heads)."""
    df = _try_load_csv(loss_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Dual-Net Training Loss (Policy & Value Heads)", fontsize=13, fontweight="bold")
    if df is not None:
        ax = axes[0]
        ax.plot(df["epoch"], df["train_p_loss"], label="Train", color=BLUE, lw=2, marker="o", ms=5)
        ax.plot(df["epoch"], df["val_p_loss"],   label="Val",   color=RED,  lw=2, marker="s", ms=5, ls="--")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Policy Head Loss")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(df["epoch"], df["train_v_loss"], label="Train", color=GREEN,  lw=2, marker="o", ms=5)
        ax.plot(df["epoch"], df["val_v_loss"],   label="Val",   color=ORANGE, lw=2, marker="s", ms=5, ls="--")
        ax.set(xlabel="Epoch", ylabel="MSE loss", title="Value Head Loss")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    else:
        for ax in axes:
            _no_data_ax(ax, f"Không có dữ liệu: {loss_csv}")
    plt.tight_layout()
    _save(fig, FIG_DIR / "dual_loss.png")
    plt.close(fig)


def plot_elo(elo_csv: str = "logs/elo_history.csv") -> None:
    """ELO ước tính theo từng iteration tự chơi."""
    df = _load_elo(elo_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("ELO Progress vs Stockfish", fontsize=13, fontweight="bold")
    if df is not None:
        ax.plot(df["iter"], df["est_elo"], "o-", color=BLUE, lw=2, ms=8)
        ax.fill_between(df["iter"], df["est_elo"], alpha=0.12, color=BLUE)
        ax.axhline(800,  ls="--", color=GRAY,   alpha=0.7, lw=1.2, label="SF skill 0 (~800)")
        ax.axhline(1000, ls="--", color=ORANGE, alpha=0.7, lw=1.2, label="SF skill 2 (~1000)")
        ax.axhline(1200, ls="--", color=RED,    alpha=0.7, lw=1.2, label="SF skill 4 (~1200)")
        for _, row in df.iterrows():
            ax.annotate(f"  iter {int(row['iter'])}\n  wr={row['win_rate']*100:.0f}%",
                        xy=(row["iter"], row["est_elo"]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=8, color=BLUE)
        ax.set_xlim(0.5, df["iter"].max() + 0.5)
        ax.set_ylim(0, max(1500, df["est_elo"].max() + 300))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set(xlabel="Iteration", ylabel="Estimated ELO")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    else:
        _no_data_ax(ax, f"Không có dữ liệu: {elo_csv}")
    plt.tight_layout()
    _save(fig, FIG_DIR / "elo_progress.png")
    plt.close(fig)


# ── chart tổng hợp (legacy) ──────────────────────────────────────────────────

def plot_combined(loss_csv: str | None, elo_csv: str, mode: str) -> None:
    """Sinh 1 ảnh tổng hợp (2 subplot): loss + ELO."""
    loss_df = _try_load_csv(loss_csv) if loss_csv else None
    elo_df  = _load_elo(elo_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Chess AI Training Progress", fontsize=13, fontweight="bold")

    if loss_df is not None:
        if mode == "dual":
            axes[0].plot(loss_df["epoch"], loss_df["train_p_loss"], label="Policy (train)", color=BLUE,  lw=2)
            axes[0].plot(loss_df["epoch"], loss_df["val_p_loss"],   label="Policy (val)",   color=BLUE,  lw=2, ls="--")
            axes[0].plot(loss_df["epoch"], loss_df["train_v_loss"], label="Value (train)",  color=GREEN, lw=2)
            axes[0].plot(loss_df["epoch"], loss_df["val_v_loss"],   label="Value (val)",    color=GREEN, lw=2, ls="--")
            axes[0].set(xlabel="Epoch", ylabel="Loss", title="Dual-net training loss")
        else:
            axes[0].plot(loss_df["epoch"], loss_df["train_loss"], label="Train", color=BLUE, lw=2)
            axes[0].plot(loss_df["epoch"], loss_df["val_loss"],   label="Val",   color=RED,  lw=2, ls="--")
            axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Supervised training loss")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    else:
        _no_data_ax(axes[0], "Không có dữ liệu loss")

    if elo_df is not None:
        axes[1].plot(elo_df["iter"], elo_df["est_elo"], "o-", color=BLUE, lw=2, ms=8)
        axes[1].fill_between(elo_df["iter"], elo_df["est_elo"], alpha=0.12, color=BLUE)
        axes[1].axhline(800,  ls="--", color=GRAY,   alpha=0.7, lw=1.2, label="SF skill 0")
        axes[1].axhline(1000, ls="--", color=ORANGE, alpha=0.7, lw=1.2, label="SF skill 2")
        axes[1].set_xlim(0.5, elo_df["iter"].max() + 0.5)
        axes[1].set_ylim(0, max(1500, elo_df["est_elo"].max() + 300))
        axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[1].set(xlabel="Iteration", ylabel="Estimated ELO", title="ELO vs Stockfish")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    else:
        _no_data_ax(axes[1], "Không có dữ liệu ELO")

    plt.tight_layout()
    _save(fig, FIG_DIR / "training_progress.png")
    plt.close(fig)


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["all", "supervised", "dual", "elo", "combined"],
                    default="all", help="Loại chart cần sinh")
    ap.add_argument("--loss",    default=None,                         help="Override đường dẫn loss CSV")
    ap.add_argument("--elo-csv", default="logs/elo_history.csv",       help="Override đường dẫn ELO CSV")
    args = ap.parse_args()

    if args.mode in ("all", "supervised"):
        plot_supervised_loss(args.loss or "logs/supervised_loss.csv")
    if args.mode in ("all", "dual"):
        plot_dual_loss(args.loss or "logs/dual_supervised_loss.csv")
    if args.mode in ("all", "elo"):
        plot_elo(args.elo_csv)
    if args.mode in ("all", "combined"):
        plot_combined(args.loss, args.elo_csv, "supervised")


if __name__ == "__main__":
    main()
