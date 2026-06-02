"""
self_play_loop.py
-----------------
Vòng lặp self-play hoàn chỉnh:
1. Sinh self-play data với model hiện tại
2. Retrain copy của model trên data đó (mixed loss)
3. Pit model mới vs cũ — nếu mới ≥55% thì giữ
4. Lặp lại N iterations

Đây là phần "AI" thực sự của project — chính là vòng lặp đặc trưng của AlphaZero.
"""
from __future__ import annotations

import copy
import pathlib
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ...model.network import PolicyValueNet
from .self_play import SelfPlayConfig, generate_self_play_data
from .pit import play_match
from .elo import update_elo


@dataclass
class LoopConfig:
    num_iterations: int = 5
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train_epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    pit_games: int = 10
    pit_simulations: int = 100
    pit_num_parallel: int = 1        # batch inference cho MCTS lúc pit (1 = như cũ)
    adjudication_margin: int = 100   # material edge để chấm thắng khi game chạm max_moves
    accept_threshold: float = 0.55
    output_dir: str = "models/selfplay"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _train_on_buffer(
    model: PolicyValueNet,
    states: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    cfg: LoopConfig,
) -> dict:
    """Train model trên data từ self-play với loss = CE(policy) + MSE(value)."""
    model.train()
    ds = TensorDataset(states, policies, values)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    metrics = {"policy_loss": 0.0, "value_loss": 0.0}
    n_batches = 0
    for epoch in range(cfg.train_epochs):
        for x, p_target, v_target in loader:
            x = x.to(cfg.device)
            p_target = p_target.to(cfg.device)
            v_target = v_target.to(cfg.device)

            logits, v_pred = model(x)
            # Cross-entropy với soft target (MCTS distribution)
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(p_target * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(v_pred, v_target)
            loss = policy_loss + cfg.value_weight * value_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metrics.items()}


def run_self_play_loop(
    initial_model_path: str | None,
    cfg: LoopConfig,
) -> dict:
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load hoặc khởi tạo model
    current = PolicyValueNet()
    if initial_model_path:
        ckpt = torch.load(initial_model_path, map_location=cfg.device, weights_only=False)
        current.load_state_dict(ckpt["model_state"])
        print(f"[loop] loaded initial model from {initial_model_path}")
    current.to(cfg.device)

    cfg.self_play.device = cfg.device

    history = []
    current_elo = 0.0  # ELO tương đối, model ban đầu = 0
    for it in range(1, cfg.num_iterations + 1):
        print(f"\n=== Iteration {it}/{cfg.num_iterations} ===")
        t0 = time.time()

        # 1. Self-play
        print(f"[loop] self-play {cfg.self_play.num_games} games...")
        data = generate_self_play_data(current, cfg.self_play)

        # 2. Retrain — copy model trước, train trên copy
        candidate = copy.deepcopy(current)
        print(f"[loop] retraining candidate on {len(data['states'])} samples...")
        train_metrics = _train_on_buffer(
            candidate, data["states"], data["policies"], data["values"], cfg
        )

        # 3. Pit
        print(f"[loop] pit candidate vs current ({cfg.pit_games} games)...")
        pit_result = play_match(
            candidate,
            current,
            num_games=cfg.pit_games,
            num_simulations=cfg.pit_simulations,
            device=cfg.device,
            adjudication_margin=cfg.adjudication_margin,
            num_parallel=cfg.pit_num_parallel,
        )
        print(
            f"[loop] pit: new {pit_result['new_wins']} - "
            f"{pit_result['old_wins']} old (draws {pit_result['draws']}), "
            f"decisive_win_rate={pit_result['decisive_win_rate']:.2f}"
        )

        accepted = pit_result["decisive_win_rate"] >= cfg.accept_threshold
        if accepted:
            current = candidate
            print(f"[loop] ACCEPT new model")
        else:
            print(f"[loop] REJECT new model, giữ model cũ")

        # ELO tương đối: cập nhật theo kết quả pit
        elo = update_elo(
            current_elo,
            pit_result["new_wins"],
            pit_result["old_wins"],
            pit_result["draws"],
            accepted,
        )
        current_elo = elo["current_elo"]
        print(
            f"[loop] ELO: candidate {elo['candidate_elo']:+.0f} "
            f"(Δ{elo['elo_diff']:+.0f}) → kept {current_elo:+.0f}"
        )

        dt = time.time() - t0
        torch.save(
            {
                "model_state": current.state_dict(),
                "iteration": it,
                "pit": pit_result,
                "train_metrics": train_metrics,
                "accepted": accepted,
                "elo": current_elo,
                "elo_detail": elo,
            },
            output_dir / f"iter_{it:03d}.pt",
        )
        torch.save(
            {"model_state": current.state_dict(), "iteration": it, "elo": current_elo},
            output_dir / "latest.pt",
        )

        history.append(
            {
                "iter": it,
                "train": train_metrics,
                "pit": pit_result,
                "accepted": accepted,
                "elo": current_elo,
                "elo_detail": elo,
                "time_s": dt,
            }
        )
        print(f"[loop] iteration done in {dt:.0f}s")

    return {"history": history, "final_elo": current_elo}
