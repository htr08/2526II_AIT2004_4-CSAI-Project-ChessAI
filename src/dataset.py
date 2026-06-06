"""Dataset và DataLoader cho supervised data, self-play data và mixed replay buffer."""

import csv
import datetime
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, random_split


class ChessDataset(Dataset):
    """Dataset bọc file .pt đã parse (X: board tensor, y: move index từ vocab)."""

    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.X: torch.Tensor = data["X"]  # (N, 17, 8, 8) float32
        self.y: torch.Tensor = data["y"]  # (N,)           int64
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {self.X.shape[0]} vs {self.y.shape[0]}"
            )
        print(f"Loaded {len(self.X):,} positions from '{pt_path}'")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _dataloader_kwargs(batch_size, num_workers, pin_memory):
    """Tạo kwargs chung cho DataLoader."""
    return dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


def get_dataloaders(
    pt_path: str,
    batch_size: int = 512,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    dataset = ChessDataset(pt_path)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    shared = _dataloader_kwargs(batch_size, num_workers, pin_memory)
    train_dl = DataLoader(train_ds, shuffle=True, **shared)
    val_dl = DataLoader(val_ds, shuffle=False, **shared)
    return train_dl, val_dl


class ChessDatasetWithValue(Dataset):
    """Như ChessDataset nhưng kèm thêm outcome (v ∈ {-1, 0, 1}). Trả về bộ 3 (X, y, v)."""

    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        if "v" not in data:
            raise ValueError(
                f"'{pt_path}' has no 'v' key. Re-run parse_pgn.py with --with_value."
            )
        self.X: torch.Tensor = data["X"]
        self.y: torch.Tensor = data["y"]
        self.v: torch.Tensor = data["v"].float()
        print(f"Loaded {len(self.X):,} positions (with value) from '{pt_path}'")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.v[idx]


def get_dual_supervised_dataloaders(
    pt_path: str,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader cho supervised training DualNet, trả về bộ 3 (X, y, v)."""
    dataset = ChessDatasetWithValue(pt_path)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    shared = _dataloader_kwargs(batch_size, num_workers, pin_memory)
    return DataLoader(train_ds, shuffle=True, **shared), DataLoader(val_ds, shuffle=False, **shared)


class SelfPlayDataset(Dataset):
    """Dataset bọc output self-play (X, policy, value). Nhận 1 hoặc nhiều file .pt."""

    def __init__(self, pt_path):
        # Accept a single path or a list of paths (replay-buffer window).
        paths = [pt_path] if isinstance(pt_path, (str, bytes)) else list(pt_path)
        Xs, Ps, Vs = [], [], []
        for p in paths:
            data = torch.load(p, map_location="cpu", weights_only=True)
            Xs.append(data["X"].float())       # (N, 17, 8, 8)
            Ps.append(data["policy"].float())  # (N, 4544) soft targets
            Vs.append(data["value"].float())   # (N,)
            print(f"  + {len(data['X']):,} positions from '{p}'")

        self.X      = torch.cat(Xs)
        self.policy = torch.cat(Ps)
        self.value  = torch.cat(Vs)

        n = len(self.X)
        if not (len(self.policy) == n and len(self.value) == n):
            raise ValueError(
                f"Length mismatch — X:{n}  policy:{len(self.policy)}  value:{len(self.value)}"
            )
        print(f"Loaded {n:,} self-play positions from {len(paths)} file(s)")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.policy[idx], self.value[idx]


def get_selfplay_dataloaders(
    pt_path: str,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    dataset = SelfPlayDataset(pt_path)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    shared = _dataloader_kwargs(batch_size, num_workers, pin_memory)
    train_dl = DataLoader(train_ds, shuffle=True,  **shared)
    val_dl   = DataLoader(val_ds,   shuffle=False, **shared)
    return train_dl, val_dl


class SupervisedAsAlphaZero(Dataset):
    """Bọc ChessDataset thành (X, policy, value=0) để trộn vào replay buffer self-play."""

    def __init__(self, pt_path: str, num_moves: int = 4544):
        self._ds = ChessDataset(pt_path)
        self.num_moves = num_moves

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        X, y = self._ds[idx]
        policy = torch.zeros(self.num_moves)
        policy[y] = 1.0
        return X, policy, torch.tensor(0.0)


def get_mixed_dataloaders(
    sup_pt: str,
    sp_pt: str,
    sp_ratio: float = 0.5,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader trộn supervised + self-play theo sp_ratio; val lấy từ self-play."""
    sup = SupervisedAsAlphaZero(sup_pt)
    sp  = SelfPlayDataset(sp_pt)

    # Hold out a val split from self-play data
    sp_val_n   = max(1, int(len(sp) * val_ratio))
    sp_train_n = len(sp) - sp_val_n
    sp_train, sp_val = random_split(sp, [sp_train_n, sp_val_n])

    # WeightedRandomSampler achieves desired sp_ratio over the concat dataset
    sup_w = (1.0 - sp_ratio) / max(len(sup), 1)
    sp_w  =        sp_ratio  / max(sp_train_n, 1)
    weights = [sup_w] * len(sup) + [sp_w] * sp_train_n
    sampler = WeightedRandomSampler(weights, len(sup) + sp_train_n, replacement=True)

    train_ds = ConcatDataset([sup, sp_train])
    shared   = _dataloader_kwargs(batch_size, num_workers, pin_memory)
    train_dl = DataLoader(train_ds, sampler=sampler, **shared)
    val_dl   = DataLoader(sp_val,   shuffle=False,   **shared)
    return train_dl, val_dl


def get_mixed_dataloader(sup_pt, sp_pt, sp_ratio=0.5, batch_size=256):
    """Wrapper 1 dataloader giữ lại để tương thích ngược."""
    train_dl, _ = get_mixed_dataloaders(sup_pt, sp_pt, sp_ratio=sp_ratio, batch_size=batch_size)
    return train_dl


def log_elo(csv_path, iteration, win_rate, est_elo):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([datetime.datetime.now().isoformat(),
                                 iteration, win_rate, est_elo])
