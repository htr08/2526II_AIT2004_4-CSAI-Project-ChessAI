"""
refactor_checkpoints.py

Quét tất cả thư mục con trong Model_dir. Mỗi thư mục chứa sẵn raw_metadata.json
với key "file_id" hoặc "drive_file_id" trỏ tới checkpoint .pt thô trên Google Drive.

Script sẽ:
  1. Đọc raw_metadata.json -> lấy file_id
  2. Download checkpoint thô từ Drive (qua gdown) -> lưu tạm checkpoint_raw.pt
  3. Trích state_dict, loại prefix "module."/"_orig_mod."
  4. Lưu checkpoint_clean.pt
  5. Ghi log (checkpoint_log.txt, tuỳ chọn checkpoint_log.json) đè vào thư mục đó

Usage:
    python refactor_checkpoints.py --root Model_dir
    python refactor_checkpoints.py --root Model_dir --json-log
    python refactor_checkpoints.py --root Model_dir --keep-raw   # giữ lại checkpoint_raw.pt
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch


POSSIBLE_STATE_DICT_KEYS = [
    "model_state_dict",
    "model",
    "state_dict",
    "net",
    "net_state_dict",
    "policy_value_net",
    "model_state",
]

STRIP_PREFIXES = ["module.", "_orig_mod."]


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint không phải dict (type={type(ckpt)})")

    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, "<root>"

    for key in POSSIBLE_STATE_DICT_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            candidate = ckpt[key]
            if candidate and all(torch.is_tensor(v) for v in candidate.values()):
                return candidate, key

    for key, value in ckpt.items():
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value, key

    raise ValueError(f"Không tìm được state_dict hợp lệ. Top-level keys: {list(ckpt.keys())}")


def clean_param_keys(state_dict):
    cleaned = {}
    renamed_count = 0
    for k, v in state_dict.items():
        new_k = k
        for prefix in STRIP_PREFIXES:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        if new_k != k:
            renamed_count += 1
        cleaned[new_k] = v
    return cleaned, renamed_count


def parse_file_id(value: str) -> str:
    """Hỗ trợ cả URL Drive đầy đủ hoặc file_id thuần."""
    value = value.strip()
    if "drive.google.com" in value:
        if "/d/" in value:
            return value.split("/d/")[1].split("/")[0]
        if "id=" in value:
            return value.split("id=")[1].split("&")[0]
    return value


def get_file_id(raw_meta: dict):
    raw = raw_meta.get("file_id") or raw_meta.get("drive_file_id")
    if not raw:
        return None
    return parse_file_id(raw)


def download_from_drive(file_id: str, out_path: Path):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, str(out_path), quiet=False)
    if result is None:
        raise RuntimeError(f"gdown download thất bại cho file_id={file_id}")


def process_folder(child_dir: Path, keep_raw: bool) -> dict:
    info = {
        "folder": child_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "UNKNOWN",
    }

    meta_path = child_dir / "raw_metadata.json"
    if not meta_path.exists():
        info["status"] = "SKIP"
        info["reason"] = "Không tìm thấy raw_metadata.json"
        return info

    try:
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        info["status"] = "ERROR"
        info["reason"] = f"Lỗi khi đọc raw_metadata.json: {e!r}"
        return info

    file_id = get_file_id(raw_meta)
    if not file_id:
        info["status"] = "ERROR"
        info["reason"] = "raw_metadata.json không có key 'file_id' hoặc 'drive_file_id'"
        return info

    info["drive_file_id"] = file_id

    raw_path = child_dir / "checkpoint_raw.pt"
    try:
        download_from_drive(file_id, raw_path)
    except Exception as e:
        info["status"] = "ERROR"
        info["reason"] = f"Lỗi khi download từ Drive: {e!r}"
        return info

    try:
        raw_ckpt = torch.load(raw_path, map_location="cpu", weights_only=False)
    except Exception as e:
        info["status"] = "ERROR"
        info["reason"] = f"Lỗi khi torch.load checkpoint_raw.pt: {e!r}"
        return info

    info["original_top_level_keys"] = (
        list(raw_ckpt.keys()) if isinstance(raw_ckpt, dict) else "non-dict"
    )

    try:
        raw_state_dict, source_key = extract_state_dict(raw_ckpt)
    except Exception as e:
        info["status"] = "ERROR"
        info["reason"] = f"Lỗi khi trích state_dict: {e!r}"
        if not keep_raw:
            raw_path.unlink(missing_ok=True)
        return info

    info["state_dict_source_key"] = source_key

    clean_sd, renamed_count = clean_param_keys(raw_state_dict)
    info["renamed_keys_count"] = renamed_count

    out_path = child_dir / "checkpoint_clean.pt"
    try:
        torch.save(clean_sd, out_path)
    except Exception as e:
        info["status"] = "ERROR"
        info["reason"] = f"Lỗi khi torch.save: {e!r}"
        if not keep_raw:
            raw_path.unlink(missing_ok=True)
        return info

    if not keep_raw:
        raw_path.unlink(missing_ok=True)

    total_params = sum(v.numel() for v in clean_sd.values() if torch.is_tensor(v))
    total_bytes = sum(v.numel() * v.element_size() for v in clean_sd.values() if torch.is_tensor(v))

    info["status"] = "OK"
    info["output_file"] = out_path.name
    info["num_tensors"] = len(clean_sd)
    info["total_params"] = total_params
    info["total_size_mb"] = round(total_bytes / (1024 * 1024), 3)
    info["param_keys"] = [
        {"name": k, "shape": list(v.shape), "dtype": str(v.dtype), "numel": v.numel()}
        for k, v in clean_sd.items()
    ]

    return info


def write_log(child_dir: Path, info: dict):
    log_path = child_dir / "checkpoint_log.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("CHECKPOINT REFACTOR LOG")
    lines.append("=" * 70)
    lines.append(f"Folder        : {info['folder']}")
    lines.append(f"Timestamp     : {info['timestamp']}")
    lines.append(f"Status        : {info['status']}")

    if "drive_file_id" in info:
        lines.append(f"Drive file_id : {info['drive_file_id']}")

    if info["status"] in ("SKIP", "ERROR"):
        lines.append(f"Reason        : {info.get('reason', '')}")
    else:
        lines.append(f"Output file   : {info['output_file']}")
        lines.append(f"Original keys : {info['original_top_level_keys']}")
        lines.append(f"State_dict key: {info['state_dict_source_key']}")
        lines.append(f"Renamed keys  : {info['renamed_keys_count']} (stripped prefixes)")
        lines.append(f"Num tensors   : {info['num_tensors']}")
        lines.append(f"Total params  : {info['total_params']:,}")
        lines.append(f"Total size    : {info['total_size_mb']} MB")
        lines.append("")
        lines.append("-" * 70)
        lines.append("PARAMETER LIST (name | shape | dtype | numel)")
        lines.append("-" * 70)
        for p in info["param_keys"]:
            lines.append(f"{p['name']:<55} | {str(p['shape']):<22} | {p['dtype']:<12} | {p['numel']:,}")

    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")


def write_json_log(child_dir: Path, info: dict):
    json_path = child_dir / "checkpoint_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Download checkpoint thô từ Drive (qua raw_metadata.json), làm sạch state_dict, ghi log."
    )
    parser.add_argument("--root", type=str, default="Model_dir")
    parser.add_argument("--json-log", action="store_true")
    parser.add_argument("--keep-raw", action="store_true", help="Giữ lại checkpoint_raw.pt sau khi xử lý")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[!] Không tìm thấy thư mục root: {root.resolve()}")
        return

    print(f"Đang quét: {root.resolve()}")
    print("=" * 70)

    summary = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        info = process_folder(child, keep_raw=args.keep_raw)
        write_log(child, info)
        if args.json_log:
            write_json_log(child, info)

        summary.append(info)

        if info["status"] == "OK":
            print(
                f"[OK]   {child.name:<25} -> {info['output_file']} "
                f"({info['num_tensors']} tensors, {info['total_params']:,} params, {info['total_size_mb']} MB)"
            )
        elif info["status"] == "SKIP":
            print(f"[SKIP] {child.name:<25} -> {info['reason']}")
        else:
            print(f"[ERR]  {child.name:<25} -> {info['reason']}")

    print("=" * 70)
    ok = sum(1 for s in summary if s["status"] == "OK")
    skip = sum(1 for s in summary if s["status"] == "SKIP")
    err = sum(1 for s in summary if s["status"] == "ERROR")
    print(f"Tổng kết: {ok} OK, {skip} SKIP, {err} ERROR / {len(summary)} thư mục")


if __name__ == "__main__":
    main()