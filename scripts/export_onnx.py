"""Export DualNet sang ONNX và benchmark tốc độ so với PyTorch.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --out checkpoints/model.onnx
"""
import argparse
import time
import numpy as np
import torch
from src.model import DualNet

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
parser.add_argument("--out", default="checkpoints/model.onnx")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--runs", type=int, default=100)
args = parser.parse_args()

model = DualNet()
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
model.eval()

in_ch = model.stem[0].in_channels  # 17
dummy = torch.zeros(args.batch, in_ch, 8, 8)

# ── Export ────────────────────────────────────────────────────────────────────
torch.onnx.export(
    model, dummy, args.out,
    input_names=["board"],
    output_names=["policy", "value"],
    dynamic_axes={"board": {0: "batch"}},
    opset_version=14,
)
print(f"Exported to {args.out}  (in_channels={in_ch})")

# ── Verify outputs match ───────────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed — skipping verify/benchmark (pip install onnxruntime)")
    raise SystemExit(0)

sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
inp = dummy.numpy()

with torch.no_grad():
    pt_policy, pt_value = model(dummy)

ort_policy, ort_value = sess.run(None, {"board": inp})
max_diff_policy = np.abs(pt_policy.numpy() - ort_policy).max()
max_diff_value  = np.abs(pt_value.numpy()  - ort_value).max()
print(f"Max diff policy={max_diff_policy:.2e}  value={max_diff_value:.2e}")
assert max_diff_policy < 1e-4 and max_diff_value < 1e-4, "ONNX output mismatch!"
print("Outputs match.")

# ── Benchmark ─────────────────────────────────────────────────────────────────
def bench(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs * 1000  # ms/call

pt_ms  = bench(lambda: model(dummy), args.warmup, args.runs)
ort_ms = bench(lambda: sess.run(None, {"board": inp}), args.warmup, args.runs)

print(f"PyTorch : {pt_ms:.2f} ms/batch  (batch={args.batch})")
print(f"ONNX RT : {ort_ms:.2f} ms/batch  (batch={args.batch})")
print(f"Speedup : {pt_ms / ort_ms:.2f}x")
