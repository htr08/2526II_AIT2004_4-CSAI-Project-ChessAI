"""
Build and save move vocabulary files needed by all other modules.

Outputs:
  data/processed/move2idx.json  — UCI string -> integer index (4544 entries)
  data/processed/idx2move.json  — integer index -> UCI string (reverse mapping for inference)

Run once before training: python -m scripts.build_vocab
"""

import json
from pathlib import Path
from src.vocab import load_or_build_move2idx, IDX2MOVE_PATH

if __name__ == "__main__":
    move2idx = load_or_build_move2idx()

    # Build and save the reverse mapping for inference (index → UCI string)
    idx2move = {str(v): k for k, v in move2idx.items()}
    out_path = Path(IDX2MOVE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(idx2move, f)
    print(f"Saved {len(idx2move)} entries -> {IDX2MOVE_PATH}")

    # Sanity check
    print(f"\nVocab size     : {len(move2idx)}")
    print(f'move2idx["e2e4"]: {move2idx["e2e4"]}')
    roundtrip = idx2move[str(move2idx["e2e4"])]
    print(f"idx2move round-trip: {roundtrip}")
    assert roundtrip == "e2e4", "Round-trip failed!"
    print("Round-trip OK")
