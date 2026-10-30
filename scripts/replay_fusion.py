"""CPU replay of the saved fusion head, not full VLA inference or catch evaluation."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture = torch.load(args.root / "artifacts/examples/fusion_replay.pt",
                         map_location="cpu", weights_only=True)
    weights = torch.load(args.root / "artifacts/models/fusion_head.pt",
                         map_location="cpu", weights_only=True)
    assert fixture["parent_sha256"] == weights["parent_sha256"]
    fusion = nn.Sequential(nn.LayerNorm(278), nn.Linear(278, 128), nn.SiLU(),
                           nn.Linear(128, 3)).eval()
    fusion.load_state_dict(weights["state"], strict=True)
    with torch.inference_mode():
        predicted = torch.tanh(fixture["base_logits"] + fusion(fixture["inputs"]))
    if predicted.shape[1:] != (12, 3) or not torch.isfinite(predicted).all():
        raise ValueError("Invalid action chunk")
    error = float((predicted - fixture["expected"]).abs().max())
    if error > 5e-6:
        raise ValueError(f"Fusion replay mismatch: {error}")
    print(json.dumps({"accepted": True, "device": "cpu",
                      "examples": len(predicted), "shape": list(predicted.shape),
                      "max_absolute_error": error,
                      "source_model_sha256": weights["parent_sha256"],
                      "scope": "Saved TRAIN features + base logits; fusion head only",
                      "full_VLA_inferences": 0, "new_catch_trials": 0}, indent=2))


if __name__ == "__main__":
    main()
