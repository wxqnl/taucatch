"""Training checkpoint of the catch policy (CatchPolicy: SpikeVLA + ball locator) -> released model directory:
config.json, model.safetensors and README.md, loadable with taucatch.spikevla alone (policy.load_release).

The frozen text model is not exported: the instruction's text states are stored instead. The running-norm statistics
are checked and kept; their training counters are dropped. Needs the training code (vendor/spikevla_icml).

    CUDA_VISIBLE_DEVICES=0 python tools/export_release.py runs_v2/saim_r1_b601/ema_3000.pt \
        --out outputs/hf/spikevla-catch-b601
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORMAT = "taucatch-spikevla/1"
SUPPORTED = {      # the configuration taucatch.spikevla implements
    "text": {"encoder_type": "smoothspike_bert"},
    "vision": {"encoder_type": "sdtv3_19m", "image_size": 224, "position_embedding": "learned_patch",
               "encode_views_separately": True, "output_grid_size": 14},
    "interaction": {"backbone_type": "spike_text_only", "text_only_spike_projections": True,
                    "padding_strategy": "key_padding_mask", "cross_attention_type": "spike_sdsa",
                    "residual_style": "normalized", "text_attention_routing": "wta"},
    "action": {"head_type": "full_spike_state8_shiftmax_continuous_inputs", "continuous_state": True,
               "spike_state_encoding": "perdim_ann", "ifr_integrate": True, "ifr_shift": 2, "ifr_split_state": False,
               "ifr_locate": True, "ifr_spatial": 0, "ifr_norm_bits": 4, "ifr_sign_field": True,
               "ifr_context_levels": 0},
}
ABSENT = {"ifr_sign_topk": 0, "ifr_position_views": (), "ifr_hand_anchor": False}
POLICY = ("step", "frames", "no_velocity", "no_image", "anchor")
DROPPED = ("num_batches_tracked", "observed_tokens")      # training counters of the spiking running norms


def release_config(ck, instruction):
    c = ck["model_config"]
    for part, values in SUPPORTED.items():
        for k, v in values.items():
            if c[part][k] != v:
                raise ValueError(f"{part}.{k} = {c[part][k]!r}: not implemented in taucatch.spikevla")
    for k, v in ABSENT.items():
        if c["action"].get(k, v) not in (v, None, [], ()):
            raise ValueError(f"action.{k} is not implemented in taucatch.spikevla")
    if c["action"]["spike_timesteps"] != c["interaction"]["cross_timesteps"]:
        raise ValueError("fusion and head timesteps differ")
    for k in ("condition", "selective", "ball", "forecast", "intercept"):
        if ck.get(k):
            raise ValueError(f"{k} policies are not exported")
    if not ck.get("locator"):
        raise ValueError("only CatchPolicy checkpoints (with the ball locator) are exported")
    inter, action = c["interaction"], c["action"]
    model = {
        "instruction": instruction,
        "text": {"encoder": "SmoothSpike-BERT (frozen; its states of the instruction are stored)",
                 "timesteps": c["text"]["timesteps"], "padding_length": c["text"]["padding_length"]},
        "vision": {"encoder": "SDT-V3 19M", "image_size": 224, "num_views": c["vision"]["num_views"]},
        "interaction": {k: inter[k] for k in ("hidden_dim", "nheads", "num_layers", "dim_feedforward",
                                              "enhancer_inner_dim", "cross_timesteps", "wta_text_topk",
                                              "spike_norm_eps")},
        "action": {k: action[k] for k in ("state_dim", "action_dim", "horizon", "num_layers", "mlp_hidden_dim",
                                          "state_hidden_dim", "spike_timesteps")},
    }
    stats = {k: np.asarray(v, np.float32).tolist() for k, v in ck["stats"].items()}
    return {"format": FORMAT, "model": model, "policy": {k: ck.get(k, False) for k in POLICY}, "stats": stats}


def release_weights(trained, config):
    """trained: the eval CatchPolicy of the checkpoint (vendor model) -> the released state dict."""
    from taucatch.spikevla import SpikeVLA
    from taucatch.vla_catch.locator import CatchPolicy
    from turbovla.models.components.wta_operators import verify_running_norms
    verify_running_norms(trained.vla)
    source = trained.state_dict()
    device = next(trained.parameters()).device
    hidden, valid, sentence = trained.vla.text_encoder.encode_bert_hidden([config["model"]["instruction"]], device)
    text = {"vla.text_encoder.hidden": hidden, "vla.text_encoder.valid": valid,
            "vla.text_encoder.sentence": sentence}
    target = CatchPolicy(SpikeVLA(config["model"]), config["stats"]).state_dict()
    out = {}
    for k, v in target.items():
        w = text[k] if k in text else source[k]
        if w.shape != v.shape or w.dtype != v.dtype:
            raise ValueError(f"{k}: {tuple(w.shape)} {w.dtype} vs {tuple(v.shape)} {v.dtype}")
        out[k] = w.detach().cpu().clone().contiguous()
    left = [k for k in source if k not in out]
    unexpected = [k for k in left if not (k.startswith("vla.text_encoder.bert.") or k.endswith(DROPPED))]
    if unexpected:
        raise ValueError(f"weights the release would drop: {unexpected[:5]}")
    return out


CARD = """---
tags:
- robotics
- spiking-neural-networks
- vision-language-action
- event-camera
library_name: pytorch
---

# SpikeVLA catcher for the reBot B601-DM

SpikeVLA policy that catches an 8 m/s ball thrown from 4 m with a reBot Arm B601-DM and a five-finger hand, from two
event cameras only. Code, simulator and real-robot steps: [taucatch](https://github.com/wxqnl/taucatch).

| Same 300 tosses, 8 m/s, 4 m (simulation, 20 ms latency, 24 V motor limits) | Caught |
|---|---|
| SpikeVLA, actions from the action head | 84.7% |
| SpikeVLA ball track + expert planner | 91.3% |

| Fastest balls, 4 m, first 100 tosses | 8 m/s | 9 m/s | 10 m/s |
|---|---|---|---|
| Expert with the true ball state | 98% | 95% | 73% |
| SpikeVLA ball track + expert planner | 87% | 45% | 13% |
| SpikeVLA, actions from the action head | 84% | 49% | 22% |

## Model

- Input: two 224 x 224 trace frames (per pixel, the time since its latest event, decayed with 10 / 50 / 250 ms as
  the three channels) and the arm state.
- SDT-V3 19M spiking backbone; an in-model ball locator on its tokens, a ballistic track over the calls;
  spiking vision-language fusion; spiking action head with Spatially Grounded Spike Attention (SGSA).
- Output: a 12 x 20 ms chunk of end-effector targets and the hand command, every 4 ms.
- {params:.1f} M parameters. The text model is not included: the instruction "{instruction}" is fixed and its
  frozen SmoothSpike-BERT states are stored in the weights.
- Trained in MuJoCo on expert demonstrations and one DAgger round on its own tosses ({source}).

## Files

| File | |
|---|---|
| `config.json` | model configuration, policy flags, state / action normalisation |
| `model.safetensors` | weights (`vla.*` SpikeVLA, `locator.*` ball locator) |

## Use

```bash
git clone https://github.com/wxqnl/taucatch.git && cd taucatch
pip install -r requirements/vla_catch.txt
python tools/fetch_spikevla_deps.py         # SDT-V3 model source
tools/fetch_b601.sh                         # B601 URDF and meshes (simulation)
python -m taucatch.vla_catch.evaluate --ckpt {repo} ...   # see the taucatch README
```

```python
from taucatch.vla_catch.policy import load_release
policy, config = load_release("{repo}")    # locator.CatchPolicy, eval it on CUDA
```
"""


def main():
    from taucatch.vla_catch.evaluate import Policy
    from taucatch.vla_catch.policy import INSTRUCTION, set_numerics
    from safetensors.torch import save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Minko030/taucatch", help="Hugging Face repo id, for the model card")
    args = ap.parse_args()
    set_numerics()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    config = release_config(ck, INSTRUCTION)
    config["policy"]["source"] = "/".join(Path(args.ckpt).parts[-2:])
    trained = Policy(args.ckpt).model
    weights = release_weights(trained, config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / "model.safetensors"), metadata={"format": FORMAT})
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    params = sum(v.numel() for k, v in weights.items() if v.is_floating_point() and "running_" not in k
                 and not k.startswith("vla.text_encoder.hidden")) / 1e6
    (out / "README.md").write_text(CARD.format(params=params, instruction=INSTRUCTION, source=config["policy"]["source"],
                                               repo=args.repo))
    size = (out / "model.safetensors").stat().st_size / 2 ** 20
    print(f"{out}: {len(weights)} tensors, {params:.2f} M parameters, {size:.0f} MiB")


if __name__ == "__main__":
    main()
