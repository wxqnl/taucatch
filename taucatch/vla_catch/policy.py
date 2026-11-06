"""SpikeVLA catch policy. Released models load with taucatch.spikevla alone (load_release); training builds on the
ICML TurboVLA code (vendor/spikevla_icml, not published) with one of three starts.

    v1       v1 pretrained weights, v1 action head
    v1_sgsa  v1 pretrained weights, SGSA read-out (F7 ifr settings, learned patch positions; new parts fresh)
    f7       F7 (SGSA final model) weights
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/spikevla_icml"
CHECKPOINTS = {name: str(ROOT / f"vendor/checkpoints/spikevla_{name}.pth") for name in ("v1", "f7")}   # pretrained starts, not published
SGSA_KEYS = ("ifr_integrate", "ifr_shift", "ifr_locate", "ifr_locate_grid", "ifr_locate_views", "ifr_norm_bits",
             "ifr_sign_field", "ifr_context_levels", "ifr_spatial", "ifr_split_state")
INSTRUCTION = "catch the ball"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def set_numerics():
    """Spike thresholds are sensitive to GEMM roundoff: full fp32, as the checkpoints were trained."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = ck.get("ema_model_state_dict") or ck["model_state_dict"]
    return ck["model_config"], {k.removeprefix("module."): v for k, v in weights.items()}


def model_config(arm):
    if arm == "f7":
        return load_checkpoint(CHECKPOINTS["f7"])
    config, weights = load_checkpoint(CHECKPOINTS["v1"])
    config = copy.deepcopy(config)
    if arm == "v1_sgsa":
        f7, _ = load_checkpoint(CHECKPOINTS["f7"])
        config["action"].update({k: f7["action"][k] for k in SGSA_KEYS})
        config["vision"]["position_embedding"] = "learned_patch"
    elif arm != "v1":
        raise ValueError(arm)
    return config, weights


def build_model(config, weights=None, condition=False):
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import build_turbovla
    config = copy.deepcopy(config)
    config["vision"]["model_source_path"] = None          # the vendored SDT-V3 source
    model = build_turbovla(TurboVLAConfig.from_mapping(config))
    if condition:
        add_query_condition(model)
    result = None
    if weights is not None:
        shapes = {k: v.shape for k, v in model.state_dict().items()}
        # v1 -> learned_patch keeps the per-view embedding, stored as [1, V, C] instead of [1, V, 1, C].
        weights = {k: v.reshape(shapes[k]) if k in shapes and v.shape != shapes[k] and v.numel() == shapes[k].numel()
                   else v for k, v in weights.items()}
        # A wider tensor along one axis (extra action channels, extra state tokens) keeps the trained part;
        # the new part starts fresh.
        init = model.state_dict()
        for k, v in list(weights.items()):
            if k not in shapes or v.shape == shapes[k] or v.ndim != len(shapes[k]):
                continue
            grow = [d for d in range(v.ndim) if v.shape[d] != shapes[k][d]]
            if len(grow) == 1 and v.shape[grow[0]] < shapes[k][grow[0]]:
                wide = init[k].clone()
                wide.narrow(grow[0], 0, v.shape[grow[0]]).copy_(v)
                weights[k] = wide
        result = model.load_state_dict(weights, strict=False)
    cache_text(model)
    return model, result


def add_query_condition(model):
    """The state also enters every action query as a constant current (state MLP, zero-initialised output).

    Through the cross-attention read-out alone, the head leaves the seven ball tokens unread: after 2000 steps a
    5 cm shift of the ball moves the chunk by 0.00 of it, and the policy catches 5% (a plain MLP on the same
    state catches 58%). The queries carry the state into the decoder's residual stream directly."""
    head = model.action_head
    dim = head.decoder.action_queries.weight.shape[1]
    head.query_condition = nn.Sequential(nn.Linear(head.state_dim, head.config.state_hidden_dim), nn.GELU(),
                                         nn.Linear(head.config.state_hidden_dim, dim))
    nn.init.zeros_(head.query_condition[2].weight)
    nn.init.zeros_(head.query_condition[2].bias)
    _condition_queries()


def _condition_queries():
    from turbovla.models.spike_action_head import SpikeACTDecoder, SpikeTurboVLAActionHead
    if getattr(SpikeACTDecoder, "_catch_condition", False):
        return
    head_forward = SpikeTurboVLAActionHead._forward

    def _forward(self, memory, state, memory_key_padding_mask, anchor=None):
        if not hasattr(self, "query_condition"):
            return head_forward(self, memory, state, memory_key_padding_mask, anchor)
        self.decoder.query_bias = self.query_condition(state)
        try:
            return head_forward(self, memory, state, memory_key_padding_mask, anchor)
        finally:
            self.decoder.query_bias = None

    def forward(self, memories, memory_valid, anchor=None):        # SpikeACTDecoder.forward with the query bias
        queries = self.action_queries.weight[None].expand(memories.shape[1], -1, -1)
        bias = getattr(self, "query_bias", None)
        if bias is not None:
            queries = queries + bias[:, None]
        sequence = queries[None].expand(memories.shape[0], -1, -1, -1)
        for layer in self.layers:
            state = {} if anchor is None else {"cross": {"anchor": anchor}}
            sequence = torch.stack([
                layer.forward_step(sequence[t], memories[t], memory_valid, state, t)
                for t in range(memories.shape[0])
            ])
        state = {}
        currents = [self.action_projection.forward_step(sequence[t], state, t) for t in range(memories.shape[0])]
        return torch.tanh(torch.stack(currents).mean(0))
    SpikeTurboVLAActionHead._forward = _forward
    SpikeACTDecoder.forward = forward
    SpikeACTDecoder._catch_condition = True


def cache_text(model):
    """The BERT text backbone is frozen and deterministic: encode each instruction once."""
    encoder = model.text_encoder
    encode = encoder.encode_bert_hidden
    cache = {}

    def cached(instructions, device):
        rows = []
        for text in instructions:
            if text not in cache:
                mode = encoder.bert.training
                encoder.bert.eval()
                with torch.no_grad():
                    cache[text] = tuple(x.detach() for x in encode([text], device))
                encoder.bert.train(mode)
            rows.append(cache[text])
        return tuple(torch.cat([r[i] for r in rows]) for i in range(3))

    encoder.encode_bert_hidden = cached


def release_dir(path):
    """A released model: a local directory with config.json and model.safetensors, or a Hugging Face repo id."""
    if (Path(path) / "config.json").is_file():
        return Path(path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(path)))


def load_release(path):
    """Released catch policy -> (locator.CatchPolicy around taucatch.spikevla.SpikeVLA, its config.json)."""
    from safetensors.torch import load_file

    from taucatch.spikevla import SpikeVLA, read_config
    from .locator import CatchPolicy
    directory = release_dir(path)
    config = read_config(directory)
    model = CatchPolicy(SpikeVLA(config["model"]), config["stats"])
    model.load_state_dict(load_file(str(directory / "model.safetensors")))
    return model, config


def pixel_values(frames):
    """uint8 [B, V, H, W, 3] -> normalized float [B, V, 3, H, W]."""
    x = frames.permute(0, 1, 4, 2, 3).float() / 255.0
    mean = x.new_tensor(IMAGE_MEAN)[:, None, None]
    std = x.new_tensor(IMAGE_STD)[:, None, None]
    return (x - mean) / std
