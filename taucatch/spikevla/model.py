"""SpikeVLA for ball catching (inference): SDT-V3 vision backbone, spiking projections and vision-language fusion, the
spiking action head with SGSA cross-attention.

The text side is fixed: the frozen SmoothSpike-BERT states of the one instruction the policy was trained on are stored
in the weights (`text_encoder.hidden`, `valid`, `sentence`), so no text model is needed at inference.
"""
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

from .action_head import ActionHead
from .fusion import TextProjection, VisionLanguageInteraction, VisionProjection

ROOT = Path(__file__).resolve().parents[2]
SDTV3_SOURCE = ROOT / "vendor/sdt_v3/models.py"            # tools/fetch_spikevla_deps.py


class SDTV3Encoder(nn.Module):
    """The 19M Spike-driven Transformer V3 without its classifier: images -> the final 14 x 14 x 360 feature map."""

    hidden_size, num_patches = 360, 196
    batch_views = False          # both views in one backbone pass (fast inference; same features)

    def __init__(self):
        super().__init__()
        if not SDTV3_SOURCE.is_file():
            raise FileNotFoundError(f"{SDTV3_SOURCE} missing: run python tools/fetch_spikevla_deps.py")
        spec = importlib.util.spec_from_file_location("sdt_v3_models", SDTV3_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.backbone = module.Efficient_Spiking_Transformer_l()
        self.backbone.head = nn.Identity()

    def forward(self, pixel_values):
        """[B, V, 3, 224, 224] -> tokens [B, V, 196, 360]; each view on its own."""
        if self.batch_views:
            tokens = self.encode(pixel_values.flatten(0, 1))
            return tokens.view(*pixel_values.shape[:2], self.num_patches, self.hidden_size)
        return torch.stack([self.encode(pixel_values[:, v]) for v in range(pixel_values.shape[1])], dim=1)

    def encode(self, images):
        return self.backbone.forward_features(images).flatten(2).transpose(1, 2).contiguous()


class TextCache(nn.Module):
    """The instruction's text encoder output, fixed: hidden [1, N, 768], valid [1, N], sentence [1, N, N]."""

    def __init__(self, instruction, length, width, dim, timesteps, eps):
        super().__init__()
        self.instruction = instruction
        self.register_buffer("hidden", torch.zeros(1, length, width))
        self.register_buffer("valid", torch.zeros(1, length, dtype=torch.bool))
        self.register_buffer("sentence", torch.zeros(1, length, length, dtype=torch.bool))
        self.text_projection = TextProjection(width, dim, timesteps, eps)

    def check(self, instructions):
        if any(text != self.instruction for text in instructions):
            raise ValueError(f"this model holds the text of {self.instruction!r} only")


class SpikeVLA(nn.Module):
    """forward(instructions, pixel_values [B, 2, 3, 224, 224], state [B, S] normalised) -> chunk [B, 12, 7] in
    [-1, 1]. Each observation is encoded on its own (batch 1), as in training's evaluation; the action head runs on
    the whole batch."""

    release = True

    def __init__(self, config):
        super().__init__()
        vision, text, inter, action = config["vision"], config["text"], config["interaction"], config["action"]
        dim, eps, self.timesteps = inter["hidden_dim"], inter["spike_norm_eps"], inter["cross_timesteps"]
        self.views, grid = vision["num_views"], 14
        self.text_encoder = TextCache(config["instruction"], text["padding_length"], 768, dim, self.timesteps, eps)
        self.vision_encoder = SDTV3Encoder()
        width = self.vision_encoder.hidden_size
        self.vision_projection = VisionProjection(width, dim, max(dim * 4, width // 2), self.timesteps, eps)
        self.view_embedding = nn.Parameter(torch.zeros(1, self.views, 1, dim))
        self.patch_position_embedding = nn.Parameter(torch.zeros(1, self.views, grid * grid, dim))
        self.patch_position_scale = nn.Parameter(torch.zeros(1, self.views, 1, 1))
        self.vision_language_interaction = VisionLanguageInteraction(
            dim, inter["enhancer_inner_dim"], inter["nheads"], inter["num_layers"], inter["wta_text_topk"],
            self.timesteps, eps)
        self.action_head = ActionHead(action, dim, inter["nheads"], inter["dim_feedforward"], eps, self.views, grid)
        self._vision_rows = None

    def _encode_vision_tokens(self, pixel_values):
        """Backbone tokens [B, V, 196, 360] and None (no stage-2 blocks). A caller that already ran the backbone
        (locator.CatchPolicy) hands its rows over in `_vision_rows`, one per observation."""
        if self._vision_rows:
            tokens, _ = self._vision_rows.pop(0)
            if tokens.shape[0] != pixel_values.shape[0]:
                raise RuntimeError("cached encoder rows do not match the call")
            return tokens, None
        return self.vision_encoder(pixel_values), None

    def encode_condition(self, pixel_values):
        """One observation [1, V, 3, H, W] -> fusion output [1, V * 196 + N, C]."""
        text = self.text_encoder
        raw, _ = self._encode_vision_tokens(pixel_values)
        valid = text.valid[..., None]
        v_state, l_state, visual, language = {}, {}, [], []
        for t in range(self.timesteps):
            current = self.vision_projection.forward_step(raw, v_state, t)
            visual.append((current + self.patch_position_scale * self.patch_position_embedding
                           + self.view_embedding).flatten(1, 2))
            language.append(text.text_projection.forward_step(text.hidden, l_state, valid, t))
        visual, language = torch.stack(visual).mean(0), torch.stack(language).mean(0)
        visual, language = self.vision_language_interaction(visual, language, ~text.valid, ~text.sentence)
        return torch.cat([visual, language], dim=1)

    def forward(self, instructions, pixel_values, state):
        self.text_encoder.check(instructions)
        batch = pixel_values.shape[0]
        condition = torch.cat([self.encode_condition(pixel_values[i:i + 1]) for i in range(batch)])
        text_valid = self.text_encoder.valid.expand(batch, -1)
        padding = torch.cat((torch.zeros((batch, condition.shape[1] - text_valid.shape[1]), dtype=torch.bool,
                                         device=condition.device), ~text_valid), dim=1)
        return self.action_head(condition, state.float(), padding)


def read_config(directory):
    return json.loads((Path(directory) / "config.json").read_text())


def load(directory, device="cpu", prefix="vla."):
    """A released model directory (config.json, model.safetensors) -> (eval SpikeVLA, config); the weights under
    `prefix` (the catch policy keeps its ball locator beside them)."""
    from safetensors.torch import load_file
    config = read_config(directory)
    model = SpikeVLA(config["model"])
    weights = load_file(str(Path(directory) / "model.safetensors"))
    model.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)})
    return model.to(device).eval(), config
