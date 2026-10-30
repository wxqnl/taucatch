"""Only RGB/text encode before release; unchanged exact fused live DVS/head."""
import torch

from .chooser import to_pixels
from .xyz_timescale_runtime_v173 import TimescaleRuntime


class PrethrowRuntime(TimescaleRuntime):
    def prepare_condition(self, text, sample):
        self.reset_episode()
        pixels = to_pixels(sample['over'][None], sample['mosaic'][None],
                           self.engine.prepare, torch.device('cuda:0'))
        self.engine.model.vla.encode_condition([text], {'pixel_values': pixels},
                                              return_distillation_features=True)
        torch.cuda.synchronize()
        return self.condition.clone(), self.padding.clone()

    def release_condition(self, rgb_t, observed_t):
        if not rgb_t <= observed_t < 0:
            raise ValueError('Condition must actually predate release')
        self.anchor_rgb_t, self.anchor_observed_t = float(rgb_t), float(observed_t)
