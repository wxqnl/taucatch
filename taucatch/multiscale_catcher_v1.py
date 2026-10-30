"""Use the existing IK/torque servo with an independently delivered fast command."""
from .action_catcher import ActionCatcher
from .action_chunk import ChunkBuffer
from .multiscale_v1 import CommandBuffer


class MultiscaleCatcher(ActionCatcher):
    def __init__(self, *args, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = CommandBuffer(ChunkBuffer(self.spec), mode)
