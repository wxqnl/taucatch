"""Fixed-shape strictFP32 CUDA graph of the unchanged learned event finder."""
import torch


class FinderGraph:
    def __init__(self, model):
        self.model = model
        self.input = torch.zeros((2, 6, 120, 160), device='cuda', dtype=torch.float32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self.output = model(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.output = model(self.input)

    @torch.inference_mode()
    def __call__(self, grid):
        if grid.shape != self.input.shape or grid.dtype != torch.float32 or not grid.is_cuda:
            raise ValueError('Expected original stereo float32 event grids on CUDA')
        self.input.copy_(grid)
        self.graph.replay()
        # Caller synchronously copies both outputs to CPU before another replay.
        return self.output
