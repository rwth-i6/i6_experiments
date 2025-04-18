import torch.nn as nn
from .common import Mode



class StreamableModule(nn.Module):
    """
    Abstract class for modules that operate differently in offline- and streaming inference mode
    """
    def __init__(self):
        super().__init__()
        self._mode = None

    def set_mode(self, mode: Mode) -> None:
        assert mode is not None, ""

        self._mode = mode

    def set_mode_cascaded(self, mode: Mode) -> None:
        assert mode is not None, ""
        
        self._mode = mode

        for m in self.modules():
            if isinstance(m, StreamableModule):
                m.set_mode(mode)

    def forward(self, *args, **kwargs):
        """
        Interface for torch modules which can operate in "streaming" and "offline" mode.

        #### Streaming mode:
            expected input shape: [B, N, C, F]
            expected output shape: [B, N, C', F']
        #### Offline mode:
            expected input shape: [B, T, F]
            expected output shape: [B, T', F']
        
        - B: batch size
        - T: number of frames
        - N: number of chunks
        - C: number of frames per chunk (with overlap)
        - F: feature dim
        """
        assert self._mode is not None, ""

        if self._mode == Mode.STREAMING:
            return self.forward_streaming(*args, **kwargs)
        else:
            return self.forward_offline(*args, **kwargs)

    def forward_offline(self, *args, **kwargs):
        raise NotImplementedError("Implement offline forward pass")

    def forward_streaming(self, *args, **kwargs):
        raise NotImplementedError("Implement streaming forward pass")
    
    def infer(self, *args, **kwargs):
        raise NotImplementedError("Implement infer")


    # TODO
    # def __pre_forward__(self, *args, **kwargs):
    #     pass