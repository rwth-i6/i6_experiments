import torch.nn as nn
from ..auxil.functional import Mode

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
        
        if self._mode == mode:
            return

        self._mode = mode

        for m in self.modules():
            if isinstance(m, StreamableModule):
                m.set_mode(mode)

    def forward(self, *args, **kwargs):
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