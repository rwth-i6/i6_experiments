import torch
from i6_models.config import ModelConfiguration

@dataclass
class UnigramConfig(ModuleConfiguration):
    n_out: int

class Unigram(torch.nn.Module):
    """
    Simple unigram, no input
    """
    def __init__(self, cfg: UnigramConfig):
        self.unigram_tensor = torch.rand(cfg.n_out, requires_grad=True)

    def forward(self):
        """
        Returns log probs of the unigram
        """
        log_probs = self.unigram_tensor.log_softmax()
        return log_probs
