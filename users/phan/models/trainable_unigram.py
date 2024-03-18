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

    def forward(self, x=None):
        """
        Returns log probs of the unigram. If input is provided
        in shape (B, S), then expand the unigram to (B, S, F)
        """
        log_probs = self.unigram_tensor.log_softmax()
        if x is not None:
            assert len(x.shape) == 2
            batch_size, max_seq_len = x.shape
            log_probs = log_probs.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, -1)
        return log_probs
