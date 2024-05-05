import torch
from dataclasses import dataclass
from i6_models.config import ModelConfiguration

@dataclass
class UnigramConfig(ModelConfiguration):
    vocab_dim: int

class Unigram(torch.nn.Module):
    """
    Simple unigram, no input
    """
    def __init__(self, step: int, cfg: UnigramConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.unigram_tensor = torch.rand(cfg.vocab_dim, requires_grad=True)
        torch.nn.init.normal_(self.unigram_tensor, std=0.1)

    def forward(self, x=None):
        """
        Returns log probs of the unigram. If input is provided
        in shape (B, S, ...), then expand the unigram to (B, S, F)
        """
        log_probs = self.unigram_tensor.log_softmax(dim=-1)
        if x is not None:
            assert len(x.shape) >= 2
            batch_size, max_seq_len = x.shape[0], x.shape[1]
            log_probs = log_probs.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, -1).to(x.device)
        return log_probs
