__all__ = ["PerplexityCallback"]

import torch
from torch import tensor

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class PerplexityCallback(ForwardCallbackIface):
    def __init__(self, vocab: str):
        self.vocab_file = vocab #TODO: do something with this... perhaprs not
        self.val_loss = tensor(0.0, dtype=torch.float64)
        self.val_elements = tensor(0)

    def init(self, *, model: torch.nn.Module):
        pass

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        scores_val = tensor(outputs["scores_ce"].raw_tensor)
        len_val = tensor(outputs["len_ce"].raw_tensor)
        self.val_loss = self.val_loss + torch.sum(scores_val, dtype=torch.float64)
        self.val_elements = self.val_elements + torch.sum(len_val)

    def finish(self):
        # Validation loss file
        with open("val_loss", "wt") as f:
            f.write(f"{self.val_loss.item():.1f}\n")

        # Sub-word perplexity file
        val_loss = self.val_loss / self.val_elements
        sw_ppl = torch.exp(val_loss)
        with open("sw_ppl", "wt") as f:
            f.write(f"{sw_ppl:.1f}\n")

        # Word-level perplexity file
        n_subwords = 1  # TODO: if not known do this in another job...!!!
        n_words = 1  # TODO: !!!
        wl_ppl = sw_ppl ** (n_subwords / n_words)
        with open("wl_ppl", "wt") as f:
            f.write(f"TODO!!!\n{wl_ppl:.1f}\n")
