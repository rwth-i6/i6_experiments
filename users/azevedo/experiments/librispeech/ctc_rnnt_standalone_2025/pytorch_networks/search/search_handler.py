import torch
from .beam_search.documented_rnnt_beam_search import RNNTBeamSearch
from .beam_search.monotonic_rnnt_beam_search_v1 import MonotonicRNNTBeamSearch



def get_beam_algo(algo: str, **kwargs) -> torch.nn.Module:
    if algo == "rnnt":
        return RNNTBeamSearch(**kwargs)
    if algo == "mrnnt":
        return MonotonicRNNTBeamSearch(**kwargs)