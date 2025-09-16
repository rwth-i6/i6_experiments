from dataclasses import dataclass
from ...rnnt.search.beam_search import RNNTSearchConfig, RNNTDecoder
from ...common import Hypothesis



"""
See `rnnt.search.beam_search` for info.
"""

@dataclass(kw_only=True)
class MonotonicRNNTSearchConfig(RNNTSearchConfig):
    def module(self):
        return MonotonicRNNTDecoder
        

class MonotonicRNNTDecoder(RNNTDecoder):
    def __init__(self, run_ctx):
        super().__init__(run_ctx)

    def _build_decoder(self):
        if self.model is None or self.blank is None:
            raise ValueError

        from .monotonic_rnnt_beam_search_v1 import MonotonicRNNTBeamSearch
        return MonotonicRNNTBeamSearch(
            model=self.model,
            blank=self.blank,
            blank_penalty=self.decoder_config.blank_log_penalty,
            device=self.run_ctx.device,
            lm_model=self.lm,
            lm_scale=self.search_config.lm_scale,
            zero_ilm_scale=self.search_config.zero_ilm_scale,
            lm_sos_token_index=0,
        )
    