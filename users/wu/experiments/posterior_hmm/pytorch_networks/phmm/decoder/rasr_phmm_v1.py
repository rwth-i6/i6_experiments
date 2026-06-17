"""
LibRASR decoder for phoneme posterior HMM models.

Ported to the **real RETURNN** forward API (``recipe/returnn``): the batched
acoustic-model forward lives in :func:`forward_step` (shared, in
``rasr_forward_base``), and the per-sequence LibRASR search + output writing live
in :class:`ForwardCallback` (a :class:`returnn.forward_iface.ForwardCallbackIface`).
See ``rasr_forward_base`` for the rationale.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sisyphus import tk

# Re-exported so the serializer can import `<module>.forward_step`.
from .rasr_forward_base import BaseRasrForwardCallback, forward_step  # noqa: F401


@dataclass
class DecoderConfig:
    rasr_config_file: Union[str, tk.Path]
    lexicon: Union[str, tk.Path]
    silence_label: str = "[SILENCE]"
    # Optional acoustic prior: decode the (scaled) likelihood instead of the raw posterior.
    # ``prior_file`` is a [num_labels] vector in +log space (phmm_zhou.PriorCallback writes
    # log(mean softmax)); the recog cost becomes -log p(l|x) + prior_scale * log p(l). prior_scale=0
    # (or prior_file=None) reproduces the raw-posterior decode.
    prior_file: Optional[Union[str, tk.Path]] = None
    prior_scale: float = 0.0


@dataclass
class ExtraConfig:
    print_rtf: bool = True
    sample_rate: int = 16000
    print_hypothesis: bool = True


class ForwardCallback(BaseRasrForwardCallback):
    """Standard (lexicon-based) LibRASR search over the AM posteriors."""

    def _make_config(self) -> DecoderConfig:
        return DecoderConfig(**self._config_kwargs)

    def _make_extra_config(self) -> ExtraConfig:
        return ExtraConfig(**self._extra_config_kwargs)

    def _setup_lexicon(self, lex):
        # Load the AM label log-prior once (in the AM-softmax index order, same as seq_logprobs).
        self._log_prior = None
        prior_file = getattr(self.config, "prior_file", None)
        prior_scale = getattr(self.config, "prior_scale", 0.0)
        if prior_file is not None and prior_scale:
            from returnn.util.basic import cf

            log_prior = np.loadtxt(cf(str(prior_file)), dtype="float32").reshape(-1)
            if log_prior.shape[-1] != self.num_labels:
                raise ValueError(
                    f"Prior dim {log_prior.shape[-1]} != lexicon inventory size {self.num_labels}"
                )
            self._log_prior = log_prior

    def build_features(self, seq_logprobs: np.ndarray) -> np.ndarray:
        if seq_logprobs.shape[-1] != self.num_labels:
            raise ValueError(
                f"Unexpected label dimension {seq_logprobs.shape[-1]} for lexicon inventory size {self.num_labels}"
            )
        # cost = -log p(l|x) + prior_scale * log p(l)  (raw posterior when no prior is configured).
        if self._log_prior is not None:
            seq_logprobs = seq_logprobs - self.config.prior_scale * self._log_prior
        return np.ascontiguousarray((-seq_logprobs).astype("float32", copy=False))
