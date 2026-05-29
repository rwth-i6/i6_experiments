"""
LibRASR decoder for phoneme posterior HMM models.

Ported to the **real RETURNN** forward API (``recipe/returnn``): the batched
acoustic-model forward lives in :func:`forward_step` (shared, in
``rasr_forward_base``), and the per-sequence LibRASR search + output writing live
in :class:`ForwardCallback` (a :class:`returnn.forward_iface.ForwardCallbackIface`).
See ``rasr_forward_base`` for the rationale.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from sisyphus import tk

# Re-exported so the serializer can import `<module>.forward_step`.
from .rasr_forward_base import BaseRasrForwardCallback, forward_step  # noqa: F401


@dataclass
class DecoderConfig:
    rasr_config_file: Union[str, tk.Path]
    lexicon: Union[str, tk.Path]
    silence_label: str = "[SILENCE]"


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

    def build_features(self, seq_logprobs: np.ndarray) -> np.ndarray:
        if seq_logprobs.shape[-1] != self.num_labels:
            raise ValueError(
                f"Unexpected label dimension {seq_logprobs.shape[-1]} for lexicon inventory size {self.num_labels}"
            )
        return np.ascontiguousarray((-seq_logprobs).astype("float32", copy=False))
