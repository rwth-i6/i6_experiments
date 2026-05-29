"""
LibRASR decoder for the lexicon-free pHMM + NN-phoneme-LM recognition path.

Mirrors `rasr_phmm_v1.py`, but pads the AM log-probability matrix from
`[T, V_am]` to `[T, V_am + 2]` with `-1e30` at the `<s>`/`</s>` indices so
the lexicon-free search can never emit them while scoring AM frames. The
search's sentence-end finalization picks `</s>` via its own logic (driven by
the LM scorer + sentence-end fall-back), so blocking those indices on
AM frames is safe.

Ported to the **real RETURNN** forward API: see `rasr_forward_base` for details.
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
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    silence_label: str = "[SILENCE]"


@dataclass
class ExtraConfig:
    print_rtf: bool = True
    sample_rate: int = 16000
    print_hypothesis: bool = True


class ForwardCallback(BaseRasrForwardCallback):
    """Lexicon-free LibRASR search; pads the AM scores at the bos/eos indices."""

    def _make_config(self) -> DecoderConfig:
        return DecoderConfig(**self._config_kwargs)

    def _make_extra_config(self) -> ExtraConfig:
        return ExtraConfig(**self._extra_config_kwargs)

    def _setup_lexicon(self, lex):
        assert self.config.bos_token in self.label_inventory and self.config.eos_token in self.label_inventory
        self.bos_index = self.label_inventory.index(self.config.bos_token)
        self.eos_index = self.label_inventory.index(self.config.eos_token)
        # AM only emits scores for indices < `am_num_labels`. The remaining
        # indices (= bos/eos) are padded with -inf at scoring time.
        self.am_num_labels = min(self.bos_index, self.eos_index)

    def build_features(self, seq_logprobs: np.ndarray) -> np.ndarray:
        am_dim = seq_logprobs.shape[-1]
        if am_dim != self.am_num_labels:
            raise ValueError(
                f"AM output dim {am_dim} does not match lexicon AM-label count {self.am_num_labels} "
                f"(total lexicon labels: {self.num_labels})"
            )
        am_features = (-seq_logprobs).astype("float32", copy=False)
        padded = np.full((am_features.shape[0], self.num_labels), 1e30, dtype="float32")
        padded[:, : self.am_num_labels] = am_features
        return np.ascontiguousarray(padded)
