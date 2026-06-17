"""
Greedy (frame-argmax) **BPE** decoder for the pHMM / CTC AMs under **real RETURNN**.

The BPE counterpart of :mod:`greedy_phon_v1`: per frame take the argmax over the AM log-probs, collapse
consecutive duplicates, drop the silence/blank label and any non-subword bracket/angle tokens, then
**de-BPE** the surviving subword stream into words. No LM, no lexicon constraint, no beam search -- the
cleanest measure of the AM's own quality. Because BPE subwords are losslessly invertible to words, the
hypothesis written to ``search_out.py`` is a WORD stream, so the standard ``search()`` ->
``SearchWordsToCTMJob`` -> ``ScliteJob`` pipeline yields a real **WER** (vs. the phoneme greedy's PER).

NOTE: the older :mod:`greedy_bpe_phmm_v1` is the **MiniReturnn** greedy used by ``phmm_bpe/baseline_128``
(``forward_init_hook``/``run_ctx`` API). This module is the real-RETURNN port (``forward_step`` +
``ForwardCallbackIface``), reusing the shared :func:`debpe.debpe_tokens` so the greedy and lexicon-free
BPE de-tokenization stay identical.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from sisyphus import tk

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict

# Re-exported so the serializer can import `<module>.forward_step` (marks `log_probs`).
from .rasr_forward_base import forward_step  # noqa: F401
from .debpe import debpe_tokens


@dataclass
class DecoderConfig:
    lexicon: Union[str, tk.Path]
    silence_label: str = "[BLANK]"


@dataclass
class ExtraConfig:
    sample_rate: int = 16000
    print_hypothesis: bool = False


class ForwardCallback(ForwardCallbackIface):
    """Frame-argmax + collapse-repeats + drop-blank + de-BPE, writing the WORD stream to ``search_out.py``."""

    def __init__(self, config: dict, extra_config: dict = None):
        self._config_kwargs = dict(config)
        self._extra_config_kwargs = dict(extra_config or {})

    def init(self, *, model):
        from i6_core.lib import lexicon
        from returnn.util.basic import cf

        self.config = DecoderConfig(**self._config_kwargs)
        self.extra_config = ExtraConfig(**self._extra_config_kwargs)

        self.recognition_file = open("search_out.py", "wt")
        self.recognition_file.write("{\n")

        lex = lexicon.Lexicon()
        lex.load(cf(self.config.lexicon))
        self.label_inventory = list(lex.phonemes.keys())  # phoneme-inventory order == AM index order
        if self.config.silence_label not in lex.phonemes:
            raise ValueError(
                f"Silence label {self.config.silence_label!r} not found in lexicon {self.config.lexicon!r}"
            )
        self.print_hypothesis = self.extra_config.print_hypothesis

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        seq_logprobs = outputs["log_probs"].raw_tensor  # [T, V_am] numpy, padding already removed
        frame_labels = np.asarray(seq_logprobs).argmax(axis=-1)
        # Collapse consecutive duplicate frame labels (standard greedy CTC: blank-separated repeats
        # survive as 2; a run of the same label without an intervening blank collapses to 1).
        if frame_labels.shape[0] > 0:
            keep = np.ones(frame_labels.shape[0], dtype=bool)
            keep[1:] = frame_labels[1:] != frame_labels[:-1]
            collapsed = frame_labels[keep]
        else:
            collapsed = frame_labels
        labels = [
            self.label_inventory[i] for i in collapsed.tolist() if 0 <= i < len(self.label_inventory)
        ]
        # Drop the silence/blank label and any non-subword tokens ([...] / <...>), then de-BPE to words.
        labels = [
            s
            for s in labels
            if s != self.config.silence_label and not s.startswith("[") and not s.startswith("<")
        ]
        hyp = debpe_tokens(labels)
        if self.print_hypothesis:
            print(hyp)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
