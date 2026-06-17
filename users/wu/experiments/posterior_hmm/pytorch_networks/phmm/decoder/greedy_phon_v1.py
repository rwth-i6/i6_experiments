"""
Greedy (frame-argmax) phoneme decoder for the pHMM / CTC AMs under **real RETURNN**.

No LM, no lexicon constraint, no beam search: per frame take the argmax over the AM log-probs,
collapse consecutive duplicates, drop the silence/blank label and any non-phoneme bracket/angle
tokens, and emit the phoneme sequence (the EOW ``#`` markers are kept). This is the cleanest measure
of the acoustic model's own phoneme quality (PER), used to separate "the CTC AM is genuinely weaker"
from "the word-constrained search degrades a fine AM".

It reuses the shared :func:`rasr_forward_base.forward_step` (which runs the AM batched and marks the
``log_probs`` output); only the per-sequence post-processing differs from the RASR decoders -- here it
is a plain argmax/collapse instead of a LibRASR search, so this callback pulls in **no** RASR/librasr
dependency and needs no ``rasr_config_file``.

The label inventory (index -> phoneme) is read from the Bliss lexicon's phoneme order, which matches
the AM output index order (index 0 = silence/blank), exactly as the no-op label scorer maps AM indices
to lexicon phonemes in the RASR decoders.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from sisyphus import tk

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict

# Re-exported so the serializer can import `<module>.forward_step` (marks `log_probs`).
from .rasr_forward_base import forward_step  # noqa: F401


@dataclass
class DecoderConfig:
    lexicon: Union[str, tk.Path]
    silence_label: str = "[SILENCE]"


@dataclass
class ExtraConfig:
    sample_rate: int = 16000
    print_hypothesis: bool = False


class ForwardCallback(ForwardCallbackIface):
    """Frame-argmax + collapse-repeats + drop-silence, writing the phoneme stream to ``search_out.py``."""

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
        # Collapse consecutive duplicate frame labels (CTC: blank-separated repeats survive as 2; a run
        # of the same label without an intervening blank collapses to 1 -- standard greedy CTC. For the
        # HMM the per-frame self-loops likewise collapse to one emission per phone occurrence.)
        if frame_labels.shape[0] > 0:
            keep = np.ones(frame_labels.shape[0], dtype=bool)
            keep[1:] = frame_labels[1:] != frame_labels[:-1]
            collapsed = frame_labels[keep]
        else:
            collapsed = frame_labels
        labels = [
            self.label_inventory[i] for i in collapsed.tolist() if 0 <= i < len(self.label_inventory)
        ]
        # Drop the silence/blank label and any non-phoneme tokens ([...] / <...>); keep EOW '#' phonemes.
        labels = [
            s
            for s in labels
            if s != self.config.silence_label and not s.startswith("[") and not s.startswith("<")
        ]
        hyp = " ".join(labels)
        if self.print_hypothesis:
            print(hyp)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
