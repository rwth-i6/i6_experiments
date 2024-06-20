"""
Sentence piece model
"""

from __future__ import annotations

from typing import Optional, Union, Any, Dict
from sisyphus import Path

from returnn_common.datasets_old_2022_10.interface import VocabConfig


class SentencePieceModel(VocabConfig):
    """
    Sentence piece model - for sentence pieces.

    See :class:`returnn.datasets.util.vocabulary.SentencePieces` for opts,
    i.e. basically the same as :class:`sentencepiece.SentencePieceProcessor` opts.
    """

    def __init__(
        self,
        dim: int,
        model_file: Union[str, Path],
        *,
        eos_idx: Optional[int] = None,
        bos_idx: Optional[int] = None,
        unknown_label: Optional[str] = None,
        other_opts: Optional[Dict[str, Any]] = None,
    ):
        super(SentencePieceModel, self).__init__()
        self.dim = dim
        self.model_file = model_file
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.unknown_label = unknown_label
        self.other_opts = other_opts

    def get_num_classes(self) -> int:
        """
        Get num classes
        """
        return self.dim

    def get_opts(self) -> Dict[str, Any]:
        """
        Get opts
        """
        d = {
            "class": "SentencePieces",
            "model_file": self.model_file,
        }
        if self.other_opts:
            d.update(self.other_opts)
            if d["class"] == "SamplingBytePairEncoding":
                # Need to fix this a bit. model_file not used here. But we need vocab_file instead.
                model_file = d.pop("model_file")
                if not d.get("vocab_file"):
                    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

                    d["vocab_file"] = ExtractSentencePieceVocabJob(model_file).out_vocab
                d.setdefault("word_prefix_symbol", "▁")
                d.setdefault("unknown_label", self.unknown_label)
                d.setdefault("bos_label", self.bos_idx)
                d.setdefault("eos_label", self.eos_idx)
        return d

    def get_eos_idx(self) -> Optional[int]:
        """EOS"""
        return self.eos_idx

    def get_bos_idx(self) -> Optional[int]:
        """BOS"""
        return self.bos_idx

    def copy(self, **kwargs) -> SentencePieceModel:
        """Copy"""
        opts = {k: getattr(self, k) for k in ["dim", "model_file", "eos_idx", "bos_idx", "unknown_label", "other_opts"]}
        opts.update(kwargs)
        if self.other_opts and kwargs.get("other_opts", None) is not None:
            opts["other_opts"] = self.other_opts.copy()
            opts["other_opts"].update(kwargs["other_opts"])
        return SentencePieceModel(**opts)
