from __future__ import annotations

from typing import Optional, Union, Any, Dict
from sisyphus import Path

from returnn_common.datasets_old_2022_10.interface import VocabConfig


class Bpe(VocabConfig):
    def __init__(
        self,
        dim: int,
        codes: Union[Path, str],  # filename, bpe_file
        vocab: Union[Path, str],  # filename
        *,
        eos_idx: Optional[int] = None,
        bos_idx: Optional[int] = None,
        unknown_label: Optional[str] = None,
        other_opts: Optional[Dict[str, Any]] = None,
    ):
        super(Bpe, self).__init__()
        self.dim = dim
        self.codes = codes
        self.vocab = vocab
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
        # For returnn.datasets.util.vocabulary.Vocabulary.create_vocab.
        # No "class" needed - when "bpe_file" is present, class=BytePairEncoding is automatically inferred.
        d = {
            "bpe_file": self.codes,
            "vocab_file": self.vocab,
            "unknown_label": self.unknown_label,
            "bos_label": self.bos_idx,
            "eos_label": self.eos_idx,
            # 'seq_postfix': [0]  # no EOS needed for RNN-T
        }
        if self.other_opts:
            d.update(self.other_opts)
            if self.other_opts.get("class") == "SamplingBytePairEncoding":
                d.pop("bpe_file")
        return d

    def get_eos_idx(self) -> Optional[int]:
        """EOS"""
        return self.eos_idx

    def get_bos_idx(self) -> Optional[int]:
        """BOS"""
        return self.bos_idx

    def copy(self, **kwargs):
        """Copy"""
        opts = {
            k: getattr(self, k) for k in ["dim", "codes", "vocab", "eos_idx", "bos_idx", "unknown_label", "other_opts"]
        }
        opts.update(kwargs)
        return Bpe(**opts)
