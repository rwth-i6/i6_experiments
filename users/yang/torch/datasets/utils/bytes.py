"""
Bytes
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from returnn_common.datasets_old_2022_10.interface import VocabConfig


class Utf8BytesVocab(VocabConfig):
    """
    Sentence piece model - for sentence pieces.

    See :class:`returnn.datasets.util.vocabulary.SentencePieces` for opts,
    i.e. basically the same as :class:`sentencepiece.SentencePieceProcessor` opts.
    """

    def __init__(self):
        self.dim = 256

    def get_num_classes(self) -> int:
        """
        Get num classes
        """
        return self.dim

    def get_opts(self) -> Dict[str, Any]:
        """
        Get opts
        """
        return {"class": "Utf8ByteTargets"}

    def get_eos_idx(self) -> Optional[int]:
        """EOS"""
        return 0

    def get_bos_idx(self) -> Optional[int]:
        """BOS"""
        return 0

    @staticmethod
    def copy() -> Utf8BytesVocab:
        """Copy"""
        return Utf8BytesVocab()
