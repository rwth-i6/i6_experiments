from sisyphus import tk
from typing import Any, Dict, Optional

from i6_core.lm.vocabulary import LmIndexVocabulary

from i6_experiments.users.rossenbach.setups.returnn_standalone.data.bpe import (
    BPESettings,
)

from .base import Datastream


class LabelDatastream(Datastream):
    """
    Defines a datastream for labels represented by indices using the default `Vocabulary` class of RETURNN

    This defines a word-(unit)-based vocabulary
    """

    def __init__(
        self,
        available_for_inference: bool,
        vocab: tk.Path,
        vocab_size: tk.Variable,
        unk_label=None,
    ):
        """

        :param bool available_for_inference:
        :param tk.Path vocab: word vocab file path (pickle)
        :Param tk.Variable|int vocab_size:
        :param str unk_label: unknown label
        """
        super().__init__(available_for_inference)
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.unk_label = unk_label

    def as_returnn_extern_data_opts(
        self, available_for_inference: Optional[bool] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(
                available_for_inference=available_for_inference
            ),
            "shape": (None,),
            "dim": self.vocab_size,
            "sparse": True,
        }
        d.update(kwargs)
        return d

    def as_returnn_targets_opts(self, **kwargs):
        """
        :rtype: dict[str]
        """
        return {"vocab_file": self.vocab, "unknown_label": self.unk_label, **kwargs}


class LmLabelDatastream(LabelDatastream):
    """
    Same as LabelDatastream but uses LmIndexVocabulary objects as input
    """
    def __init__(self, available_for_inference:bool, lm_index_vocab: LmIndexVocabulary):
        """

        :param available_for_inference:
        :param lm:
        """
        super().__init__(
            available_for_inference=available_for_inference,
            vocab=lm_index_vocab.vocab,
            vocab_size=lm_index_vocab.vocab_size,
            unk_label=lm_index_vocab.unknown_token
        )


class BpeDatastream(LabelDatastream):
    """
    This defines a datastream using the BytePairEncoding(Vocabulary) class of RETURNN
    """

    def __init__(
        self, available_for_inference, bpe_settings, seq_postfix=0, use_unk_label=False
    ):
        """
        :param BPESettings bpe_settings:
        :param Path vocab: vocab file path
        :param bool use_unk_label: unk_label should never be used for training
        """
        super(BpeDatastream, self).__init__(
            available_for_inference=available_for_inference,
            vocab=bpe_settings.bpe_vocab,
            vocab_size=bpe_settings.bpe_vocab_size,
            unk_label=bpe_settings.unk_label,
        )
        self.codes = bpe_settings.bpe_codes
        assert isinstance(seq_postfix, int)
        self.seq_postfix = seq_postfix
        self.use_unk_label = use_unk_label

    def register_outputs(self, prefix):
        tk.register_output("%s.codes" % prefix, self.codes)
        tk.register_output("%s.vocab" % prefix, self.vocab)

    def as_returnn_targets_opts(self):
        opts = {
            "class": "BytePairEncoding",
            "bpe_file": self.codes,
            "vocab_file": self.vocab,
            "unknown_label": self.unk_label if self.use_unk_label else None,
        }
        if self.seq_postfix is not None:
            opts["seq_postfix"] = [self.seq_postfix]
        return opts


class SentencePieceDatastream(Datastream):
    """
    Defines a label datastream for sentence-pieces. This does not inherit from LabelDatastream as it uses
    """

    def __init__(self, available_for_inference, spm_model, vocab_size):
        super().__init__(available_for_inference)
        self.vocab_size = vocab_size
        self.spm_model = spm_model

    def as_returnn_extern_data_opts(
        self, available_for_inference: Optional[bool] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(
                available_for_inference=available_for_inference
            ),
            "shape": (None,),
            "dim": self.vocab_size,
            "sparse": True,
        }
        d.update(kwargs)
        return d

    def as_returnn_targets_opts(self):
        opts = {
            "class": "SentencePieces",
            "model_file": self.spm_model,
            "add_eos": True,
        }
        return opts
