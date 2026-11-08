"""SentencePiece label datastream with an explicit EOS choice and optional subword sampling."""

from typing import Any, Dict, Optional, Union

from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream

SPM_WORD_PREFIX = chr(0x2581)  # the symbol SentencePiece puts in front of a piece that starts a word


class SentencePieceDatastreamV2(SentencePieceDatastream):
    """SentencePieceDatastream without EOS unless asked for, with unigram or BPE subword sampling on request."""

    def __init__(
        self,
        available_for_inference: bool,
        spm_model: tk.Path,
        vocab_size: Union[int, tk.Variable],
        *,
        add_eos: bool = False,
        sampling_alpha: Optional[float] = None,
        sampling_nbest_size: int = -1,
    ):
        """
        :param available_for_inference: whether the stream is available at inference
        :param spm_model: the SentencePiece model, e.g. from TrainSentencePieceJob.out_model
        :param vocab_size: the vocabulary size of the model
        :param add_eos: whether RETURNN appends EOS to every sequence
        :param sampling_alpha: the smoothing of the unigram sampling, None for the best segmentation
        :param sampling_nbest_size: the segmentations the sampling draws from, below 0 for all of them
        """
        super().__init__(available_for_inference=available_for_inference, spm_model=spm_model, vocab_size=vocab_size)
        self.add_eos = add_eos
        self.sampling_alpha = sampling_alpha
        self.sampling_nbest_size = sampling_nbest_size

    def as_returnn_targets_opts(self) -> Dict[str, Any]:
        """
        :return: the SentencePieces vocabulary options, sampling when an alpha is set
        """
        opts: Dict[str, Any] = {
            "class": "SentencePieces",
            "model_file": self.spm_model,
            "add_eos": self.add_eos,
        }
        if self.sampling_alpha is not None:
            opts["enable_sampling"] = True
            opts["alpha"] = self.sampling_alpha
            opts["nbest_size"] = self.sampling_nbest_size
        return opts

    def as_returnn_sbpe_targets_opts(self, breadth_prob: float) -> Dict[str, Any]:
        """
        :param breadth_prob: the breadth_prob of RETURNN's SamplingBytePairEncoding
        :return: the SamplingBytePairEncoding options over the same pieces, with the label ids of SentencePieces
        """
        from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

        return {
            "class": "SamplingBytePairEncoding",
            "vocab_file": ExtractSentencePieceVocabJob(self.spm_model).out_vocab,
            "breadth_prob": breadth_prob,
            "word_prefix_symbol": SPM_WORD_PREFIX,
            "unknown_label": "<unk>",
        }
