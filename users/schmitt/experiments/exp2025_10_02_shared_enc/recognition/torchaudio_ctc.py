__all__ = ["CtcModel", "CtcForwardStepV1"]

from abc import abstractmethod
from os import PathLike
from typing import List, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from returnn.datasets.util.vocabulary import Vocabulary
import returnn.frontend as rf
from returnn.tensor import TensorDict, batch_dim


class CtcModel(Protocol):
    @abstractmethod
    def forward_ctc(self, features: Tensor, features_len: Tensor) -> Tuple[Union[List[Tensor], Tensor], Tensor]:
        raise NotImplementedError


class CtcForwardStepV1:
    """
    A forward_step function that performs CTC recognition using Flashlight/torchaudio.

    The search runs on CPU, therefore it's best to run the model on CPU as well, otherwise
    GPU time will be wasted.
    """

    def __init__(
        self,
        *,
        beam_size: int,
        beam_size_token: int,
        beam_threshold: float,
        vocab: str,
        arpa_lm: Optional[Union[str, PathLike]] = None,
        lexicon: Optional[Union[str, PathLike]] = None,
        lm_scale: float = 0.0,
        sil_score: float = 0.0,
        word_score: float = 0.0,
        blank_log_penalty: Optional[float] = None,
        n_best: int = 1,
        prior_scale: float = 0.0,
        prior_file: Optional[PathLike] = None,
        ctc_layer_idx: int = -1,
        blank_token: str = "<blank>",
        **kwargs,
    ):
        """
        Initializes the forward step.

        For the CTC-related parameters, see
        - https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        - https://pytorch.org/audio/stable/generated/torchaudio.models.decoder.ctc_decoder.html#torchaudio.models.decoder.ctc_decoder

        :param vocab: Vocabulary file suitable for initializing a RETURNN vocab with,
            e.g. the output of `i6_core.text.label.sentencepiece.ExtractSentencePieceVocabJob`.
            When `blank_token`  is not in the vocab, it will be added as the last token.
        :param n_best: the number of hypotheses to return per seq
        :param blank_log_penalty: penalty applied to <blank> in log space
        :param prior_scale: scale applied to priors before applying to outputs
        :param prior_file: prior file compatible w/ numpy.loadtxt containing CTC label priors
        :param ctc_layer_idx: in case the model returns multiple outputs, the index of the output
            to use. Defaults to the last one (i.e. -1).
        """

        assert beam_size > 0
        assert beam_size_token > 0
        assert beam_threshold > 0
        assert n_best > 0

        self.decoder = None
        self.vocab: Vocabulary = None

        self.beam_size = beam_size
        self.beam_size_token = beam_size_token
        self.beam_threshold = beam_threshold
        self.arpa_lm = arpa_lm
        self.lexicon = lexicon
        self.lm_weight = lm_scale
        self.sil_score = sil_score
        self.word_score = word_score
        self.vocab_file = vocab
        self.ctc_layer_idx = ctc_layer_idx
        self.n_best = n_best
        self.blank_token = blank_token
        self.blank_log_penalty = blank_log_penalty

        self.priors = None
        if prior_file is not None:
            self.priors = np.loadtxt(prior_file, dtype=np.float32)
        self.prior_scale = prior_scale

        self._beam_dim = rf.Dim(self.n_best, name="beam")
        self._time_dim = rf.Dim(None, name="time")

    def _init_decoder(self):
        from torchaudio.models.decoder import ctc_decoder

        if self.decoder is not None:
            return

        self.vocab = Vocabulary.create_vocab(vocab_file=self.vocab_file, unknown_label=None)
        self._vocab_dim = rf.Dim(self.vocab.num_labels, name="vocab")
        self.decoder = ctc_decoder(
            nbest=self.n_best,
            beam_size=self.beam_size,
            beam_size_token=self.beam_size_token,
            beam_threshold=self.beam_threshold,
            lexicon=self.lexicon,
            lm=self.arpa_lm,
            lm_weight=self.lm_weight,
            tokens=self.vocab.labels + [self.blank_token]
            if self.blank_token not in self.vocab.labels
            else self.vocab.labels,
            blank_token=self.blank_token,
            sil_token=self.blank_token,
            sil_score=self.sil_score,
            word_score=self.word_score,
        )

    def __call__(self, *, model: CtcModel, extern_data: TensorDict, **kwargs):
        from torchaudio.models.decoder import CTCHypothesis

        self._init_decoder()

        data = extern_data["data"]
        seq_len = data.dims[1].dyn_size_ext
        logits, logits_len = model.forward_ctc(
            data.raw_tensor,
            seq_len.raw_tensor.to(device=data.raw_tensor.device),
        )
        if isinstance(logits, list):
            logits = logits[self.ctc_layer_idx]
        logits = F.log_softmax(logits, dim=-1)
        logits = logits.float().cpu()
        if self.blank_log_penalty is not None:
            logits[:, :, -1] -= self.blank_log_penalty  # blank must be last output
        if self.priors is not None:
            logits -= self.prior_scale * self.priors

        hypotheses: List[List[CTCHypothesis]] = self.decoder(logits, logits_len.cpu())
        assert all(len(beam) == self.n_best for beam in hypotheses)

        lens = [torch.tensor([len(hyp.tokens) for hyp in beam]) for beam in hypotheses]
        padded_lens = pad_sequence(lens, batch_first=True)  # Batch, Beam
        lens_dyn_seq_ext = rf.convert_to_tensor(padded_lens, dims=[batch_dim, self._beam_dim])

        padded_token_beams = [pad_sequence([hyp.tokens for hyp in beam]) for beam in hypotheses]
        padded_tokens = pad_sequence(padded_token_beams, batch_first=True)  # Batch, Time, Beam
        padded_tokens = padded_tokens.transpose(1, 2)  # Batch, Beam, Time
        token_output_dims = [batch_dim, self._beam_dim, rf.Dim(lens_dyn_seq_ext, name="time")]
        tokens = rf.convert_to_tensor(padded_tokens, dims=token_output_dims, sparse_dim=self._vocab_dim)
        rf.get_run_ctx().mark_as_output(tokens, "tokens", dims=token_output_dims)

        padded_align_beams = [pad_sequence([hyp.timesteps for hyp in beam]) for beam in hypotheses]
        padded_alignment = pad_sequence(padded_align_beams, batch_first=True)  # Batch, Time, Beam
        padded_alignment = padded_alignment.transpose(1, 2)  # Batch, Beam, Time
        alignment = rf.convert_to_tensor(padded_alignment, dims=token_output_dims, sparse_dim=data.dims[1])
        rf.get_run_ctx().mark_as_output(alignment, "alignment", dims=token_output_dims)

        scores = [torch.tensor([hyp.score for hyp in beam]) for beam in hypotheses]
        padded_scores = pad_sequence(scores, batch_first=True)  # Batch, Beam
        rf.get_run_ctx().mark_as_output(padded_scores, "scores", dims=[batch_dim, self._beam_dim])
