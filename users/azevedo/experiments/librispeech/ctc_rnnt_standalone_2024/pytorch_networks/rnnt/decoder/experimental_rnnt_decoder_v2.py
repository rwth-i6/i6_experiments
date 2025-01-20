"""
Experimental RNNT decoder
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
from torch import nn

from torchaudio.models import RNNT
from .rnnt_beam_search import ModifiedRNNTBeamSearch
from .chunk_handler import AudioStreamer


@dataclass
class DecoderConfig:
    # files
    returnn_vocab: Union[str, Any]

    # search related options:
    beam_size: int

    # chunk size definitions for streaming in #samples
    left_size: Optional[int] = None
    right_size: Optional[int] = None
    stride: Optional[int] = None

    pad_value: Optional[float] = None

    # keep states/ hyps for next chunk inference?
    keep_states: Optional[bool] = None
    keep_hyps: Optional[bool] = None

    # for new hash
    test_version: Optional[float] = None

    eos_penalty: Optional[float] = None

    # prior correction
    blank_log_penalty: Optional[float] = None

    # batched encoder config
    batched_encoder = False

    # extra compile options
    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class Transcriber(nn.Module):
    def __init__(
        self,
        feature_extraction: nn.Module,
        encoder: nn.Module,
        mapping: nn.Module,
        left_size: Optional[int] = None,
        ) -> None:
        super().__init__()
        self.feature_extraction = feature_extraction
        self.encoder = encoder
        self.mapping = mapping

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param input:
        :param lengths:
        :return:
        """

        squeezed_features = torch.squeeze(input)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, lengths)

        mask = mask_tensor(audio_features, audio_features_len)

        encoder_out, out_mask = self.encoder(audio_features, mask)
        encoder_out = self.mapping(encoder_out)
        encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

        # encoder implements own infer method
        infer_func = getattr(self.encoder, "infer", None)
        assert infer_func is not None and callable(infer_func), "Encoder requires function 'infer' for decoding."

        squeezed_features = torch.squeeze(input)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, lengths)

            encoder_out, out_mask, state = self.encoder.infer(audio_features, audio_features_len, states)
            encoder_out = self.mapping(encoder_out) # (1, C', F'')
            encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths, [state]


# EXPL: this is called before the .forward_step (see below) of the model is called
#       run_ctx stores information between the hooks (and also current training info)
def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.config = config

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    run_ctx.blank_log_penalty = config.blank_log_penalty

    from returnn.datasets.util.vocabulary import Vocabulary

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    print("create RNNT model...")
    model = run_ctx.engine._model   # class Model from ...conformer_* with trained encoder (model.conformer)
    rnnt_model = RNNT(              # copies Model structure from ...conformer_* to be in torch's RNNT-wrapper
        transcriber=Transcriber(
            feature_extraction=model.feature_extraction,
            encoder=model.conformer,
            mapping=model.encoder_out_linear,
            left_size=config.left_size
        ),
        predictor=model.predictor,
        joiner=model.joiner,
    )
    run_ctx.rnnt_decoder = ModifiedRNNTBeamSearch(
        model=rnnt_model,
        blank=model.cfg.label_target_size,
        blank_penalty=run_ctx.blank_log_penalty,
        eos_penalty=config.eos_penalty
    )
    print("done!")

    run_ctx.beam_size = config.beam_size

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis


# EXPL: this is called after the .forward_step of the model executed
def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print(
            "Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s)
        )


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', 1]
    raw_audio_len = data["raw_audio:size1"].cpu()  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

        start = time.time()

    config: DecoderConfig = run_ctx.config
    if config.left_size is None:
        raise TypeError("DecoderConfig.left_size can't be None!")

    hyps = []
    for i in range(raw_audio.shape[0]):
        # init generator for chunks of our raw_audio according to DecoderConfig
        chunk_streamer = AudioStreamer(
            raw_audio=raw_audio[i],
            raw_audio_len=raw_audio_len[i],
            left_size=config.left_size,
            right_size=config.right_size,
            stride=config.stride,
            pad_value=config.pad_value,
        )

        hypothesis, state = None, None
        hypo = []

        for chunk, eff_chunk_len in chunk_streamer:
            hypothesis, state = run_ctx.rnnt_decoder.infer(
                input=chunk,
                length=torch.tensor(eff_chunk_len),
                beam_width=run_ctx.beam_size,
                state=state if config.keep_states else None,
                hypothesis=hypothesis if config.keep_hyps else None,
            )

            if not config.keep_hyps:
                # every hypothesis is independent from preceding ones
                hypo += hypothesis[0][0][:-1]

        if config.keep_hyps:
            hyps.append(hypothesis[0][0][:-1])
        else:
            hyps.append(hypo)

    if run_ctx.print_rtf:
        total_time = time.time() - start
        run_ctx.total_time += total_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (total_time, total_time / audio_len_batch))

    for hyp, tag in zip(hyps, data["seq_tag"]):
        sequence = [run_ctx.labels[idx] for idx in hyp if idx < len(run_ctx.labels)]
        text = " ".join(sequence).replace("@@ ", "")
        if run_ctx.print_hypothesis:
            print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))
