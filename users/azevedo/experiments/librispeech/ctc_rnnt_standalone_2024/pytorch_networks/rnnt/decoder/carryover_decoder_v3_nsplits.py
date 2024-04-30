"""
Experimental RNNT decoder
"""

import math
import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
from torch import nn

from torchaudio.models import RNNT
from .rnnt_beam_search import ModifiedRNNTBeamSearch
from .chunk_handler import AudioStreamer

from ..auxil.functional import num_samples_to_frames, Mode


@dataclass
class DecoderConfig:
    # files
    returnn_vocab: Union[str, Any]

    # search related options:
    beam_size: int

    # chunk size definitions for streaming in #samples
    chunk_size: Optional[int] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

    carry_over_size: Optional[float] = None

    pad_value: Optional[float] = None

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
        chunk_size: Optional[int] = None,
        lookahead_size: Optional[int] = None,
        carry_over_size: Optional[int] = None,
        num_splits: Optional[int] = None,
        left_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.feature_extraction = feature_extraction
        self.encoder = encoder
        self.mapping = mapping

        self.chunk_size = chunk_size
        self.lookahead_size = lookahead_size
        self.carry_over_size = carry_over_size
        self.num_splits = num_splits

        self.left_size = left_size

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param input:
        :param lengths:
        :return:
        """

        squeezed_features = torch.squeeze(input)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, lengths)

            chunk_size_frames = num_samples_to_frames(
                n_fft=self.feature_extraction.n_fft,
                hop_length=self.feature_extraction.hop_length,
                center=self.feature_extraction.center,
                num_samples=int(self.left_size)
            )

            time_dim_pad = -audio_features.size(1) % chunk_size_frames
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, time_dim_pad), "constant", 0)

        mask = mask_tensor(audio_features, audio_features_len)

        encoder_out, out_mask, _ = self.encoder(
            audio_features, mask,
            lookahead_size=self.lookahead_size, carry_over_size=self.carry_over_size,
            k=self.num_splits
        )[1]
        encoder_out = self.mapping(encoder_out)
        encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

        if self.chunk_size is None or self.chunk_size <= 0:
                # non-streaming mode
                output, out_lengths = self.forward(input, lengths)
                return output, out_lengths, [[]]

        # encoder implements own infer method
        infer_func = getattr(self.encoder, "infer", None)
        assert infer_func is not None and callable(infer_func), "Encoder requires function 'infer' for decoding."

        squeezed_features = torch.squeeze(input)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, lengths)

            chunk_size_frames = num_samples_to_frames(
                n_fft=self.feature_extraction.n_fft,
                hop_length=self.feature_extraction.hop_length,
                center=self.feature_extraction.center,
                num_samples=int(self.left_size)
            )

            time_dim_pad = -audio_features.size(1) % chunk_size_frames
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, time_dim_pad), "constant", 0)

            encoder_out, out_mask, state = self.encoder.infer(audio_features, audio_features_len,
                                                              states,
                                                              chunk_size=chunk_size_frames,
                                                              lookahead_size=self.lookahead_size)
            encoder_out = self.mapping(encoder_out)  # (1, C', F'')
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
    model = run_ctx.engine._model

    try:
        carry_over_size = model.carry_over_size
    except AttributeError:
        carry_over_size = None

    rnnt_model = RNNT(
        transcriber=Transcriber(
            feature_extraction=model.feature_extraction,
            encoder=model.conformer,
            mapping=model.encoder_out_linear,
            chunk_size=config.chunk_size,
            lookahead_size=model.lookahead_size,
            carry_over_size=carry_over_size,
            num_splits=model.num_splits,
            left_size=int(model.cfg.chunk_size)
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

    hyps = []
    if config.chunk_size is None or config.chunk_size <= 0:
        # non-streaming mode
        for i in range(raw_audio.shape[0]):
            hypothesis, states = run_ctx.rnnt_decoder.infer(
                input=raw_audio[[i]],
                length=raw_audio_len[[i]],
                beam_width=run_ctx.beam_size,
            )
            hyps.append(hypothesis[0][0][:-1])
    else:
        for i in range(raw_audio.shape[0]):
            # init generator for chunks of our raw_audio according to DecoderConfig
            chunk_streamer = AudioStreamer(
                raw_audio=raw_audio[i],
                raw_audio_len=raw_audio_len[i],
                left_size=config.chunk_size,
                right_size=config.lookahead_size,
                stride=config.stride,
                pad_value=config.pad_value,
            )

            hypothesis, state = None, None
            states = collections.deque(maxlen=math.ceil(config.carry_over_size))

            for chunk, eff_chunk_len in chunk_streamer:
                hypothesis, state = run_ctx.rnnt_decoder.infer(
                    input=chunk,
                    length=torch.tensor(eff_chunk_len),
                    beam_width=run_ctx.beam_size,
                    state=tuple(states) if len(states) > 0 else None,
                    hypothesis=hypothesis,
                )
                states.append(state)

            hyps.append(hypothesis[0][0][:-1])

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
