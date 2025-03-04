"""
Experimental RNNT decoder
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
from torch import nn

from torchaudio.models import RNNT
from .rnnt_beam_search import ModifiedRNNTBeamSearch

from ..auxil.functional import num_samples_to_frames, Mode, mask_tensor, process_offline_sample, process_streaming_sample


@dataclass
class DecoderConfig:
    # files
    returnn_vocab: Union[str, Any]

    # search related options:
    beam_size: int

    mode: Mode

    # streaming definitions if mode == Mode.STREAMING
    chunk_size: Optional[int] = None
    carry_over_size: Optional[float] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

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


class Transcriber(nn.Module):
    def __init__(
            self,
            feature_extraction: nn.Module,
            encoder: nn.Module,
            mapping: nn.Module,
            chunk_size: Optional[int] = None,
            lookahead_size: Optional[int] = None,
            carry_over_size: Optional[int] = None
    ) -> None:
        super().__init__()

        self.feature_extraction = feature_extraction
        self.encoder = encoder
        self.mapping = mapping

        self.chunk_size = chunk_size
        self.lookahead_size = lookahead_size
        self.carry_over_size = carry_over_size

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param input:
        :param lengths:
        :return:
        """
        squeezed_features = torch.squeeze(input)
        with torch.no_grad():
            self.feature_extraction.set_mode(Mode.OFFLINE)
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, lengths)

        mask = mask_tensor(audio_features, audio_features_len)

        self.encoder.set_mode(Mode.OFFLINE)
        encoder_out, out_mask = self.encoder(
            audio_features, mask,
            lookahead_size=self.lookahead_size, carry_over_size=self.carry_over_size,
        )
        encoder_out = self.mapping(encoder_out)
        encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
                
        with torch.no_grad():            
            # TODO: TEST!
            chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(self.chunk_size))
            audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)

            encoder_out, out_mask, state = self.encoder.infer(
                audio_features, audio_features_len,
                states,
                chunk_size=chunk_size_frames,
                lookahead_size=self.lookahead_size
            )
            encoder_out = self.mapping(encoder_out)  # (1, C', F'')
            encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths, [state]


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

    rnnt_model = RNNT(
        transcriber=Transcriber(
            feature_extraction=model.feature_extraction,
            encoder=model.conformer,
            mapping=model.encoder_out_linear,
            chunk_size=config.chunk_size,
            lookahead_size=model.lookahead_size,
            carry_over_size=config.carry_over_size,
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
    for i in range(raw_audio.shape[0]):
        if config.mode == Mode.OFFLINE:
            hyp = process_offline_sample(raw_audio=raw_audio[i], raw_audio_len=raw_audio_len[i], run_ctx=run_ctx)
        else:
            hyp = process_streaming_sample(raw_audio=raw_audio[i], raw_audio_len=raw_audio_len[i], config=config, run_ctx=run_ctx)

        hyps.append(hyp)

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