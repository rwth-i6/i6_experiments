"""
Experimental RNNT decoder
"""

from typing import Callable, Dict, List, Optional, Tuple
import time
import numpy as np
import torch
from torch import nn

from torchaudio.models import RNNT, RNNTBeamSearch

import torch



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
            mapping: nn.Module
    ):
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
        output, out_lengths = self.forward(input, lengths)
        return output, out_lengths, [[]]



def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    run_ctx.labels = vocab.labels

    run_ctx.rnnt_decoder = None
    run_ctx.beam_size = kwargs["beam_size"]

    run_ctx.running_audio_len_s = 0
    run_ctx.total_time = 0

def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):

    if run_ctx.rnnt_decoder is None:
        print("create RNNT model...")
        rnnt_model = RNNT(
            transcriber=Transcriber(
                feature_extraction=model.feature_extraction,
                encoder=model.conformer,
                mapping=model.encoder_out_linear),
            predictor=model.predictor,
            joiner=model.joiner,
        )
        run_ctx.rnnt_decoder = RNNTBeamSearch(
            model=rnnt_model,
            blank=model.cfg.label_target_size,
        )
        print("done!")


    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
    run_ctx.running_audio_len_s += audio_len_batch

    start = time.time()
    tags = data["seq_tag"]

    hyps = []
    for i in range(raw_audio.shape[0]):
        hypothesis, states = run_ctx.rnnt_decoder.infer(
            input=raw_audio[[i]],
            length=raw_audio_len[[i]],
            beam_width=run_ctx.beam_size,
        )
        hyps.append(hypothesis[0][0][:-1])  # exclude last sentence end token

    total_time = time.time() - start
    run_ctx.total_time += total_time


    print("Batch-time: %.2f, Batch-RTF: %.3f" % (total_time, total_time / audio_len_batch))

    for hyp, tag in zip(hyps, tags):
        sequence = [run_ctx.labels[idx] for idx in hyp if idx < len(run_ctx.labels)]
        text = " ".join(sequence).replace("@@ ","")
        print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))