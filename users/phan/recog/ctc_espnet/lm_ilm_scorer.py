import random

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, to_device
from espnet.nets.pytorch_backend.rnn.attentions import initial_att

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim


class LM_ILM_Scorer(AbsDecoder):
    def __init__(self, model, batch_dims, enc_spatial_dim, lm_scale=0.0, ilm_scale=0.0):
        super().__init__()
        self.model = model
        self.batch_dims = batch_dims
        self.enc_spatial_dim = enc_spatial_dim
        self.initial_call = True
        self.lm_scale = lm_scale
        self.ilm_scale = ilm_scale

    def init_state(self, x):
        new_state = rf.State()

        if self.lm_scale > 0:
            new_state.lm = self.model.language_model.default_initial_state(batch_dims=self.batch_dims)
        if self.ilm_scale > 0:
            new_state.ilm = self.model.ilm.default_initial_state(batch_dims=self.batch_dims)

        return new_state

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def score(self, yseq, state, enc_args):

        target_raw = yseq[-1].unsqueeze(0)
        target = rf.constant(self.model.bos_idx, dims=self.batch_dims, dtype="int32", sparse_dim=self.model.target_dim)
        target.raw_tensor = target_raw.to(torch.int32)

        out_log_prob = None
        new_state = rf.State()

        # if self.initial_call:
        #     input_embed = rf.zeros(self.batch_dims + [self.model.target_embed.out_dim], feature_dim=self.model.target_embed.out_dim)
        #     self.initial_call = False
        # else:
        #     input_embed = self.model.target_embed(target)

        if self.lm_scale > 0:
            lm_out = self.model.language_model(target, state=state.lm, spatial_dim=single_step_dim)
            lm_state = lm_out["state"]
            new_state.lm = lm_state
            lm_log_prob = rf.log_softmax(lm_out["output"], axis=self.model.target_dim)

            out_log_prob = (
                self.lm_scale * lm_log_prob
            )

        if self.ilm_scale > 0:
            ilm_out = self.model.ilm(target, state=state.ilm, spatial_dim=single_step_dim)
            ilm_state = ilm_out["state"]
            new_state.ilm = ilm_state
            ilm_log_prob = rf.log_softmax(ilm_out["output"], axis=self.model.target_dim)

            if out_log_prob is None:
                out_log_prob = (
                    self.ilm_scale * ilm_log_prob
                )
            else:
                out_log_prob = (
                    out_log_prob - self.ilm_scale * ilm_log_prob
                )

        return out_log_prob.raw_tensor[0], new_state
