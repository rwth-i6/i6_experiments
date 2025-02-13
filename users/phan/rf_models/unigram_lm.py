from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection, Dict
import tree
import math
import numpy as np
import torch
import torch.nn as nn
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import (
    trafo_lm_kazuki_import,
)

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )


class Unigram_LM_RF(rf.Module):
    r"""Context 0 LM. It is the logits.
    """

    def __init__(
        self,
        output_dim: Dim,
        preload_tensor: Optional[str] = None,
        preload_is_log_prob: Optional[bool] = False,
        preload_blank_idx: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        :param output_dim: output vocab size
        :param preload_tensor: preload the unigram using a TXT (compat with np.loadtxt)
        :param preload_is_log_prob: True if preload is in log prob space, False if in probability space
        :param preload_blank_idx: If the preload is a frame-level prior, pass its blank idx here.
            The blank idx will be removed from the preloaded tensor, then renormalize over the rest true labels.
        """
        super().__init__()
        self.output_dim = output_dim
        if preload_tensor is None:
            raw_output = torch.normal(mean=0.0, std=0.1, size=(output_dim.capacity,), requires_grad=True)
        else:
            preload_unigram = np.loadtxt(preload_tensor, dtype="float32")
            if not preload_is_log_prob:
                preload_unigram = np.log(preload_unigram)
            if preload_blank_idx is not None:
                preload_unigram = np.delete(preload_unigram, preload_blank_idx, 0)
            raw_output = torch.tensor(preload_unigram).log_softmax(0)
        assert raw_output.shape == (output_dim.capacity,)
        self.output = rf.Tensor("output", dims=[output_dim], dtype="float32", raw_tensor=raw_output)
        for name, param in self.named_parameters():
            param.initial = rf.init.Normal(stddev=0.1) # mean already 0.0

    def default_initial_state(
        self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
    ) -> rf.State:
        """
        Unigram is stateless. Return this just to be compatible with search.
        """
        state = rf.zeros(
            dims=batch_dims,
            sparse_dim=self.output_dim,
        )
        return state

    def select_state(self, state: rf.State, backrefs) -> rf.State:
        """
        Unigram is stateless. Return this just to be compatible with search.
        """
        state = rf.gather(state, indices=backrefs)
        return state

    def __call__(
        self,
        input: rf.Tensor,
        spatial_dim: Optional[Dim] = None,
        state: Optional[rf.State] = None,
        batch_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.
        Returns (batch dims, V) logits
        """
        
        self.output.raw_tensor = self.output.raw_tensor.to(input.raw_tensor.device)
        output = self.output
        for dim in input.dims:
            output = rf.expand_dim(output, dim)
        if spatial_dim == single_step_dim:
            batch_dims = list(input.dims)
            dim_order = list(input.dims) + [self.output_dim] # here input is (B), no spatial dim
        else:
            batch_dims = input.remaining_dims(spatial_dim)
            dim_order = [spatial_dim] + batch_dims + [self.output_dim] # we want [T, B, V] like lstm and ffnn
        output = output.copy_transpose(dim_order)

        return {
            "output": output,
            "state": self.default_initial_state(batch_dims=batch_dims),
        }


def get_model(*, epoch: int, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    lm_cfg = config.typed_value("lm_cfg", {})
    lm_cls = lm_cfg.pop("class")
    assert lm_cls == "Unigram_LM_RF", "Only Unigram LM is supported"
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name="data", **extern_data_dict["data"])
    return Unigram_LM_RF(
        output_dim=data.sparse_dim,
        **lm_cfg,
    )

from returnn.tensor import batch_dim
from i6_experiments.users.phan.utils.masking import get_seq_mask
def train_step(*, model: Unigram_LM_RF, extern_data: TensorDict, **_kwargs_unused):
    targets = extern_data["data"]
    delayed = extern_data["delayed"]
    spatial_dim = delayed.get_time_dim_tag()
    targets_len_rf = spatial_dim.dyn_size_ext
    targets_len = targets_len_rf.raw_tensor
    out = model(delayed, spatial_dim=spatial_dim)
    logits: Tensor = out["output"]
    logits_raw = logits.raw_tensor # (T, B, V)
    targets_raw = targets.raw_tensor.long() # (B, T)
    ce = torch.nn.functional.cross_entropy(
        input = logits_raw.permute(1, 2, 0),
        target = targets_raw,
        reduction="none",
    )
    seq_mask = get_seq_mask(seq_lens=targets_len, max_seq_len=targets_len.max(), device=logits_raw.device)
    loss = (ce*seq_mask).sum()
    ppl = torch.exp(loss/targets_len.sum())
    rf.get_run_ctx().mark_as_loss(
        name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
    rf.get_run_ctx().mark_as_loss(
        name="ppl", loss=ppl, as_error=True,
    )


