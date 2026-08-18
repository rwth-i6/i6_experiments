__all__ = ["forward_step_v1"]

import torch.nn.functional as F

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
import returnn.frontend as rf
from returnn.tensor import TensorDict


def forward_step_v1(
    *,
    model: Model,
    extern_data: TensorDict,
):
    encoder_out = extern_data["encoder_output"].raw_tensor
    encoder_lens = extern_data["encoder_output"].dims[1].dyn_size_ext.raw_tensor

    tokens = extern_data["tokens"].raw_tensor
    tokens_len = extern_data["tokens"].dims[1].dyn_size_ext.raw_tensor

    logits = model.decode_seq(tokens, tokens_len, encoder_out, encoder_lens)
    logits = logits[..., : model.text_out_dim]
    last_logits = logits[:, tokens_len.long().max() - 1]
    scores = -F.log_softmax(last_logits, dim=-1)  # Batch, Vocab
    run_ctx = rf.get_run_ctx()
    run_ctx.mark_as_output(name="scores", tensor=scores)
