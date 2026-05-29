"""
Forward module that computes log-perplexity over an LmDataset for the
kazuki_trafo_zijian_variant_v1 LM. Writes a single-key JSON `ppl.json`
with total log-prob, total token count and the resulting perplexity.
"""
import json
import math
from dataclasses import dataclass


@dataclass
class PplConfig:
    pass


def forward_init_hook(run_ctx, **kwargs):
    run_ctx.total_neg_log_prob = 0.0
    run_ctx.total_tokens = 0
    run_ctx.num_sequences = 0


def forward_finish_hook(run_ctx, **kwargs):
    nll_per_token = run_ctx.total_neg_log_prob / max(run_ctx.total_tokens, 1)
    ppl = math.exp(nll_per_token)
    payload = {
        "total_neg_log_prob": run_ctx.total_neg_log_prob,
        "total_tokens": int(run_ctx.total_tokens),
        "num_sequences": int(run_ctx.num_sequences),
        "nll_per_token": nll_per_token,
        "perplexity": ppl,
    }
    with open("ppl.json", "wt") as f:
        json.dump(payload, f, indent=2)


def forward_step(*, model, extern_data, **kwargs):
    import torch

    labels = extern_data["data"].raw_tensor.long()  # [B, T]
    labels_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor  # [B]
    delayed_labels = extern_data["delayed"].raw_tensor.long()  # [B, T]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    with torch.no_grad():
        max_len = labels.shape[1]
        seq_mask = torch.arange(max_len, device=labels.device)[None, :] < labels_len.to(labels.device)[:, None]
        out = model(delayed_labels, seq_mask)
        lm_logits = out[0] if isinstance(out, tuple) else out

        ce = torch.nn.functional.cross_entropy(
            lm_logits.transpose(1, 2), labels, reduction="none"
        )
        ce = ce * seq_mask.to(ce.dtype)

        run_ctx.total_neg_log_prob += float(ce.sum().detach().cpu())
        run_ctx.total_tokens += int(labels_len.sum().detach().cpu())
        run_ctx.num_sequences += int(labels.shape[0])
