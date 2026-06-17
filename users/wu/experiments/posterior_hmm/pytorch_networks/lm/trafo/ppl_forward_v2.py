"""
Forward step + callback computing phoneme-LM perplexity over an LmDataset, reporting BOTH
with-EOS and without-EOS numbers plus the raw total negative log-prob (in nats) needed for the
word-equivalent perplexity. For kazuki_trafo_zijian_variant_v1, under **real RETURNN**.

Real RETURNN drives the forward task with a `forward_step` (which MARKS outputs) plus a
`ForwardCallback` (a `returnn.forward_iface.ForwardCallbackIface`); the run context is reset every
step, so cross-step accumulation lives in the callback. `forward_step` marks the per-token negative
log-prob `nll` on the data time-dim, so RETURNN hands each sequence to `process_seq` trimmed to its
real length (padding removed).

Convention (matches RASR lm-util `compute-perplexity-from-text-file`):
  The `LmDataset` appends `</s>` (seq_end_symbol) so `data = [t_1, .., t_N, </s>]` and
  `delayed = [<s>, t_1, .., t_N]`; the model predicts `data` from `delayed`. Hence the LAST target
  of every sequence (the last element of the per-seq `nll`) is `</s>`.
    * WITH-EOS    : sum NLL over all N+1 targets, normalize by sum(N+1)  (= RASR `perplexity`)
    * WITHOUT-EOS : drop the `</s>` target of each sequence -> sum over N, normalize by sum(N)
                    (= RASR `perplexity_without_eos`)
  The phoneme LM has a closed vocabulary (no unknown), so "without-unknowns" == "with".
"""
import json
import math

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict

_NLL = "nll"  # per-token negative log-prob, marked by forward_step


def forward_step(*, model, extern_data: TensorDict, **kwargs):
    """Run the LM on a batch and mark the per-token NLL (nats) on the data time-dim."""
    import torch
    import returnn.frontend as rf

    data = extern_data["data"]
    batch_dim = data.dims[0]
    time_dim = data.dims[1]  # dyn_size_ext == seq lengths (incl. the appended </s>)
    labels = data.raw_tensor.long()  # [B, T] = [t_1..t_N, </s>]
    labels_len = time_dim.dyn_size_ext.raw_tensor  # [B]
    delayed_labels = extern_data["delayed"].raw_tensor.long()  # [B, T] = [<s>, t_1..t_N]

    with torch.no_grad():
        device = labels.device
        max_len = labels.shape[1]
        seq_mask = torch.arange(max_len, device=device)[None, :] < labels_len.to(device)[:, None]
        out = model(delayed_labels, seq_mask)
        lm_logits = out[0] if isinstance(out, tuple) else out  # [B, T, V]
        # fp32 cross-entropy for accurate perplexity (the forward runs under bf16 autocast).
        ce = torch.nn.functional.cross_entropy(
            lm_logits.float().transpose(1, 2), labels, reduction="none"
        )  # [B, T] in nats

    # Mark on the data time-dim so RETURNN trims each sequence to its real length in process_seq.
    rf.convert_to_tensor(ce, dims=[batch_dim, time_dim], name=_NLL).mark_as_output(
        _NLL, shape=[batch_dim, time_dim]
    )


class ForwardCallback(ForwardCallbackIface):
    """Accumulate the per-sequence NLL (with/without the final </s>) and write ppl.json."""

    def init(self, *, model):
        self.total_nll_with_eos = 0.0   # nats, summed over all targets incl. </s>
        self.total_nll_eos_only = 0.0   # nats, summed over only the </s> targets
        self.total_tokens_with_eos = 0  # = sum(seq_len) incl. </s>
        self.num_sequences = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        nll = outputs[_NLL].raw_tensor  # [seq_len] numpy, padding already removed
        n = int(nll.shape[0])
        if n <= 0:
            return
        self.total_nll_with_eos += float(nll.sum())
        self.total_nll_eos_only += float(nll[-1])  # the </s> target is last
        self.total_tokens_with_eos += n
        self.num_sequences += 1

    def finish(self):
        nll_we = float(self.total_nll_with_eos)
        nll_eos = float(self.total_nll_eos_only)
        nll_woe = nll_we - nll_eos
        n_we = int(self.total_tokens_with_eos)
        n_seq = int(self.num_sequences)
        n_woe = n_we - n_seq  # one </s> per sequence removed
        payload = {
            "num_sequences": n_seq,
            # with EOS (predicts </s>)
            "total_neg_log_prob_with_eos": nll_we,
            "total_tokens_with_eos": n_we,
            "perplexity_with_eos": math.exp(nll_we / max(n_we, 1)),
            # without EOS (</s> target dropped)
            "total_neg_log_prob_without_eos": nll_woe,
            "total_tokens_without_eos": n_woe,
            "perplexity_without_eos": math.exp(nll_woe / max(n_woe, 1)),
            # diagnostic
            "total_neg_log_prob_eos_only": nll_eos,
        }
        with open("ppl.json", "wt") as f:
            json.dump(payload, f, indent=2)
