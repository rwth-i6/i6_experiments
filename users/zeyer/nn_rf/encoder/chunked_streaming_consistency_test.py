"""Sisyphus job: verify that streaming CTC log-probs match the offline output.

Builds a :class:`ReturnnForwardJobV2` whose custom ``forward_step`` runs the
offline ``model.encode_and_get_ctc_log_probs`` and the streaming variant on the
same audio, then writes the per-sequence ``max_abs_diff`` and ``mean_abs_diff``
of the log-prob tensors to a JSON file.

If the streaming implementation is correct, both diffs should be at the level
of float32 numerical noise (~1e-5 .. 1e-6). Larger values indicate a bug in
the streaming forward path.
"""

from __future__ import annotations

import json
from typing import List

import torch
from sisyphus import tk  # noqa: F401  (kept for symmetry with other helpers)

from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf

from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg, ModelWithCheckpoint
from i6_experiments.users.zeyer.returnn.config import config_dict_update_
from i6_experiments.users.zeyer.serialization_v2 import ReturnnConfigWithNewSerialization
from i6_experiments.users.zeyer.recog import _returnn_v2_get_model
from i6_experiments.users.zeyer import tools_paths

from .chunked_streaming import streaming_encode_and_get_ctc_log_probs


_DIFF_OUTPUT_FILE = "diff_stats.json"


def _streaming_consistency_forward_step(*, model, extern_data, **_kwargs):
    """Forward step: compute offline + streaming log-probs, diff, mark as output."""
    from returnn.config import get_global_config

    config = get_global_config()
    segment_seconds = config.float("streaming_segment_seconds", 10.0)

    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()

    # Offline path (the existing implementation)
    log_probs_off, _enc, enc_dim_off = model.encode_and_get_ctc_log_probs(
        data, in_spatial_dim=data_spatial_dim
    )

    # Streaming path (the new implementation)
    log_probs_str, enc_dim_str = streaming_encode_and_get_ctc_log_probs(
        model, data, data_spatial_dim, segment_seconds=segment_seconds
    )

    # Bring both raw tensors to a consistent [B, T, V+1] layout for comparison.
    def _to_btv(lp: Tensor, enc_dim: Dim) -> torch.Tensor:
        order = lp.dims
        bi = order.index(batch_dim)
        ti = order.index(enc_dim)
        fi = order.index(lp.feature_dim)
        return lp.raw_tensor.permute(bi, ti, fi).contiguous()

    off_t = _to_btv(log_probs_off, enc_dim_off)
    str_t = _to_btv(log_probs_str, enc_dim_str)

    t_off, t_str = off_t.shape[1], str_t.shape[1]
    t_min = min(t_off, t_str)
    diff = (off_t[:, :t_min, :] - str_t[:, :t_min, :]).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    print(
        f"streaming-consistency: T_off={t_off} T_str={t_str} "
        f"max_abs_diff={max_diff:.6g} mean_abs_diff={mean_diff:.6g}",
        flush=True,
    )

    device = off_t.device
    rf.get_run_ctx().mark_as_output(
        Tensor(
            "max_diff",
            dims=[batch_dim],
            dtype="float32",
            raw_tensor=torch.tensor([max_diff], dtype=torch.float32, device=device),
        ),
        "max_diff",
        dims=[batch_dim],
    )
    rf.get_run_ctx().mark_as_output(
        Tensor(
            "mean_diff",
            dims=[batch_dim],
            dtype="float32",
            raw_tensor=torch.tensor([mean_diff], dtype=torch.float32, device=device),
        ),
        "mean_diff",
        dims=[batch_dim],
    )
    rf.get_run_ctx().mark_as_output(
        Tensor(
            "t_off",
            dims=[batch_dim],
            dtype="int32",
            raw_tensor=torch.tensor([t_off], dtype=torch.int32, device=device),
        ),
        "t_off",
        dims=[batch_dim],
    )
    rf.get_run_ctx().mark_as_output(
        Tensor(
            "t_str",
            dims=[batch_dim],
            dtype="int32",
            raw_tensor=torch.tensor([t_str], dtype=torch.int32, device=device),
        ),
        "t_str",
        dims=[batch_dim],
    )


def _streaming_consistency_forward_callback():
    """Callback: collect per-sequence diff stats, write JSON at the end."""
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.tensor import TensorDict

    class Callback(ForwardCallbackIface):
        def __init__(self):
            self.results: dict = {}

        def init(self, *, model):
            self.results = {}

        def process_seq(self, *, seq_tag: str, outputs: "TensorDict"):
            self.results[seq_tag] = {
                "max_abs_diff": float(outputs["max_diff"].raw_tensor.item()),
                "mean_abs_diff": float(outputs["mean_diff"].raw_tensor.item()),
                "t_off": int(outputs["t_off"].raw_tensor.item()),
                "t_str": int(outputs["t_str"].raw_tensor.item()),
            }

        def finish(self):
            summary = None
            if self.results:
                max_vals = [r["max_abs_diff"] for r in self.results.values()]
                mean_vals = [r["mean_abs_diff"] for r in self.results.values()]
                summary = {
                    "num_seqs": len(self.results),
                    "max_of_max_abs_diff": max(max_vals),
                    "mean_of_max_abs_diff": sum(max_vals) / len(max_vals),
                    "mean_of_mean_abs_diff": sum(mean_vals) / len(mean_vals),
                }
            out = {"summary": summary, "per_seq": self.results}
            with open(_DIFF_OUTPUT_FILE, "w") as f:
                json.dump(out, f, indent=2)

    return Callback()


def make_streaming_consistency_test_job(
    *,
    model: ModelWithCheckpoint,
    dataset,
    aux_loss_layers: List[int],
    segment_seconds: float = 10.0,
    mem_rqmt: float = 8,
    version: int = 1,
) -> ReturnnForwardJobV2:
    """Build a ReturnnForwardJobV2 that runs offline-vs-streaming log-prob diff."""
    model_def = model.definition

    config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        extern_data=dataset.get_extern_data(),
        forward_data=dataset.get_main_dataset(),
    )
    if isinstance(model_def, ModelDefWithCfg):
        config_dict["_model_def"] = model_def.model_def
        config_dict_update_(config_dict, model_def.config)
    else:
        config_dict["_model_def"] = model_def
    config_dict["get_model"] = _returnn_v2_get_model
    config_dict["forward_step"] = _streaming_consistency_forward_step
    config_dict["forward_callback"] = _streaming_consistency_forward_callback
    config_dict["streaming_segment_seconds"] = segment_seconds
    config_dict["aux_loss_layers"] = list(aux_loss_layers)
    config_dict["max_seqs"] = 1  # streaming impl requires batch=1
    config_dict["batch_size"] = 1_000_000 * model_def.batch_size_factor

    post_config_dict = dict(
        log_batch_size=True,
        torch_log_memory_usage=True,
        watch_memory=True,
    )

    returnn_config = ReturnnConfigWithNewSerialization(config_dict, post_config_dict)

    job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=returnn_config,
        output_files=[_DIFF_OUTPUT_FILE],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=mem_rqmt,
    )
    return job
