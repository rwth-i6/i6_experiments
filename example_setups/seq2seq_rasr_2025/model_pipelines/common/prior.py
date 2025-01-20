__all__ = ["compute_priors"]

from typing import Protocol, Tuple

import numpy as np
import torch
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_experiments.common.setups.serialization import Collection, Import
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from ...data.base import DataConfig
from ...tools import returnn_python_exe, returnn_root
from ..common.imports import recipe_imports


class EncoderModel(Protocol):
    def forward(
        self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


class ComputePriorCallback(ForwardCallbackIface):
    def init(self, *, model: EncoderModel):
        self.n = 1
        self.avg_probs = None

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        log_prob_tensor = outputs["log_probs"].raw_tensor
        assert log_prob_tensor is not None
        prob_tensor_iter = iter(np.exp(log_prob_tensor))

        if self.avg_probs is None:
            self.avg_probs = next(prob_tensor_iter)
            print("Create probs collection tensor of shape", self.avg_probs.shape)

        for prob_tensor in prob_tensor_iter:
            self.n += 1
            self.avg_probs += (prob_tensor - self.avg_probs) / self.n

    def finish(self):
        prob_array = self.avg_probs
        log_prob_array = np.log(prob_array)  # type: ignore
        log_prob_strings = ["%.20e" % s for s in log_prob_array]

        # Write txt file
        with open("prior.txt", "wt") as f:
            f.write(" ".join(log_prob_strings))

        # Write xml file
        with open("prior.xml", "wt") as f:
            f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="{len(log_prob_array)}">\n')
            f.write(" ".join(log_prob_strings))
            f.write("\n</vector-f32>")

        # Plot png file
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xdata = range(len(prob_array))  # type: ignore
        plt.semilogy(xdata, prob_array)
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig("prior.png")


def _prior_step(*, model: EncoderModel, extern_data: TensorDict, **_):
    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]  # [B, T, 1]
    audio_samples_size = raw_data["data:size1"]  # [B]

    log_probs, sequence_lengths = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size.to(device=audio_samples.device),
    )  # [B, T, V], [B]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext.raw_tensor = sequence_lengths
    run_ctx.mark_as_output(log_probs, name="log_probs")


def compute_priors(
    prior_data_config: DataConfig,
    model_serializers: Collection,
    checkpoint: PtCheckpoint,
) -> tk.Path:
    prior_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
            },
            "backend": "torch",
            "batch_size": 20_000 * 160,
        },
        python_prolog=recipe_imports,
        python_epilog=[
            model_serializers,
            Import(
                f"{_prior_step.__module__}.{_prior_step.__name__}",
                import_as="forward_step",
            ),
            Import(
                f"{ComputePriorCallback.__module__}.{ComputePriorCallback.__name__}",
                import_as="forward_callback",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    prior_returnn_config.update(prior_data_config.get_returnn_data("forward_data"))

    prior_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=prior_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )

    return prior_job.out_files["prior.txt"]
