"""Two identical resumable decodings; manually interrupt only ``interrupt_me``.

Uses the normal digital QAT lexfree recipe. The instance ID changes only the
Sisyphus job identity, so the two jobs have separate persistent work directories.
Wait for interrupt_me/work/forward_cache.json to contain sequences before
interrupting it, then resume the same job without clearing its work directory.
"""

from dataclasses import replace
from functools import partial
from unittest.mock import patch

from i6_core.returnn import PtCheckpoint
from i6_experiments.users.azim_javed.common.jobs.returnn import ResumableReturnnForwardJobV2
from sisyphus import Job, tk
from sisyphus.job import get_args

from ....model_pipelines.common import recog
from ....model_pipelines.common.train import TrainedModel
from ..recognition import qat_ffnn_transducer_bpe as recognition
from ..training.qat_ffnn_transducer_bpe import get_model_config


PREFIX = "diagnostics/resumable_forward/digital_qat_lexfree"
CHECKPOINT = "/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.s9sXAl53j9fG/output/models/epoch.2000.pt"


class DiagnosticResumableReturnnForwardJob(ResumableReturnnForwardJobV2):
    def __init__(self, *, instance_id: str, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def hash(cls, kwargs):
        # Sisyphus stores **kwargs as a nested dict; apply the inherited defaults
        # before using ReturnnForwardJobV2.hash (e.g. log_verbosity=5).
        forward_kwargs = get_args(ResumableReturnnForwardJobV2.__init__, (), kwargs["kwargs"])
        return Job.hash({
            "forward_job_hash": super().hash(forward_kwargs),
            "instance_id": kwargs["instance_id"],
        })


def run():
    model = TrainedModel(
        descriptor=PREFIX,
        model_config=get_model_config(
            weight_bit_prec=8, activation_bit_prec=8, weight_dropout=0.0, weight_pruning_config=None,
        ),
        checkpoints={2000: PtCheckpoint(tk.Path(CHECKPOINT))},
    )
    variant = recognition.default_offline_lexfree_recog_variant()
    variant.search_mode_params = replace(
        variant.search_mode_params,
        resumable=True, batched_decoder=True, gpu_mem_rqmt=24, mem_rqmt=32,
    )
    results = []
    for instance_id in ("uninterrupted", "interrupt_me"):
        # Override only job construction locally; all config/decoder setup stays
        # in the existing recognition recipe. The context restores it afterwards.
        with patch.object(
            recog, "ResumableReturnnForwardJobV2",
            partial(DiagnosticResumableReturnnForwardJob, instance_id=instance_id),
        ):
            results.extend(recognition.run(
                model=replace(model, descriptor=f"{PREFIX}/{instance_id}"),
                variants=[variant], corpora=["dev-other"],
            ))
    return results
