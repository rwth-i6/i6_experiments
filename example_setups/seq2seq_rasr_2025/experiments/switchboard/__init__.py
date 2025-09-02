from typing import List

from sisyphus import gs, tk

from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import create_base_recog_report
from . import bpe_aed, bpe_ctc, bpe_ffnn_transducer, phoneme_ctc


def run_all() -> List[RecogResult]:
    recog_results = []

    with ExperimentContext("baselines"):
        recog_results.extend(bpe_ctc.run_all())
        recog_results.extend(bpe_aed.run_all())
        recog_results.extend(bpe_ffnn_transducer.run_all())
        recog_results.extend(phoneme_ctc.run_all())

        tk.register_report(
            f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt", values=create_base_recog_report(recog_results), required=True
        )

    return recog_results
