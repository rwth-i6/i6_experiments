from typing import List

from sisyphus import tk

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import create_report

from .bpe_aed_baseline import run_bpe_aed_baseline

from .bpe_ctc_baseline import run_bpe_ctc_baseline
from .bpe_phoneme_ctc_baseline import run_bpe_phoneme_ctc_baseline
from .phoneme_ctc_baseline import run_phoneme_ctc_baseline
from .bpe_ffnn_transducer_baseline import run_bpe_ffnn_transducer_baseline


def run_switchboard_baselines(prefix: str = "switchboard") -> List[RecogResult]:
    recog_results = []
    recog_results.extend(run_bpe_ctc_baseline())
    recog_results.extend(run_bpe_phoneme_ctc_baseline())
    recog_results.extend(run_phoneme_ctc_baseline())
    recog_results.extend(run_bpe_ffnn_transducer_baseline())
    recog_results.extend(run_bpe_aed_baseline())

    tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)

    return recog_results
