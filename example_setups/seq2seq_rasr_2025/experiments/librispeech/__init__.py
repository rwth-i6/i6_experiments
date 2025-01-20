from typing import List

from sisyphus import tk

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import create_report
from .bpe_aed_baseline import run_bpe_aed_baseline
from .bpe_combination_baseline import run_bpe_combination_baseline
from .bpe_ctc_baseline import run_bpe_ctc_baseline
from .bpe_ffnn_transducer_baseline import run_bpe_ffnn_transducer_baseline
from .bpe_lstm_lm_baseline import run_bpe_lstm_lm_baseline


def run_librispeech_baselines(prefix: str = "librispeech") -> List[RecogResult]:
    run_bpe_lstm_lm_baseline()

    recog_results = []
    recog_results.extend(run_bpe_ctc_baseline())
    recog_results.extend(run_bpe_ffnn_transducer_baseline())
    recog_results.extend(run_bpe_aed_baseline())
    recog_results.extend(run_bpe_combination_baseline())

    tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)

    return recog_results
