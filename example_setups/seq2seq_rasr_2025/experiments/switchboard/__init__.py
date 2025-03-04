from typing import List

from sisyphus import tk

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import create_report
from .bpe_combination_baseline import run_bpe_combination_baseline
from .bpe_lstm_lm_baseline import run_bpe_lstm_lm_baseline


def run_switchboard_baselines(prefix: str = "switchboard") -> List[RecogResult]:
    run_bpe_lstm_lm_baseline()

    recog_results = []
    recog_results.extend(run_bpe_combination_baseline())

    tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)

    return recog_results
