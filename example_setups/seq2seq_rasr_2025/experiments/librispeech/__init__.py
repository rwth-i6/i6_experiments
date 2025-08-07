from typing import List

from sisyphus import tk

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import create_report

from .bpe_aed_baseline import run_bpe_aed_baseline

# from .bpe_combination_baseline import run_bpe_combination_baseline
from .bpe_ctc_baseline import run_bpe_ctc_baseline
from .bpe_phoneme_ctc_baseline import run_bpe_phoneme_ctc_baseline
from .phoneme_ctc_baseline import run_phoneme_ctc_baseline
from .bpe_ffnn_transducer_baseline import run_bpe_ffnn_transducer_baseline
from .bpe_transformer_lm_baseline import run_bpe_transformer_lm_baseline

# from .bpe_full_ctx_transducer_baseline import run_bpe_full_ctx_transducer_baseline
from .word_transformer_lm_baseline import run_word_transformer_lm_baseline


def run_librispeech_baselines(prefix: str = "librispeech") -> List[RecogResult]:
    run_word_transformer_lm_baseline(num_layers=24)
    # run_word_transformer_lm_baseline(num_layers=48)
    run_word_transformer_lm_baseline(num_layers=96)

    run_bpe_transformer_lm_baseline(bpe_size=128, num_layers=24)
    # run_bpe_transformer_lm_baseline(bpe_size=128, num_layers=48)
    run_bpe_transformer_lm_baseline(bpe_size=128, num_layers=96)

    run_bpe_transformer_lm_baseline(bpe_size=5000, num_layers=24)
    # run_bpe_transformer_lm_baseline(bpe_size=5000, num_layers=48)
    run_bpe_transformer_lm_baseline(bpe_size=5000, num_layers=96)

    recog_results = []
    recog_results.extend(run_bpe_ctc_baseline())
    recog_results.extend(run_bpe_phoneme_ctc_baseline())
    recog_results.extend(run_phoneme_ctc_baseline())
    recog_results.extend(run_bpe_ffnn_transducer_baseline())
    # recog_results.extend(run_bpe_full_ctx_transducer_baseline())
    recog_results.extend(run_bpe_aed_baseline())
    # recog_results.extend(run_bpe_combination_baseline())

    tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)

    return recog_results
