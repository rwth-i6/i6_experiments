from .bpe_ctc_baseline import run_bpe_ctc_baseline
from .bpe_ffnn_transducer_baseline import run_bpe_ffnn_transducer_baseline
from .bpe_aed_baseline import run_bpe_aed_baseline

# from .bpe_lstm_lm_baseline import run_bpe_lstm_lm_baselines


def run_switchboard_baselines() -> None:
    run_bpe_ctc_baseline()
    run_bpe_ffnn_transducer_baseline()
    run_bpe_aed_baseline()
    # run_bpe_lstm_lm_baselines()
