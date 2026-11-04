from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext


def run_librispeech() -> None:
    from .librispeech import run

    run()


def run_loquacious_small() -> None:
    from .loquacious import run_small

    run_small()


def run_loquacious_medium() -> None:
    from .loquacious import run_medium

    run_medium()


def run_open_asr() -> None:
    from .open_asr import run

    run()


def run_wordwise_lm() -> None:
    from .wordwise_lm import py

    py()


def run_qwen_tuning() -> None:
    from .qwen_tuning import py

    py()


def run_earnings_qwen() -> None:
    from .earnings_qwen import py

    py()


def run_pretrained_corrections() -> None:
    from .pretrained_corrections import py
    py()


def run_speech_qwen() -> None:
    from .speech_qwen import py
    py()


def run_controlled_comparison() -> None:
    from .controlled_comparison import py
    py()


def run_error_complementarity() -> None:
    from .error_complementarity import py
    py()


def run_research_diagnostics() -> None:
    from .research_diagnostics import py
    py()


def main() -> None:
    with ExperimentContext("librispeech"):
        run_librispeech()

    with ExperimentContext("loquacious_small"):
        run_loquacious_small()

    with ExperimentContext("loquacious_medium"):
        run_loquacious_medium()

    # with ExperimentContext("loquacious_large"):
    #     run_loquacious_large()

    # The suite already prefixes its aliases and outputs with "open-asr/".
    run_open_asr()
    # Delayed-fusion word-LM pilots are parked; keep their exports for reports.
    run_qwen_tuning()
    run_earnings_qwen()
    run_pretrained_corrections()
    run_speech_qwen()
    run_controlled_comparison()
    run_error_complementarity()
    run_research_diagnostics()
    from .results import py as register_reports

    register_reports()


def py() -> None:
    main()
