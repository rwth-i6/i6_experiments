from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import run_librispeech_baselines

# from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.switchboard import run_switchboard_baselines
# from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.tedlium2 import run_tedlium2_baselines


def main() -> None:
    run_librispeech_baselines()
    # run_switchboard_baselines()
    # run_tedlium2_baselines()
