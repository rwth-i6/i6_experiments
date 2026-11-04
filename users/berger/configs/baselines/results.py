"""Live report collection only: never schedules recognitions or downloads."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.result_collection import (
        register_result_collection,
    )

    register_result_collection()


def main():
    py()
