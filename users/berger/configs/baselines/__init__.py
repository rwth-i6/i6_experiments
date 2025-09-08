from i6_core.returnn import PtCheckpoint
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments import librispeech as librispeech_experiments
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments import switchboard as switchboard_experiments
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments import tedlium2 as tedlium_experiments
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.report import (
    create_offline_recog_report_with_search_errors,
    create_streaming_recog_report,
)
from sisyphus import gs, tk


def main() -> None:
    with ExperimentContext("librispeech"):
        librispeech_experiments.run_all()

        with ExperimentContext("bpe_ctc"):
            with ExperimentContext("score_threshold_tuning"):
                train_job, model_config = librispeech_experiments.bpe_ctc.run_training()
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore

                recog_results = []
                for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]:

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_offline_tree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_tedlium_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )

                for score_threshold in [2.0, 4.0, 6.0, 8.0]:
                    for intermediate_score_threshold in [score_threshold // 2, score_threshold, score_threshold + 2]:
                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_lexiconfree_lstm(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-{score_threshold}_inter-score-{intermediate_score_threshold}",
                                score_threshold=score_threshold,
                                intermediate_score_threshold=intermediate_score_threshold,
                                max_beam_size=64,
                                intermediate_max_beam_size=64,
                            )
                        )

                for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                    # recog_results.extend(
                    #     librispeech_experiments.bpe_ctc.run_recognitions_offline_lexiconfree_lstm(
                    #         checkpoint=checkpoint,
                    #         model_config=model_config,
                    #         corpora=["dev-other"],
                    #         descriptor=f"recog_beam-{max_beam_size}",
                    #         score_threshold=...,
                    #         intermediate_score_threshold=...,
                    #         max_beam_size=max_beam_size,
                    #     )
                    # )

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_offline_tree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=6.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-13.0_beam-{max_beam_size}",
                            score_threshold=13.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-16.0_beam-{max_beam_size}",
                            score_threshold=16.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ctc.run_recognitions_tedlium_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=14.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                tk.register_report(
                    f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                    create_offline_recog_report_with_search_errors(recog_results),
                )

            with ExperimentContext("latency_tuning"):
                train_job, model_config = librispeech_experiments.bpe_ctc.run_training()
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore

                recog_results = []
                for chunk_history in [3, 5, 10]:
                    for chunk_center in [0.5, 1, 1.5]:
                        for chunk_future in [0.5, 1, 1.5]:
                            for maximum_delay in [5, 10, 15, 20]:
                                recog_results.extend(
                                    librispeech_experiments.bpe_ctc.run_recognitions_streaming_tree_4gram(
                                        checkpoint=checkpoint,
                                        model_config=model_config,
                                        corpora=["dev-other"],
                                        descriptor=f"ch-{chunk_history}_cc-{chunk_center}_cf-{chunk_future}_md-{maximum_delay}",
                                        chunk_center_seconds=chunk_center,
                                        chunk_history_seconds=chunk_history,
                                        chunk_future_seconds=chunk_future,
                                        maximum_stable_delay=maximum_delay,
                                    )
                                )

                tk.register_report(
                    f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt", create_streaming_recog_report(recog_results)
                )

            with ExperimentContext("bpe_5k"):
                model_config = librispeech_experiments.bpe_ctc.get_model_config(bpe_size=5000)
                train_options = librispeech_experiments.bpe_ctc.get_train_options(bpe_size=5000)

                train_job, model_config = librispeech_experiments.bpe_ctc.run_training(
                    model_config=model_config, train_options=train_options
                )
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
                librispeech_experiments.bpe_ctc.run_base_recognition_suite(
                    checkpoint=checkpoint,
                    model_config=model_config,
                    lexiconfree_lstm_search=False,
                    tree_4gram_lstm_search=False,
                    tree_trafo_search=False,
                    tree_trafo_kazuki_search=False,
                )

                with ExperimentContext("beam-size-tuning"):
                    recog_results = []
                    for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_tree(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_beam-{max_beam_size}",
                                score_threshold=6.0,
                                max_beam_size=max_beam_size,
                            )
                        )

                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-13.0_beam-{max_beam_size}",
                                score_threshold=13.0,
                                max_beam_size=max_beam_size,
                            )
                        )

                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-16.0_beam-{max_beam_size}",
                                score_threshold=16.0,
                                max_beam_size=max_beam_size,
                            )
                        )
                    tk.register_report(
                        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                        create_offline_recog_report_with_search_errors(recog_results),
                    )

            with ExperimentContext("small"):
                model_config = librispeech_experiments.bpe_ctc.get_model_config(layer_size=384, num_att_heads=6)

                ctc_small_train_job, _ = librispeech_experiments.bpe_ctc.run_training(model_config=model_config)
                checkpoint: PtCheckpoint = ctc_small_train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
                librispeech_experiments.bpe_ctc.run_base_recognition_suite(
                    checkpoint=checkpoint,
                    model_config=model_config,
                    tree_trafo_search=False,
                    tree_trafo_kazuki_search=False,
                )

                with ExperimentContext("beam-size-tuning"):
                    recog_results = []
                    for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_tree(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_beam-{max_beam_size}",
                                score_threshold=6.0,
                                max_beam_size=max_beam_size,
                            )
                        )

                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-13.0_beam-{max_beam_size}",
                                score_threshold=13.0,
                                max_beam_size=max_beam_size,
                            )
                        )

                        recog_results.extend(
                            librispeech_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-16.0_beam-{max_beam_size}",
                                score_threshold=16.0,
                                max_beam_size=max_beam_size,
                            )
                        )
                    tk.register_report(
                        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                        create_offline_recog_report_with_search_errors(recog_results),
                    )

        with ExperimentContext("bpe_aed"):
            with ExperimentContext("bpe_5k"):
                model_config = librispeech_experiments.bpe_aed.get_model_config(bpe_size=5000)
                train_options = librispeech_experiments.bpe_aed.get_train_options(bpe_size=5000)

                train_job, model_config = librispeech_experiments.bpe_aed.run_training(
                    model_config=model_config, train_options=train_options
                )
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
                librispeech_experiments.bpe_aed.run_base_recognition_suite(
                    checkpoint=checkpoint,
                    model_config=model_config,
                    lexiconfree_lstm_search=False,
                    tree_search=False,
                    tree_4gram_search=False,
                    tree_4gram_tedlium_search=False,
                    tree_trafo_search=False,
                    tree_trafo_kazuki_search=False,
                )

        with ExperimentContext("phoneme_ctc"):
            with ExperimentContext("score_threshold_tuning"):
                train_job, model_config = librispeech_experiments.phoneme_ctc.run_training()
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
                recog_results = []
                for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]:
                    recog_results.extend(
                        librispeech_experiments.phoneme_ctc.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            descriptor=f"recog_score-{score_threshold}",
                            corpora=["dev-other"],
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )

                for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                    recog_results.extend(
                        librispeech_experiments.phoneme_ctc.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            descriptor=f"recog_beam-{max_beam_size}",
                            corpora=["dev-other"],
                            score_threshold=12.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                tk.register_report(
                    f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                    create_offline_recog_report_with_search_errors(recog_results),
                )

        with ExperimentContext("bpe_ffnn_transducer"):
            with ExperimentContext("score_threshold_tuning"):
                train_job, model_config = librispeech_experiments.bpe_ffnn_transducer.run_training()
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
                recog_results = []
                for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]:
                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_lexiconfree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_tree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )
                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=5000,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_tedlium_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                        )
                    )

                for score_threshold in [2.0, 4.0, 6.0, 8.0]:
                    for intermediate_score_threshold in [score_threshold // 2, score_threshold, score_threshold + 2]:
                        recog_results.extend(
                            librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_lexiconfree_lstm(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-{score_threshold}_inter-score-{intermediate_score_threshold}",
                                score_threshold=score_threshold,
                                max_beam_size=64,
                                intermediate_max_beam_size=64,
                                intermediate_score_threshold=intermediate_score_threshold,
                            )
                        )
                for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_lexiconfree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=6.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_tree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=6.0,
                            max_beam_size=max_beam_size,
                        )
                    )
                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=12.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_tedlium_offline_tree_4gram(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=13.0,
                            max_beam_size=max_beam_size,
                        )
                    )

                for max_beam_size in [2, 4, 8, 16, 32, 64, 128]:
                    recog_results.extend(
                        librispeech_experiments.bpe_ffnn_transducer.run_recognitions_offline_lexiconfree_lstm(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_beam-{max_beam_size}",
                            score_threshold=6.0,
                            intermediate_score_threshold=4.0,
                            max_beam_size=max_beam_size,
                            intermediate_max_beam_size=max_beam_size,
                        )
                    )

                tk.register_report(
                    f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                    create_offline_recog_report_with_search_errors(recog_results),
                )

            with ExperimentContext("latency_tuning"):
                train_job, model_config = librispeech_experiments.bpe_ffnn_transducer.run_training()
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore

                recog_results = []
                for chunk_history in [3, 5, 10]:
                    for chunk_center in [0.5, 1, 1.5]:
                        for chunk_future in [0.5, 1, 1.5]:
                            for maximum_delay in [5, 10, 15, 20]:
                                recog_results.extend(
                                    librispeech_experiments.bpe_ffnn_transducer.run_recognitions_streaming_tree_4gram(
                                        checkpoint=checkpoint,
                                        model_config=model_config,
                                        corpora=["dev-other"],
                                        descriptor=f"ch-{chunk_history}_cc-{chunk_center}_cf-{chunk_future}_md-{maximum_delay}",
                                        chunk_center_seconds=chunk_center,
                                        chunk_history_seconds=chunk_history,
                                        chunk_future_seconds=chunk_future,
                                        maximum_stable_delay=maximum_delay,
                                    )
                                )

                tk.register_report(
                    f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt", create_streaming_recog_report(recog_results)
                )

        with ExperimentContext("bpe_aed"):
            with ExperimentContext("score_threshold_tuning"):
                train_job, model_config = librispeech_experiments.bpe_aed.run_training()
                checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
                recog_results = []
                for score_threshold in [0.1, 0.3]:
                    recog_results.extend(
                        librispeech_experiments.bpe_aed.run_recognitions_offline_lexiconfree(
                            checkpoint=checkpoint,
                            model_config=model_config,
                            corpora=["dev-other"],
                            descriptor=f"recog_score-{score_threshold}",
                            score_threshold=score_threshold,
                            max_beam_size=64,
                        )
                    )
                    for intermediate_score_threshold in [score_threshold // 2, score_threshold, score_threshold * 2]:
                        recog_results.extend(
                            librispeech_experiments.bpe_aed.run_recognitions_offline_lexiconfree_lstm(
                                checkpoint=checkpoint,
                                model_config=model_config,
                                corpora=["dev-other"],
                                descriptor=f"recog_score-{score_threshold}_inter-score-{intermediate_score_threshold}",
                                score_threshold=score_threshold,
                                intermediate_score_threshold=intermediate_score_threshold,
                                max_beam_size=64,
                                intermediate_max_beam_size=64,
                            )
                        )

                tk.register_report(
                    f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                    create_offline_recog_report_with_search_errors(recog_results),
                )

    with ExperimentContext("switchboard"):
        switchboard_experiments.run_all()

    with ExperimentContext("tedlium2"):
        tedlium_experiments.run_all()

        with ExperimentContext("bpe_ctc"):
            train_job, model_config = tedlium_experiments.bpe_ctc.run_training()
            checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
            recog_results = []

            for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0]:
                recog_results.extend(
                    tedlium_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                        checkpoint=checkpoint,
                        model_config=model_config,
                        corpora=["dev"],
                        descriptor=f"recog_score-{score_threshold}",
                        score_threshold=score_threshold,
                        max_beam_size=5000,
                    )
                )

            for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 200, 300, 400, 500]:
                recog_results.extend(
                    tedlium_experiments.bpe_ctc.run_recognitions_offline_tree_4gram(
                        checkpoint=checkpoint,
                        model_config=model_config,
                        corpora=["dev"],
                        descriptor=f"recog_score-13.0_beam-{max_beam_size}",
                        score_threshold=13.0,
                        max_beam_size=max_beam_size,
                    )
                )
            tk.register_report(
                f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
                create_offline_recog_report_with_search_errors(recog_results),
            )
