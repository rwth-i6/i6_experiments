from i6_experiments.example_setups.seq2seq_rasr_2025.data.loquacious.recog import LoquaciousTreeTimesyncRecogParams
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import (
    recognition,
    training,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import (
    run_medium as _run_medium,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import (
    run_small as _run_small,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.learning_rates import (
    ConstConstDecayLRConfig,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import (
    LexiconfreeLabelsyncRecogParams,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.report import register_recog_report


def run_small() -> None:
    _run_small()

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_ffnn_transducer/baseline_model"):
            bpe_transducer_model = training.small.bpe_ffnn_transducer.run()

            with ExperimentContext("search_scale_tuning"):
                recog_variants = []
                for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4]:
                    for lm_scale in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
                        variant = recognition.bpe_ffnn_transducer.default_offline_tree_4gram_recog_variant()
                        variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
                        variant.ilm_scale = ilm_scale
                        assert isinstance(variant.search_algorithm_params, LoquaciousTreeTimesyncRecogParams)
                        assert variant.search_algorithm_params.word_lm_params is not None
                        variant.search_algorithm_params.word_lm_params.scale = lm_scale
                        recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=bpe_transducer_model,
                        train_corpus_key="train.small",
                        corpora=["dev.short"],
                        variants=recog_variants,
                    )
                )

            with ExperimentContext("search_space_tuning"):
                recog_variants = []
                for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
                    for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                        variant = recognition.bpe_ffnn_transducer.default_offline_tree_4gram_recog_variant()
                        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
                        variant.search_algorithm_params.score_thresholds = [score_threshold]
                        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                        recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=bpe_transducer_model,
                        variants=recog_variants,
                        train_corpus_key="train.small",
                        corpora=["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"],
                    )
                )


def run_medium() -> None:
    _run_medium()

    with ExperimentContext("training"):
        with ExperimentContext("bpe_ffnn_transducer/bpe_10k"):
            model_config = training.medium.bpe_ffnn_transducer.get_model_config(bpe_size=10000)
            model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]
            train_options = training.medium.bpe_ffnn_transducer.get_train_options(bpe_size=10000)
            train_options.batch_size = 12_000 * 160
            train_options.accum_grad_multiple_step = 2
            train_options.gpu_mem_rqmt = 48
            bpe_10k_transducer_model = training.medium.bpe_ffnn_transducer.run(
                model_config=model_config, train_options=train_options
            )

        with ExperimentContext("bpe_aed"):
            with ExperimentContext("no_ctc"):
                train_options = training.medium.bpe_aed.get_train_options()
                train_options.ctc_loss_scale = 0.0
                assert isinstance(train_options.lr_config, ConstConstDecayLRConfig)
                train_options.lr_config.const_lr_1 = 1e-05
                train_options.lr_config.const_lr_2 = 1e-04
                train_options.lr_config.decayed_lr = 1e-05
                no_ctc_aed_model = training.medium.bpe_aed.run(train_options=train_options)

            with ExperimentContext("bpe_10k"):
                model_config = training.medium.bpe_aed.get_model_config(bpe_size=10000)
                model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]
                train_options = training.medium.bpe_aed.get_train_options(bpe_size=10000)
                train_options.gpu_mem_rqmt = 48
                bpe_10k_aed_model = training.medium.bpe_aed.run(model_config=model_config, train_options=train_options)

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_ffnn_transducer"):
            with ExperimentContext("baseline_model"):
                bpe_transducer_model = training.medium.bpe_ffnn_transducer.run()
                with ExperimentContext("search_space_tuning"):
                    recog_variants = []
                    for lm_scale in [0.3, 0.6, 0.8]:
                        for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
                            for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                                variant = recognition.bpe_ffnn_transducer.default_offline_tree_4gram_recog_variant()
                                variant.descriptor += f"_lm-{lm_scale}_score-{score_threshold}_beam-{max_beam_size}"
                                variant.search_algorithm_params.score_thresholds = [score_threshold]
                                variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                                assert isinstance(variant.search_algorithm_params, LoquaciousTreeTimesyncRecogParams)
                                assert variant.search_algorithm_params.word_lm_params is not None
                                variant.search_algorithm_params.word_lm_params.scale = lm_scale
                                recog_variants.append(variant)
                    register_recog_report(
                        recognition.bpe_ffnn_transducer.run(
                            model=bpe_transducer_model,
                            variants=recog_variants,
                            train_corpus_key="train.medium",
                            corpora=["dev.short"],
                        )
                    )

                with ExperimentContext("search_scale_tuning"):
                    recog_variants = []
                    for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4]:
                        for lm_scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                            variant = recognition.bpe_ffnn_transducer.default_offline_tree_4gram_recog_variant()
                            variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
                            variant.ilm_scale = ilm_scale
                            assert isinstance(variant.search_algorithm_params, LoquaciousTreeTimesyncRecogParams)
                            assert variant.search_algorithm_params.word_lm_params is not None
                            variant.search_algorithm_params.word_lm_params.scale = lm_scale
                            recog_variants.append(variant)
                    register_recog_report(
                        recognition.bpe_ffnn_transducer.run(
                            model=bpe_transducer_model,
                            variants=recog_variants,
                            train_corpus_key="train.medium",
                            corpora=["dev.short"],
                        )
                    )

            with ExperimentContext("bpe_10k/baseline_recog"):
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(model=bpe_10k_transducer_model, train_corpus_key="train.medium")
                )

        with ExperimentContext("phoneme_ffnn_transducer/baseline_model"):
            phoneme_transducer_model = training.medium.phoneme_ffnn_transducer.run()
            with ExperimentContext("search_scale_tuning"):
                recog_variants = []
                for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4]:
                    for lm_scale in [0.3, 0.5, 0.6, 0.7, 0.8]:
                        variant = recognition.phoneme_ffnn_transducer.default_offline_4gram_recog_variant()
                        variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
                        variant.ilm_scale = ilm_scale
                        assert isinstance(variant.search_algorithm_params, LoquaciousTreeTimesyncRecogParams)
                        assert variant.search_algorithm_params.word_lm_params is not None
                        variant.search_algorithm_params.word_lm_params.scale = lm_scale
                        recog_variants.append(variant)
                register_recog_report(
                    recognition.phoneme_ffnn_transducer.run(
                        model=phoneme_transducer_model, variants=recog_variants, corpora=["dev.short"]
                    )
                )

            with ExperimentContext("search_space_tuning"):
                recog_variants = []
                for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
                    for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                        variant = recognition.phoneme_ffnn_transducer.default_offline_4gram_recog_variant()
                        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
                        variant.search_algorithm_params.score_thresholds = [score_threshold]
                        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                        recog_variants.append(variant)
                register_recog_report(
                    recognition.phoneme_ffnn_transducer.run(
                        model=phoneme_transducer_model, variants=recog_variants, corpora=["dev.short"]
                    )
                )

        with ExperimentContext("bpe_aed"):
            with ExperimentContext("baseline_model"):
                bpe_aed_model = training.medium.bpe_aed.run()

                with ExperimentContext("search_space_tuning"):
                    recog_variants = []
                    for max_beam_size in [1, 4, 8, 16, 32, 64, 128]:
                        for score_threshold in [None, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0]:
                            for length_norm_scale in [1.2, 1.5]:
                                variant = recognition.bpe_aed.default_lexfree_recog_variant()
                                variant.descriptor += (
                                    f"_beam-{max_beam_size}_score-{score_threshold}_ln-{length_norm_scale}"
                                )
                                variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                                variant.search_algorithm_params.score_thresholds = (
                                    [score_threshold] if score_threshold is not None else None
                                )
                                assert isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)
                                variant.search_algorithm_params.length_norm_scale = length_norm_scale
                                recog_variants.append(variant)
                    register_recog_report(
                        recognition.bpe_aed.run(
                            model=bpe_aed_model,
                            variants=recog_variants,
                            train_corpus_key="train.medium",
                            corpora=["dev.short"],
                        )
                    )

                with ExperimentContext("aed_ctc_search"):
                    recog_variants = []
                    for ctc_score_scale in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
                        variant = recognition.bpe_aed.default_lexfree_aed_ctc_recog_variant()
                        variant.descriptor += f"_ctc-scale-{ctc_score_scale}"
                        variant.ctc_score_scale = ctc_score_scale
                        recog_variants.append(variant)
                    register_recog_report(
                        recognition.bpe_aed.run(
                            model=bpe_aed_model,
                            variants=recog_variants,
                            train_corpus_key="train.medium",
                            corpora=["dev.short"],
                        )
                    )

            with ExperimentContext("no-ctc/baseline_recog"):
                variants = list(
                    filter(lambda variant: variant.ctc_score_scale == 0, recognition.bpe_aed.default_recog_variants())
                )
                register_recog_report(
                    recognition.bpe_aed.run(model=no_ctc_aed_model, train_corpus_key="train.medium", variants=variants)
                )

            with ExperimentContext("bpe_10k/baseline_recog"):
                register_recog_report(recognition.bpe_aed.run(model=bpe_10k_aed_model, train_corpus_key="train.medium"))
