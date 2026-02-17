from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import recognition, run_all, training
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.learning_rates import (
    ConstConstDecayLRConfig,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import (
    LexiconfreeLabelsyncRecogParams,
    LexiconfreeTimesyncRecogParams,
    TreeTimesyncRecogParams,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.report import register_recog_report


def run() -> None:
    run_all()

    with ExperimentContext("training"):
        with ExperimentContext("bpe_ctc"):
            with ExperimentContext("bpe_5k"):
                model_config = training.bpe_ctc.get_model_config(bpe_size=5000)
                train_options = training.bpe_ctc.get_train_options(bpe_size=5000)
                bpe_5k_ctc_model = training.bpe_ctc.run(model_config=model_config, train_options=train_options)

            with ExperimentContext("bpe_10k"):
                model_config = training.bpe_ctc.get_model_config(bpe_size=10000)
                model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]
                train_options = training.bpe_ctc.get_train_options(bpe_size=10000)
                bpe_10k_ctc_model = training.bpe_ctc.run(model_config=model_config, train_options=train_options)

            with ExperimentContext("small"):
                model_config = training.bpe_ctc.get_model_config(layer_size=384)
                model_config.conformer_cfg.block_cfg.mhsa_cfg.num_att_heads = 6
                small_ctc_model = training.bpe_ctc.run(model_config=model_config)

        with ExperimentContext("bpe_aed"):
            with ExperimentContext("bpe_5k"):
                model_config = training.bpe_aed.get_model_config(bpe_size=5000)
                train_options = training.bpe_aed.get_train_options(bpe_size=5000)
                bpe_5k_aed_model = training.bpe_aed.run(model_config=model_config, train_options=train_options)

            with ExperimentContext("bpe_10k"):
                model_config = training.bpe_aed.get_model_config(bpe_size=10000)
                train_options = training.bpe_aed.get_train_options(bpe_size=10000)
                bpe_10k_aed_model = training.bpe_aed.run(model_config=model_config, train_options=train_options)

            with ExperimentContext("no-ctc"):
                train_options = training.bpe_aed.get_train_options()
                train_options.ctc_loss_scale = 0.0
                assert isinstance(train_options.lr_config, ConstConstDecayLRConfig)
                train_options.lr_config.const_lr_1 = 1e-05 / 10
                train_options.lr_config.const_lr_2 = 1e-05
                train_options.lr_config.decayed_lr = 1e-05 / 10

                no_ctc_aed_model = training.bpe_aed.run(train_options=train_options)

        with ExperimentContext("bpe_ffnn_transducer/bpe_10k"):
            model_config = training.bpe_ffnn_transducer.get_model_config(bpe_size=10000)
            model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]
            train_options = training.bpe_ffnn_transducer.get_train_options(bpe_size=10000)
            train_options.gpu_mem_rqmt = 48
            bpe_10k_transducer_model = training.bpe_ffnn_transducer.run(
                model_config=model_config, train_options=train_options
            )

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_ctc"):
            with ExperimentContext("baseline_model"):
                bpe_ctc_model = training.bpe_ctc.run()
                with ExperimentContext("lexfree_lstm_tuning"):
                    recog_variants = []

                    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0]:
                        variant = recognition.bpe_ctc.default_offline_lexfree_lstm_recog_variant()
                        variant.descriptor += f"_lm-{lm_scale}"
                        assert isinstance(variant.search_algorithm_params, LexiconfreeTimesyncRecogParams)
                        variant.search_algorithm_params.max_beam_sizes = [2048, 2048]
                        variant.search_algorithm_params.score_thresholds = [16.0, 16.0]
                        variant.bpe_lstm_lm_scale = lm_scale
                        recog_variants.append(variant)

                    register_recog_report(
                        recognition.bpe_ctc.run(model=bpe_ctc_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("tree_tuning"):
                    recog_variants = []

                    for word_end_score_threshold in [0.1, 0.3, 0.5, 0.7]:
                        variant = recognition.bpe_ctc.default_offline_tree_recog_variant()
                        variant.descriptor += f"_we-score-{word_end_score_threshold}"
                        assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                        variant.search_algorithm_params.score_thresholds = [6.0]
                        variant.search_algorithm_params.max_beam_sizes = [2048]
                        variant.search_algorithm_params.max_word_end_beam_size = None
                        variant.search_algorithm_params.word_end_score_threshold = word_end_score_threshold
                        recog_variants.append(variant)

                    register_recog_report(
                        recognition.bpe_ctc.run(model=bpe_ctc_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("tree_4gram_tuning"):
                    recog_variants = []

                    for word_end_score_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        variant = recognition.bpe_ctc.default_offline_tree_4gram_recog_variant()
                        variant.descriptor += f"_we-score-{word_end_score_threshold}"
                        assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                        assert variant.search_algorithm_params.word_lm_params is not None
                        variant.search_algorithm_params.word_lm_params.scale = 0.6
                        variant.search_algorithm_params.score_thresholds = [12.0]
                        variant.search_algorithm_params.max_beam_sizes = [256]
                        variant.search_algorithm_params.word_end_score_threshold = word_end_score_threshold
                        recog_variants.append(variant)

                    register_recog_report(
                        recognition.bpe_ctc.run(model=bpe_ctc_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("tree_lstm_tuning"):
                    recog_variants = []

                    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0]:
                        variant = recognition.bpe_ctc.default_offline_tree_lstm_recog_variant()
                        variant.descriptor += f"_lm-{lm_scale}"
                        variant.search_algorithm_params.max_beam_sizes = [2048, 2048]
                        variant.search_algorithm_params.score_thresholds = [16.0, 16.0]
                        variant.bpe_lstm_lm_scale = lm_scale
                        recog_variants.append(variant)

                    register_recog_report(
                        recognition.bpe_ctc.run(model=bpe_ctc_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("tree_4gram_lstm_tuning"):
                    recog_variants = []

                    for word_lm_scale in [0.2, 0.3, 0.4, 0.6, 0.8]:
                        for bpe_lm_scale in [0.3, 0.6, 0.8, 1.0]:
                            variant = recognition.bpe_ctc.default_offline_tree_lstm_4gram_recog_variant()
                            variant.descriptor += f"_word-lm-{word_lm_scale}_bpe-lm-{bpe_lm_scale}"
                            variant.search_algorithm_params.max_beam_sizes = [2048, 2048]
                            variant.search_algorithm_params.score_thresholds = [16.0, 16.0]
                            assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                            assert variant.search_algorithm_params.word_lm_params is not None
                            variant.search_algorithm_params.word_lm_params.scale = word_lm_scale
                            variant.bpe_lstm_lm_scale = bpe_lm_scale
                            recog_variants.append(variant)

                    register_recog_report(
                        recognition.bpe_ctc.run(model=bpe_ctc_model, variants=recog_variants, corpora=["dev-other"])
                    )

            with ExperimentContext("bpe_5k/baseline_recog"):
                variants = list(
                    filter(lambda variant: variant.bpe_lstm_lm_scale == 0, recognition.bpe_ctc.default_recog_variants())
                )
                register_recog_report(recognition.bpe_ctc.run(model=bpe_5k_ctc_model, variants=variants))

            with ExperimentContext("bpe_10k/baseline_recog"):
                variants = list(
                    filter(lambda variant: variant.bpe_lstm_lm_scale == 0, recognition.bpe_ctc.default_recog_variants())
                )
                register_recog_report(recognition.bpe_ctc.run(model=bpe_10k_ctc_model, variants=variants))

            with ExperimentContext("small/baseline_recog"):
                register_recog_report(recognition.bpe_ctc.run(model=small_ctc_model))

        with ExperimentContext("bpe_aed"):
            aed_model = training.bpe_aed.run()

            with ExperimentContext("baseline_model"):
                with ExperimentContext("lexfree_tuning"):
                    recog_variants = []
                    for beam_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                        for length_norm_scale in [0.8, 1.0, 1.2]:
                            recog_variant = recognition.bpe_aed.default_lexfree_recog_variant()
                            recog_variant.descriptor += f"_beam-{beam_size}_ln-{length_norm_scale}"
                            recog_variant.search_algorithm_params.max_beam_sizes = [beam_size]
                            recog_variant.search_algorithm_params.score_thresholds = None
                            assert isinstance(recog_variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)
                            recog_variant.search_algorithm_params.length_norm_scale = length_norm_scale
                            recog_variants.append(recog_variant)
                    register_recog_report(
                        recognition.bpe_aed.run(model=aed_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("lexfree_lstm_tuning"):
                    recog_variants = []
                    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0]:
                        for length_norm_scale in [0.8, 1.0, 1.2]:
                            recog_variant = recognition.bpe_aed.default_lexfree_lstm_recog_variant()
                            recog_variant.descriptor += f"_lm-{lm_scale}_ln-{length_norm_scale}"
                            recog_variant.search_algorithm_params.max_beam_sizes = [1024, 128]
                            recog_variant.search_algorithm_params.score_thresholds = None
                            assert isinstance(recog_variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)
                            recog_variant.search_algorithm_params.length_norm_scale = length_norm_scale
                            recog_variants.append(recog_variant)
                    register_recog_report(
                        recognition.bpe_aed.run(model=aed_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("lexfree_aed_ctc_labelsync_tuning"):
                    recog_variants = []
                    for ctc_scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        for length_norm_scale in [0.8, 1.0, 1.2]:
                            recog_variant = recognition.bpe_aed.default_lexfree_aed_ctc_recog_variant()
                            recog_variant.descriptor += f"_ctc-{ctc_scale}_ln-{length_norm_scale}"
                            recog_variant.search_algorithm_params.max_beam_sizes = [1024, 128]
                            recog_variant.search_algorithm_params.score_thresholds = None
                            assert isinstance(recog_variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)
                            recog_variant.search_algorithm_params.length_norm_scale = length_norm_scale
                            recog_variant.ctc_score_scale = ctc_scale
                            recog_variants.append(recog_variant)
                    register_recog_report(
                        recognition.bpe_aed.run(model=aed_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("lexfree_aed_ctc_timesync_tuning"):
                    recog_variants = []
                    for ctc_scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        recog_variant = recognition.bpe_aed.default_lexfree_aed_ctc_timesync_recog_variant()
                        recog_variant.descriptor += f"_ctc-{ctc_scale}"
                        assert isinstance(recog_variant.search_algorithm_params, LexiconfreeTimesyncRecogParams)
                        recog_variant.search_algorithm_params.max_beam_sizes = [1024, 128]
                        recog_variant.search_algorithm_params.score_thresholds = None
                        recog_variant.ctc_score_scale = ctc_scale
                        recog_variants.append(recog_variant)
                    register_recog_report(
                        recognition.bpe_aed.run(model=aed_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("tree_aed_ctc_tuning"):
                    recog_variants = []
                    for ctc_scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        recog_variant = recognition.bpe_aed.default_tree_aed_ctc_recog_variant()
                        recog_variant.descriptor += f"_ctc-{ctc_scale}"
                        assert isinstance(recog_variant.search_algorithm_params, TreeTimesyncRecogParams)
                        recog_variant.search_algorithm_params.max_beam_sizes = [1024, 128]
                        recog_variant.search_algorithm_params.score_thresholds = None
                        recog_variant.search_algorithm_params.max_word_end_beam_size = None
                        recog_variant.search_algorithm_params.word_end_score_threshold = None
                        recog_variant.ctc_score_scale = ctc_scale
                        recog_variants.append(recog_variant)
                    register_recog_report(
                        recognition.bpe_aed.run(model=aed_model, variants=recog_variants, corpora=["dev-other"])
                    )

                with ExperimentContext("tree_aed_ctc_4gram_tuning"):
                    recog_variants = []
                    for ctc_scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        for lm_scale in [0.3, 0.6, 0.9]:
                            recog_variant = recognition.bpe_aed.default_tree_aed_ctc_4gram_recog_variant()
                            recog_variant.descriptor += f"_ctc-{ctc_scale}_lm-{lm_scale}"
                            assert isinstance(recog_variant.search_algorithm_params, TreeTimesyncRecogParams)
                            assert recog_variant.search_algorithm_params.word_lm_params is not None
                            recog_variant.search_algorithm_params.max_beam_sizes = [1024, 128]
                            recog_variant.search_algorithm_params.word_lm_params.scale = lm_scale
                            recog_variant.search_algorithm_params.score_thresholds = None
                            recog_variant.search_algorithm_params.max_word_end_beam_size = None
                            recog_variant.search_algorithm_params.word_end_score_threshold = None
                            recog_variant.ctc_score_scale = ctc_scale
                            recog_variants.append(recog_variant)
                    register_recog_report(
                        recognition.bpe_aed.run(model=aed_model, variants=recog_variants, corpora=["dev-other"])
                    )

            with ExperimentContext("bpe_5k/baseline_recog"):
                variants = list(
                    filter(lambda variant: variant.bpe_lstm_lm_scale == 0, recognition.bpe_aed.default_recog_variants())
                )
                register_recog_report(recognition.bpe_aed.run(model=bpe_5k_aed_model, variants=variants))

            with ExperimentContext("bpe_10k/baseline_recog"):
                variants = list(
                    filter(lambda variant: variant.bpe_lstm_lm_scale == 0, recognition.bpe_aed.default_recog_variants())
                )
                register_recog_report(recognition.bpe_aed.run(model=bpe_10k_aed_model, variants=variants))

            with ExperimentContext("no-ctc/baseline_recog"):
                register_recog_report(
                    recognition.bpe_aed.run(
                        model=no_ctc_aed_model,
                        variants=[recognition.bpe_aed.default_lexfree_recog_variant()],
                    )
                )

        with ExperimentContext("phoneme_ctc/baseline_model/tree_4gram_tuning"):
            phoneme_ctc_model = training.phoneme_ctc.run()
            recog_variants = []
            for word_end_score_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                variant = recognition.phoneme_ctc.default_offline_4gram_recog_variant()
                variant.descriptor += f"_we-score-{word_end_score_threshold}"
                assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                variant.search_algorithm_params.max_beam_sizes = [512]
                variant.search_algorithm_params.score_thresholds = [12.0]
                variant.search_algorithm_params.max_word_end_beam_size = None
                variant.search_algorithm_params.word_end_score_threshold = word_end_score_threshold
                recog_variants.append(variant)

            register_recog_report(
                recognition.phoneme_ctc.run(model=phoneme_ctc_model, variants=recog_variants, corpora=["dev-other"])
            )

        with ExperimentContext("bpe_ffnn_transducer/baseline_model"):
            transducer_model = training.bpe_ffnn_transducer.run()

            with ExperimentContext("lexfree_lstm_tuning"):
                recog_variants = []
                for ilm_scale in [0.0, 0.1, 0.2, 0.3]:
                    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0]:
                        variant = recognition.bpe_ffnn_transducer.default_offline_lexfree_lstm_recog_variant()
                        variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
                        variant.search_algorithm_params.max_beam_sizes = [1024, 1024]
                        variant.search_algorithm_params.score_thresholds = [16.0, 16.0]
                        variant.ilm_scale = ilm_scale
                        variant.bpe_lstm_lm_scale = lm_scale
                        recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=transducer_model, variants=recog_variants, corpora=["dev-other"]
                    )
                )

            with ExperimentContext("tree_tuning"):
                recog_variants = []
                for max_beam_size in [2, 4, 8, 16, 32, 64, 256, 512, 1024]:
                    variant = recognition.bpe_ffnn_transducer.default_offline_tree_recog_variant()
                    variant.descriptor += f"_beam-{max_beam_size}"
                    variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                    variant.search_algorithm_params.score_thresholds = [6.0]
                    recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=transducer_model, variants=recog_variants, corpora=["dev-other"]
                    )
                )

            with ExperimentContext("tree_4gram_tuning"):
                recog_variants = []
                for word_end_score_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]:
                    variant = recognition.bpe_ffnn_transducer.default_offline_tree_4gram_recog_variant()
                    variant.descriptor += f"_we-score-{word_end_score_threshold}"
                    assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                    assert variant.search_algorithm_params.word_lm_params is not None
                    variant.search_algorithm_params.word_lm_params.scale = 0.6
                    variant.search_algorithm_params.max_beam_sizes = [128]
                    variant.search_algorithm_params.score_thresholds = [12.0]
                    variant.search_algorithm_params.max_word_end_beam_size = None
                    variant.search_algorithm_params.word_end_score_threshold = word_end_score_threshold
                    recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=transducer_model, variants=recog_variants, corpora=["dev-other"]
                    )
                )

            with ExperimentContext("tree_lstm_tuning"):
                recog_variants = []
                for ilm_scale in [0.0, 0.1, 0.2, 0.3]:
                    for lm_scale in [0.6, 0.8]:
                        variant = recognition.bpe_ffnn_transducer.default_offline_tree_lstm_recog_variant()
                        variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
                        assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                        variant.search_algorithm_params.max_beam_sizes = [1024, 1024]
                        variant.search_algorithm_params.score_thresholds = [16.0, 16.0]
                        variant.search_algorithm_params.max_word_end_beam_size = None
                        variant.search_algorithm_params.word_end_score_threshold = None
                        variant.ilm_scale = ilm_scale
                        variant.bpe_lstm_lm_scale = lm_scale
                        recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=transducer_model, variants=recog_variants, corpora=["dev-other"]
                    )
                )

            with ExperimentContext("tree_4gram_lstm_tuning"):
                recog_variants = []
                for ilm_scale in [0.0, 0.1, 0.2, 0.3]:
                    for word_lm_scale in [0.2, 0.3, 0.6]:
                        for bpe_lm_scale in [0.3, 0.6, 0.8]:
                            variant = recognition.bpe_ffnn_transducer.default_offline_tree_lstm_4gram_recog_variant()
                            variant.descriptor += f"_ilm-{ilm_scale}_word-lm-{word_lm_scale}_bpe-lm-{bpe_lm_scale}"
                            assert isinstance(variant.search_algorithm_params, TreeTimesyncRecogParams)
                            assert variant.search_algorithm_params.word_lm_params is not None
                            variant.search_algorithm_params.max_beam_sizes = [1024, 1024]
                            variant.search_algorithm_params.score_thresholds = [16.0, 16.0]
                            variant.search_algorithm_params.max_word_end_beam_size = None
                            variant.search_algorithm_params.word_end_score_threshold = None
                            variant.ilm_scale = ilm_scale
                            variant.bpe_lstm_lm_scale = lm_scale
                            variant.search_algorithm_params.word_lm_params.scale = word_lm_scale
                            recog_variants.append(variant)
                register_recog_report(
                    recognition.bpe_ffnn_transducer.run(
                        model=transducer_model, variants=recog_variants, corpora=["dev-other"]
                    )
                )

        with ExperimentContext("bpe_10k/baseline_recog"):
            variants = list(
                filter(
                    lambda variant: variant.bpe_lstm_lm_scale == 0,
                    recognition.bpe_ffnn_transducer.default_recog_variants(),
                )
            )
            register_recog_report(
                recognition.bpe_ffnn_transducer.run(model=bpe_10k_transducer_model, variants=variants)
            )
