from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import recognition, run_all, training
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.report import register_recog_report


def _max_beam_size_values(base: int):
    sizes = [1, 2]
    while True:
        b = sizes[-1] * 2
        if b > base:
            break
        sizes.append(b)
    return sizes


def _max_beam_size_recog_variants(make_variants):
    variants = []
    for make_variant in make_variants:
        base_variant = make_variant()
        base = base_variant.search_algorithm_params.max_beam_sizes[-1]
        for max_beam_size in _max_beam_size_values(base):
            max_beam_sizes = [1024] * (len(base_variant.search_algorithm_params.max_beam_sizes) - 1) + [max_beam_size]
            variant = make_variant()
            variant.search_algorithm_params.max_beam_sizes = max_beam_sizes
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
            variant.descriptor += f"_max-beam-size-{max_beam_size}"
            variants.append(variant)
    return variants


def _ctc_bpe_max_beam_size_variants():
    return _max_beam_size_recog_variants(
        [
            recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant,
            recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant,
            recognition.ctc_bpe.default_offline_tree_recog_variant,
            recognition.ctc_bpe.default_offline_tree_4gram_recog_variant,
            recognition.ctc_bpe.default_offline_tree_lstm_recog_variant,
            recognition.ctc_bpe.default_offline_tree_trafo_recog_variant,
        ]
    )


def _ctc_phoneme_max_beam_size_variants():
    return _max_beam_size_recog_variants(
        [
            recognition.ctc_phoneme.default_offline_4gram_recog_variant,
        ]
    )


def _ffnn_transducer_bpe_max_beam_size_variants():
    return _max_beam_size_recog_variants(
        [
            recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_tree_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_tree_lstm_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant,
        ]
    )


def _full_ctx_transducer_bpe_max_beam_size_variants():
    return _max_beam_size_recog_variants(
        [
            recognition.full_ctx_transducer_bpe.default_offline_lexfree_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_lexfree_lstm_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_tree_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_tree_4gram_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_tree_lstm_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant,
            recognition.full_ctx_transducer_bpe.default_offline_tree_trafo_recog_variant,
            # recognition.full_ctx_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu,
        ]
    )


def run() -> None:
    base_models, recog_results = run_all()

    # ================
    # === Training ===
    # ================

    bpe_5k_ctc_model = training.ctc_bpe.run(
        descriptor="ctc_bpe-5k",
        model_config=training.ctc_bpe.get_model_config(bpe_size=5000),
        train_options=training.ctc_bpe.get_train_options(bpe_size=5000),
    )

    model_config = training.ctc_bpe.get_model_config(bpe_size=10000)
    model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]  # type: ignore
    bpe_10k_ctc_model = training.ctc_bpe.run(
        descriptor="ctc_bpe-10k",
        model_config=model_config,
        train_options=training.ctc_bpe.get_train_options(bpe_size=10000),
    )

    model_config = training.ctc_bpe.get_model_config(layer_size=384)
    model_config.conformer_cfg.block_cfg.mhsa_cfg.num_att_heads = 6
    small_ctc_model = training.ctc_bpe.run(descriptor="ctc_bpe_small", model_config=model_config)

    bpe_5k_aed_model = training.aed_bpe.run(
        descriptor="aed_bpe-5k",
        model_config=training.aed_bpe.get_model_config(bpe_size=5000),
        train_options=training.aed_bpe.get_train_options(bpe_size=5000),
    )

    bpe_10k_aed_model = training.aed_bpe.run(
        descriptor="aed_bpe-10k",
        model_config=training.aed_bpe.get_model_config(bpe_size=10000),
        train_options=training.aed_bpe.get_train_options(bpe_size=10000),
    )

    train_options = training.aed_bpe.get_train_options()
    train_options.ctc_loss_scale = 0.0
    train_options.lr_config.const_lr_1 = 1e-05 / 10  # type: ignore
    train_options.lr_config.const_lr_2 = 1e-05  # type: ignore
    train_options.lr_config.decayed_lr = 1e-05 / 10  # type: ignore
    no_ctc_aed_model = training.aed_bpe.run(descriptor="aed_no-ctc-loss", train_options=train_options)

    model_config = training.ffnn_transducer_bpe.get_model_config(bpe_size=10000)
    model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]  # type: ignore
    train_options = training.ffnn_transducer_bpe.get_train_options(bpe_size=10000)
    train_options.gpu_mem_rqmt = 48
    bpe_10k_transducer_model = training.ffnn_transducer_bpe.run(
        descriptor="ffnn_transducer_bpe-10k", model_config=model_config, train_options=train_options
    )

    # ===================
    # === Recognition ===
    # ===================

    variants = list(
        filter(
            lambda variant: variant.bpe_lstm_lm_scale == 0 and variant.bpe_trafo_lm_scale == 0,
            recognition.ctc_bpe.default_recog_variants(),
        )
    )
    recog_results.extend(recognition.ctc_bpe.run(model=bpe_5k_ctc_model, variants=variants))
    recog_results.extend(recognition.ctc_bpe.run(model=bpe_10k_ctc_model, variants=variants))
    recog_results.extend(recognition.ctc_bpe.run(model=small_ctc_model))

    variants = list(
        filter(lambda variant: variant.bpe_lstm_lm_scale == 0, recognition.aed_bpe.default_recog_variants())
    )
    recog_results.extend(recognition.aed_bpe.run(model=bpe_5k_aed_model, variants=variants))
    recog_results.extend(recognition.aed_bpe.run(model=bpe_10k_aed_model, variants=variants))
    recog_results.extend(
        recognition.aed_bpe.run(model=no_ctc_aed_model, variants=[recognition.aed_bpe.default_lexfree_recog_variant()])
    )

    variants = []
    # for beam_size_1 in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     for beam_size_2 in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for beam_size_1 in [512]:
        for beam_size_2 in [512]:
            if beam_size_2 > beam_size_1:
                continue
            for score_threshold_1 in [2.0, 6.0, 10.0, 14.0]:
                for score_threshold_2 in [2.0, 6.0, 10.0, 14.0]:
                    variant = recognition.aed_bpe.default_lexfree_aed_ctc_recog_variant()
                    variant.descriptor += (
                        f"_beam-{beam_size_1}-{beam_size_2}_score-{score_threshold_1}-{score_threshold_2}"
                    )
                    variant.ctc_score_scale = 0.3
                    variant.search_algorithm_params.max_beam_sizes = [beam_size_1, beam_size_2]
                    variant.search_algorithm_params.score_thresholds = [score_threshold_1, score_threshold_2]
                    variants.append(variant)

                    variant = recognition.aed_bpe.default_lexfree_aed_ctc_timesync_recog_variant()
                    variant.descriptor += (
                        f"_beam-{beam_size_1}-{beam_size_2}_score-{score_threshold_1}-{score_threshold_2}"
                    )
                    variant.ctc_score_scale = 0.3
                    variant.search_algorithm_params.max_beam_sizes = [beam_size_1, beam_size_2]
                    variant.search_algorithm_params.score_thresholds = [score_threshold_1, score_threshold_2]
                    variants.append(variant)
    recog_results.extend(
        recognition.aed_bpe.run(model=base_models["aed_bpe"], variants=variants, corpora=["dev-other"])
    )

    variants = list(
        filter(
            lambda variant: variant.bpe_lstm_lm_scale == 0,
            recognition.ffnn_transducer_bpe.default_recog_variants(),
        )
    )
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=bpe_10k_transducer_model, variants=variants))

    # recog_results.extend(
    #     recognition.ctc_bpe.run(
    #         model=base_models["ctc_bpe"],
    #         variants=_ctc_bpe_max_beam_size_variants(),
    #         corpora=["dev-other"],
    #     )
    # )
    # recog_results.extend(
    #     recognition.ctc_phoneme.run(
    #         model=base_models["ctc_phoneme"],
    #         variants=_ctc_phoneme_max_beam_size_variants(),
    #         corpora=["dev-other"],
    #     )
    # )
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"],
    #         variants=_ffnn_transducer_bpe_max_beam_size_variants(),
    #         corpora=["dev-other"],
    #     )
    # )
    # recog_results.extend(
    #     recognition.full_ctx_transducer_bpe.run(
    #         model=base_models["full_ctx_transducer_bpe"],
    #         variants=_full_ctx_transducer_bpe_max_beam_size_variants(),
    #         corpora=["dev-other"],
    #     )
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64]:
    # for beam_size in [4, 16, 64]:
    #     for score_threshold in [8.0, 10.0, None]:
    #         variant = recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}"
    #         variant.search_algorithm_params.max_beam_sizes = [64, beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds[-1] = score_threshold
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64]:
    # for beam_size in [4, 16, 64]:
    #     for score_threshold in [8.0, 10.0, None]:
    #         variant = recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}_gpu"
    #         variant.search_mode_params.gpu_mem_rqmt = 24
    #         variant.search_algorithm_params.max_beam_sizes = [64, beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds[-1] = score_threshold
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [14.0, 16.0, None]:
    #         variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}"
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [14.0, 16.0, None]:
    #         variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}_gpu"
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [12.0, None]:
    #         variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}"
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64]:
    # for beam_size in [4, 16, 64]:
    #     for score_threshold in [8.0, 10.0, None]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}"
    #         variant.search_algorithm_params.max_beam_sizes = [64, beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds[-1] = score_threshold
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
    #     )
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64]:
    # for beam_size in [4, 16, 64]:
    #     for score_threshold in [8.0, 10.0, None]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}_gpu"
    #         variant.search_mode_params.gpu_mem_rqmt = 24
    #         variant.search_algorithm_params.max_beam_sizes = [64, beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds[-1] = score_threshold
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
    #     )
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [14.0, 16.0, None]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}"
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
    #     )
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [14.0, 16.0, None]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}_gpu"
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
    #     )
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [12.0, None]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}"
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
    #     )
    # )
    #
    # variants = []
    # # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    # for beam_size in [4, 16, 64, 256]:
    #     for score_threshold in [12.0, None]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    #         variant.descriptor += f"_beam-{beam_size}_score-{score_threshold}_gpu"
    #         variant.search_mode_params.gpu_mem_rqmt = 24
    #         variant.search_algorithm_params.max_beam_sizes = [beam_size]
    #         if score_threshold is None:
    #             variant.search_algorithm_params.score_thresholds = None
    #             variant.search_algorithm_params.word_end_score_threshold = None
    #         else:
    #             variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
    #     )
    # )

    # Table 1
    variants = []
    variant = recognition.ctc_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
    variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other", "test-other"])
    )

    variants = []
    variant = recognition.ctc_phoneme.default_offline_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_phoneme.default_offline_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_phoneme.default_offline_trafo_gpu_recog_variant()
    variants.append(variant)

    recog_results.extend(
        recognition.ctc_phoneme.run(
            model=base_models["ctc_phoneme"], variants=variants, corpora=["dev-other", "test-other"]
        )
    )

    variants = []

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
    variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other", "test-other"]
        )
    )

    # Table 3
    variants = []
    for score_threshold, max_beam_size in [
        (0.0, 1),
        (2.0, 16),
        (4.0, 16),
        (8.0, 128),
        (12.0, 256),
        (16.0, 256),
        (16.0, 1024),
    ]:
        variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        variant.search_algorithm_params.score_thresholds = [score_threshold]
        variant.compute_search_errors = True
        variants.append(variant)

        # variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        # variant.search_algorithm_params.score_thresholds = [score_threshold]
        # variant.compute_search_errors = True
        # variants.append(variant)
    recog_results.extend(
        recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    )

    # Table 4
    variants = []
    for score_threshold, max_beam_size in [(0.0, 1), (2.0, 16), (4.0, 16), (8.0, 128), (12.0, 256), (16.0, 1024)]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        variant.search_algorithm_params.score_thresholds = [score_threshold]
        variant.compute_search_errors = True
        variants.append(variant)

        # variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        # variant.search_algorithm_params.score_thresholds = [score_threshold]
        # variant.compute_search_errors = True
        # variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    # Table 7
    variants = []
    for score_threshold in [True, False]:
        for max_beam_size in [4, 16, 64, 256]:
            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_beam-{max_beam_size}"
            if score_threshold:
                variant.descriptor += "_dyn"
            else:
                variant.descriptor += "_fixed"
                variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)

            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_beam-{max_beam_size}_gpu"
            variant.search_mode_params.gpu_mem_rqmt = 24
            if score_threshold:
                variant.descriptor += "_dyn"
            else:
                variant.descriptor += "_fixed"
                variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    # Table 8
    variants = []
    for score_threshold in [True, False]:
        for max_beam_size in [4, 16, 64, 128, 512, 1024]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
            # variant.descriptor += f"_beam-{max_beam_size}"
            # variant.search_mode_params.gpu_mem_rqmt = 24
            # if score_threshold:
            #     variant.descriptor += "_dyn"
            # else:
            #     variant.descriptor += "_fixed"
            #     variant.search_algorithm_params.score_thresholds = None
            #     variant.search_algorithm_params.word_end_score_threshold = None
            # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            # variants.append(variant)

            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
            variant.descriptor += f"_beam-{max_beam_size}"
            variant.search_mode_params.gpu_mem_rqmt = 24
            variant.search_mode_params.mem_rqmt = 32
            if score_threshold:
                variant.descriptor += "_dyn"
            else:
                variant.descriptor += "_fixed"
                variant.search_algorithm_params.score_thresholds = None
                variant.search_algorithm_params.word_end_score_threshold = None
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    # Table 9
    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_lexfree_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.search_algorithm_params.max_beam_sizes = [128]
        variant.search_mode_params.mem_rqmt = 32
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(model=base_models["ctc_bpe"], variants=variants, corpora=["dev-other"])
    )

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ctc_phoneme.default_offline_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_phoneme.default_offline_trafo_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_phoneme.default_offline_trafo_gpu_recog_variant()
        # variant.search_algorithm_params.max_beam_sizes = [128]
        variant.search_mode_params.mem_rqmt = 32
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    recog_results.extend(
        recognition.ctc_phoneme.run(model=base_models["ctc_phoneme"], variants=variants, corpora=["dev-other"])
    )

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.search_algorithm_params.max_beam_sizes = [512]
        variant.search_mode_params.mem_rqmt = 32
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    # Figure 2
    variants = []

    for score_threshold in [None, 2.0, 4.0, 6.0]:
        for max_beam_size in [4, 8, 16, 32, 64, 128]:
            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold, score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)

    for score_threshold in [None, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for max_beam_size in [4, 8, 16, 32, 64, 128]:
            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_gpu"
            variant.search_mode_params.gpu_mem_rqmt = 24
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold, score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)

    for score_threshold in [None, 4.0, 8.0, 12.0, 16.0]:
        for max_beam_size in [16, 64, 256, 512, 1024]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            variant.search_mode_params.mem_rqmt = 32
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
                variant.search_algorithm_params.word_end_score_threshold = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

    for score_threshold in [None, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0]:
        for max_beam_size in [16, 32, 64, 128, 256, 512, 1024]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            variant.search_mode_params.mem_rqmt = 32
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
                variant.search_algorithm_params.word_end_score_threshold = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=base_models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    register_recog_report(recog_results)
