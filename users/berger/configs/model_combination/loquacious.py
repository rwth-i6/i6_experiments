from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import recognition, training
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_large
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_medium as _run_medium
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_small as _run_small
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
            recognition.ctc_bpe.default_offline_lexfree_recog_variant,
            recognition.ctc_bpe.default_offline_tree_recog_variant,
            recognition.ctc_bpe.default_offline_tree_4gram_recog_variant,
        ]
    )


def _ffnn_transducer_bpe_max_beam_size_variants():
    return _max_beam_size_recog_variants(
        [
            recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_tree_recog_variant,
            recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant,
        ]
    )


def _ffnn_transducer_phoneme_max_beam_size_variants():
    return _max_beam_size_recog_variants(
        [
            recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant,
        ]
    )


def run_small() -> None:
    models, recog_results = _run_small()

    recog_variants = []
    for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for lm_scale in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
            variant.search_algorithm_params.word_lm_params.scale = lm_scale  # type: ignore
            recog_variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.small",
            variants=recog_variants,
            corpora=["dev.short"],
        )
    )

    recog_variants = []
    for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
        for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            recog_variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.small",
            variants=recog_variants,
            corpora=["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"],
        )
    )

    register_recog_report(recog_results)


def run_medium() -> None:
    models, recog_results = _run_medium()

    model_config = training.medium.ffnn_transducer_bpe.get_model_config(bpe_size=10000)
    model_config.conformer_cfg.frontend.cfg.pool_kernel_sizes = [(3, 1), (2, 1)]  # type: ignore
    train_options = training.medium.ffnn_transducer_bpe.get_train_options(bpe_size=10000)
    train_options.beam_size = 12_000 * 160
    train_options.accum_grad_multiple_step = 2
    train_options.gpu_mem_rqmt = 48
    ffnn_transducer_bpe_10k = training.medium.ffnn_transducer_bpe.run(
        descriptor="ffnn_transducer_bpe-10k",
        model_config=model_config,
        train_options=train_options,
    )

    train_options = training.medium.aed_bpe.get_train_options()
    train_options.ctc_loss_scale = 0.0
    train_options.lr_config.const_lr_1 = 1e-05  # type: ignore
    train_options.lr_config.const_lr_2 = 1e-04  # type: ignore
    train_options.lr_config.decayed_lr = 1e-05  # type: ignore
    aed_no_ctc = training.medium.aed_bpe.run(descriptor="aed_no-ctc", train_options=train_options)

    train_options = training.medium.aed_bpe.get_train_options(bpe_size=10000)
    train_options.gpu_mem_rqmt = 48
    aed_bpe_10k = training.medium.aed_bpe.run(
        descriptor="aed_bpe-10k",
        model_config=training.medium.aed_bpe.get_model_config(bpe_size=10000),
        train_options=train_options,
    )

    # recog_results.extend(
    #     recognition.ctc_bpe.run(
    #         model=models["ctc_bpe"],
    #         train_corpus_key="train.medium",
    #         variants=_ctc_bpe_max_beam_size_variants(),
    #         corpora=["dev.short"],
    #     )
    # )

    # variants = []
    # for score_threshold in [6.0, 8.0, 10.0]:
    #     for max_beam_size in [4, 16, 32, 64, 128, 256, 512, 1024]:
    #         variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    #         variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
    #         variant.search_algorithm_params.score_thresholds = [score_threshold]
    #         variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
    #         variants.append(variant)
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=models["ffnn_transducer_bpe"],
    #         train_corpus_key="train.medium",
    #         variants=variants,
    #         corpora=["dev.short"],
    #     )
    # )

    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=models["ffnn_transducer_bpe"],
    #         train_corpus_key="train.medium",
    #         variants=_ffnn_transducer_bpe_max_beam_size_variants(),
    #         corpora=["dev.short"],
    #     )
    # )
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(
    #         model=ffnn_transducer_bpe_10k,
    #         train_corpus_key="train.medium",
    #         variants=_ffnn_transducer_bpe_max_beam_size_variants(),
    #         corpora=["dev.short"],
    #     )
    # )
    # recog_results.extend(
    #     recognition.ffnn_transducer_phoneme.run(
    #         model=models["ffnn_transducer_phoneme"],
    #         variants=_ffnn_transducer_phoneme_max_beam_size_variants(),
    #         corpora=["dev.short"],
    #     )
    # )
    # recog_results.extend(
    #     recognition.ffnn_transducer_bpe.run(model=ffnn_transducer_bpe_10k, train_corpus_key="train.medium")
    # )
    #
    # recog_variants = []
    # for max_beam_size in [1, 4, 8, 16, 32, 64, 128]:
    #     for score_threshold in [None, 1.0, 2.0, 3.0, 4.0, 8.0]:
    #         for length_norm_scale in [None, 1.2, 1.5]:
    #             variant = recognition.aed_bpe.default_lexfree_aed_ctc_recog_variant()
    #             variant.descriptor += f"_beam-{max_beam_size}_score-{score_threshold}_ln-{length_norm_scale}"
    #             variant.search_algorithm_params.max_beam_sizes = [128, max_beam_size]
    #             variant.search_algorithm_params.score_thresholds = (
    #                 [8.0, score_threshold] if score_threshold is not None else None
    #             )
    #             variant.search_algorithm_params.length_norm_scale = length_norm_scale  # type: ignore
    #             recog_variants.append(variant)
    # recog_results.extend(
    #     recognition.aed_bpe.run(
    #         model=models["aed_bpe"], train_corpus_key="train.medium", variants=recog_variants, corpora=["dev.short"]
    #     )
    # )
    #
    # variants = []
    # for beam_size in [2, 4, 8, 16, 32, 64, 256, 512]:
    #     for score_threshold in [6.0, 8.0, 10.0, None]:
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
    #     recognition.ctc_bpe.run(
    #         model=models["ctc_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.short"]
    #     )
    # )
    #
    # variants = []
    # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     for score_threshold in [6.0, 8.0, 10.0, None]:
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
    #         model=models["ffnn_transducer_bpe"],
    #         train_corpus_key="train.medium",
    #         variants=variants,
    #         corpora=["dev.short"],
    #     )
    # )
    #
    # variants = []
    # for beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     for score_threshold in [6.0, 8.0, 10.0, None]:
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
    #         model=models["ffnn_transducer_bpe"],
    #         train_corpus_key="train.medium",
    #         variants=variants,
    #         corpora=["dev.short"],
    #     )
    # )

    # recog_variants = list(
    #     filter(lambda variant: variant.ctc_score_scale == 0.0, recognition.aed_bpe.default_recog_variants())
    # )
    # recog_results.extend(
    #     recognition.aed_bpe.run(model=aed_no_ctc, train_corpus_key="train.medium", variants=recog_variants)
    # )
    #
    # recog_results.extend(recognition.aed_bpe.run(model=aed_bpe_10k, train_corpus_key="train.medium"))

    # Table 2
    variants = []
    variant = recognition.ctc_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
    variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(
            model=models["ctc_bpe"],
            train_corpus_key="train.medium",
            variants=variants,
            corpora=["dev.all", "test.all"],
        )
    )

    variants = []
    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
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

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.medium",
            variants=variants,
            corpora=["dev.all", "test.all"],
        )
    )

    variants = []
    variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_phoneme.run(
            model=models["ffnn_transducer_phoneme"],
            variants=variants,
            corpora=["dev.all", "test.all"],
        )
    )

    # Table 5
    variants = []
    for score_threshold, max_beam_size in [(0.0, 1), (2.0, 16), (4.0, 32), (6.0, 32), (8.0, 32)]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        variant.search_algorithm_params.score_thresholds = [score_threshold]
        variant.compute_search_errors = True
        variants.append(variant)

        # variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        # variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_gpu_search_err"
        # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        # variant.search_algorithm_params.score_thresholds = [score_threshold]
        # variant.search_mode_params.gpu_mem_rqmt = 24
        # variant.compute_search_errors = True
        # variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.all"]
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
        variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(
            model=models["ctc_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.all"]
        )
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
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
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
        variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.all"]
        )
    )

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_phoneme.run(
            model=models["ffnn_transducer_phoneme"], variants=variants, corpora=["dev.all"]
        )
    )

    # Figure 3
    variants = []

    for score_threshold in [None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for max_beam_size in [2, 4, 8, 16, 32, 64, 128]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_gpu"
            variant.search_mode_params.gpu_mem_rqmt = 24
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.medium",
            variants=variants,
            corpora=["dev.all"],
        )
    )

    register_recog_report(recog_results)
