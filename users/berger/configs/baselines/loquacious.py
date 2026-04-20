from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import recognition, training
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_medium as _run_medium
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_small as _run_small
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.report import register_recog_report


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
    train_options.batch_size = 12_000 * 160
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

    recog_variants = []
    for lm_scale in [0.3, 0.6, 0.8]:
        for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
            for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
                variant.descriptor += f"_lm-{lm_scale}_score-{score_threshold}_beam-{max_beam_size}"
                variant.search_algorithm_params.score_thresholds = [score_threshold]
                variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                variant.search_algorithm_params.word_lm_params.scale = lm_scale  # type: ignore
                recog_variants.append(variant)

    for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for lm_scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
            variant.ilm_scale = ilm_scale
            variant.search_algorithm_params.word_lm_params.scale = lm_scale  # type: ignore
            recog_variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.medium",
            variants=recog_variants,
            corpora=["dev.short"],
        )
    )

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(model=ffnn_transducer_bpe_10k, train_corpus_key="train.medium")
    )

    recog_variants = []
    for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for lm_scale in [0.3, 0.5, 0.6, 0.7, 0.8]:
            variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
            variant.descriptor += f"_ilm-{ilm_scale}_lm-{lm_scale}"
            variant.ilm_scale = ilm_scale
            variant.search_algorithm_params.word_lm_params.scale = lm_scale  # type: ignore
            recog_variants.append(variant)

    for score_threshold in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
        for max_beam_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            recog_variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_phoneme.run(
            model=models["ffnn_transducer_phoneme"], variants=recog_variants, corpora=["dev.short"]
        )
    )

    recog_variants = []
    for max_beam_size in [1, 4, 8, 16, 32, 64, 128]:
        for score_threshold in [None, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0]:
            for length_norm_scale in [1.2, 1.5]:
                variant = recognition.aed_bpe.default_lexfree_recog_variant()
                variant.descriptor += f"_beam-{max_beam_size}_score-{score_threshold}_ln-{length_norm_scale}"
                variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
                variant.search_algorithm_params.score_thresholds = (
                    [score_threshold] if score_threshold is not None else None
                )
                variant.search_algorithm_params.length_norm_scale = length_norm_scale  # type: ignore
                recog_variants.append(variant)
    for ctc_score_scale in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        variant = recognition.aed_bpe.default_lexfree_aed_ctc_recog_variant()
        variant.descriptor += f"_ctc-scale-{ctc_score_scale}"
        variant.ctc_score_scale = ctc_score_scale
        recog_variants.append(variant)
    recog_results.extend(
        recognition.aed_bpe.run(
            model=models["aed_bpe"], train_corpus_key="train.medium", variants=recog_variants, corpora=["dev.short"]
        )
    )

    recog_variants = list(
        filter(lambda variant: variant.ctc_score_scale == 0.0, recognition.aed_bpe.default_recog_variants())
    )
    recog_results.extend(
        recognition.aed_bpe.run(model=aed_no_ctc, train_corpus_key="train.medium", variants=recog_variants)
    )

    recog_results.extend(recognition.aed_bpe.run(model=aed_bpe_10k, train_corpus_key="train.medium"))

    register_recog_report(recog_results)
