from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import recognition, run_all, training
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.report import register_recog_report


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

    register_recog_report(recog_results)
