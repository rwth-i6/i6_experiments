from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import recognition, run_all, training
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.learning_rates import (
    ConstConstDecayLRConfig,
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

        with ExperimentContext("bpe_transformer_lm/small"):
            model_config = training.bpe_transformer_lm.get_model_config()
            model_config.num_layers = 24
            training.bpe_transformer_lm.run(model_config=model_config)

        with ExperimentContext("bpe_transformer_lm/bpe_10k"):
            model_config = training.bpe_transformer_lm.get_model_config(bpe_size=10000)
            train_options = training.bpe_transformer_lm.get_train_options(bpe_size=10000)
            training.bpe_transformer_lm.run(model_config=model_config, train_options=train_options)

        with ExperimentContext("bpe_transformer_lm/bpe_10k_small"):
            model_config = training.bpe_transformer_lm.get_model_config(bpe_size=10000)
            model_config.num_layers = 24
            train_options = training.bpe_transformer_lm.get_train_options(bpe_size=10000)
            training.bpe_transformer_lm.run(model_config=model_config, train_options=train_options)

        with ExperimentContext("word_transformer_lm/small"):
            model_config = training.word_transformer_lm.get_model_config()
            model_config.num_layers = 24
            training.word_transformer_lm.run(model_config=model_config)

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_ctc"):
            with ExperimentContext("bpe_5k/baseline_recog"):
                variants = list(
                    filter(
                        lambda variant: variant.bpe_lstm_lm_scale == 0 and variant.bpe_trafo_lm_scale == 0,
                        recognition.bpe_ctc.default_recog_variants(),
                    )
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

        with ExperimentContext("bpe_ffnn_transducer"):
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
