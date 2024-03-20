from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_core.report.report import _Report_Type

from .data import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ..data import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config

def flash_bpe_rnnt_report_format(report: _Report_Type) -> str:
    extra_ls = []
    out = [(" ".join(recog.split("/")[-3:]), str(report[recog])) for recog in report if not any(extra in recog for extra in extra_ls)]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = [out[0]]
    for extra in extra_ls:
        out2 = [(" ".join(recog.split("/")[-3:]), str(report[recog])) for recog in report if extra in recog]
        out2 = sorted(out2, key=lambda x: float(x[1]))
        if len(out2) > 0:
            out.append((extra, ""))
            out.extend(out2)
            best_ls.append(out2[0])
    best_ls = sorted(best_ls, key=lambda x: float(x[1]))
    out.append(("Best Results", ""))
    out.extend(best_ls)
    return "\n".join([f"{pair[0]}:  {str(pair[1])}" for pair in out])

def conformer_rnnt_baseline():
    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/baseline/"

    BPE_SIZE = 1000

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=5, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_bpe_training_datasets(
        bpe_size=BPE_SIZE,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
        )
    from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm

    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    arpa_ted_lm = lm.ngram_lm

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(
        ft_name,
        datasets,
        train_args,
        search_args=None,
        num_epochs=250,
        decoder="rnnt.decoder.experimental_rnnt_decoder",
        with_prior=False,
        evaluate_epoch=None,
        eval_best=True,
    ):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs
        search_job_ls = []
        report = {}
        returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)
        format_string_report, values_report, search_jobs = search(
            ft_name + "/default_%i" % evaluate_epoch,
            returnn_search_config,
            train_job.out_checkpoints[evaluate_epoch],
            test_dataset_tuples,
            RETURNN_EXE,
            MINI_RETURNN_ROOT,
            use_gpu=search_args.get("use_gpu", False),
        )
        search_job_ls += search_jobs
        report.update(values_report)

        from i6_core.returnn import GetBestPtCheckpointJob
        if eval_best:
            best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_rnnt")
            best_job.add_alias(ft_name + "/get_best_job")
            format_string_report, values_report, search_jobs = search(
                ft_name + "/best_chkpt",
                returnn_search_config,
                best_job.out_checkpoint,
                test_dataset_tuples,
                RETURNN_EXE,
                MINI_RETURNN_ROOT,
            )
            search_job_ls += search_jobs
            report.update(values_report)

        return train_job, search_job_ls, format_string_report, report

    def generate_report(results, exp_name):
        from i6_core.report import GenerateReportStringJob, MailJob

        report = GenerateReportStringJob(report_values=results, report_template=flash_bpe_rnnt_report_format)
        report.add_alias(f"report/report/{exp_name}")
        mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
        mail.add_alias(f"report/mail/{exp_name}")
        tk.register_output("mail/" + exp_name, mail.out_status)

    train_args_adamw03_accum2_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
            + list(np.linspace(7e-4, 7e-5, 110))
            + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
        "debug": True,
    }

    default_search_args = {
        "lexicon": get_text_lexicon(bpe_size=BPE_SIZE),  # TODO: cleanup
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 14,
    }

    #### New experiments with corrected FF-Dim

    from ..pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
    )

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=10,
    )

    model_config_sub6 = copy.deepcopy(model_config)
    model_config_sub6.frontend_config.pool1_stride = (3, 1)
    model_config_sub6.frontend_config.pool1_kernel_size = (3, 1)

    model_config_sub6_later = copy.deepcopy(model_config_sub6)
    model_config_sub6_later.specauc_start_epoch = 40

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v4_transparent_latepredictor",
        "net_args": {"model_config_dict": asdict(model_config_sub6_later)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v4_JJLR_sub6_transparent_latepredictor/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 14.9
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v4_JJLR_sub6_transparent_latepredictor/bs12"
    )
    del results
    from ..pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v5_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        PredictorConfig,
    )

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        num_lstm_layers=1,
        lstm_hidden_dim=1024,
        lstm_dropout=0.3,
    )
    model_config_v5_sub6 = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=20,
        joiner_dim=512,
        joiner_activation="relu",
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v5_transparent",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub6_start20_transparent/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 11.3
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub6_start20_transparent/bs12"
    )
    del results

    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub6_start20_transparent/bs12_ep134",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
        evaluate_epoch=134,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 13.5
        results=results, exp_name=prefix_name +"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub6_start20_transparent/bs12_ep134",
    )
    del results

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.3,
    )
    model_config_v5_sub6_512lstm = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=20,
        joiner_dim=512,
        joiner_activation="relu",
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v5_transparent",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub6_start20_lstm512_transparent/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 10.4
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub6_start20_lstm512_transparent/bs12"
    )
    from ..pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        PredictorConfig,
    )

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.1,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.3,
    )
    model_config_v5_sub6_512lstm = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=20,
        joiner_dim=512,
        joiner_activation="relu",
        joiner_dropout=0.1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_transparent",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 10.1
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs12"
    )
    del results

    results = {}
    for beam_size in [1, 2, 4, 8, 12, 16, 20, 24, 32, 64, 128]:
        search_args_gpu = {
            "beam_size": beam_size,
            "returnn_vocab": label_datastream.vocab,
            "use_gpu": True,  # also for new hash
        }
        _, _, _, wer_values = run_exp(
            prefix_name
            + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs%u_gpu"
            % beam_size,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args_gpu,
            with_prior=False,
        )
        results.update(wer_values)
        del wer_values
    generate_report(  # 10.1
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/base"
    )
    del results

    search_args_gpu = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
        "use_gpu": True,  # also for new hash
        "batched_encoder_decoding": True,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name
        + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs12_gpu_batched",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args_gpu,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 10.1
        results=results, exp_name=prefix_name +  "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs12_gpu_batched"
    )
    del results

    results = {}
    for blank_log_penalty in [0.1, 0.2, 0.3]:
        search_args_gpu = {
            "beam_size": 16,
            "returnn_vocab": label_datastream.vocab,
            "use_gpu": True,  # also for new hash
            "blank_log_penalty": blank_log_penalty,
        }
        _, _, _, wer_values = run_exp(
            prefix_name
            + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs16_bp%.1f_gpu"
            % blank_log_penalty,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args_gpu,
            with_prior=False,
        )
        results.update(wer_values)
        del wer_values
    generate_report(  # 10.0
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_sub6_start20_lstm512_transparent/bs_16_penalty"
    )
    del results

    train_args_const20 = copy.deepcopy(train_args)
    train_args_const20["config"]["learning_rates"] = (
        list(np.linspace(1e-4, 1e-4, 20))
        + list(np.linspace(1e-4, 7e-4, 90))
        + list(np.linspace(7e-4, 7e-5, 110))
        + list(np.linspace(7e-5, 1e-8, 30))
    )
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name
        + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_const20_sub6_start20_lstm512_transparent/bs12",
        datasets=train_data,
        train_args=train_args_const20,
        search_args=search_args,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 10.1
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_const20_sub6_start20_lstm512_transparent/bs12"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7_transparent",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    _, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_transparent/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    results.update(wer_values)
    del wer_values
    generate_report(  # 9.8
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_transparent/bs12"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    train_job, _, _, wer_values= run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    train_job.rqmt["gpu_mem"] = 24
    results.update(wer_values)
    del wer_values
    generate_report(  # 9.6
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512/bs12"
    )
    del results
    # TODO: This here above is the best baseline with 9.3%, with the accum step 3 setting also runnable on 11GB GPU

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["learning_rates"] = (
            list(np.linspace(7e-6, 7e-4, 220)) +
            list(np.linspace(7e-4, 7e-5, 220)) +
            list(np.linspace(7e-5, 1e-8, 60)))
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    train_job, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_longer/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
        num_epochs=500
    )
    train_job.rqmt["gpu_mem"] = 24
    results.update(wer_values)
    del wer_values
    generate_report(  # 9.1
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_longer/bs12"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["batch_size"] = 180 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 2
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    train_job, _, _, wer_values = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_r2/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    train_job.rqmt["gpu_mem"] = 24
    results.update(wer_values)
    del wer_values
    generate_report(  # 9.5
        results=results, exp_name=prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_r2/bs12"
    )
    del results
