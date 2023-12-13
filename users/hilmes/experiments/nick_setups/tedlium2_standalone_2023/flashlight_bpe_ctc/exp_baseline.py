from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional

from i6_core.report.report import _Report_Type

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from .data import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ..data import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config


def flash_bpe_ctc_report_format(report: _Report_Type) -> str:
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


def conformer_baseline():
    prefix_name = "experiments/rescale/tedliumv2/flashlight_bpe_ctc/"

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
        with_prior=False,
        num_epochs=250,
        decoder="ctc.decoder.flashlight_bpe_ctc",
        eval_epochs: Optional[List] = None,
    ):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=eval_epochs, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if eval_epochs is None:
            eval_epochs = [num_epochs]
        search_job_ls = []
        report = {}
        for epoch in eval_epochs:
            if with_prior:
                prior_args = copy.deepcopy(train_args)
                if "max_seqs" in prior_args["config"]:
                    del prior_args["config"]["max_seqs"]
                returnn_config = get_prior_config(training_datasets=datasets, **prior_args)
                prior_file = compute_prior(
                    ft_name,
                    returnn_config,
                    checkpoint=train_job.out_checkpoints[epoch],
                    returnn_exe=RETURNN_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    epoch=str(epoch)  # just for alias generation
                )
                tk.register_output(training_name + f"/prior/{epoch}.txt", prior_file)
                search_args["prior_file"] = prior_file

            returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)
            format_string_report, values_report, search_jobs = search(
                ft_name + "/default_%i" % epoch,
                returnn_search_config,
                train_job.out_checkpoints[epoch],
                test_dataset_tuples,
                RETURNN_EXE,
                MINI_RETURNN_ROOT,
            )
            search_job_ls += search_jobs
            report.update(values_report)
        from i6_core.returnn import GetBestPtCheckpointJob
        best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_ctc")
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

        report = GenerateReportStringJob(report_values=results, report_template=flash_bpe_ctc_report_format)
        report.add_alias(f"report/report/{exp_name}")
        mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
        mail.add_alias(f"report/mail/{exp_name}")
        tk.register_output("mail/" + exp_name, mail.out_status)

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw03_accum2 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 125)) + list(np.linspace(1e-3, 1e-6, 125)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
        "debug": False,
    }

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
        "debug": False,
    }

    default_search_args = {
        "lexicon": get_text_lexicon(bpe_size=BPE_SIZE),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 14,
    }

    #### New experiments with corrected FF-Dim

    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_cfg import (
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
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v2",
        "net_args": {"model_config_dict": asdict(model_config)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v2_JJLR/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 97.9, not converged
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v2_JJLR"
    )
    del results

    from ..pytorch_networks.ctc.conformer_0923 import i6modelsV1_VGG4LayerActFrontendV1_v4_cfg

    model_config_v4_start11 = i6modelsV1_VGG4LayerActFrontendV1_v4_cfg.ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        specauc_start_epoch=11,
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
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v4",
        "net_args": {"model_config_dict": asdict(model_config_v4_start11)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v4_JJLR_specstart11/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.0
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v4_JJLR_specstart11"
    )
    del results
    # TODO: This here is the subsampling 4 baseline giving 8.0% with LM 1.6 and prior 0.5
    results = {}
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "net_args": {"model_config_dict": asdict(model_config_v4_start11)},
    }
    train_args["config"]["learning_rates"] = (
            list(np.linspace(7e-6, 7e-4, 130)) + list(np.linspace(7e-4, 7e-5, 230)) + list(np.linspace(7e-5, 1e-8, 140))
    )
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_specstart11_longer/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=500,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.7
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_specstart11_longer"
    )
    del results

    results = {}
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "net_args": {"model_config_dict": asdict(model_config_v4_start11)},
    }
    train_args["config"]["learning_rates"] = (
            list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + [7e-5])
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_specstart11_longerend/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=500,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.6
        results=results,
        exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_specstart11_longerend"
    )
    del results

    results = {}
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "net_args": {"model_config_dict": asdict(model_config_v4_start11)},
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_specstart11/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.9, most likely better due to noise
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_specstart11"
    )
    del results
    # --------------------------------------------------------------------------------------------------------------- #
    # SUB 6 from here

    model_config_v4_sub6_start11 = copy.deepcopy(model_config_v4_start11)
    model_config_v4_sub6_start11.frontend_config.pool1_stride = (3, 1)
    model_config_v4_sub6_start11.frontend_config.pool1_kernel_size = (3, 1)

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v4",
        "net_args": {"model_config_dict": asdict(model_config_v4_sub6_start11)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v4_JJLR_sub6_specstart11/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # did not converge 98.0
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v4_JJLR_sub6_specstart11"
    )
    del results

    model_config_sub6 = copy.deepcopy(model_config)
    model_config_sub6.frontend_config.pool1_stride = (3, 1)
    model_config_sub6.frontend_config.pool1_kernel_size = (3, 1)

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3_transparent",
        "net_args": {"model_config_dict": asdict(model_config_sub6)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_transparent_sub6/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 97.8 not converged
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_transparent_sub6"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3_posenc_transparent",
        "net_args": {"model_config_dict": asdict(model_config_sub6)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_posenc_transparent_sub6/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 99.2, not converged
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_posenc_transparent_sub6"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3_posenc_transparent_latespecaug",
        "net_args": {"model_config_dict": asdict(model_config_sub6)},
    }
    results = {}
    for lm_weight in [1.4, 1.6, 1.8, 2.0]:
        for prior_scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for beam_size in [512, 1024]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                    "beam_size_token": 128,
                    "beam_size": beam_size,
                }
                _, _, _, wer_values = run_exp(
                    prefix_name
                    + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_posenc_transparent_sub6_latespecaug/lm%.1f_prior%.2f_bs%i_th14"
                    % (lm_weight, prior_scale, beam_size),
                    datasets=train_data,
                    train_args=train_args,
                    search_args=search_args,
                    with_prior=True,
                )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.4
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_posenc_transparent_sub6_latespecaug"
    )
    del results

    train_args_debug = copy.deepcopy(train_args)
    train_args_debug["debug"] = True
    # greedy
    search_args = {
        "returnn_vocab": label_datastream.vocab,
    }
    run_exp(
        prefix_name
        + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_posenc_transparent_sub6_latespecaug/greedy",
        datasets=train_data,
        train_args=train_args_debug,
        search_args=search_args,
        with_prior=True,
        decoder="ctc.decoder.greedy_bpe_ctc_v2",
    )
