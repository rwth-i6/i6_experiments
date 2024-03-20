from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Optional, List

from i6_core.report.report import _Report_Type

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from .data import build_phon_training_datasets, TrainingDatasetSettings, get_eow_text_lexicon
from ..data import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config

def flash_phon_ctc_report_format(report: _Report_Type) -> str:
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


def pretrained_experiments():
    prefix_name = "experiments/rescale/tedliumv2/flashlight_phon_ctc/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=5, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_phon_training_datasets(settings=train_settings)
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
    # TODO: Add binary conversion job

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(
        ft_name,
        datasets,
        train_args,
        search_args=None,
        with_prior=False,
        num_epochs=250,
        decoder="ctc.decoder.flashlight_phoneme_ctc",
        eval_epochs: Optional[List] = None,
        eval_best: bool = True,
        eval_average: Optional[List] = None,
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
                returnn_config = get_prior_config(training_datasets=datasets, **train_args)
                prior_file = compute_prior(
                    ft_name,
                    returnn_config,
                    checkpoint=train_job.out_checkpoints[epoch],
                    returnn_exe=RETURNN_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    epoch=str(epoch)
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
        if eval_best:
            best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_ctc")
            best_job.add_alias(ft_name + "/get_best_job")
            if with_prior:
                returnn_config = get_prior_config(training_datasets=datasets, **train_args)
                prior_file = compute_prior(
                    ft_name,
                    returnn_config,
                    checkpoint=best_job.out_checkpoint,
                    returnn_exe=RETURNN_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    epoch="best"
                )
                tk.register_output(training_name + f"/prior/best.txt", prior_file)
                search_args["prior_file"] = prior_file
            returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)
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
        from i6_core.returnn import AverageTorchCheckpointsJob
        if eval_average:
            chkpts = []
            for x in [0, 1, 2, 3]:
                best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_ctc", index=x)
                best_job.add_alias(ft_name + f"/get_best_job_{x}")
                chkpts.append(best_job.out_checkpoint)
            avrg_job = AverageTorchCheckpointsJob(checkpoints=chkpts, returnn_python_exe=RETURNN_EXE, returnn_root=MINI_RETURNN_ROOT)
            avrg_job.add_alias(ft_name + "/avrg_chkpt_job")
            format_string_report, values_report, search_jobs = search(
                ft_name + "/avrg_chkpt",
                returnn_search_config,
                avrg_job.out_checkpoint,
                test_dataset_tuples,
                RETURNN_EXE,
                MINI_RETURNN_ROOT,
            )
            search_job_ls += search_jobs
            report.update(values_report)

        return train_job, search_job_ls, format_string_report, report

    def generate_report(results, exp_name):
        from i6_core.report import GenerateReportStringJob, MailJob

        report = GenerateReportStringJob(report_values=results, report_template=flash_phon_ctc_report_format)
        report.add_alias(f"report/report/{exp_name}")
        mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
        mail.add_alias(f"report/mail/{exp_name}")
        tk.register_output("mail/" + exp_name, mail.out_status)

    default_search_args = {
        "lexicon": get_eow_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 64,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 50,
    }
    from ..pytorch_networks.ctc.conformer_0923 import hubert_pretrained_v1_cfg, hubert_pretrained_v2_cfg

    # hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
    #     finetune_layer=2,
    #     name="base-ls960",
    # )
    # model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
    #     specauc_start_epoch=0,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     hubert_cfg=hubert_cfg_2,
    # )
    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
    #                           + list(np.linspace(7e-4, 7e-5, 110))
    #                           + list(np.linspace(7e-5, 1e-8, 30)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 25,
    #     },
    #     "debug": True,
    # }
    # eval_epochs = [250]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.5, 0.7]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #         }
    #         search_args["beam_size"] = 1024
    #         search_args["beam_threshold"] = 14
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "hubert/pretrain_v3_base_tune2_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (
    #             lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #             eval_best=True
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 6.6
    #     results=results, exp_name=prefix_name + "hubert/pretrain_v3_base_tune2_jjlr"
    # )
    # del results

    hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=2,
        name="large-ls960-ft",
    )
    model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_2,
    )
    train_args_hubert_adam_accum25_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
                              + list(np.linspace(7e-4, 7e-5, 110))
                              + list(np.linspace(7e-5, 1e-8, 30)) + [7e-5],
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 3,
            "accum_grad_multiple_step": 25,
        },
        "debug": False,
    }
    eval_epochs = [250, 300, 350, 400, 450, 500]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.5, 0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "hubert/pretrain_v3_large960_tune2_jjlr_longer/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
                eval_best=True,
                num_epochs=500,
                eval_average=[496, 497, 499, 500]
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(  # 5.0
        results=results, exp_name=prefix_name + "hubert/pretrain_v3_large960_tune2_jjlr_longer"
    )
    del results

    # for layer_num in range(0, 24):
    #     hubert_cfg_pick = hubert_pretrained_v2_cfg.HubertConfig(
    #         finetune_layer=True,
    #         name="large-ls960-ft",
    #         keep_layers=layer_num
    #     )
    #     model_config_hubert_pick = hubert_pretrained_v2_cfg.ModelConfig(
    #         specauc_start_epoch=0,
    #         label_target_size=vocab_size_without_blank,
    #         final_dropout=0.2,
    #         hubert_cfg=hubert_cfg_pick,
    #     )
    #     train_args_hubert_adam_accum25_jjlr = {
    #         "config": {
    #             "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #             "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
    #                               + list(np.linspace(7e-4, 7e-5, 110))
    #                               + list(np.linspace(7e-5, 1e-8, 30)) + [7e-5],
    #             #############
    #             "batch_size": 180 * 16000,
    #             "max_seq_length": {"audio_features": 35 * 16000},
    #             "max_seqs": 3 if layer_num < 21 else 2,
    #             "accum_grad_multiple_step": 25 if layer_num < 21 else 50,
    #         },
    #         "debug": False,
    #     }
    #     eval_epochs = [250, 300, 400, 500]
    #     train_args = {
    #         **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #         "network_module": "ctc.conformer_0923.hubert_pretrained_v4",
    #         "net_args": {"model_config_dict": asdict(model_config_hubert_pick)},
    #     }
    #     results = {}
    #     for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #         for prior_scale in [0.5, 0.7]:
    #             search_args = {
    #                 **default_search_args,
    #                 "lm_weight": lm_weight,
    #                 "prior_scale": prior_scale,
    #             }
    #             search_args["beam_size"] = 1024
    #             search_args["beam_threshold"] = 14
    #             train_job, _, _, wer_values = run_exp(
    #                 prefix_name
    #                 + f"hubert/pretrain_v3_large960_keepuntil_{layer_num}_jjlr_longer/lm%.1f_prior%.2f_bs1024_th14" % (
    #                     lm_weight, prior_scale),
    #                 datasets=train_data,
    #                 train_args=train_args,
    #                 search_args=search_args,
    #                 with_prior=True,
    #                 eval_epochs=eval_epochs,
    #                 eval_best=True,
    #                 num_epochs=500,
    #                 eval_average=True,
    #             )
    #             if layer_num > 6:
    #                 train_job.rqmt["gpu_mem"] = 24
    #             results.update(wer_values)
    #             del wer_values
    #     generate_report(
    #         results=results, exp_name=prefix_name + f"hubert/pretrain_v3_large960_keepuntil_{layer_num}_jjlr_longer"
    #     )
    #     del results

    for layer_num in range(0, 24):
        hubert_cfg_pick = hubert_pretrained_v2_cfg.HubertConfig(
            finetune_layer=True,
            name="large-ls960-ft",
            keep_layers=layer_num
        )
        model_config_hubert_pick = hubert_pretrained_v2_cfg.ModelConfig(
            specauc_start_epoch=0,
            label_target_size=vocab_size_without_blank,
            final_dropout=0.2,
            hubert_cfg=hubert_cfg_pick,
        )
        train_args_hubert_adam_accum25_jjlr = {
            "config": {
                "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
                "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
                                  + list(np.linspace(7e-4, 7e-5, 110))
                                  + list(np.linspace(7e-5, 1e-8, 30)),
                #############
                "batch_size": 180 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "max_seqs": 3 if layer_num < 21 else 2,
                "accum_grad_multiple_step": 25 if layer_num < 21 else 50,
            },
            "debug": False,
        }
        eval_epochs = [50, 100, 150, 200, 250]
        train_args = {
            **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
            "network_module": "ctc.conformer_0923.hubert_pretrained_v4",
            "net_args": {"model_config_dict": asdict(model_config_hubert_pick)},
        }
        results = {}
        for lm_weight in [1.0, 1.4, 1.6, 1.8, 2.0, 2.2]:
            for prior_scale in [0.5, 0.7]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                search_args["beam_size"] = 1024
                search_args["beam_threshold"] = 14
                train_job, _, _, wer_values = run_exp(
                    prefix_name
                    + f"hubert/pretrain_v3_large960_keepuntil_{layer_num}_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=train_data,
                    train_args=train_args,
                    search_args=search_args,
                    with_prior=True,
                    eval_epochs=eval_epochs,
                    eval_best=True,
                    num_epochs=250,
                    eval_average=True,
                )
                if layer_num > 6:
                    train_job.rqmt["gpu_mem"] = 24
                results.update(wer_values)
                del wer_values
        generate_report(
            results=results, exp_name=prefix_name + f"hubert/pretrain_v3_large960_keepuntil_{layer_num}_jjlr"
        )
        del results

    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 130))
    #                           + list(np.linspace(7e-4, 7e-5, 230))
    #                           + list(np.linspace(7e-5, 1e-8, 140)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 25,
    #     },
    #     "debug": False,
    # }
    # eval_epochs = [250, 300, 350, 400, 450, 500]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.5, 0.7]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #         }
    #         search_args["beam_size"] = 1024
    #         search_args["beam_threshold"] = 14
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "hubert/pretrain_v3_large960_tune2_jjlr_longflat/lm%.1f_prior%.2f_bs1024_th14" % (
    #             lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #             eval_best=True,
    #             num_epochs=500
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 5.0
    #     results=results, exp_name=prefix_name + "hubert/pretrain_v3_large960_tune2_jjlr_longflat"
    # )
    # del results

    hubert_cfg_full = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=True,
        name="large-ls960-ft",
    )

    model_config_hubert_full = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_full,
    )

    train_args_hubert_adam_accum25_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 130))
                              + list(np.linspace(7e-4, 7e-5, 230))
                              + list(np.linspace(7e-5, 1e-8, 140)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 2,
            "accum_grad_multiple_step": 50,
        },
        "debug": False,
    }
    eval_epochs = [250, 300, 350, 400, 450, 500]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_full)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.5, 0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "hubert/pretrain_v3_large960_tunefull_jjlr_longflat/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
                eval_best=True,
                num_epochs=500,
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "hubert/pretrain_v3_large960_tunefull_jjlr_longflat"
    )
    del results

    from ..pytorch_networks.ctc.conformer_0923 import parakeet_pretrained_v1_cfg
    parakeet_cfg_full = parakeet_pretrained_v1_cfg.ParakeetConfig(
        finetune_layer=True,
        name="ctc-1.1b",
        keep_layers=None,
    )

    model_config_parakeet_full = parakeet_pretrained_v1_cfg.ModelConfig(
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        parakeet_config=parakeet_cfg_full,
    )

    train_args_parakeet_adam_accum25_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
                              + list(np.linspace(7e-4, 7e-5, 110))
                              + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 2,
            "accum_grad_multiple_step": 50,
        },
        "debug": True,
    }
    eval_epochs = [50, 100, 150, 200, 250]
    train_args = {
        **copy.deepcopy(train_args_parakeet_adam_accum25_jjlr),
        "network_module": "ctc.conformer_0923.parakeet_pretrained_v1",
        "net_args": {"model_config_dict": asdict(model_config_parakeet_full)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.5, 0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "parakeet/pretrain_v1_1.1b_tunefull_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
                eval_best=True,
                num_epochs=250,
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "hubert/pretrain_v1_1.1b_tunefull_jjlr"
    )
    del results

