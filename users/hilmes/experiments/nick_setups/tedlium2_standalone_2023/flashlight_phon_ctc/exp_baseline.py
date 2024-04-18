from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Optional, List, Dict

from i6_core.report.report import _Report_Type

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from .data import build_phon_training_datasets, TrainingDatasetSettings, get_eow_text_lexicon
from ..data import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior, quantize_static

from .config import get_training_config, get_search_config, get_prior_config, get_static_quant_config

def calc_stat(ls):
    avrg = np.average([float(x[1]) for x in ls])
    min = np.min([float(x[1]) for x in ls])
    max = np.max([float(x[1]) for x in ls])
    median = np.median([float(x[1]) for x in ls])
    std = np.std([float(x[1]) for x in ls])
    ex_str = f"Avrg: {avrg}, Min {min}, Max {max}, Median {median}, Std {std}"
    return ex_str

def flash_phon_ctc_report_format(report: _Report_Type) -> str:
    extra_ls = ["quantize_static"]
    out = [(" ".join(recog.split("/")[6:]), str(report[recog])) for recog in report if not any(extra in recog for extra in extra_ls)]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = [out[0]]
    for extra in extra_ls:
        if extra == "quantize_static":
            tmp = {recog: report[recog] for recog in report if extra in recog}
            iters = set()
            for recog in tmp:
                x = recog.split("/")
                for sub in x:
                    if "samples" in sub:
                        iters.add(sub[len("samples_"):])
            for samples in iters:
                out2 = [(" ".join(recog.split("/")[6:]), str(report[recog])) for recog in report if f"samples_{samples}/" in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append((extra + f"_samples_{samples}", ex_str))
                    out.extend(out2)
                    best_ls.append(out2[0])
        else:
            out2 = [(" ".join(recog.split("/")[6:]), str(report[recog])) for recog in report if extra in recog]
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
    from i6_core.lm.kenlm import CreateBinaryLMJob, CompileKenLMJob
    from i6_core.tools.git import CloneGitRepositoryJob
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=arpa_ted_lm,
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_binary_lm_job.add_alias("experiments/jaist_project/lm/create_4gram_binary_lm")
    arpa_4gram_binary_lm = arpa_4gram_binary_lm_job.out_lm
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
        eval_average: bool = True,
        quant_args: Optional[Dict] = None,
    ):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                epoch=str(num_epochs),
            )
            tk.register_output(training_name + f"/prior/{str(num_epochs)}.txt", prior_file)
            search_args["prior_file"] = prior_file

        returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)

        if eval_epochs is None:
            eval_epochs = [num_epochs]
        search_job_ls = []
        report = {}
        for epoch in eval_epochs:
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
        if eval_average is not None:
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

        if quant_args is not None:
            # best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_ctc")
            # best_job.add_alias(ft_name + "/get_best_job")
            if with_prior:
                returnn_config = get_prior_config(training_datasets=datasets, **train_args)
                prior_file = compute_prior(
                    ft_name,
                    returnn_config,
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    returnn_exe=RETURNN_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    epoch=str(num_epochs)
                )
                tk.register_output(training_name + f"/prior/{num_epochs}.txt", prior_file)
                search_args["prior_file"] = prior_file
            tmp_quant_args = copy.deepcopy(quant_args)
            quant_decoder = tmp_quant_args.pop("decoder", decoder)
            sample_ls = tmp_quant_args.pop("sample_ls", [10])
            iterations = tmp_quant_args.pop("num_iterations", 1)
            for num_samples in sample_ls:
                for seed in range(iterations):
                    it_name = ft_name + f"/quantize_static/samples_{num_samples}/seed_{seed}/"
                    tmp_quant_args["dataset_seed"] = seed
                    tmp_quant_args["num_samples"] = num_samples
                    quant_config = get_static_quant_config(training_datasets=datasets, quant_args=tmp_quant_args, **train_args)
                    quant_model = quantize_static(
                        prefix_name=it_name + "default_250/",
                        returnn_config=quant_config,
                        checkpoint=train_job.out_checkpoints[num_epochs],
                        returnn_exe=RETURNN_EXE,
                        returnn_root=MINI_RETURNN_ROOT,
                    )
                    returnn_search_config = get_search_config(
                        **train_args,
                        decoder_args=search_args,
                        decoder=quant_decoder,
                        quant_args=tmp_quant_args
                    )
                    format_string_report, values_report, search_jobs = search(
                        it_name + "default_250/",
                        returnn_search_config,
                        quant_model,
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

    from ..pytorch_networks.ctc.conformer_0923.transparent_i6modelsV1_2x1D_frontend_xavierinit_cfg import (
        SpecaugConfig,
        TwoLayer1DFrontendConfig,
        ModelConfig,
    )

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=256,
        conv2_channels=384,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=2,
        conv2_stride=2,
        dropout=0.1,
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

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw03_accum2 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 125)) + list(np.linspace(1e-3, 1e-6, 125)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
    }

    default_search_args = {
        "lexicon": get_eow_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 64,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 50,
    }

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }

    results = {}
    for lm_weight in [1.5, 2.0, 2.5]:
        for prior_scale in [0.3, 0.5, 0.75, 1.0]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.2f"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values

    for pruning in [10, 20, 30, 40, 50]:
        search_args = {
            **default_search_args,
            "lm_weight": 2.0,
            "prior_scale": 0.5,
        }
        search_args["beam_size"] = 256
        search_args["beam_threshold"] = pruning
        _, _, _, wer_values = run_exp(
            prefix_name
            + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm2.0_prior0.5_bs256_prune%i" % pruning,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args,
            with_prior=True,
            eval_best=False,
        )
        results.update(wer_values)
        del wer_values

    for pruning in [10, 12, 14, 16, 18, 20]:
        # 10 = 10.0
        # 12 = 9.9
        # 14 = 9.9
        # 16 = 9.8
        search_args = {
            **default_search_args,
            "lm_weight": 2.0,
            "prior_scale": 0.5,
        }
        search_args["beam_size"] = 1024
        search_args["beam_threshold"] = pruning
        _, _, _, wer_values = run_exp(
            prefix_name
            + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm2.0_prior0.5_bs1024_prune%i" % pruning,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args,
            with_prior=True,
            eval_best=False,
        )
        results.update(wer_values)
        del wer_values

    generate_report(  # 9.8
        results=results, exp_name=prefix_name + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit"
    )
    del results

    results = {}
    # re-tune prior and lm-weight using beampruning 16
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.0, 0.3, 0.4, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.1f_bs1024_prune16"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 9.7
        results=results, exp_name=prefix_name + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit_bs1024_prune16"
    )
    del results

# Ted-Lium can be larger
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=512,
        conv2_channels=512,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=2,
        conv2_stride=2,
        dropout=0.1,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw03_accum2 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 125)) + list(np.linspace(1e-3, 1e-6, 125)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
    }

    default_search_args = {
        "lexicon": get_eow_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 64,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 50,
    }

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }

    results = {}
    for lm_weight in [1.5, 2.0, 2.5]:
        for prior_scale in [0.3, 0.5, 0.75, 1.0]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/transparent_12x512_i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.2f"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 10.1
        results=results, exp_name=prefix_name + "conformer_0923/transparent_12x512_i6modelsV1_2x1D_frontend_xavierinit"
    )
    del results

    # TODO: not converging same with AMP
    # train_args_amp = copy.deepcopy(train_args)
    # train_args_amp["config"]["torch_amp_options"] = {"dtype": "float16"}  # Pascal / 1080 GPUs can only do float16
    # for lm_weight in [1.5, 2.0, 2.5]:
    #    for prior_scale in [0.3, 0.5, 0.75, 1.0]:
    #        search_args = {
    #            **default_search_args,
    #            "lm_weight": lm_weight,
    #            "prior_scale": prior_scale,
    #        }
    #        run_exp(prefix_name + "conformer_0923/transparent_12x512_i6modelsV1_2x1D_frontend_xavierinit_amp/lm%.1f_prior%.2f" % (
    #        lm_weight, prior_scale),
    #                datasets=train_data, train_args=train_args_amp, search_args=search_args, with_prior=True)

    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_2x1D_frontend_xavierinit_cfg import (
        SpecaugConfig,
        TwoLayer1DFrontendConfig,
        ModelConfig,
    )

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=256,
        conv2_channels=384,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=2,
        conv2_stride=2,
        dropout=0.1,
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
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.2f" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 9.1
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_2x1D_frontend_xavierinit"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["optimizer"] = {"class": "adam", "epsilon": 1e-16}
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_2x1D_frontend_xavierinit_adam/lm%.1f_prior%.2f" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 9.3
        results=results,
        exp_name=prefix_name + "conformer_0923/i6modelsV1_2x1D_frontend_xavierinit_adam"
    )
    del results

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
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1/lm%.1f_prior%.2f" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1/lm%.1f_prior%.2f_bs1024" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.1
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1"
    )
    del results

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_posenc",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc/lm%.1f_prior%.2f" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.1
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc"
    )
    del results

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_convfirst",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_convfirst/lm%.1f_prior%.2f"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.3
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_convfirst"
    )
    del results

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_posenc_convfirst",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc_convfirst/lm%.1f_prior%.2f"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.0
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc_convfirst"
    )
    del results

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_xavierinit/lm%.1f_prior%.2f"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False
            )
            results.update(wer_values)
            del wer_values
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_xavierinit/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values

    generate_report(  # 7.9
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_xavierinit"
    )
    del results

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc_xavierinit/lm%.1f_prior%.2f"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_best=False,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 8.2
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc_xavierinit"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values

    generate_report(  # 7.8
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR"
    )
    del results
    ######################################################

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["optimizer"] = {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2}
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_decay-2/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.9
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_decay-2"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["optimizer"] = {"class": "adamw", "epsilon": 1e-16, "weight_decay": 5e-3}
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_decay5-3/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.8
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_decay5-3"
    )
    del results

    #############################################

    # Train long basic
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(1e-5, 1e-3, 250)) + list(np.linspace(1e-3, 1e-6, 250))
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_ep500/lm%.1f_prior%.2f_bs1024"
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
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_ep500"
    )
    del results

    # Train long skewed
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(1e-5, 1e-3, 200)) + list(np.linspace(1e-3, 1e-7, 300))
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_ep500skewed/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=500,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.5
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_ep500skewed"
    )
    del results

    bene_model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=6,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=9,
        final_dropout=0.2,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(bene_model_config),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_bene_param/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 92.4, not converged
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_bene_param"
    )
    del results

    # No Subsampling
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
    frontend_config_nosub = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(1, 1),
        pool1_stride=(1, 1),
        pool1_padding=None,
        pool2_kernel_size=(1, 1),
        pool2_stride=(1, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config_nosub = ModelConfig(
        frontend_config=frontend_config_nosub,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_nosub),
        },
    }
    train_args["config"]["batch_size"] = 150 * 12000
    train_args["config"]["accum_grad_multiple_step"] = 5
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_nosub/lm%.1f_prior%.2f_bs1024"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(  # 10.0
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_nosub"
    )
    del results

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
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v2",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
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
    generate_report(  # 7.2
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v2_JJLR"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values

            # beam search token
            if lm_weight == 2.0 and prior_scale == 0.5:
                for bst in [10, 20, 30, 40, 50]:
                    search_args = copy.deepcopy(search_args)
                    search_args["beam_size_token"] = bst
                    _, _, _, wer_values = run_exp(
                        prefix_name
                        + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/lm%.1f_prior%.2f_bs1024_th14_bst_%i"
                        % (lm_weight, prior_scale, bst),
                        datasets=train_data,
                        train_args=train_args,
                        search_args=search_args,
                        with_prior=True,
                    )
                    results.update(wer_values)
                    del wer_values
                    # if bst == 20:  # Does currently not work since SFTF cannot be onnx exported
                    #     _, search_jobs, _, _ = run_exp(
                    #         prefix_name
                    #         + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/lm%.1f_prior%.2f_bs1024_th14_bst_%i_exp1"
                    #         % (lm_weight, prior_scale, bst),
                    #         datasets=train_data,
                    #         train_args=train_args,
                    #         search_args=search_args,
                    #         with_prior=True,
                    #         decoder="ctc.decoder.flashlight_experimental_phoneme_ctc",
                    #     )

    # Search GRID
    for lm_weight in [1.6, 1.8, 2.0, 2.2, 2.4]:  # 5
        for prior_scale in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:  # 5
            for beam_threshold in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:  # 12
                # for beam_size in [256, 1024, 4096, 8192]:  # 4
                for beam_size in [256, 1024]:  # 4
                    continue
                    search_args = {
                        **copy.deepcopy(default_search_args),
                        "lm_weight": lm_weight,
                        "prior_scale": prior_scale,
                    }
                    search_args["beam_size"] = beam_size
                    search_args["beam_threshold"] = beam_threshold
                    search_args["node"] = "intel"
                    _, search_jobs, _, wer_values = run_exp(
                        prefix_name
                        + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/search_grid_intel_full/lm%.1f_prior%.2f_bs%i_th%i"
                        % (lm_weight, prior_scale, beam_size, beam_threshold),
                        datasets=train_data,
                        train_args=train_args,
                        search_args=search_args,
                        with_prior=True,
                    )
                    results.update(wer_values)
                    del wer_values
                    for search_job in search_jobs:
                        search_job.rqmt["sbatch_args"] = "-p rescale_intel -A rescale_speed"
                        if beam_size > 1024:
                            search_job.rqmt["mem"] = 12
                        elif beam_size > 4096:
                            search_job.rqmt["mem"] = 16

    generate_report(  # 7.2
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR"
    )
    del results

    # with speed perturbation
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
        "use_speed_perturbation": True,
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_speed/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.4
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_speed"
    )
    del results

    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import (
        ModelConfig as ModelConfigV4,
    )

    model_config_v4 = ModelConfigV4(
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
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_v4),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000
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
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.2
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR"
    )
    del results
    # TODO: this here above is the best baseline, use as starting point, giving 7.2% with LM 2.2 and Prior 0.7

    train_args = copy.deepcopy(train_args)
    train_args["config"]["learning_rates"] = (
            list(np.linspace(7e-6, 7e-4, 220)) + list(np.linspace(7e-4, 7e-5, 220)) + list(np.linspace(7e-5, 1e-8, 60))
    )
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
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_longerJJLR_500ep/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=500
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.3, 7.1 avrg.
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_longerJJLR_500ep"
    )
    del results

    train_args = copy.deepcopy(train_args)
    train_args["config"]["learning_rates"] = (
            list(np.linspace(7e-6, 7e-4, 130)) + list(np.linspace(7e-4, 7e-5, 230)) + list(np.linspace(7e-5, 1e-8, 140))
    )
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
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_flatterJJLR_500ep/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=500
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 6.8
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_flatterJJLR_500ep"
    )
    del results

    train_args = copy.deepcopy(train_args)
    train_args["config"]["learning_rates"] = (
            list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 31) + [7e-5])
    )
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
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_endJJLR_500ep/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=500
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 6.9
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_endJJLR_500ep"
    )
    del results

    # model_config_v4_start11 = copy.deepcopy(model_config_v4)
    # model_config_v4_start11.specauc_start_epoch = 11
    # train_args = copy.deepcopy(train_args)
    # train_args["net_args"]["model_config_dict"] = asdict(model_config_v4_start11)
    # train_args["config"]["learning_rates"] = list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-5, 150))
    # train_args["config"]["batch_size"] = 500 * 16000
    # train_args["config"]["accum_grad_multiple_step"] = 1
    # train_args["config"]["optimizer"]["weight_decay"] = 1e-2
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
    #             + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_24gb_bs500/lm%.1f_prior%.2f_bs1024_th14"
    #             % (lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_best=False,
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 7.8
    #     results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_24gb_bs500"
    # )
    # del results

    frontend_config_large = VGG4LayerActFrontendV1Config_mod(
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
        out_features=512,
        activation=None,
    )
    model_config_large = ModelConfig(
        frontend_config=frontend_config_large,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=4,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_large),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 100 * 16000
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum2/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.2
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum2"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_large),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 100 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum3/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 94.4, not converged
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum3"
    )
    del results

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_large),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 135)) + list(np.linspace(7e-4, 7e-5, 135)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 100 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 4
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
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum4_300ep/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=300,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 7.2
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum4_300ep"
    )
    del results

    frontend_config_sub3 = VGG4LayerActFrontendV1Config_mod(
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
        pool2_kernel_size=(1, 2),
        pool2_stride=(1, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    specaug_config_sub3 = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    model_config_v4 = ModelConfigV4(
        frontend_config=frontend_config_sub3,
        specaug_config=specaug_config_sub3,
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
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "debug": False,
        "net_args": {
            "model_config_dict": asdict(model_config_v4),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000
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
            _, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub3/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )
            results.update(wer_values)
            del wer_values
    generate_report(  # 9.2 but diverges, running now with more spec
        results=results, exp_name=prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR_sub3"
    )
    del results

    from ..pytorch_networks.ctc.conformer_0923.baseline_quant_v1_cfg import QuantModelTrainConfigV1, QuantModelConfigV1
    model_train_config_quant_v1 = QuantModelTrainConfigV1(
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
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.baseline_quant_v1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_train_config_quant_v1),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000

    results = {}
    model_config_quant_v1 = QuantModelConfigV1(
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor",
        moving_average=0.01,
        bit_prec=8
    )
    #for num_samples in [10, 30, 100, 500, 1000]:
    quant_args = {
        "sample_ls": [10, 30, 100, 500, 1000],
        "quant_config_dict": asdict(model_config_quant_v1),
        "decoder": "ctc.decoder.flashlight_quant_stat_phoneme_ctc",
        "num_iterations": 100,
    }
    # hardcode scales for now since parameter search over quant already is rough
    for lm_weight in [2.2]:
        for prior_scale in [0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["arpa_lm"] = arpa_4gram_binary_lm
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            _, _, _, wer_values = run_exp(
                prefix_name
                + f"conformer_0923/baseline_quant_JJLR/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                quant_args=quant_args,
            )
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "conformer_0923/baseline_quant_JJLR"
    )
    del results

    for bit_prec in [4, 5, 6, 7]:
        results = {}
        model_config_quant_v1 = QuantModelConfigV1(
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor",
            moving_average=0.01,
            bit_prec=bit_prec
        )
        #for num_samples in [10, 30, 100, 500, 1000]:
        quant_args = {
            "sample_ls": [10],
            "quant_config_dict": asdict(model_config_quant_v1),
            "decoder": "ctc.decoder.flashlight_quant_stat_phoneme_ctc",
            "num_iterations": 100,
        }
        # hardcode scales for now since parameter search over quant already is rough
        for lm_weight in [2.2]:
            for prior_scale in [0.7]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                search_args["arpa_lm"] = arpa_4gram_binary_lm
                search_args["beam_size"] = 1024
                search_args["beam_threshold"] = 14
                _, _, _, wer_values = run_exp(
                    prefix_name
                    + f"conformer_0923/baseline_quant_JJLR_bitprec_{bit_prec}/lm%.1f_prior%.2f_bs1024_th14"
                    % (lm_weight, prior_scale),
                    datasets=train_data,
                    train_args=train_args,
                    search_args=search_args,
                    with_prior=True,
                    quant_args=quant_args,
                )
                results.update(wer_values)
                del wer_values
        generate_report(
            results=results, exp_name=prefix_name + f"conformer_0923/baseline_quant_JJLR_bitprec_{bit_prec}"
        )
        del results

    specaug_config_less = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    model_train_config_quant_v2 = QuantModelTrainConfigV1(
        frontend_config=frontend_config,
        specaug_config=specaug_config_less,
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
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.baseline_quant_v1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_train_config_quant_v2),
        },
    }
    train_args["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    train_args["config"]["batch_size"] = 180 * 16000

    results = {}
    model_config_quant_v1 = QuantModelConfigV1(
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor",
        moving_average=0.01,
        bit_prec=8
    )
    #for num_samples in [10, 30, 100, 500, 1000]:
    quant_args = {
        "sample_ls": [10, 30, 100, 500, 1000],
        "quant_config_dict": asdict(model_config_quant_v1),
        "decoder": "ctc.decoder.flashlight_quant_stat_phoneme_ctc",
        "num_iterations": 100,
    }
    # hardcode scales for now since parameter search over quant already is rough
    for lm_weight in [2.2]:
        for prior_scale in [0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["arpa_lm"] = arpa_4gram_binary_lm
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            _, _, _, wer_values = run_exp(
                prefix_name
                + f"conformer_0923/baseline_quant_JJLR_less_spec/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                quant_args=quant_args,
            )
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "conformer_0923/baseline_quant_JJLR_less_spec"
    )
    del results

    for bit_prec in []:
        results = {}
        model_config_quant_v1 = QuantModelConfigV1(
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor",
            moving_average=0.01,
            bit_prec=bit_prec
        )
        #for num_samples in [10, 30, 100, 500, 1000]:
        quant_args = {
            "sample_ls": [10],
            "quant_config_dict": asdict(model_config_quant_v1),
            "decoder": "ctc.decoder.flashlight_quant_stat_phoneme_ctc",
            "num_iterations": 100,
        }
        # hardcode scales for now since parameter search over quant already is rough
        for lm_weight in [2.2]:
            for prior_scale in [0.7]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                search_args["arpa_lm"] = arpa_4gram_binary_lm
                search_args["beam_size"] = 1024
                search_args["beam_threshold"] = 14
                _, _, _, wer_values = run_exp(
                    prefix_name
                    + f"conformer_0923/baseline_quant_JJLR_less_spec_bitprec_{bit_prec}/lm%.1f_prior%.2f_bs1024_th14"
                    % (lm_weight, prior_scale),
                    datasets=train_data,
                    train_args=train_args,
                    search_args=search_args,
                    with_prior=True,
                    quant_args=quant_args,
                )
                results.update(wer_values)
                del wer_values
        generate_report(
            results=results, exp_name=prefix_name + f"conformer_0923/baseline_quant_JJLR_less_spec_bitprec_{bit_prec}"
        )
        del results
