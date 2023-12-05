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


def pretrained_rnnt():
    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/"

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

        return train_job, search_job_ls, format_string_report, report

    def generate_report(results, exp_name):
        from i6_core.report import GenerateReportStringJob, MailJob

        report = GenerateReportStringJob(report_values=results, report_template=flash_bpe_rnnt_report_format)
        report.add_alias(f"report/report/{exp_name}")
        mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
        mail.add_alias(f"report/mail/{exp_name}")
        tk.register_output("mail/" + exp_name, mail.out_status)

    from ..pytorch_networks.rnnt.conformer_1023 import hubert_pretrain_v1_cfg

    predictor_config = hubert_pretrain_v1_cfg.PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.1,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.3,
    )

    hubert_cfg_2 = hubert_pretrain_v1_cfg.HubertConfig(
        finetune_layer=2,
        name="base-ls960",
    )
    model_config_hubert_2 = hubert_pretrain_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_2,
        predictor_config=predictor_config,
        joiner_dim=512,
        joiner_activation="relu",
        joiner_dropout=0.1,
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
            "max_seqs": 3,
            "accum_grad_multiple_step": 25,
        },
        "debug": False,
    }
    eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
        "network_module": "rnnt.conformer_1023.hubert_pretrain_v1",
        "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    }
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    results = {}
    train_job, _, _, wer_values = run_exp(
        prefix_name
        + "conformer_1023/hubert_pretrain_v3_base_tune2_jjlr/bs12",
        datasets=train_data,
        train_args=train_args,
        search_args=search_args,
        with_prior=False,
    )
    train_job.rqmt["gpu_mem"] = 24
    results.update(wer_values)
    del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "conformer_1023/hubert_pretrain_v3_base_tune2_jjlr"
    )
    del results
