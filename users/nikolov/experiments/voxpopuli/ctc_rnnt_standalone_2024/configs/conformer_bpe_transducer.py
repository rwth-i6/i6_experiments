import copy
from dataclasses import dataclass
from sisyphus import tk

from dataclasses import asdict
import numpy as np

from i6_experiments.users.jxu.experiments.transducer.voxpopuli.data import get_voxpopuli_data
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.users.jxu.experiments.transducer.voxpopuli.pytorch_networks.i6modelsV1_VGG4LayerActFrontendV1_v7 import get_model_config

from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training, search, compute_prior
from i6_experiments.users.jxu.experiments.transducer.voxpopuli.configs.config import get_training_config, get_search_config, get_prior_config


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset


def conformer_rnnt_baseline():
    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/baseline/"

    train_data = TrainingDatasets(
        train=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon",
                                    split="train",
                                    partition_epoch=20),
        cv=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon",
                                    split="dev",
                                    partition_epoch=1)
    )

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    #MINI_RETURNN_ROOT = tk.Path("/u/rossenbach/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    # ---------------------------------------------------------------------------------------------------------------- #
    def run_exp(ft_name, datasets, train_args, search_args=None, num_epochs=600,
                decoder="rnnt.decoder.experimental_rnnt_decoder", with_prior=False, evaluate_epoch=None):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)
        # _, _, search_jobs = search(ft_name + "/default_%i" % evaluate_epoch, returnn_search_config,
        #                            train_job.out_checkpoints[evaluate_epoch], test_dataset_tuples, RETURNN_EXE,
        #                            MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False))
        # return train_job, search_jobs

        return train_job

    model_config = get_model_config(vocab_size_without_blank=81919,network_args={})

    train_args_adamw03_accum2_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
                np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
        "debug": True,
    }

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config)},
    }
    train_args["config"]["batch_size"] = 120 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
    }

    train_job = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_transparent/bs12",
        datasets=train_data, train_args=train_args, search_args=search_args, with_prior=False)

    train_job.rqmt["gpu_mem"] = 24
    tk.register_output("output/learning_rates", train_job.out_learning_rates)
    tk.register_output("output/out_model_dir", train_job.out_model_dir)





