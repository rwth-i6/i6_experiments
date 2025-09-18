from __future__ import annotations
import copy
from sisyphus import gs, tk
import i6_core.rasr as rasr
from i6_core.lib.lexicon import Lexicon, Lemma
import i6_core.lm as lm
from apptek_asr.meta.nn import NNSystemV2
from apptek_asr.meta.rasr import (
    PyrasrVenvBuilder,
    SentencepieceRasrFsaExporterConfigBuilder,
)
from apptek_asr.artefacts import ArtefactSpecification
from apptek_asr.artefacts.factory import AbstractArtefactRepository

from i6_core.tools.git import CloneGitRepositoryJob

# ********** Settings **********
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

# Atanas BPE setup
num_outputs = 10237
num_subepochs = 625  # 50 full ep * 100 partition / 8 gpu

### General ###
# ISO language code of the system (e.g. "EN_US")
# used to determine the namespace of most artefacts
lang_code_es = "ES"
lang_code_es_us = "ES_US"  # for test set
lang_code_es_es = "ES_ES"

# Frequency of the audio data: either "16kHz" or "8kHz"
sampling_rate = "16kHz"
sampling_rate_int = int(sampling_rate[: -len("kHz")]) * 1000

# Name for the experiment. This will name the output folder
experiment_name = "CTC-40ms-sentencepiece"

### Training ###
# Specify lists of training HDF artefact names from the appropriate namespace here
# as Dict[str, List[str]]. These will be concatenated and used for training
#
# feature + alignment pairs can either be stored separately in two different hdf
# files (then set both train_corpora_alignment and train_corpora_features) or
# together in a single file (in this case only set train_corpora_alignment).
feature_hdfs_16k_corpora = {
    "hdf.features.ES.f16kHz": [
        "AMARA-batch.1.upv.v1-hdf-raw_wav.16kHz.split-17",
        "dotsub-batch.1.v1-hdf-raw_wav.16kHz.split-6",
        "LangMedia-batch.1.v1-hdf-raw_wav.16kHz.split-1",
        "TedTalks-batch.1.v2-hdf-raw_wav.16kHz.split-1",
        "commonvoice-validated.v1-20250414-hdf-raw_wav.16kHz.split-55",
        "MLS-train.v1-20250414-hdf-raw_wav.16kHz.split-92",
        "YODAS-es000.v1-20250414-hdf-raw_wav.16kHz.split-278",
        "VoxPopuli-train.v1-20250414-hdf-raw_wav.16kHz.split-15",
        "Fleurs-train.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "Heroico-usma.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "Heroico-heroico.recordings.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "Heroico-heroico.answers.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "MediaSpeech-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "SpokenWikipedia-train.v1-20250414-hdf-raw_wav.16kHz.split-3",
    ],
    "hdf.features.ES_AR.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_CL.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_CO.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_ES.f16kHz": [
        "Audiria-batch.1.v1-hdf-raw_wav.16kHz.split-4",
        "RTVE-batch.1.common.v2-hdf-raw_wav.16kHz.split-220",
        "RTVE-batch.2.upv.v2-hdf-raw_wav.16kHz.split-70",
        "Europarl-ST-train.v1-20250414-hdf-raw_wav.16kHz.split-2",
        "Europarl-ST-train-noisy.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_MX.f16kHz": [
        "ExcelsiorTV-batch.1.v1-hdf-raw_wav.16kHz.split-40",
        "Mexican.TV.Broadcast-batch.1.v1-hdf-raw_wav.16kHz.split-42",
        "CIEMPIESS-balance.train.v1-20250414-hdf-raw_wav.16kHz.split-2",
        "CIEMPIESS-fem.train.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "CIEMPIESS-light.train.v1-20250414-hdf-raw_wav.16kHz.split-2",
    ],
    "hdf.features.ES_PE.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_PR.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_VE.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
    "hdf.features.ES_US.f16kHz": [
        "AllMDScript-batch.1.v2-hdf-raw_wav.16kHz.split-3",
        "CNN-batch.1.v1-hdf-raw_wav.16kHz.split-1",
        "ReadNews.Latino-batch.1.v1-hdf-raw_wav.16kHz.split-1",
        "Hub4-batch.1.v2-hdf-raw_wav.16kHz.split-3",
        "NJHealth-batch.1.train.v2-hdf-raw_wav.16kHz.split-4",
        "NJHealth-batch.3.MTurk.v1-hdf-raw_wav.16kHz.split-3",
        "Speecon-batch.1.v1-hdf-raw_wav.16kHz.split-17",
        "OFTravelRecordings-batch.1.v1-hdf-raw_wav.16kHz.split-1",
        "TelemundoNoticias-batch.1.v1-hdf-raw_wav.16kHz.split-5",
        "Telenovela-batch.1.v1-hdf-raw_wav.16kHz.split-22",
    ],
}
feature_hdfs_8k_corpora = {
    "hdf.features.ES.f8kHz_as_16kHz": [
        "Fisher-batch.1.train.v2-20250414-hdf-raw_wav.16kHz.split-17",
    ],
    "hdf.features.ES_AR.f8kHz_as_16kHz": [
        "Callcenter.AcVe-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-25",
    ],
    "hdf.features.ES_CL.f8kHz_as_16kHz": [
        "Callcenter.AcVe-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-17",
    ],
    "hdf.features.ES_CO.f8kHz_as_16kHz": [
        "Callcenter.AcVe-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-22",
    ],
    "hdf.features.ES_ES.f8kHz_as_16kHz": [
        "Appen-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-10",
    ],
    "hdf.features.ES_PE.f8kHz_as_16kHz": [
        "Callcenter.AcVe-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-22",
    ],
    "hdf.features.ES_US.f8kHz_as_16kHz": [
        "AAA-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "CallFriend-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-5",
        "CallHome-batch.1.train.v2-20250414-hdf-raw_wav.16kHz.split-1",
        "CollinCounty-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-9",
        "Hub5-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
        "Ignite.ATNT-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-6",
        "Ignite.ATNT-batch.2.v1-20250414-hdf-raw_wav.16kHz.split-8",
        "Ignite.HomeShopping-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-32",
        "Ignite.LiveAgent-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-2",
        "ListenTrust-batch.1.v2-20250414-hdf-raw_wav.16kHz.split-40",
        "ListenTrust-batch.2.v2-20250414-hdf-raw_wav.16kHz.split-40",
        "ListenTrust-batch.3.capital.v3-20250414-hdf-raw_wav.16kHz.split-3",
        "ListenTrust-batch.4.funeral.v3-20250414-hdf-raw_wav.16kHz.split-1",
        "ListenTrust-batch.5.immigration.v3-20250414-hdf-raw_wav.16kHz.split-4",
        "ListenTrust-batch.6.real.v3-20250414-hdf-raw_wav.16kHz.split-26",
        "ListenTrust-batch.7.simulated.v2-20250414-hdf-raw_wav.16kHz.split-20",
        "NameAddr-batch.1.v1-20250414-hdf-raw_wav.16kHz.split-1",
    ],
}
train_corpora_features = feature_hdfs_16k_corpora | feature_hdfs_8k_corpora

dev_hdf_8k_artefact_specs = ArtefactSpecification(
    "hdf.features.ES.f8kHz_as_16kHz",
    "Fisher-batch.2.dev.v2-20250414-hdf-raw_wav.16kHz.split-1",
)
dev_hdf_16k_europarl_artefact_specs = ArtefactSpecification(
    "hdf.features.ES_ES.f16kHz",
    "Europarl-ST-dev.v1-20250414-hdf-raw_wav.16kHz.split-1",
)
dev_hdf_16k_voxpopuli_artefact_specs = ArtefactSpecification(
    "hdf.features.ES.f16kHz",
    "VoxPopuli-dev.v1-20250414-hdf-raw_wav.16kHz.split-1",
)
dev_hdf_16k_commonvoice_artefact_specs = ArtefactSpecification(
    "hdf.features.ES.f16kHz",
    "commonvoice-dev.v1-20250414-hdf-raw_wav.16kHz.split-3",
)

# collection of bliss corpus files to be used for fsa training
corpora_16k = {
    "corpus.ES.f16kHz": [
        "AMARA-batch.1.upv.v1",
        "dotsub-batch.1.v1",
        "LangMedia-batch.1.v1",
        "TedTalks-batch.1.v2",
        "commonvoice-validated.v1",
        "commonvoice-dev.v1",
        "MLS-train.v1",
        "YODAS-es000.v1",
        "VoxPopuli-train.v1",
        "VoxPopuli-dev.v1",
        "Fleurs-train.v1",
        "Heroico-usma.v1",
        "Heroico-heroico.recordings.v1",
        "Heroico-heroico.answers.v1",
        "MediaSpeech-batch.1.v1",
        "SpokenWikipedia-train.v1",
    ],
    "corpus.ES_AR.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1",
    ],
    "corpus.ES_CL.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1",
    ],
    "corpus.ES_CO.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1",
    ],
    "corpus.ES_ES.f16kHz": [
        "Audiria-batch.1.v1",
        "RTVE-batch.1.common.v2",
        "RTVE-batch.2.upv.v2",
        "Europarl-ST-train.v1",
        "Europarl-ST-dev.v1",
        "Europarl-ST-train-noisy.v1",
    ],
    "corpus.ES_MX.f16kHz": [
        "ExcelsiorTV-batch.1.v1",
        "Mexican.TV.Broadcast-batch.1.v1",
        "CIEMPIESS-balance.train.v1",
        "CIEMPIESS-fem.train.v1",
        "CIEMPIESS-light.train.v1",
    ],
    "corpus.ES_PE.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1",
    ],
    "corpus.ES_PR.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1",
    ],
    "corpus.ES_VE.f16kHz": [
        "GoogleCrowdsourced-batch.1.v1",
    ],
    "corpus.ES_US.f16kHz": [
        "AllMDScript-batch.1.v2",
        "CNN-batch.1.v1",
        "ReadNews.Latino-batch.1.v1",
        "Hub4-batch.1.v2",
        "NJHealth-batch.1.train.v2",
        "NJHealth-batch.3.MTurk.v1",
        "Speecon-batch.1.v1",
        "OFTravelRecordings-batch.1.v1",
        "TelemundoNoticias-batch.1.v1",
        "Telenovela-batch.1.v1",
    ],
}
corpora_8k = {
    "corpus.ES.f8kHz": [
        "Fisher-batch.1.train.v2",
        "Fisher-batch.2.dev.v2",
    ],
    "corpus.ES_AR.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",
    ],
    "corpus.ES_CL.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",
    ],
    "corpus.ES_CO.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",
    ],
    "corpus.ES_ES.f8kHz": [
        "Appen-batch.1.v1",
    ],
    "corpus.ES_PE.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",
    ],
    "corpus.ES_US.f8kHz": [
        "AAA-batch.1.v1",
        "CallFriend-batch.1.v1",
        "CallHome-batch.1.train.v2",
        "CollinCounty-batch.1.v2",
        "Hub5-batch.1.v1",
        "Ignite.ATNT-batch.1.v2",
        "Ignite.ATNT-batch.2.v1",
        "Ignite.HomeShopping-batch.1.v2",
        "Ignite.LiveAgent-batch.1.v2",
        "ListenTrust-batch.1.v2",
        "ListenTrust-batch.2.v2",
        "ListenTrust-batch.3.capital.v3",
        "ListenTrust-batch.4.funeral.v3",
        "ListenTrust-batch.5.immigration.v3",
        "ListenTrust-batch.6.real.v3",
        "ListenTrust-batch.7.simulated.v2",
        "NameAddr-batch.1.v1",
    ],
}
corpora_bliss = corpora_16k | corpora_8k

# sentencepiece model
spm_ns = "subword_units.sentencepiece.ES"
spm_name = "2025-04-spm_10240-nmt_nfkc_cf-mbw"

fsa_topology_config = rasr.RasrConfig()
fsa_topology_config.lib_rasr.alignment_fsa_exporter.allophone_state_graph_builder.topology = (
    "ctc"
)
fsa_exporter_kwargs = {
    "extra_fsa_exporter_post_config": fsa_topology_config,
}

# Extra options for the HDF dataset.
# Relevant: partition_epoch splits one epoch into N subepochs.
#           This generates checkpoints faster but has implications for LR scheduling.
dataset_extra_opts = {
    "partition_epoch": 100,  # about 1.5h
    "data_dtype": "int16",
    "train_set_sorting": "laplace:.1000",
    "cv_hdf_artefact_specs": {
        "dev-8k": dev_hdf_8k_artefact_specs,
        "dev-16k-1": dev_hdf_16k_europarl_artefact_specs,
        "dev-16k-2": dev_hdf_16k_voxpopuli_artefact_specs,
        "dev-16k-3": dev_hdf_16k_commonvoice_artefact_specs,
    },
}


# Network config artefact.
# Most conformer nets are too deep an need a special python_prolog:
# "config_params": {"python_prolog": "import sys\nsys.setrecursionlimit(10000)"},
network_config_ns = "network_config.pt_conformer_rel_pos"
network_config_name = "2024-09-conformer_rel_pos-9001_cart-16kHz_logmel"
network_config_kwargs = {
    "num_heads": 14,
    "num_layers": 20,
    "num_outputs": num_outputs,
    "auxloss_layers": [10, 20],
    "internal_subsampling_rate": 4,
    "output_upsampling": False,
    "upsampler_output_dropout": 0.0,
    "i6_models_artefact_spec": ArtefactSpecification(
        "software.i6_models", "i6_models-2025-05-13", commit="8c5460f2398889abb3fe605e9180e9d03ad216ce"
    ),
    #"i6_models_commit": "8c5460f2398889abb3fe605e9180e9d03ad216ce",
}

# training returnn config components
training_config_ns = "returnn_config.training"
training_config_name = "modular-training-pt-600-v1"
training_config_kwargs = {
    "num_epochs": num_subepochs,
    "training_args": {
        "mem_rqmt": 64,
        "horovod_num_processes": 8,  # num gpus to use
        "distributed_launch_cmd": "torchrun",
    },
    "config_params": {
        "config": {
            "use_horovod": False,
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},
            "torch_dataloader_opts": {"num_workers": 1},
            # important for usage of the simple HDFDataset
            "cache_size": "0",
            "torch_log_memory_usage": True,
            "max_seq_length": {"data": 30 * 16000},  # 30s covers >99.9% seqs
            "torch_amp": {"dtype": "bfloat16", "grad_scaler": None},
            "accum_grad_multiple_step": 6,  # effective batch size: 150k frames
        },
    },
}

returnn_config_batch_size_ns = "returnn_config.batch_size"
returnn_config_batch_size_name = "raw-wav-16kHz-batch_size-2080"
returnn_config_batch_size_kwargs = {"batch_size": 25_000}

# Similar to 1b English model: warm-up(8%) + constant(64%) + decay(28%)
# new paln: warm-up(8%) + 300 constant(48%) + 275 decay(44%)
returnn_config_learning_rates_ns = "returnn_config.learning_rates"
returnn_config_learning_rates_name = "finetune-1e-5"
returnn_config_learning_rates_kwargs = {
    "learning_rate": None,
    "learning_rates": [1e-6 + (5e-4 - 1e-6) / 50 * i for i in range(50)]
    + [5e-4] * 300
    + [5e-4 + (1e-6 - 5e-4) / 275 * (i + 1) for i in range(275)],
}

# Training chunking setting (not compatible with fullsum yet)
returnn_config_chunking_ns = "returnn_config.chunking"
returnn_config_chunking_name = "no-chunking"
returnn_config_chunking_kwargs = {}

returnn_config_optimizer_ns = "returnn_config.optimizer"
returnn_config_optimizer_name = "adam"
returnn_config_optimizer_kwargs = {"class": "adamW", "weight_decay": 0.05}


# Torch train step function
i6_native_ops_repo = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_native_ops",
    commit="0cdaf6038b5b348c242156aae24bf5d8fcde48c5",
    checkout_folder_name="i6_native_ops",
).out_repository

train_step_func_ns = "returnn_config.pytorch"
train_step_func_name = "fullsum-max-likelihood-loss-v2"
train_step_func_kwargs = {
    "label_posterior_scale": 1.0,
    "transition_scale": 1.0,
    "label_smoothing_scale": 0.05,
    "label_smoothing_exclude_labels": [0],
    "mainloss_scale": 0.7,
    "zero_infinity": True,
    "i6_native_ops_repo": i6_native_ops_repo,
}

# acoustic_model_config artefact
training_acoustic_model_config_ns = "am_config"
training_acoustic_model_config_name = "e2e-monophone-am-config-v1"
training_acoustic_model_config_kwargs = {}

# prior computation torch forward step function
prior_forward_step_func_ns = "returnn_config.pytorch"
prior_forward_step_func_name = "recognition-log-softmax"
# prior computation returnn callback function
prior_callback_ns = "returnn_config.pytorch"
prior_callback_name = "prior-callback"
# recognition torch forward step function
forward_step_func_ns = "returnn_config.pytorch"
forward_step_func_name = "recognition-log-softmax"


### Software ###
# RASR artefact and the runtime used to run RASR
# usually the most recent version available should be used
asrmon_name = "asrmon-2024-10-17"
rasr_name = "streaming-rasr-2025-07-12"
returnn_name = "returnn-2025-04-22"
runtime_name = "ApptekCluster-ubuntu2204-tf2.15.1-pt2.3.0-2024-04-24"


def py():
    aar = AbstractArtefactRepository()

    navigation = ["config_params", "post_config", "cleanup_old_models"]
    nested_dict = training_config_kwargs
    for param in navigation:
        nested_dict.setdefault(param, {})
        nested_dict = nested_dict[param]
    nested_dict["keep"] = sorted(
        list(
            set(
                nested_dict.get("keep", [])
                + list(range(50, 351, 10))
                + [
                    450,
                    550,
                    625,
                ]  # keep every 10 epoch during constant LR to possibly stop early
            )
        )
    )

    training_args, training_config = (
        ArtefactSpecification(
            training_config_ns, training_config_name, **training_config_kwargs
        )
        .get_factory(aar)
        .build()
    )

    training_config.update(
        ArtefactSpecification(
            network_config_ns, network_config_name, **network_config_kwargs
        )
        .get_factory(aar)
        .build()
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_batch_size_ns,
            returnn_config_batch_size_name,
            **returnn_config_batch_size_kwargs,
        ).build(aar)
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_chunking_ns,
            returnn_config_chunking_name,
            **returnn_config_chunking_kwargs,
        ).build(aar)
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_learning_rates_ns,
            returnn_config_learning_rates_name,
            **returnn_config_learning_rates_kwargs,
        ).build(aar)
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_optimizer_ns,
            returnn_config_optimizer_name,
            optimizer_kwargs=returnn_config_optimizer_kwargs,
        ).build(aar)
    )

    prior_config = copy.deepcopy(training_config)
    prior_config.config.pop("torch_distributed", None)
    prior_config.config.pop("max_seq_length", None)
    prior_config.config["model_outputs"] = {
        "output": {
            "dim": num_outputs,
        }
    }
    prior_config.update(
        ArtefactSpecification(
            prior_forward_step_func_ns, prior_forward_step_func_name
        ).build(aar)
    )
    prior_config.update(
        ArtefactSpecification(prior_callback_ns, prior_callback_name).build(aar)
    )

    recog_network_config_kwargs = copy.deepcopy(network_config_kwargs)
    recog_network_config_kwargs.update({"mode": "recognition"})
    recognition_config = ArtefactSpecification(
        network_config_ns, network_config_name, **recog_network_config_kwargs
    ).build(aar)
    recognition_config.update(
        ArtefactSpecification(forward_step_func_ns, forward_step_func_name).build(aar)
    )

    returnn_root = aar.get_artefact_factory("software.returnn", returnn_name).build()[
        "returnn_root"
    ]

    artefacts = {
        "runtime_spec": ArtefactSpecification("runtime", runtime_name),
        "rasr_spec": ArtefactSpecification("rasr", rasr_name),
    }
    gs.worker_wrapper = artefacts["runtime_spec"].build(aar).worker_wrapper

    feature_hdf_artefact_specs = {
        hdf_ns_ + corpus: ArtefactSpecification(hdf_ns_, corpus)
        for hdf_ns_, corpora_ in train_corpora_features.items()
        for corpus in corpora_
    }

    # Prepare training dataset
    dataset_opts = {
        "dataset_type": "DistributeHDFDataset",
        "hdf_artefact_specs": feature_hdf_artefact_specs,
        **dataset_extra_opts,
    }

    corpus_specs = {
        f"{ns}.{name}": ArtefactSpecification(ns, name)
        for ns, corpus_list in corpora_bliss.items()
        for name in corpus_list
    }
    spm_spec = ArtefactSpecification(spm_ns, spm_name)
    training_am_config_spec = ArtefactSpecification(
        training_acoustic_model_config_ns,
        training_acoustic_model_config_name,
        **training_acoustic_model_config_kwargs,
    )

    nonword_lex = Lexicon()
    for phon in ["[SILENCE]", "[NOISE]", "[MUSIC]"]:
        nonword_lex.add_phoneme(phon, variation="none")
    nonword_lex.add_lemma(
        Lemma(orth=["<blank>"], phon=["[SILENCE]"], special="blank", synt=[], eval=[[]])
    )
    nonword_lex.add_lemma(
        Lemma(orth=["[silence]"], phon=["[SILENCE]"], special="silence")
    )
    nonword_lex.add_lemma(Lemma(orth=["[noise]"], phon=["[NOISE]"]))
    nonword_lex.add_lemma(Lemma(orth=["[vocalized-noise]"], phon=["[NOISE]"]))
    nonword_lex.add_lemma(Lemma(orth=["[vocalized-unknown]"], phon=["[NOISE]"]))
    nonword_lex.add_lemma(
        Lemma(orth=["[unknown]"], phon=["[NOISE]", "[MUSIC]"], special="unknown")
    )
    nonword_lex.add_lemma(
        Lemma(
            orth=["[sentence-begin]"], synt=["<s>"], eval=[[]], special="sentence-begin"
        )
    )
    nonword_lex.add_lemma(
        Lemma(orth=["[sentence-end]"], synt=["</s>"], eval=[[]], special="sentence-end")
    )

    fsa_exporter_opts = {
        "corpus_specs": corpus_specs,
        "sentencepiece_spec": spm_spec,
        "am_config_spec": training_am_config_spec,
        "meta_lexicon_extra_kwargs": {"nonword_lex": nonword_lex},
        **fsa_exporter_kwargs,
    }
    fsa_exporter_config_builder = SentencepieceRasrFsaExporterConfigBuilder(
        aar=aar, **fsa_exporter_opts
    )
    fsa_exporter_config_path = fsa_exporter_config_builder.build()
    training_lexicon = fsa_exporter_config_builder.corpus_merger.out["lexicon"]
    tk.register_output("training_lexicon", training_lexicon)
    # filter by unknown
    dataset_opts["segment_list"] = fsa_exporter_config_builder.out_segments

    py_rasr_arg = copy.deepcopy(artefacts)
    returnn_exe = PyrasrVenvBuilder(aar, **py_rasr_arg).build(
        tk.Path("/usr/bin/python3")
    )

    train_step_config = ArtefactSpecification(
        train_step_func_ns,
        train_step_func_name,
        fsa_exporter_config_path=fsa_exporter_config_path,
        **train_step_func_kwargs,
    ).build(aar)

    training_config.update(train_step_config)

    # Run training
    nn_system = NNSystemV2(
        aar,
        model_type=NNSystemV2.ModelType.onnx_from_pt,
        training_args=training_args,
        training_config=training_config,
        recognition_config=recognition_config,
        prior_config=prior_config,
        dataset_options=dataset_opts,
        exp_name=experiment_name,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        **artefacts,
    )
    nn_system.run()
    nn_system.training_job.rqmt.update({"gpu_mem": 48})
    return nn_system.training_job, spm_spec.build(aar), nn_system.training_job.out_checkpoints[num_subepochs]#.out_checkpoints[625]


import returnn.frontend as rf
import returnn.torch.frontend as rtf
from typing import Dict, Any, Optional
from returnn.tensor import Tensor, Dim, single_step_dim, batch_dim
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
import torch

NETWORK_CONFIG_KWARGS = {
    # Required by RETURNN torch engine, could be used to build epoch/step dependent model, not used yet, see networks.conformer class
    "epoch": 625, # Dummy
    "step": 0, # Dummy
    #
    "num_layers": 20,
    "num_heads": 14,
    "enc_input_dim": 896,
    "dropout": 0.1,
    "sampling_rate": 16000,
    "conv1_channels": 32,
    "conv2_channels": 64,
    "conv3_channels": 64,
    "conv4_channels": 32,
    "num_channels": 80,
    "enc_hidden_dim": 3584,
    "conv_filter_size": 33,
    "num_outputs": 10237,
    "extract_logmel_features": True,
    "use_specaugment": True,
    "auxloss_layers": [10, 20],
    "internal_subsampling_rate": 4,
    "output_upsampling": False,
    "with_bias": True,
    "learnable_pos_emb": True,
    "rel_pos_clip": 16,
    "with_linear_pos": False,
    "with_pos_bias": False,
    "separate_pos_emb_per_head": False,
    "pos_emb_dropout": 0.0,
    "encoder_output_dropout": 0.1,
    "upsampler_output_dropout": 0.0,
    "dropout_broadcast_axes": "T",
}

BLANK_IDX = 3

def get_model_and_vocab(fine_tuned_model: bool = False):
    from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoints
    if fine_tuned_model:
        from i6_experiments.users.zhang.experiments.apptek.am.ctc_streaming_finetuning import py as FT_py, ctc_model_def as ctc_FT_model_def
        training_job, vocab, *_ = FT_py()
        model_def = ctc_FT_model_def
    else:
        training_job, vocab, *_ = py()
        model_def = ctc_model_def
    i6_models_repo = CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/i6_models",
        commit='8c5460f2398889abb3fe605e9180e9d03ad216ce',#network_config_kwargs["i6_models_commit"],
        checkout_folder_name="i6_models",
    ).out_repository
    from i6_core.serialization.base import ExternalImport
    i6_models = ExternalImport(import_path=i6_models_repo)
    return ModelWithCheckpoints.from_training_job(definition=model_def, training_job=training_job), vocab, i6_models

def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    return Model(**_get_ctc_model_kwargs_from_global_config(target_dim=target_dim))


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 22
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = 120


def _get_ctc_model_kwargs_from_global_config(*, target_dim: Dim) -> Dict[str, Any]:
    from returnn.config import get_global_config
    from i6_experiments.users.zhang.experiments.lm.ffnn import FeedForwardLm
    from returnn.frontend.decoder.transformer import TransformerDecoder

    config = get_global_config()  # noqa
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=NETWORK_CONFIG_KWARGS["enc_input_dim"], kind=Dim.Types.Feature)


    network_config = config.typed_value("network_config_kwargs", None)
    assert network_config, "Must provide network_config_kwargs for this model loading"
    recog_lm = None
    recog_language_model = config.typed_value("recog_language_model", None)
    preload_dict = config.typed_value("preload_from_files", None)
    recog_lm_load = None
    if preload_dict is not None:
        recog_lm_load = preload_dict.get("recog_lm", None)

    if recog_language_model and recog_lm_load:
        assert isinstance(recog_language_model, dict)
        recog_language_model = recog_language_model.copy()
        cls_name = recog_language_model.pop("class")
        #raise NotImplementedError
        if cls_name == "FeedForwardLm":
            recog_lm = FeedForwardLm(vocab_dim=target_dim, **recog_language_model)
        elif cls_name == "TransformerLm":
            recog_lm = TransformerDecoder(encoder_dim=None,vocab_dim=target_dim, **recog_language_model)

    kwargs = dict(
        in_dim=in_dim,
        target_dim=target_dim,
        blank_idx=BLANK_IDX,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        recog_language_model=recog_lm,
    )
    kwargs.update(network_config)
    return kwargs


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx

from torch import nn
def promote_buffers_to_params(root: nn.Module):
    """
    Replace every buffer on every submodule with a nn.Parameter(requires_grad=False).
    This unblocks RETURNN's PTModuleAsRFModule bridge that expects Parameters.
    Only for inference use.
    """
    for _, submod in root.named_modules():  # includes root
        # iterate only direct buffers of this submodule
        for buf_name, buf in list(submod.named_buffers(recurse=False)):
            # remove buffer first (otherwise setattr/register may keep it as buffer)
            if buf_name in submod._buffers:
                submod._buffers.pop(buf_name)
            # register parameter with same local name (no dots here)
            # ensure it does not require grad (ok for any dtype; grads won’t be created)
            param = nn.Parameter(buf, requires_grad=False)
            submod.register_parameter(buf_name, param)

from returnn.torch.frontend.bridge import PTModuleAsRFModule
from i6_experiments.users.zhang.experiments.apptek.am.hot_fix import permute_logits_16khz_spm10k
class Model(rf.Module):
    """
    RF Module wrapper around the PyTorch ConformerRelPosModel for RETURNN compatibility.
    """
    def __init__(
        self,
        epoch: int,
        step: int,
        *,
        sampling_rate: int,
        num_channels: int,
        conv1_channels: int,
        conv2_channels: int,
        conv3_channels: int,
        conv4_channels: int,
        enc_input_dim: int,
        enc_hidden_dim: int,
        dropout: float,
        num_heads: int,
        conv_filter_size: int,
        num_layers: int,
        num_outputs: int,
        use_specaugment: bool = True,
        specaugment_kwargs: dict = None,
        extract_logmel_features: bool = True,
        auxloss_layers: list = None,
        internal_subsampling_rate: int = 3,
        output_upsampling: bool = True,
        with_bias: bool = True,
        learnable_pos_emb: bool = True,
        rel_pos_clip: int = 16,
        with_linear_pos: bool = False,
        with_pos_bias: bool = False,
        separate_pos_emb_per_head: bool = False,
        pos_emb_dropout: float = 0.0,
        encoder_output_dropout: float = 0.1,
        upsampler_output_dropout: float = 0.1,
        dropout_broadcast_axes: str = "T",
        #--------args for wrapper------------
        target_dim: Dim = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        recog_language_model: Optional[Any] = None,
        **kwargs
    ):
        super(Model, self).__init__()
        from returnn.config import get_global_config
        from apptek_asr.lib.pytorch.networks.conformer_rel_pos import ConformerRelPosModel

        config = get_global_config(return_empty_if_none=True)
        self.ctc_am_scale = config.float("ctc_am_scale", 1.0)
        self.ctc_prior_scale = config.float("ctc_prior_scale", 0.0)
        self.blank_logit_shift = None
        self.out_blank_separated = False # Assume standard case
        self.target_dim = target_dim
        self.wb_target_dim = target_dim # The vocab contains <blank>
        self.wb_target_dim.vocab = self.target_dim.vocab

        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx
        self.recog_language_model = recog_language_model

        print(f"eos_idx: {eos_idx}, bos_idx: {bos_idx}, blank_idx: {blank_idx}, wb_target_dim.vocab: {self.wb_target_dim.vocab}")
        # Instantiate the underlying PyTorch Conformer model
        assert target_dim.dimension == num_outputs + 3, f"target_dim {target_dim.dimension} != num_outputs {num_outputs}"
        pt = ConformerRelPosModel(
            epoch=epoch,
            step=step,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            conv3_channels=conv3_channels,
            conv4_channels=conv4_channels,
            enc_input_dim=enc_input_dim,
            enc_hidden_dim=enc_hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            conv_filter_size=conv_filter_size,
            num_layers=num_layers,
            num_outputs=num_outputs,
            use_specaugment=use_specaugment,
            specaugment_kwargs=specaugment_kwargs,
            extract_logmel_features=extract_logmel_features,
            auxloss_layers=auxloss_layers,
            internal_subsampling_rate=internal_subsampling_rate,
            output_upsampling=output_upsampling,
            with_bias=with_bias,
            learnable_pos_emb=learnable_pos_emb,
            rel_pos_clip=rel_pos_clip,
            with_linear_pos=with_linear_pos,
            with_pos_bias=with_pos_bias,
            separate_pos_emb_per_head=separate_pos_emb_per_head,
            pos_emb_dropout=pos_emb_dropout,
            encoder_output_dropout=encoder_output_dropout,
            upsampler_output_dropout=upsampler_output_dropout,
            dropout_broadcast_axes=dropout_broadcast_axes,
            **kwargs
        )
        promote_buffers_to_params(pt)
        self.AM = PTModuleAsRFModule(pt_module=pt)

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        features_len: Tensor = None,
        collected_outputs: dict = None
    ):
        """
        Forward pass: accepts RETURNN frontend Tensor inputs, converts to torch, runs the PyTorch model, and converts back to rf.Tensor.
        """
        # Convert rf.Tensor to torch.Tensor

        src_torch = source.raw_tensor * 32768.0 # We need PCM int16,
        # -> RETURNN dataloader uses soundfile to load audio, which normalize to [-1,1] by default

        # get mask via in_spatial_dim API (try both variants, some APIs accept the source)
        try:
            # some implementations require the source tensor; others don't
            mask_rf = in_spatial_dim.get_mask(source)
        except TypeError:
            mask_rf = in_spatial_dim.get_mask()

        # Defensive: if mask_rf is None, fall back to full-length later
        if mask_rf is None:
            # fallback: infer length from full time dimension of `source`
            features_len_t = torch.full((src_torch.size(0),), src_torch.size(1), dtype=torch.long, device=src_torch.device)
        else:
            # mask_rf might be boolean or 0/1 float. Sum over in_spatial_dim to get lengths.
            # cast to float if needed then reduce sum:
            mask_float = rf.cast(mask_rf, dtype=rf.get_default_float_dtype()) #if not getattr(mask_rf, "dtype",None) else mask_rf
            features_len_rf = rf.reduce_sum(mask_float, axis=in_spatial_dim)  # shape [B]
            # convert to torch long
            features_len_t = features_len_rf.raw_tensor.long()

        # Test
        # def test_lp(scales = [1.0, 4.0, 16.0, 128.0, 32768.0]):
        #     x = source.raw_tensor  # [B, T, 1]
        #     for gain in scales:
        #         x2 = x * gain
        #         outputs, out_len = self._pt_module(x2, features_len_t)  # (or path you use)
        #         logits_t = outputs[-1]
        #         lp = torch.log_softmax(logits_t, dim=-1)
        #         top = lp.argmax(-1)
        #         ratio = (top != self.blank_idx).float().mean().item()
        #         print(f"gain={gain} non_blank_ratio={ratio:.4f}")

        # with torch.no_grad():
        #     x = source.raw_tensor  # [B, T, 1]
        #     for gain in [1.0, 4.0, 16.0, 128.0, 32768.0]:
        #         x2 = x * gain
        #         outputs, out_len = self._pt_module(x2, features_len_t)  # (or path you use)
        #         logits_t = outputs[-1]
        #         lp = torch.log_softmax(logits_t, dim=-1)
        #         top = lp.argmax(-1)
        #         ratio = (top != self.blank_idx).float().mean().item()
        #         print(f"gain={gain} non_blank_ratio={ratio:.4f}")

        # Run PyTorch model
        outputs, out_len = self.AM(src_torch, features_len_t)  # _pt_module is the original PT model

        # Convert back to rf.Tensor
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 0:
                raise ValueError("Model returned empty output list")
            # Take the final head
            logits_torch = outputs[-1]  # shape [B, T, F]
        else:
            assert isinstance(outputs, torch.Tensor)
            logits_torch = outputs

        logits_torch = permute_logits_16khz_spm10k(logits_torch)
        #import pdb;pdb.set_trace()
        out_len_rf = rtf.TorchBackend.convert_to_tensor(out_len, dims=[batch_dim], name="output_length", dtype="int64")
        enc_spatial_dim = Dim(None, name="enc_spatial", dyn_size_ext=out_len_rf)

        logits = rtf.TorchBackend.convert_to_tensor(logits_torch, dims=[batch_dim, enc_spatial_dim, self.wb_target_dim], name=f"output", dtype=rf.get_default_float_dtype())
        logits.feature_dim = self.wb_target_dim
        enc = None # For now Will not be used anyway
        return logits, enc, enc_spatial_dim

    def log_probs_wb_from_logits(self, logits: rf.Tensor) -> rf.Tensor:
        """
        Minimal inference-oriented version:
          - supports joint (blank in vector) and out_blank_separated modes,
          - applies blank_logit_shift (non-inplace),
          - computes stable log-probs,
          - applies ctc_am_scale and ctc_prior_scale (batch/static).
        Returns an rf.Tensor with feature_dim == self.wb_target_dim.
        """
        if not self.out_blank_separated:
            if self.blank_logit_shift:
                # Avoid in-place ops: use `+` not `+=`
                logits = logits + rf.sparse_to_dense(
                    self.blank_idx, label_value=self.blank_logit_shift, other_value=0, axis=self.wb_target_dim
                )
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)

        else:
            # separate blank handling
            assert self.blank_idx == self.target_dim.dimension
            dummy_blank_feat_dim = rf.Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            # emission normalized conditionally over target_dim
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            # we could call self._maybe_apply_on_log_probs on emissions, but it's identity for inference
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank)
            if self.blank_logit_shift:
                logits_blank = logits_blank + self.blank_logit_shift
            log_probs_blank = rf.log_sigmoid(logits_blank)  # log P(blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)  # log P(not-blank)
            # joint log-prob for emissions = log P(emission|not-blank) + log P(not-blank)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
        log_probs.feature_dim = self.wb_target_dim

        # no gradient-side ops in inference (kept as identity)
        log_probs = self._maybe_apply_on_log_probs(log_probs)

        # apply scaling/prior adjustments (these matter for scoring in inference)
        if self.ctc_am_scale == 1 and self.ctc_prior_scale == 0:
            return log_probs

        log_probs_am = log_probs
        log_probs = log_probs_am * self.ctc_am_scale

        if self.ctc_prior_scale:
            if self.ctc_prior_type == "batch":
                axis = [dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
                log_prob_prior = rf.reduce_logsumexp(log_probs_am, axis=axis)
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "static":
                log_prob_prior = self.static_prior
                assert log_prob_prior.dims == (self.wb_target_dim,)
            else:
                raise ValueError(f"invalid ctc_prior_type {self.ctc_prior_type!r}")
            log_probs = log_probs - (log_prob_prior * self.ctc_prior_scale)

        return log_probs

    def _maybe_apply_log_probs_normed_grad(self, log_probs: rf.Tensor) -> rf.Tensor:
        """
        No-op for inference. Real implementation modifies gradients and is only needed for training.
        """
        return log_probs

    def _maybe_apply_on_log_probs(self, log_probs: rf.Tensor) -> rf.Tensor:
        """
        Identity wrapper for inference; keeps method interface compatible with rf_conformer.Model.
        """
        # We keep the same checks to be defensive, but do not change values.
        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim
        # skip gradient-only transforms
        return log_probs

    def get_default_name(self):
        return "conformer_relpos_rf"
