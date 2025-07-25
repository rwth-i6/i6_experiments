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
    "i6_models_commit": "8c5460f2398889abb3fe605e9180e9d03ad216ce",
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
    return nn_system.training_job.out_checkpoints[625]
