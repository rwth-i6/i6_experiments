import copy
from dataclasses import asdict

from sisyphus import tk

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...config import get_pca_fit_config, get_pca_label_stats_config
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets_95_5_split
from ...default_tools import MINI_RETURNN_ROOT, RETURNN_EXE
from ...pipeline import ASRModel, compute_pca_label_stats, compute_pca_state, search
from ...pytorch_networks.hsmm.wav2vec2_hf_pca_dump_cfg import ModelConfig

from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.data.phmm_phon import (
    get_phmm_eow_lexicon,
    get_phmm_eow_vocab_datastream,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.phmm_default_tools import (
    LIBRASR_WHEEL,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.phmm_lm import (
    get_4gram_lm_rasr_config,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.phmm_rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    CreateLibrasrVenvJob,
    build_librasr_phmm_recognition_config,
)


def _dataset_opts_without_cache_manager(dataset):
    opts = copy.deepcopy(dataset.as_returnn_opts())

    def visit(value):
        if isinstance(value, dict):
            if value.get("use_cache_manager") is True:
                value["use_cache_manager"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(opts)
    return opts


def _stats_output_files(target_layer: int, feature_name: str):
    return [
        f"layer_{target_layer}_{feature_name}_pooled_variance.txt",
        f"layer_{target_layer}_{feature_name}_pooled_variance.npy",
        f"layer_{target_layer}_{feature_name}_pooled_variance.pt",
        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.txt",
        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.npy",
        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.pt",
    ]


def _stats_forward_args(target_layer: int, feature_name: str, label_datastream, *, pca_dims=None):
    config = {
        "target_layer": target_layer,
        "returnn_vocab": label_datastream.vocab,
        "log_interval_batches": 10,
        "alignment_subsample_factor": 2,
    }
    if pca_dims is None:
        config.update(
            {
                "pooled_variance_filename": f"layer_{target_layer}_{feature_name}_pooled_variance.txt",
                "pooled_variance_npy_filename": f"layer_{target_layer}_{feature_name}_pooled_variance.npy",
                "pooled_variance_pt_filename": f"layer_{target_layer}_{feature_name}_pooled_variance.pt",
                "phoneme_means_filename": f"layer_{target_layer}_{feature_name}_phoneme_feature_means.txt",
                "phoneme_means_npy_filename": f"layer_{target_layer}_{feature_name}_phoneme_feature_means.npy",
                "phoneme_means_pt_filename": f"layer_{target_layer}_{feature_name}_phoneme_feature_means.pt",
            }
        )
    else:
        config.update(
            {
                "pca_dims": pca_dims,
                "pooled_variance_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_pooled_variance.txt",
                "pooled_variance_npy_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_pooled_variance.npy",
                "pooled_variance_pt_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_pooled_variance.pt",
                "phoneme_means_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_phoneme_feature_means.txt",
                "phoneme_means_npy_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_phoneme_feature_means.npy",
                "phoneme_means_pt_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_phoneme_feature_means.pt",
            }
        )
    return {"config": config}


def _run_rasr_search(
    *,
    root_prefix: str,
    target_layer: int,
    feature_name: str,
    model_config: ModelConfig,
    stats_outputs,
    label_datastream,
    recog_lexicon,
    test_dataset_tuples,
    phmm_returnn_exe,
    lm_scales,
    network_module: str,
):
    asr_model = ASRModel(
        checkpoint=None,
        net_args={"model_config_dict": asdict(model_config)},
        network_module=network_module,
        prior_file=None,
        prior_files=None,
        prefix_name=root_prefix,
    )

    for lm_scale in lm_scales:
        recog_config = build_librasr_phmm_recognition_config(
            lexicon_path=recog_lexicon,
            lm_config=get_4gram_lm_rasr_config(lexicon_file=recog_lexicon, scale=lm_scale),
            logfile_suffix=f"pca_gaussian_layer{target_layer}_{feature_name}_lm{lm_scale:g}",
        )
        search(
            prefix_name=f"{root_prefix}/layer_{target_layer}/{feature_name}/search_lm{lm_scale:g}",
            forward_config={"num_workers_per_gpu": 0},
            asr_model=copy.deepcopy(asr_model),
            decoder_module="hsmm.decoder.rasr_pca_gaussian_hmm",
            decoder_args={
                "config": {
                    "rasr_config_file": recog_config,
                    "lexicon": recog_lexicon,
                    "returnn_vocab": label_datastream.vocab,
                    "phoneme_means_pt": stats_outputs[
                        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.pt"
                    ],
                    "pooled_variance_pt": stats_outputs[
                        f"layer_{target_layer}_{feature_name}_pooled_variance.pt"
                    ],
                    "target_layer": target_layer,
                    "use_label_priors": True,
                    "missing_label_score": -1.0e6,
                    "normalize_frame_scores": True,
                },
                "extra_config": {
                    "print_hypothesis": False,
                },
            },
            test_dataset_tuples=test_dataset_tuples,
            returnn_exe=phmm_returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
            use_gpu=True,
            debug=False,
        )


def rasr_search_wav2vec2_large_pca_gaussian():
    root_prefix = "example_setups/librispeech/feature_dump/ls960_wav2vec2_large_pca_gaussian_phmm_eow_rasr_search"
    stats_prefix = "example_setups/librispeech/feature_dump/ls960_wav2vec2_large_pca_label_stats_phmm_eow_new_alignments"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )

    librispeech_key = "train-other-960"
    target_layer = 16
    lm_scales = [0.0, 0.5, 1.0, 1.5]

    label_datastream = get_phmm_eow_vocab_datastream(
        prefix=stats_prefix,
        g2p_librispeech_key=librispeech_key,
    )

    alignment_hdf = [
        tk.Path(f"/u/zyang/setups/mini/output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf")
        for i in range(1, 201)
    ]
    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    train_data = build_eow_phon_training_datasets_95_5_split(
        prefix=stats_prefix,
        librispeech_key=librispeech_key,
        settings=train_settings,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
    )

    network_module = "hsmm.wav2vec2_hf_pca_dump"
    layer_prefix = f"{root_prefix}/wav2vec2_large_layer_{target_layer}"

    common_model_config = {
        "label_target_size": label_datastream.vocab_size,
        "hf_model_name": "facebook/wav2vec2-large",
        "pretrained": True,
        "freeze_feature_encoder": True,
        "freeze_encoder": True,
        "apply_spec_augment": False,
        "aux_ctc_loss_layers": [target_layer],
    }

    phmm_returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    phmm_lexicon = get_phmm_eow_lexicon(g2p_librispeech_key=librispeech_key)
    tk.register_output(root_prefix + "/phmm_eow_phon_lexicon.xml.gz", phmm_lexicon)
    recog_lexicon = AddSentenceBoundaryLemmataToPhmmLexiconJob(phmm_lexicon).out_lexicon
    tk.register_output(root_prefix + "/phmm_eow_phon_recog_lexicon.xml.gz", recog_lexicon)

    test_dataset_tuples = {
        dataset_key: build_test_dataset(dataset_key=dataset_key, settings=train_settings)
        for dataset_key in ["dev-clean", "dev-other"]
    }
    _ = get_bliss_corpus_dict(audio_format="ogg")

    pca_dim = 512
    pca_fit_model_config = ModelConfig(
        **common_model_config,
        pca_dim=pca_dim,
    )
    pca_fit_model_config.update_pca_during_training = True
    pca_returnn_config = get_pca_fit_config(
        forward_dataset=train_data.train,
        network_module=network_module,
        config={
            "forward": _dataset_opts_without_cache_manager(train_data.train),
            "batch_size": 400 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 240,
            "num_workers_per_gpu": 1,
        },
        net_args={"model_config_dict": asdict(pca_fit_model_config)},
        forward_args={"config": {"log_interval_batches": 10}},
        debug=False,
    )
    pca_outputs, pca_job = compute_pca_state(
        layer_prefix + f"/fit_pca_{pca_dim}",
        pca_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        use_gpu=True,
    )
    pca_job.rqmt["gpu_mem"] = 24

    pca_feature_name = f"pca_{pca_dim}"
    pca_stats_model_config = copy.deepcopy(pca_fit_model_config)
    pca_stats_model_config.pca_state_path = pca_outputs["pca_state.pt"]
    pca_stats_model_config.update_pca_during_training = False
    pca_stats_returnn_config = get_pca_label_stats_config(
        forward_dataset=train_data.train,
        network_module=network_module,
        config={
            "forward": _dataset_opts_without_cache_manager(train_data.train),
            "batch_size": 800 * 16000,
            "max_seqs": 240,
            "num_workers_per_gpu": 1,
        },
        net_args={"model_config_dict": asdict(pca_stats_model_config)},
        forward_args=_stats_forward_args(
            target_layer,
            pca_feature_name,
            label_datastream,
            pca_dims=[pca_dim],
        ),
        debug=False,
    )
    pca_stats_outputs, pca_stats_job = compute_pca_label_stats(
        prefix_name=layer_prefix + f"/{pca_feature_name}_label_stats",
        returnn_config=pca_stats_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=_stats_output_files(target_layer, pca_feature_name),
        mem_rqmt=24,
        use_gpu=True,
    )
    pca_stats_job.rqmt["gpu_mem"] = 24

    _run_rasr_search(
        root_prefix=root_prefix,
        target_layer=target_layer,
        feature_name=pca_feature_name,
        model_config=pca_stats_model_config,
        stats_outputs=pca_stats_outputs,
        label_datastream=label_datastream,
        recog_lexicon=recog_lexicon,
        test_dataset_tuples=test_dataset_tuples,
        phmm_returnn_exe=phmm_returnn_exe,
        lm_scales=lm_scales,
        network_module=network_module,
    )

    full_feature_name = "full"
    full_model_config = ModelConfig(
        **common_model_config,
        pca_dim=None,
    )
    full_model_config.update_pca_during_training = False
    full_stats_returnn_config = get_pca_label_stats_config(
        forward_dataset=train_data.train,
        network_module=network_module,
        config={
            "forward": _dataset_opts_without_cache_manager(train_data.train),
            "batch_size": 800 * 16000,
            "max_seqs": 240,
            "num_workers_per_gpu": 1,
        },
        net_args={"model_config_dict": asdict(full_model_config)},
        forward_args=_stats_forward_args(
            target_layer,
            full_feature_name,
            label_datastream,
            pca_dims=None,
        ),
        debug=False,
    )
    full_stats_outputs, full_stats_job = compute_pca_label_stats(
        prefix_name=layer_prefix + "/full_label_stats",
        returnn_config=full_stats_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=_stats_output_files(target_layer, full_feature_name),
        mem_rqmt=24,
        use_gpu=True,
    )
    full_stats_job.rqmt["gpu_mem"] = 24

    _run_rasr_search(
        root_prefix=root_prefix,
        target_layer=target_layer,
        feature_name=full_feature_name,
        model_config=full_model_config,
        stats_outputs=full_stats_outputs,
        label_datastream=label_datastream,
        recog_lexicon=recog_lexicon,
        test_dataset_tuples=test_dataset_tuples,
        phmm_returnn_exe=phmm_returnn_exe,
        lm_scales=lm_scales,
        network_module=network_module,
    )


py = rasr_search_wav2vec2_large_pca_gaussian
