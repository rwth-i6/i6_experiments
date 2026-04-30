from dataclasses import asdict

from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.data.hdf_seq_whitelist import (
    ExtractSeqListFromHDFJob,
)

from ...config import get_feature_class_stats_config
from ...data.phmm_common import DatasetSettings
from ...data.phmm_phon import (
    build_eow_phon_phmm_training_dataset_with_optional_hdf,
    get_phmm_eow_vocab_datastream,
)
from ...phmm_default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pytorch_networks.phmm.wav2vec2_hf_phmm_v2_cfg import ModelConfig


def eow_phon_phmm_ls960_wav2vec2_class_stats():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_eow_phon_wav2vec2_class_stats"

    settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )
    librispeech_key = "train-other-960"

    label_datastream = get_phmm_eow_vocab_datastream(
        prefix=prefix_name,
        g2p_librispeech_key=librispeech_key,
    )
    alignment_hdf = [
        tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf")
        for i in range(1, 201)
    ]
    alignment_seq_whitelist = ExtractSeqListFromHDFJob(alignment_hdf).out_seq_list
    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )
    forward_dataset, _datastreams = build_eow_phon_phmm_training_dataset_with_optional_hdf(
        prefix=prefix_name,
        librispeech_key=librispeech_key,
        settings=settings,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
        segment_file=alignment_seq_whitelist,
    )

    model_config = ModelConfig(
        label_target_size=label_datastream.vocab_size,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        aux_loss_layers=[3, 6, 9, 12],
        aux_loss_scales=[1.0, 1.0, 1.0, 1.0],
    )
    compute_class_variance = True
    output_files = (
        ["class_stats_state.pt"]
        + [f"pooled_mean_{layer}.txt" for layer in model_config.aux_loss_layers]
        + [f"class_counts_{layer}.txt" for layer in model_config.aux_loss_layers]
    )
    if compute_class_variance:
        output_files += [f"pooled_variance_{layer}.txt" for layer in model_config.aux_loss_layers]

    returnn_config = get_feature_class_stats_config(
        forward_dataset=forward_dataset,
        network_module="phmm.wav2vec2_hf_phmm_v2",
        config={"batch_size": 300 * 16000},
        net_args={"model_config_dict": asdict(model_config)},
        forward_args={
            "alignment_stream_name": "alignments",
            "downsample_factor": 2,
            "compute_variance": compute_class_variance,
        },
        debug=False,
    )

    forward_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=24,
        time_rqmt=24,
        device="gpu",
        cpu_rqmt=4,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=output_files,
    )
    forward_job.add_alias(prefix_name + "/class_stats_job")
    forward_job.rqmt["gpu_mem"] = 24
    for output_file, out_file in forward_job.out_files.items():
        tk.register_output(prefix_name + f"/{output_file}", out_file)


py = eow_phon_phmm_ls960_wav2vec2_class_stats
