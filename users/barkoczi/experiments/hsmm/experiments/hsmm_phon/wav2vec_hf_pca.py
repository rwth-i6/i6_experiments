import copy
from dataclasses import asdict
from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings
from ...config import get_pca_fit_config, get_forward_config
from ...data.phon import build_eow_phon_training_datasets_95_5_split, get_eow_vocab_datastream
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

# Import the blocks from your updated pipeline file
from ...pipeline import compute_pca_state, dump_pca_features
from ...pytorch_networks.hsmm.wav2vec2_hf_pca_dump_cfg import ModelConfig


def dump_wav2vec2_pca_features_with_alignments():
    prefix_name = "example_setups/librispeech/feature_dump/ls960_wav2vec2_pca512"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        # sorted_reverse is fine for dumping, though you might want regular sorting
        train_seq_ordering="sorted_reverse", 
    )

    librispeech_key = "train-other-960"

    # 1. Setup Label Datastream for Alignments
    label_datastream = get_eow_vocab_datastream(
        prefix=prefix_name,
        g2p_librispeech_key=librispeech_key,
    )

    alignment_hdf = [
        tk.Path(f"/work/asr3/zyang/share/joerg/generative/hsmm/alignments/lbs_mono_phone/train_960/alignment_{i}.hdf")
        for i in range(1, 201)
    ]

    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    # 2. Build Dataset combining Audio and HDF alignments
    train_data = build_eow_phon_training_datasets_95_5_split(
        prefix=prefix_name,
        librispeech_key=librispeech_key,
        settings=train_settings,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="classes",
    )

    base_model_config = ModelConfig(
        label_target_size=label_datastream.vocab_size,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False, # Must be False for clean feature extraction
        pca_dim=512,
        aux_ctc_loss_layers=[3, 6, 9, -1],
    )

    # Adjust this path to wherever you saved the PyTorch network file 
    network_module = "hsmm.wav2vec2_hf_pca_dump"
    
    # ---------------------------------------------------------
    # STAGE 1: FIT PCA ON THE DATASET
    # ---------------------------------------------------------
    pca_model_config = copy.deepcopy(base_model_config)
    pca_model_config.update_pca_during_training = True

    pca_forward_config = {
        "batch_size": 1000 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "num_workers_per_gpu": 2,
    }

    pca_returnn_config = get_pca_fit_config(
        forward_dataset=train_data.train,
        network_module=network_module,
        config=pca_forward_config,
        net_args={"model_config_dict": asdict(pca_model_config)},
        debug=False,
    )

    pca_outputs, pca_job = compute_pca_state(
        prefix_name + "/fit_pca",
        pca_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        use_gpu=True,
    )
    pca_job.rqmt["gpu_mem"] = 24


    # ---------------------------------------------------------
    # STAGE 2: EXTRACT FEATURES & ALIGNMENTS TO HDF5
    # ---------------------------------------------------------
    dump_model_config = copy.deepcopy(base_model_config)
    dump_model_config.pca_state_path = pca_outputs["pca_state.pt"]
    dump_model_config.update_pca_during_training = False

    output_hdf5_name = "ls960_wav2vec2_features.hdf5"

    dump_returnn_config = get_forward_config(
        network_module=network_module,
        config={
            "forward": train_data.train.as_returnn_opts(), 
            "batch_size": 500 * 16000,
            "num_workers_per_gpu": 2,
        },
        net_args={"model_config_dict": asdict(dump_model_config)},
        # Point decoder to the same module so it finds `forward_init_hook` 
        decoder=network_module, 
        decoder_args={
            "config": {
                "output_filename": output_hdf5_name,
                "target_layer": -1, # The layer index from aux_ctc_loss_layers you want to dump
                "alignment_subsample_factor": 2,
            }
        },
        debug=False,
    )

    # Call your newly created pipeline block
    dump_outputs, dump_job = dump_pca_features(
        prefix_name=prefix_name,
        returnn_config=dump_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_filename=output_hdf5_name,
        mem_rqmt=24,
        use_gpu=True,
    )
    # Adjust resources as needed
    dump_job.rqmt["gpu_mem"] = 24
    dump_job.rqmt["cpu"] = 4


py = dump_wav2vec2_pca_features_with_alignments
