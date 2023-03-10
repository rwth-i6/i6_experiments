import os
from sisyphus import tk

from .data import get_tts_data_from_ctc_align, TTSForwardData
from .config import get_training_config, get_forward_config
from .pipeline import tts_training, tts_forward, gl_swer, synthesize_with_splits, extract_speaker_embedding_hdf, create_tts, synthesize_arbitrary_corpus, TTSInferenceSystem

from ..default_tools import RETURNN_EXE, RETURNN_ROOT, RETURNN_COMMON
from ..ctc_aligner.experiments import get_baseline_ctc_alignment_v2
from i6_experiments.users.rossenbach.datasets.librispeech import get_bliss_corpus_dict

from ..storage import duration_alignments


def get_optimized_tts_models():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    base_name = "experiments/unnormalized/default_tts/ctc_based"

    alignment_hdf = duration_alignments["oclr_v2_rec_0.00"]
    training_datasets, durations = get_tts_data_from_ctc_align(alignment_hdf=alignment_hdf)



    short_name = "gauss_ctc_tts_spp"
    name = base_name + short_name

    network_args = {
        "model_type": "nar_nonatt_taco",
    }

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=True, **network_args,
    )  # implicit reconstruction loss
    training_config.config["learning_rates"] = [0.0001, 0.001]

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        # vocoder=get_default_vocoder(name=name),
        # debug=True,
    )
    
    
    short_name = "gauss_ctc_tts_spp_nobroadcast"
    name = base_name + short_name
    network_args = {
        "model_type": "nar_nonatt_taco_nobroadcast",
    }

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=True, **network_args,
    )  # implicit reconstruction loss
    training_config.config["learning_rates"] = [0.0001, 0.001]

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        # vocoder=get_default_vocoder(name=name),
        # debug=True,
    )

    # ls_360 = get_bliss_corpus_dict()["train-clean-360"]

    # synthesize_arbitrary_corpus(
    #     prefix_name=name,
    #     export_name=short_name,
    #     random_corpus=ls_360,
    #     tts_model=tts_inference_data
    # )