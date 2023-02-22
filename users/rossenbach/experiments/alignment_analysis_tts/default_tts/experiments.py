import os
import numpy as np
from sisyphus import tk

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict

from .data import get_tts_data_from_ctc_align, TTSForwardData, get_tts_forward_data_legacy
from .config import get_training_config, get_forward_config
from .pipeline import tts_training, tts_forward, gl_swer, synthesize_with_splits, extract_speaker_embedding_hdf, create_tts, synthesize_arbitrary_corpus, TTSInferenceSystem

from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT, RETURNN_COMMON, RETURNN_DATA_ROOT
from ..ctc_aligner.experiments import get_baseline_ctc_alignment_v2
from ..gl_vocoder.default_vocoder import get_default_vocoder

from ..storage import add_ogg_zip, duration_alignments

def get_ctc_based_tts():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    name = "experiments/alignment_analysis_tts/default_tts/ctc_based/baseline"

    alignment_hdf = get_baseline_ctc_alignment_v2()
    training_datasets, durations = get_tts_data_from_ctc_align(alignment_hdf=alignment_hdf)

    network_args = {
        "model_type": "tts_model"
    }

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
    )  # implicit reconstruction loss

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        vocoder=get_default_vocoder(name=name)
    )

    forward_data = get_tts_forward_data_legacy(
        "train-clean-360",
        speaker_embedding_hdf=tts_inference_data.speaker_hdf,
        segment_file=None
    )

    from recipe.i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

    synth_xml = synthesize_with_splits(
        name=name,
        reference_corpus=get_bliss_corpus_dict()["train-clean-360"],
        corpus_name="train-clean-360",
        job_splits=20,
        datasets=forward_data,
        returnn_root=RETURNN_RC_ROOT,
        returnn_exe=RETURNN_EXE,
        returnn_common_root=RETURNN_COMMON,
        checkpoint=tts_inference_data.train_job.out_checkpoints[200],
        tts_model_kwargs=tts_inference_data.network_args,
        vocoder=tts_inference_data.vocoder,
        peak_normalization=False,
    )

    from i6_core.returnn.oggzip import BlissToOggZipJob
    ogg_zip = BlissToOggZipJob(
        bliss_corpus=synth_xml,
        segments=None,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
    ).out_ogg_zip

    add_ogg_zip("default_ctc_tts", ogg_zip)

    #synthesize_arbitrary_corpus(
    #    prefix_name=name,
    #    export_name="default_ctc_tts",
    #    random_corpus=ls_360,
    #    tts_model=tts_inference_data
    #)

def get_optimized_tts_models():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    base_name = "experiments/alignment_analysis_tts/default_tts/ctc_based/"


    # for silence_pp in [True, False]:
    for silence_pp in [True]:
        short_name = "gauss_ctc_tts_" + ("spp" if silence_pp else "nospp")
        name = base_name + short_name

        alignment_hdf = get_baseline_ctc_alignment_v2(silence_preprocessed=silence_pp)
        training_datasets, durations = get_tts_data_from_ctc_align(alignment_hdf=alignment_hdf, silence_preprocessed=silence_pp)

        network_args = {
            "model_type": "tts_model",
            "gauss_up": True,
        }

        training_config = get_training_config(
            returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
        )  # implicit reconstruction loss
        training_config.config["learning_rates"] = [0.0001, 0.001]

        tts_inference_data = create_tts(
            name=name,
            training_config=training_config,
            network_args=network_args,
            training_datasets=training_datasets,
            vocoder=get_default_vocoder(name=name),
            # debug=True,
        )

        ls_360 = get_bliss_corpus_dict()["train-clean-360"]

        synthesize_arbitrary_corpus(
            prefix_name=name,
            export_name=short_name,
            random_corpus=ls_360,
            tts_model=tts_inference_data
        )
        
        
def get_ls460_models():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    base_name = "experiments/alignment_analysis_tts/default_tts/ctc_ls460_pe3_oclrv1_based/"


    short_name = "gauss_tts"
    name = base_name + short_name

    alignment_hdf = duration_alignments["ls460_pe3_oclrv1"]
    training_datasets, durations = get_tts_data_from_ctc_align(
        ls_corpus_key="train-clean-460",
        alignment_hdf=alignment_hdf,
        silence_preprocessed=True,
        partition_epoch=3)

    network_args = {
        "model_type": "tts_model",
        "gauss_up": True,
        "embedding_size": 384,
        "speaker_embedding_size": 384,
        "enc_lstm_size": 384,
        "dec_lstm_size": 1536,
    }

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
    )  # implicit reconstruction loss
    training_config.config["learning_rates"] = [0.0001, 0.001]

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        vocoder=get_default_vocoder(name=name),
        # debug=True,
    )
    
    short_name = "gauss_tts_oclrv1"
    name = base_name + short_name
    
    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
    )  # implicit reconstruction loss
    training_config.config["learning_rates"] = list(np.linspace(0.0003, 0.003, 40)) + list(np.linspace(0.003, 0.00003, 60))

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        vocoder=get_default_vocoder(name=name),
        # debug=True,
    )

    short_name = "gauss_tts_oclrv2"
    name = base_name + short_name

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
    )  # implicit reconstruction loss
    training_config.config["learning_rates"] = list(np.linspace(0.0002, 0.002, 40)) + list(np.linspace(0.002, 0.00002, 60))

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        vocoder=get_default_vocoder(name=name, corpus_key="train-clean-460"),
        # debug=True,
    )

def get_ls960_models():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    base_name = "experiments/alignment_analysis_tts/default_tts/ctc_ls960_pe6_oclrv1_based/"


    alignment_hdf = duration_alignments["ls960_pe6_oclrv1"]
    training_datasets, durations = get_tts_data_from_ctc_align(
        ls_corpus_key="train-other-960",
        alignment_hdf=alignment_hdf,
        silence_preprocessed=True,
        partition_epoch=6)

    network_args = {
        "model_type": "tts_model",
        "gauss_up": True,
        "embedding_size": 384,
        "speaker_embedding_size": 384,
        "enc_lstm_size": 384,
        "dec_lstm_size": 1536,
    }


    short_name = "gauss_tts_oclr"
    name = base_name + short_name

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
    )  # implicit reconstruction loss
    training_config.config["learning_rates"] = list(np.linspace(0.0002, 0.002, 40)) + list(np.linspace(0.002, 0.00002, 60))

    tts_inference_data = create_tts(
        name=name,
        training_config=training_config,
        network_args=network_args,
        training_datasets=training_datasets,
        vocoder=get_default_vocoder(name=name, corpus_key="train-other-960"),
        # debug=True,
    )