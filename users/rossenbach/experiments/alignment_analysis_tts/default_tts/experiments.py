import os
from sisyphus import tk

from .data import get_tts_data_from_ctc_align, TTSForwardData, get_tts_forward_data_legacy
from .config import get_training_config, get_forward_config
from .pipeline import tts_training, tts_forward, gl_swer, synthesize_with_splits, extract_speaker_embedding_hdf

from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT, RETURNN_COMMON, RETURNN_DATA_ROOT

from ..ctc_aligner.experiments import get_baseline_ctc_alignment_v2

from ..gl_vocoder.default_vocoder import get_default_vocoder

from ..synthetic_storage import add_ogg_zip

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
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, **network_args
    )  # implicit reconstruction loss

    forward_config = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=TTSForwardData(
            dataset=training_datasets.cv, datastreams=training_datasets.datastreams
        ),
        **network_args
    )

    train_job = tts_training(
        config=training_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name,
        num_epochs=200,
    )
    forward_job = tts_forward(
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name
    )
    tk.register_output(os.path.join(name, "test.hdf"), forward_job.out_default_hdf)
    vocoder = get_default_vocoder(name=name)
    gl_swer(
        name=name,
        vocoder=vocoder,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
        returnn_root=RETURNN_RC_ROOT,
        returnn_exe=RETURNN_EXE
    )

    speaker_hdf = extract_speaker_embedding_hdf(
        train_job.out_checkpoints[200],
        returnn_common_root=RETURNN_COMMON,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        datasets=training_datasets,
        prefix=name,
        network_args=network_args
    )
    tk.register_output(name + "/exctracted_speakers.hdf", speaker_hdf)

    forward_data = get_tts_forward_data_legacy(
        "train-clean-360",
        speaker_embedding_hdf=speaker_hdf,
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
        checkpoint=train_job.out_checkpoints[200],
        tts_model_kwargs=network_args,
        vocoder=vocoder,
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

