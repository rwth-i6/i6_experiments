import numpy as np

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict

from .data import get_tts_data_from_ctc_align
from .config import get_training_config
from .pipeline import synthesize_with_splits, create_tts, synthesize_arbitrary_corpus

from ..default_tools_2023 import RETURNN_ROOT, RETURNN_COMMON, RETURNN_EXE
from ..ctc_aligner.experiments import get_baseline_ctc_alignment_v2
from ..gl_vocoder.default_vocoder import get_default_vocoder

from ..storage import add_ogg_zip


def architecture_experiments():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    base_name = "experiments/alignment_analysis_tts/architecture_experiments/ctc_based/"


    alignment_hdf = get_baseline_ctc_alignment_v2(silence_preprocessed=True)
    training_datasets, durations = get_tts_data_from_ctc_align(alignment_hdf=alignment_hdf, silence_preprocessed=True)

    def run_exp(name, short_name, training_config):
        synth_xml = synthesize_with_splits(
            name=name,
            reference_corpus=get_bliss_corpus_dict()["train-clean-360"],
            corpus_name="train-clean-360",
            job_splits=20,
            datasets=forward_data,
            returnn_root=RETURNN_ROOT,
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

    network_args = {
        "model_type": "nar_tts.blstm_v1",
    }

        training_config = get_training_config(
            returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, debug=False, **network_args
        )  # implicit reconstruction loss
        training_config.config["learning_rates"] = list(np.linspace(2e-4, 2e-3, 90)) + list(np.linspace(2e-3, 2e-4, 90)) + list(np.linspace(2e-4, 1e-6, 20))

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