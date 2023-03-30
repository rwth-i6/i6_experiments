"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from dataclasses import dataclass
from typing import Dict
from sisyphus import tk

from i6_core.text.processing import PipelineJob
from i6_core.corpus.transform import MergeCorporaJob, MergeStrategy
from i6_core.meta.system import CorpusObject


from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
    get_bliss_corpus_dict,
    get_arpa_lm_dict,
    get_bliss_lexicon,
    get_g2p_augmented_bliss_lexicon_dict,
    constants,
)
from i6_experiments.common.setups.rasr import RasrDataInput

from i6_experiments.common.baselines.librispeech.data import CorpusData




def get_corpus_data_inputs(
    synthetic_data_bliss: tk.Path, use_g2p_training: bool = True, use_stress_marker: bool = False
) -> CorpusData:
    """
    Create the corpus data for any LibriSpeech RASR setup

    :param corpus_key: which LibriSpeech subset to use e.g. train-other-960, refer to common/datasets/librispeech.py
    :param use_g2p_training: If true, uses Sequitur to generate full lexicon coverage for the training data
    :param use_stress_marker: If the phoneme representation should include the ARPA stress marker
        Sometimes this is also referred to as "unfolded" lexicon.
    :return: (train_data, dev_data, test_data)
    """
    corpus_key = "train-clean-460"

    corpus_object_dict = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")

    bliss_corpus_dict = get_bliss_corpus_dict(
        audio_format="ogg", output_prefix="corpora"
    )

    train_100_v1_bliss = bliss_corpus_dict["train-clean-100"]
    train_100_v2_bliss = PipelineJob(train_100_v1_bliss, ["sed 's/train-clean-100/train-clean-100-v2/g'"],
                                      zip_output=True).out
    train_100_v3_bliss = PipelineJob(train_100_v1_bliss, ["sed 's/train-clean-100/train-clean-100-v3/g'"],
                                      zip_output=True).out

    combined_training = MergeCorporaJob(
        bliss_corpora=[
            train_100_v1_bliss,
            train_100_v2_bliss,
            train_100_v3_bliss,
            synthetic_data_bliss,
            ],
            name="librispeech-combined",
        merge_strategy=MergeStrategy.SUBCORPORA,
    )
    
    combined_corpus_object = CorpusObject()
    combined_corpus_object.corpus_file = combined_training.out_merged_corpus
    combined_corpus_object.audio_format = "ogg"
    combined_corpus_object.audio_dir = None
    combined_corpus_object.duration = 660

    # Definition of the official 4-gram LM to be used as default LM
    lm = {
        "filename": get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    # This is the standard LibriSpeech lexicon
    lexicon = {
        "filename": get_bliss_lexicon(
            use_stress_marker=use_stress_marker,
            add_unknown_phoneme_and_mapping=not use_g2p_training,
        ),
        "normalize_pronunciation": False,
    }

    # In case we train with a G2P-augmented lexicon we do not use the same lexicon for training and recognition.
    # The recognition Lexion is always without G2P to ensure better comparability
    if use_g2p_training:
        train_lexicon = {
            "filename": get_g2p_augmented_bliss_lexicon_dict(
                use_stress_marker=use_stress_marker,
                add_unknown_phoneme_and_mapping=False,
            )[corpus_key],
            "normalize_pronunciation": False,
        }
    else:
        train_lexicon = lexicon

    # Here we define all corpora that are used.
    # The training corpus is dynamic based on which subset we want to use,
    # but dev and test are fixed.
    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[corpus_key] = RasrDataInput(
        corpus_object=combined_corpus_object,
        concurrent=120,
        lexicon=train_lexicon,
        lm=None,
    )

    for dev_key in ["dev-clean", "dev-other"]:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=constants.concurrent[dev_key],
            lexicon=lexicon,
            lm=lm,
        )

    for test_key in ["test-clean", "test-other"]:
        test_data_inputs[test_key] = RasrDataInput(
            corpus_object=corpus_object_dict[test_key],
            concurrent=constants.concurrent[test_key],
            lexicon=lexicon,
            lm=lm,
        )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )