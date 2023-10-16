from typing import Dict
from i6_core import audio
from i6_core.meta.system import CorpusObject
import i6_experiments.common.datasets.librispeech as lbs_dataset
from sisyphus import tk
import copy
from typing import Dict, List, Tuple
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
    MakeBlankLexiconJob,
)


# align_hdf = ls_corpus.get_alignment_hdf(returnn_root)


def get_corpus_object_dict() -> Dict[str, CorpusObject]:
    bliss_corpus_dict = {
        key: tk.Path(f"/work/asr4/berger/dependencies/librispeech/corpus/{name}_no_timing.gz")
        for key, name in [
            ("sms-train-other-960", "train_960"),
            ("sms-dev-clean", "dev_clean"),
            ("sms-test-clean", "test_clean"),
        ]
    }
    durations = {
        "sms-train-other-960": 1920,
        "sms-dev-clean": 11,
        "sms-test-clean": 11,
    }

    corpus_object_dict = {}

    for corpus_key, bliss_corpus in bliss_corpus_dict.items():
        corpus_object = CorpusObject()
        j = audio.BlissFfmpegJob(
            bliss_corpus,
            ffmpeg_options=["-map_channel", "0.0.0"],
            recover_duration=False,
            output_format="wav",
        )
        corpus_object.corpus_file = j.out_corpus
        corpus_object.audio_format = "wav"
        corpus_object.audio_dir = None
        corpus_object.duration = durations[corpus_key]

        corpus_object_dict[corpus_key] = corpus_object

    return corpus_object_dict


def get_data_inputs(
    train_key: str = "sms-train-other-960",
    dev_keys: List[str] = ["sms-dev-clean"],
    test_keys: List[str] = ["sms-test-clean"],
    use_stress: bool = False,
    add_unknown_phoneme_and_mapping: bool = True,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    corpus_object_dict = copy.deepcopy(get_corpus_object_dict())

    lm = helpers.ArpaLMData(10, lbs_dataset.get_arpa_lm_dict()["4gram"])

    original_bliss_lexicon = lbs_dataset.get_bliss_lexicon(
        use_stress_marker=use_stress,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
    )

    if use_augmented_lexicon:
        bliss_lexicon = lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=use_stress,
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        )[train_key]
    else:
        bliss_lexicon = original_bliss_lexicon

    if ctc_lexicon:
        bliss_lexicon = EnsureSilenceFirstJob(bliss_lexicon).out_lexicon
        bliss_lexicon = DeleteEmptyOrthJob(bliss_lexicon).out_lexicon
        bliss_lexicon = MakeBlankLexiconJob(bliss_lexicon).out_lexicon

    lexicon_config = helpers.LexiconConfig(
        filename=bliss_lexicon,
        normalize_pronunciation=False,
        add_all_allophones=add_all_allophones,
        add_allophones_from_lexicon=not add_all_allophones,
    )

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[train_key] = helpers.RasrDataInput(
        corpus_object=corpus_object_dict[train_key],
        concurrent=300,
        lexicon=lexicon_config,
    )

    for dev_key in dev_keys:
        dev_data_inputs[dev_key] = helpers.RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=20,
            lexicon=lexicon_config,
            lm=lm,
        )

    for test_key in test_keys:
        test_data_inputs[test_key] = helpers.RasrDataInput(
            corpus_object=corpus_object_dict[test_key],
            concurrent=20,
            lexicon=lexicon_config,
            lm=lm,
        )

    return train_data_inputs, dev_data_inputs, test_data_inputs


def get_scoring_corpora() -> Dict[str, tk.Path]:
    return {
        key: tk.Path(f"/work/asr4/berger/dependencies/librispeech/corpus/{name}_transcriptions.gz")
        for key, name in [
            ("sms-train-other-960", "train_960"),
            ("sms-dev-clean", "dev_clean"),
            ("sms-test-clean", "test_clean"),
        ]
    }
