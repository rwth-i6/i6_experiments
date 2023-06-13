from typing import Dict, Tuple

import i6_experiments.common.datasets.tedlium2.constants as tdl_constants
import i6_experiments.common.datasets.tedlium2.corpus as tdl_corpus
import i6_experiments.common.datasets.tedlium2.lexicon as tdl_lexicon
from i6_experiments.common.datasets.tedlium2.download import download_data_dict
from i6_experiments.common.setups.rasr import util as rasr_util
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.recipe import lexicon


def get_data_inputs(
        add_unknown_phoneme_and_mapping: bool = False,
        ctc_lexicon: bool = False,
        use_augmented_lexicon: bool = True,
        add_all_allophones: bool = False,
        audio_format: str = "wav",
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    corpus_object_dict = tdl_corpus.get_corpus_object_dict(
        audio_format=audio_format,
    )

    lm = helpers.ArpaLMData(10, download_data_dict().lm_dir)

    original_bliss_lexicon = tdl_lexicon.get_bliss_lexicon(
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping)

    if use_augmented_lexicon:
        bliss_lexicon = tdl_lexicon.get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping)
    else:
        bliss_lexicon = original_bliss_lexicon

    bliss_lexicon = lexicon.EnsureSilenceFirstJob(bliss_lexicon).out_lexicon

    if ctc_lexicon:
        bliss_lexicon = lexicon.DeleteEmptyOrthJob(bliss_lexicon).out_lexicon
        bliss_lexicon = lexicon.MakeBlankLexiconJob(bliss_lexicon).out_lexicon

    lexicon_config = helpers.LexiconConfig(
        filename=bliss_lexicon,
        normalize_pronunciation=False,
        add_all_allophones=add_all_allophones,
        add_allophones_from_lexicon=not add_all_allophones,
    )

    train_data_inputs = {
        "train": helpers.RasrDataInput(
            corpus_object=corpus_object_dict["train"],
            concurrent=tdl_constants.CONCURRENT["train"],
            lexicon=lexicon_config,
        )
    }

    dev_data_inputs = {
        "dev": helpers.RasrDataInput(
            corpus_object=corpus_object_dict["dev"],
            concurrent=tdl_constants.CONCURRENT["dev"],
            lexicon=lexicon_config,
            lm=lm,
        )
    }

    test_data_inputs = {
        "test": helpers.RasrDataInput(
            corpus_object=corpus_object_dict["test"],
            concurrent=tdl_constants.CONCURRENT["test"],
            lexicon=lexicon_config,
            lm=lm,
        )
    }

    return train_data_inputs, dev_data_inputs, test_data_inputs


def get_final_gmm_output():
    output_args = rasr_util.OutputArgs("final")

    for ck in ["train", "dev", "test"]:
        output_args.define_corpus_type(ck, ck)

    output_args.add_feature_to_extract("gt")

    return output_args
