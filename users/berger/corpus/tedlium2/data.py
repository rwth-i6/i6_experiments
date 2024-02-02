from typing import Dict, Tuple

from i6_core.bpe.train import ReturnnTrainBpeJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict
import i6_experiments.common.datasets.tedlium2.lexicon as tdl_lexicon
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs
from i6_experiments.common.setups.rasr import util as rasr_util
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.recipe import lexicon


def get_data_inputs(
    add_unknown_phoneme_and_mapping: bool = False,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    data_inputs = get_corpus_data_inputs(add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping)

    original_bliss_lexicon = tdl_lexicon.get_bliss_lexicon(
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping
    )

    if use_augmented_lexicon:
        bliss_lexicon = tdl_lexicon.get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping
        )
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

    train_data_input = data_inputs["train"]["train"]
    dev_data_input = data_inputs["dev"]["dev"]
    test_data_input = data_inputs["test"]["test"]

    assert dev_data_input.lm

    lm = helpers.ArpaLMData(filename=dev_data_input.lm["filename"], scale=dev_data_input.lm.get("scale", 1.0))

    train_data_inputs = {
        "train": helpers.RasrDataInput(
            corpus_object=train_data_input.corpus_object,
            concurrent=train_data_input.concurrent,
            lexicon=lexicon_config,
        )
    }

    dev_data_inputs = {
        "dev": helpers.RasrDataInput(
            corpus_object=dev_data_input.corpus_object,
            concurrent=dev_data_input.concurrent,
            lexicon=lexicon_config,
            lm=lm,
        )
    }

    test_data_inputs = {
        "test": helpers.RasrDataInput(
            corpus_object=test_data_input.corpus_object,
            concurrent=test_data_input.concurrent,
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


def get_bpe(size: int) -> ReturnnTrainBpeJob:
    txt_file = get_text_data_dict()["background-data"]
    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository

    return ReturnnTrainBpeJob(txt_file, size, subword_nmt_repo=subword_nmt_repo)
