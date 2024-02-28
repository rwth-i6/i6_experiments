import copy
from typing import Dict, List, Optional, Tuple

from i6_core.bpe.train import ReturnnTrainBpeJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict
import i6_experiments.common.datasets.tedlium2.lexicon as tdl_lexicon
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs
from i6_experiments.common.setups.rasr import util as rasr_util
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.recipe import lexicon
from i6_experiments.users.berger.corpus.general.helpers import filter_unk_in_corpus_object
from .lm_data import get_lm


def get_data_inputs(
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    lm_names: Optional[List[str]] = None,
    add_unknown_phoneme_and_mapping: bool = False,
    ctc_lexicon: bool = False,
    filter_unk_from_corpus: bool = False,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    if cv_keys is None:
        cv_keys = ["dev"]
    if dev_keys is None:
        dev_keys = ["dev"]
    if test_keys is None:
        test_keys = ["test"]
    if lm_names is None:
        lm_names = ["4gram"]

    data_inputs = get_corpus_data_inputs(add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping)

    assert data_inputs["dev"]["dev"].lm

    lms = {key: get_lm(key) for key in lm_names}

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

    train_data_inputs = {}
    cv_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_corpus_object = data_inputs[train_key][train_key].corpus_object
    if filter_unk_from_corpus:
        train_corpus_object = copy.deepcopy(train_corpus_object)
        filter_unk_in_corpus_object(train_corpus_object, bliss_lexicon)

    train_data_inputs[train_key] = helpers.RasrDataInput(
        corpus_object=helpers.convert_legacy_corpus_object_to_scorable(train_corpus_object),
        concurrent=data_inputs[train_key][train_key].concurrent,
        lexicon=lexicon_config,
    )

    for cv_key in cv_keys:
        cv_corpus_object = data_inputs[cv_key][cv_key].corpus_object
        if filter_unk_from_corpus:
            cv_corpus_object = copy.deepcopy(cv_corpus_object)
            filter_unk_in_corpus_object(cv_corpus_object, bliss_lexicon)
        cv_data_inputs[cv_key] = helpers.RasrDataInput(
            corpus_object=helpers.convert_legacy_corpus_object_to_scorable(cv_corpus_object),
            concurrent=data_inputs[cv_key][cv_key].concurrent,
            lexicon=lexicon_config,
        )

    for dev_key in dev_keys:
        for lm_name, lm in lms.items():
            dev_data_inputs[f"{dev_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=helpers.convert_legacy_corpus_object_to_scorable(
                    data_inputs[dev_key][dev_key].corpus_object
                ),
                concurrent=data_inputs[dev_key][dev_key].concurrent,
                lexicon=lexicon_config,
                lm=lm,
            )

    for test_key in test_keys:
        for lm_name, lm in lms.items():
            dev_data_inputs[f"{test_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=helpers.convert_legacy_corpus_object_to_scorable(
                    data_inputs[test_key][test_key].corpus_object
                ),
                concurrent=data_inputs[test_key][test_key].concurrent,
                lexicon=lexicon_config,
                lm=lm,
            )

    return train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs


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
