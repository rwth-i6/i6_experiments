import copy
from typing import Dict, List, Tuple, Optional
from i6_core.meta.system import CorpusObject
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.common.setups.rasr import util as rasr_util
from i6_experiments.users.berger.corpus.general.helpers import filter_unk_in_corpus_object
from .lm_data import get_lm
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
    MakeBlankLexiconJob,
)
from sisyphus import tk

sep_libric_css_corpus_dev = CorpusObject()
sep_libric_css_corpus_dev.corpus_file = tk.Path(
    "/work/asr4/berger/dependencies/librispeech/corpus/libriCSS_singlechannel_77_62000_124.dev.xml.gz"
)
sep_libric_css_corpus_dev.audio_format = "wav"
sep_libric_css_corpus_dev.duration = 1.0
sep_libric_css_corpus_eval = CorpusObject()
sep_libric_css_corpus_eval.corpus_file = tk.Path(
    "/work/asr4/berger/dependencies/librispeech/corpus/libriCSS_singlechannel_77_62000_124.eval.xml.gz"
)
sep_libric_css_corpus_eval.audio_format = "wav"
sep_libric_css_corpus_eval.duration = 1.0

extra_corpus_object_dict = {
    "sep-libri-css-dev": sep_libric_css_corpus_dev,
    "sep-libri-css-eval": sep_libric_css_corpus_eval,
}


def get_data_inputs(
    train_key: str = "train-other-960",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    lm_names: Optional[List[str]] = None,
    use_stress: bool = False,
    add_unknown_phoneme_and_mapping: bool = True,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    use_wei_lexicon: bool = False,
    filter_unk_from_corpus: bool = True,
    add_all_allophones: bool = False,
    audio_format: str = "wav",
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    if cv_keys is None:
        cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]
    if lm_names is None:
        lm_names = ["4gram"]

    corpus_object_dict = copy.deepcopy(
        lbs_dataset.get_corpus_object_dict(
            audio_format=audio_format,
            output_prefix="corpora",
        )
    )
    corpus_object_dict.update(extra_corpus_object_dict)

    lms = {lm_name: get_lm(lm_name) for lm_name in lm_names}

    if use_wei_lexicon:
        bliss_lexicon = tk.Path(
            "/work/asr4/berger/dependencies/librispeech/lexicon/train-dev.lexicon.wei.xml",
            hash_overwrite="LS_train-lex_wei",
        )
    else:
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

        bliss_lexicon = EnsureSilenceFirstJob(bliss_lexicon).out_lexicon

        if ctc_lexicon:
            bliss_lexicon = DeleteEmptyOrthJob(bliss_lexicon).out_lexicon
            bliss_lexicon = MakeBlankLexiconJob(bliss_lexicon).out_lexicon

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

    concurrent = {
        "train-clean-100": 40,
        "train-clean-360": 100,
        "train-clean-460": 150,
        "train-other-500": 150,
        "train-other-960": 300,
    }[train_key]

    train_corpus_object = corpus_object_dict[train_key]
    if filter_unk_from_corpus:
        train_corpus_object = copy.deepcopy(train_corpus_object)
        filter_unk_in_corpus_object(train_corpus_object, bliss_lexicon)
    train_data_inputs[train_key] = helpers.RasrDataInput(
        corpus_object=train_corpus_object,
        concurrent=concurrent,
        lexicon=lexicon_config,
    )

    for cv_key in cv_keys:
        cv_corpus_object = corpus_object_dict[cv_key]
        if filter_unk_from_corpus:
            cv_corpus_object = copy.deepcopy(cv_corpus_object)
            filter_unk_in_corpus_object(cv_corpus_object, bliss_lexicon)
        cv_data_inputs[cv_key] = helpers.RasrDataInput(
            corpus_object=cv_corpus_object,
            concurrent=20,
            lexicon=lexicon_config,
        )

    for dev_key in dev_keys:
        for lm_name, lm in lms.items():
            dev_data_inputs[f"{dev_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=corpus_object_dict[dev_key],
                concurrent=20,
                lexicon=lexicon_config,
                lm=lm,
            )

    for test_key in test_keys:
        for lm_name, lm in lms.items():
            test_data_inputs[f"{test_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=corpus_object_dict[test_key],
                concurrent=20,
                lexicon=lexicon_config,
                lm=lm,
            )

    return train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs


def get_final_gmm_output():
    output_args = rasr_util.OutputArgs("final")

    for tc in (
        "train-clean-100",
        "train-clean-360",
        "train-clean-460",
        "train-other-500",
        "train-other-960",
    ):
        output_args.define_corpus_type(tc, "train")
    for dc in ("dev-clean", "dev-other"):
        output_args.define_corpus_type(dc, "dev")
    for tc in ("test-clean", "test-other"):
        output_args.define_corpus_type(tc, "test")

    output_args.add_feature_to_extract("gt")

    return output_args
