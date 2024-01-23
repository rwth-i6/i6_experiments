import copy
from typing import Dict, List, Tuple
from i6_core.meta.system import CorpusObject
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.common.setups.rasr import util as rasr_util
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
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    test_keys: List[str] = ["test-clean", "test-other"],
    lm_name: str = "4gram",
    use_stress: bool = False,
    add_unknown_phoneme_and_mapping: bool = True,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    use_wei_lexicon: bool = False,
    add_all_allophones: bool = False,
    audio_format: str = "wav",
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    corpus_object_dict = copy.deepcopy(
        lbs_dataset.get_corpus_object_dict(
            audio_format=audio_format,
            output_prefix="corpora",
        )
    )
    corpus_object_dict.update(extra_corpus_object_dict)

    lm = get_lm(lm_name)

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
    dev_data_inputs = {}
    test_data_inputs = {}

    concurrent = {
        "train-clean-100": 40,
        "train-clean-360": 100,
        "train-clean-460": 150,
        "train-other-500": 150,
        "train-other-960": 300,
    }[train_key]

    train_data_inputs[train_key] = helpers.RasrDataInput(
        corpus_object=corpus_object_dict[train_key],
        concurrent=concurrent,
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
