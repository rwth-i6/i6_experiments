from typing import Dict, List, Tuple
import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
    MakeBlankLexiconJob,
)


def get_data_inputs(
    train_key: str = "train-other-960",
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    test_keys: List[str] = ["test-clean", "test-other"],
    use_stress: bool = False,
    add_unknown_phoneme_and_mapping: bool = True,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = True,
    audio_format: str = "wav",
) -> Tuple[Dict[str, rasr_util.RasrDataInput], ...]:
    corpus_object_dict = lbs_dataset.get_corpus_object_dict(
        audio_format=audio_format,
        output_prefix="corpora",
    )

    lm = {
        "filename": lbs_dataset.get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

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

    lexicon_config = {
        "filename": bliss_lexicon,
        "normalize_pronunciation": False,
        "add_all": add_all_allophones,
        "add_from_lexicon": not add_all_allophones,
    }

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

    train_data_inputs[train_key] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[train_key],
        concurrent=concurrent,
        lexicon=lexicon_config,
    )

    for dev_key in dev_keys:
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=20,
            lexicon=lexicon_config,
            lm=lm,
        )

    for test_key in test_keys:
        test_data_inputs[test_key] = rasr_util.RasrDataInput(
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
