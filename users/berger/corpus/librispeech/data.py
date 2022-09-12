import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
)


def get_data_inputs(
    train_key="train-other-960",
    add_unknown_phoneme_and_mapping=True,
    use_eval_data_subset: bool = False,
    delete_empty_orth: bool = False,
    use_augmented_lexicon: bool = True,
):
    corpus_object_dict = lbs_dataset.get_corpus_object_dict(
        audio_format="wav",
        output_prefix="corpora",
    )

    lm = {
        "filename": lbs_dataset.get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    use_stress_marker = False

    original_bliss_lexicon = lbs_dataset.get_bliss_lexicon(
        use_stress_marker=use_stress_marker,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
    )

    augmented_bliss_lexicon = {
        "filename": lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=use_stress_marker,
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        )[train_key],
        "normalize_pronunciation": False,
    }

    bliss_lexicon = (
        augmented_bliss_lexicon if use_augmented_lexicon else original_bliss_lexicon
    )
    bliss_lexicon["filename"] = EnsureSilenceFirstJob(
        bliss_lexicon["filename"]
    ).out_lexicon

    if delete_empty_orth:
        bliss_lexicon["filename"] = DeleteEmptyOrthJob(
            bliss_lexicon["filename"]
        ).out_lexicon

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
        lexicon=bliss_lexicon,
    )

    dev_corpus_keys = (
        ["dev-other"] if use_eval_data_subset else ["dev-clean", "dev-other"]
    )
    test_corpus_keys = [] if use_eval_data_subset else ["test-clean", "test-other"]

    for dev_key in dev_corpus_keys:
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=20,
            lexicon=bliss_lexicon,
            lm=lm,
        )

    for tst_key in test_corpus_keys:
        test_data_inputs[tst_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[tst_key],
            concurrent=20,
            lexicon=bliss_lexicon,
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
