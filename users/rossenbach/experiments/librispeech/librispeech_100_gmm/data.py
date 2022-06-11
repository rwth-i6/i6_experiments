from i6_experiments.common.datasets.librispeech import get_corpus_object_dict, get_arpa_lm_dict, get_bliss_lexicon, get_g2p_augmented_bliss_lexicon_dict
from i6_experiments.common.setups.rasr import RasrDataInput


def get_corpus_data_inputs(use_g2p_training=False, use_stress_marker=True, unknown_phoneme=True):
    """

    :param bool use_g2p_training:
    :param bool use_stress_marker:
    :return: (train_data, dev_data, test_data)
    :rtype: tuple(RasrDataInput, RasrDataInput, RasrDataInput)
    """

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    lm = {
        "filename": get_arpa_lm_dict()['4gram'],
        'type': "ARPA",
        'scale': 10,
    }
    lexicon = {
        'filename': get_bliss_lexicon(use_stress_marker=use_stress_marker, add_unknown_phoneme_and_mapping=unknown_phoneme),
        'normalize_pronunciation': False,
    }

    if use_g2p_training:
        train_lexicon = {
            "filename": get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=use_stress_marker, add_unknown_phoneme_and_mapping=unknown_phoneme)["train-clean-100"],
            "normalize_pronunciation": False,
        }
    else:
        train_lexicon = lexicon

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs['train-clean-100'] = RasrDataInput(
        corpus_object=corpus_object_dict['train-clean-100'],
        concurrent=10,
        lexicon=train_lexicon,
        lm=lm,
    )

    for dev_key in ['dev-clean', 'dev-other']:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=10,
            lexicon=lexicon,
            lm=lm
        )

    test_data_inputs['test-clean'] = RasrDataInput(
        corpus_object=corpus_object_dict['test-clean'],
        concurrent=10,
        lexicon=lexicon,
        lm=lm,
    )

    return train_data_inputs, dev_data_inputs, test_data_inputs
