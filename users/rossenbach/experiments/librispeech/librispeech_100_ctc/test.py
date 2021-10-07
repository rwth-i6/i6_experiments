from sisyphus import gs, tk

from i6_experiments.common.setups.hybrid.util import RasrDataInput, RasrInitArgs

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict,\
    get_corpus_object_dict, get_arpa_lm_dict
from i6_experiments.users.rossenbach.lexicon.modification import AddBoundaryMarkerToLexiconJob

from .ctc_system import CtcSystem

rasr_args = RasrInitArgs(
    costa_args={
        'eval_recordings': True,
        'eval_lm': False
    },
    am_args={
        'states_per_phone'   : 1, # single state
        'state_tying'        : 'monophone-eow', # different class for final phoneme of a word
        'tdp_transition'     : (0, 0, 0, 0), # no tdps
        'tdp_silence'        : (0, 0, 0, 0),
    },
    feature_extraction_args={
        'gt': {
            'gt_options': {
                'minfreq'   : 100,
                'maxfreq'   : 7500,
                'channels'  : 50,
                'do_specint': False
            }
        }
    }
)


def create_eow_lexicon():
    ls100_bliss_lexicon = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=True)['train-clean-100']
    add_boundary_marker_job = AddBoundaryMarkerToLexiconJob(
        bliss_lexicon=ls100_bliss_lexicon,
        add_eow=True,
        add_sow=False
    )
    ls100_eow_bliss_lexicon = add_boundary_marker_job.out_lexicon

    return ls100_eow_bliss_lexicon


def get_corpus_data_inputs():
    """

    :return:
    """

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    lm = {
        "filename": get_arpa_lm_dict()['4gram'],
        'type': "ARPA",
        'scale': 10,
    }
    lexicon = {
        'filename': create_eow_lexicon(),
        'normalize_pronunciation': False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs['train-clean-100'] = RasrDataInput(
        corpus_object=corpus_object_dict['train-clean-100'],
        concurrent=10,
        lexicon=lexicon,
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


def ctc_test():
    eow_lexicon = create_eow_lexicon()
    tk.register_output("experiments/librispeech_100_ctc/eow_lexicon.xml", eow_lexicon)

    system = CtcSystem()
    train_data, dev_data, test_data = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", ))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""

