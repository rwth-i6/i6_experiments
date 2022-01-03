"""
This is the unfolded GMM Monophone baseline, which is not intended as ASR System but
for LibriSpeech-100h TTS preprocessing
"""
import copy
import time

from sisyphus import gs

from i6_core.meta.system import select_element

from i6_experiments.common.setups.rasr.util import RasrDataInput
from i6_experiments.common.setups.rasr import gmm_system

from .baseline_args import get_init_args, get_monophone_args
from .data import get_corpus_data_inputs


def run_baseline_training():

    train, dev, test = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_gmm/unfolded_monophone_baseline'
    system = gmm_system.GmmSystem()
    start = time.time()
    system.init_system(hybrid_init_args=get_init_args(),
                       monophone_args=get_monophone_args(),
                       triphone_args=None,
                       vtln_args=None,
                       sat_args=None,
                       vtln_sat_args=None,
                       train_data=train,
                       dev_data=dev,
                       test_data=test)
    print("init_system took: %.1f" % (time.time()-start))
    start = time.time()
    system.run(["extract", "mono"])
    print("run took: %.1f" % (time.time()-start))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ''

    return system

monophone_aligner_system = None

def get_monophone_aligner_system():
    """
    :return: default monophone GMM baseline system for aligning librispeech
    :rtype: gmm_system.GmmSystem
    """
    global monophone_aligner_system
    if monophone_aligner_system is None:
        monophone_aligner_system = run_baseline_training()
    return monophone_aligner_system


def get_monophone_ls100_training_alignment_and_allophones():
    aligner_system = get_monophone_aligner_system()
    return aligner_system.alignments["train-clean-100"]["train_mono"][0].alternatives["bundle"], aligner_system.allophone_files["base"]


def align_any_data(corpus_object, name, concurrent, lexicon=None, uncached=True):
    """

    :param CorpusObject corpus_object
    :param str name:
    :param int concurrent:
    :return:
    """
    system = get_monophone_aligner_system()
    train_corpus_name = system.train_corpora[0]

    if name not in system.crp.keys():
        if lexicon is None:
            lexicon = system.crp[train_corpus_name].lexicon_config.file

        rasr_input = RasrDataInput(corpus_object=corpus_object, concurrent=concurrent, lexicon={'filename': lexicon, 'normalize_pronunciation': False})
        system.add_corpus(name, rasr_input, add_lm=False)

        system.crp[name].lexicon_config = system.crp[train_corpus_name].lexicon_config
        system.crp[name].lexicon_config.file = lexicon

        feature_extraction_args = copy.deepcopy(system.hybrid_init_args.feature_extraction_args)
        feature_extraction_args['mfcc']['mfcc_options']['samples_options']["audio_format"] = corpus_object.audio_format
        if corpus_object.audio_format == "ogg":
            feature_extraction_args['mfcc']['mfcc_options']['samples_options']["scale_input"] = 32768
        elif corpus_object.audio_format == "wav":
            pass
        else:
            assert False, "Currently unsupported audio format for aligning %s" % corpus_object.audio_format
        system.extract_features_for_corpus(name, feature_extraction_args)
        system.feature_scorers[name] = system.feature_scorers[train_corpus_name]
    else:
        print("Corpus with name %s already added" % name)
    prefix = "uncached_" if uncached and name not in system.train_corpora else ""
    system.align(name + "_align", name, prefix + system.monophone_args.training_args['feature_flow'], "train_mono")
    alignment = select_element(system.alignments, name, name + '_align').alternatives['bundle']
    return alignment

