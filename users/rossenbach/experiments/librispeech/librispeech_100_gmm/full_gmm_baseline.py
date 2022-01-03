import time

from sisyphus import gs

from i6_experiments.common.setups.rasr import gmm_system

from .baseline_args import get_init_args, get_monophone_args
from .data import get_corpus_data_inputs


def run_baseline_training():

    train, dev, test = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_gmm/full_gmm_baseline'
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