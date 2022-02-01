import time

from sisyphus import gs

from i6_experiments.common.setups.rasr import gmm_system

from .baseline_args import get_init_args, get_monophone_args, get_cart_args, get_triphone_args, get_vtln_args, get_sat_args, get_vtln_sat_args
from .data import get_corpus_data_inputs


def run_nodc_baseline():

    train, dev, test = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_gmm/full_gmm_nodc_baseline'
    system = gmm_system.GmmSystem()
    start = time.time()
    system.init_system(hybrid_init_args=get_init_args(dc_detection=False),
                       monophone_args=get_monophone_args(),
                       cart_args=get_cart_args(folded=False),
                       triphone_args=get_triphone_args(),
                       vtln_args=get_vtln_args(),
                       sat_args=get_sat_args(),
                       vtln_sat_args=get_vtln_sat_args(),
                       train_data=train,
                       dev_data=dev,
                       test_data=test)
    print("init_system took: %.1f" % (time.time()-start))
    start = time.time()
    system.run(["extract", "mono", "cart", "tri", "vtln", "sat", "vtln+sat"])
    print("run took: %.1f" % (time.time()-start))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ''

    return system