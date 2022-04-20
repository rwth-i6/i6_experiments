# An addition to the hybrid baseline, only real purpose it to register and handle job outputs
# Note, there is some duplicate logic with librispeech_hybrid_baseline, this should prob be merged

from this import d
from recipe.i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob
import inspect
import hashlib

from sisyphus import tk

SALT = "42"

# Steps: 
# 0: Make returnn config
#
# 1: Training -> make_and_register_returnn_rasr_train
#
# 2: Recognition ->
#
#
#

def make_and_hash_returnn_rtc_config(
    network_func,    # A reference to a function that can create the network ( the returnn_common way )
    config_base_args
):


    rtc_network_and_config_code = network_func()
    network_code = inspect.getsource(network_func) # TODO: we might wanna hash a more complte version

    print("TBS: hashing this net code:")
    print(network_code)

    returnn_train_config = ReturnnConfig(
        config_base_args,
        python_epilog=[
        rtc_network_and_config_code,
"""
import resource
import sys
try:
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
sys.setrecursionlimit(10 ** 6)
"""
        ],
        post_config=dict(cleanup_old_models=True),
        python_epilog_hash=hashlib.sha256(SALT.encode() + network_code.encode()).hexdigest(), # TODO: let user specify salt if he want's to 'rerun' experiment
        sort_config=False,
    )

    return returnn_train_config


def make_and_register_returnn_rasr_train(
    returnn_train_config,
    returnn_rasr_config_args,
    output_path,
):
    returnn_rasr_train = ReturnnRasrTrainingJob(returnn_config=returnn_train_config, **returnn_rasr_config_args)

    tk.register_output(f"{output_path}/returnn.config", returnn_rasr_train.out_returnn_config_file)
    tk.register_output(f"{output_path}/score_and_error.png", returnn_rasr_train.out_plot_se)
    tk.register_output(f"{output_path}/learning_rate.png", returnn_rasr_train.out_plot_lr)
    return returnn_rasr_train

def make_and_register_returnn_rasr_search(
  recog_for_epochs,
  recog_corpus,
  train_job,
  output_path,
):
    pass