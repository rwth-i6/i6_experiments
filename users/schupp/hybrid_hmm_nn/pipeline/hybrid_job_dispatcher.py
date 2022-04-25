# An addition to the hybrid baseline, only real purpose it to register and handle job outputs
# Note, there is some duplicate logic with librispeech_hybrid_baseline, this should prob be merged

from this import d
from recipe.i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob
import inspect
import hashlib
import returnn.tf.engine

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

def test_network_contruction(
    network_func,    # A reference to a function that can create the network ( the returnn_common way )
    config_base_args
):
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset
    from recipe.returnn_common.tests.returnn_helpers import config_net_dict_via_serialized
    from returnn.config import Config

    rtc_network_and_config_code = network_func()
    print(rtc_network_and_config_code)

    config, net_dict = config_net_dict_via_serialized(rtc_network_and_config_code)

    extern_data_opts = config["extern_data"]
    n_data_dim = extern_data_opts["data"]["dim_tags"][-1].dimension
    n_classes_dim = extern_data_opts["classes"]["sparse_dim"].dimension if "classes" in extern_data_opts else 7

    config = Config({
    "train": {
      "class": "DummyDataset", "input_dim": n_data_dim, "output_dim": n_classes_dim,
      "num_seqs": 2, "seq_len": 5},
        **config
    })

    dataset = init_dataset(config.typed_value("train"))

    engine = Engine(config=config)
    engine.init_train_from_config(train_data=dataset)


    print(net_dict)

def make_and_hash_returnn_rtc_config(
    network_func,    # A reference to a function that can create the network ( the returnn_common way )
    config_base_args
):


    rtc_network_and_config_code = network_func()
    network_code = inspect.getsource(network_func) # TODO: we might wanna hash a more complte version

    print("TBS: returnn code")
    print(rtc_network_and_config_code)

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

# Takes oldstyle network generating code
def make_returnn_train_config_old(
    network_func,
    config_base_args
):
    return None


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