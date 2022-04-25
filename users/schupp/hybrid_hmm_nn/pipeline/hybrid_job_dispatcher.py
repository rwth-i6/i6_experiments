# An addition to the hybrid baseline, only real purpose it to register and handle job outputs
# Note, there is some duplicate logic with librispeech_hybrid_baseline, this should prob be merged

from this import d
from i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob
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
    network=None,
    config_base_args=None,
    post_config_args=None
):

    # We want all functions from ../helpers/specaugment_new.py
    from ..helpers import specaugment_new

    # Net trick to filter all functions that are not build ins
    functions = [ f for f in dir(specaugment_new) if not f[:2] == "__"]
    code = "\n".join([ inspect.getsource(getattr(specaugment_new, f)) for f in functions ])

    print(code)

    returnn_train_config = ReturnnConfig(
        config={
            "network" : network,
            **config_base_args
        },
        python_prolog=code,
        post_config=post_config_args
    )
    return returnn_train_config



def test_net_contruction(
    rt_config : ReturnnConfig
):

    from ..helpers.returnn_test_helper import make_scope, make_feed_dict
    from returnn.config import Config
    from returnn.tf.util.data import Dim, SpatialDim, FeatureDim, BatchInfo
    from returnn.util.basic import hms, NumbersDict, BackendEngine, BehaviorVersion
    from returnn.tf.network import TFNetwork


    from ..helpers import specaugment_new

    # Net trick to filter all functions that are not build ins
    functions = [ f for f in dir(specaugment_new) if not f[:2] == "__"]
    funcs = {key:value for key in functions for value in [getattr(specaugment_new, k) for k in functions]}

    #from recipe.returnn_common.tests.returnn_helpers import config_net_dict_via_serialized
    config = Config({
        **rt_config.config,
        **funcs
    })

    BehaviorVersion.set(config.int("behavior_version", 12))

    with make_scope() as session:
        net = TFNetwork(config=config,  train_flag=True) #extern_data=extern_data,
        net.construct_from_dict(rt_config.config["network"])
        out = net.get_default_output_layer().output
        net.initialize_params(session)
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))

def make_and_register_returnn_rasr_train(
    #system,
    returnn_train_config,
    returnn_rasr_config_args,
    output_path,

):
    returnn_rasr_train = ReturnnRasrTrainingJob(
        returnn_config=returnn_train_config, 
        log_verbosity=5, # So we get all error outputs and co
        keep_epochs=None, # We use cleanup old models instead
        **returnn_rasr_config_args
    )

    #system.jobs[train_corpus_key]['train_nn_%s' % name] = j
    #system.nn_models[train_corpus_key][name] = j.out_models
    #system.nn_configs[train_corpus_key][name] = j.out_returnn_config_file
    returnn_rasr_train.add_alias(f"{output_path}/train.job")

    tk.register_output(f"{output_path}/returnn.config", returnn_rasr_train.out_returnn_config_file)
    tk.register_output(f"{output_path}/score_and_error.png", returnn_rasr_train.out_plot_se)
    tk.register_output(f"{output_path}/learning_rate.png", returnn_rasr_train.out_plot_lr)
    return returnn_rasr_train

import copy
def make_and_register_returnn_rasr_search(
    system = None,
    returnn_train_config = None,
    train_job = None ,
    recog_corpus_key = None,
    feature_name = None,
    limit_eps=None,
    exp_name = None
):
    # train_job.out_models
    for id in train_job.out_models:
        if id not in limit_eps:
            # I mean I think we can just leave this here 
            #and the searches on epochs that are not stored will never be executed?
            # Nope actually we need to limit this for now
            continue
        model = train_job.out_models[id]
        print(model)
        returnn_search_config = copy.deepcopy(returnn_train_config)

        # change config for recog
        # 1 - remove num epochs
        returnn_search_config.post_config.pop("num_epochs", None)
        # 2 - change output layer
        if returnn_search_config.config['network']['output'].get('class', None) == 'softmax':
            # set output to log-softmax
            returnn_search_config.config['network']['output']['class'] = 'linear'
            returnn_search_config.config['network']['output']['activation'] = 'log_softmax'
            returnn_search_config.config['network']['output'].pop('target', None)

        tf_feature_flow_args = copy.deepcopy(system.tf_feature_flow_args)

        tf_graph_feature_scorer_args = copy.deepcopy(system.tf_graph_feature_scorer_args)

        nn_recog_args = copy.deepcopy(system.nn_recog_args)

        nn_recog_args['flow'] = system.get_full_tf_feature_flow(
          base_flow=system.feature_flows[recog_corpus_key][feature_name],
          crnn_config=returnn_search_config,
          nn_model=model,#self.nn_models[train_corpus_key][train_job_name][epoch],
          **tf_feature_flow_args
        )
        nn_recog_args['feature_scorer'] = system.get_precomputed_hybrid_feature_scorer(
          '', recog_corpus_key, **tf_graph_feature_scorer_args)

        nn_recog_args['corpus'] = recog_corpus_key
        nn_recog_args['name'] = f"{exp_name}/{id:03}"

        setattr(system.crp[recog_corpus_key], 'flf_tool_exe', system.RASR_FLF_TOOL) # Only way to set this...

        system.recog(**nn_recog_args)
        # Aaaand we want to also optimize lm scale per default !TODO!

