# An addition to the hybrid baseline, only real purpose it to register and handle job outputs
# Note, there is some duplicate logic with librispeech_hybrid_baseline, this should prob be merged

from torch import exp_
from i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob
import inspect
import hashlib
import returnn.tf.engine
import os
from typing import OrderedDict

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
    post_config_args=None,
    recoursion_depth = None,
    extra_code_string = None # TODO this is new
):

    # We want all functions from ../helpers/specaugment_new.py
    from ..helpers import specaugment_new

    # Net trick to filter all functions that are not build ins
    functions = [ f for f in dir(specaugment_new) if not f[:2] == "__"]
    code = "\n".join([ inspect.getsource(getattr(specaugment_new, f)) for f in functions ])

    if extra_code_string:
        code += extra_code_string

    if not recoursion_depth is None:
        code += f"import sys\nsys.setrecursionlimit({recoursion_depth})\n"

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
    print("TBS: test config args:")
    print(rt_config.config)

    BehaviorVersion.set(config.int("behavior_version", 12))

    with make_scope() as session:
        net = TFNetwork(config=config,  train_flag=True) #extern_data=extern_data,
        net.construct_from_dict(rt_config.config["network"])
        out = net.get_default_output_layer().output
        net.initialize_params(session)
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))

        net.print_network_info()

def test_net_construction_advanced(
    rt_config : ReturnnConfig

):
    from returnn.tf.engine import Engine
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
    print("TBS: test config args:")
    print(rt_config.config)

    BehaviorVersion.set(config.int("behavior_version", 12))

    import tensorflow as tf

    with tf.Graph().as_default() as graph:
        assert isinstance(graph, tf.Graph)
        Engine.create_network(
            config=config, rnd_seed=1,
            train_flag=False, eval_flag=False, search_flag=False,
            net_dict=rt_config.config["network"])

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

from ..helpers.returnn_helpers import ReturnnRasrTrainingJobDevtrain

def make_and_register_returnn_rasr_train_02_devtrain(
    #system,
    returnn_train_config,
    returnn_rasr_config_args,
    output_path,
    set_rqmt = False # This was falsely not set on other jobs, TODO add as default in next itteration

):
    rqmt = {}
    if set_rqmt:
        rqmt = {
            'device' : "gpu",
            'time_rqmt': 168,
            'mem_rqmt': 12,
            'cpu_rqmt': 3,
        }

    returnn_rasr_config_args.update(rqmt)

    returnn_rasr_train = ReturnnRasrTrainingJobDevtrain(
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

# TODO use this:
MIN_LM_OPTIMIZE_EP = 120 # Min epoch from where to maybe optimize lm scale

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
            continue # TODO: we need also a way to just check which epochs are there and run only those
        model = train_job.out_models[id]
        #print(model)
        system_search_for_model(
            model = model,
            system = system,
            returnn_train_config=returnn_train_config,
            recog_corpus_key=recog_corpus_key,
            feature_name=feature_name,
            recog_name = f"{exp_name}/{id:03}"
        )


def make_and_register_returnn_rasr_search_02(
    system = None,
    returnn_train_config = None,
    train_job = None ,
    recog_corpus_key = None,
    feature_name = None,
    limit_eps=None,
    exp_name = None,
    amount_paralel_searches=15, # This was 10 per default earlier
    use_gpu_and_extra_mem=False
):
    # train_job.out_models
    for id in train_job.out_models:
        if id not in limit_eps:
            # I mean I think we can just leave this here 
            #and the searches on epochs that are not stored will never be executed?
            # Nope actually we need to limit this for now
            continue # TODO: we need also a way to just check which epochs are there and run only those
        model = train_job.out_models[id]
        #print(model)
        system_search_for_model(
            model = model,
            system = system,
            returnn_train_config=returnn_train_config,
            recog_corpus_key=recog_corpus_key,
            feature_name=feature_name,
            recog_name = f"{exp_name}/{id:03}",
            amount_paralel_searches=amount_paralel_searches,
            use_gpu_and_extra_mem=use_gpu_and_extra_mem
        )

def system_search_for_model(
    model = None,
    system = None,
    returnn_train_config = None,
    recog_corpus_key = None,
    feature_name = None,
    recog_name = None,
    amount_paralel_searches=None,
    use_gpu_and_extra_mem=False
):
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
    nn_recog_args['name'] = recog_name

    setattr(system.crp[recog_corpus_key], 'flf_tool_exe', system.RASR_FLF_TOOL) # Only way to set this...

    if amount_paralel_searches:
        nn_recog_args["rtf"] = amount_paralel_searches

    if use_gpu_and_extra_mem:
        nn_recog_args.update(OrderedDict(
                use_gpu=True,
                rtf=10,
                mem=32,
                lmgc_mem=32
            ))

    system.recog(**nn_recog_args)

    system.optimize_am_lm(
        f"recog_{nn_recog_args['name']}", 
        recog_corpus_key, nn_recog_args['pronunciation_scale'], 
        nn_recog_args['lm_scale'], '', 
        opt_only_lm_scale=False)

    # When this is done we can dispatch a score summary job or similar
    # The wer is acessibe under: self.jobs[corpus]["scorer_%s" % name].out_wer

    # Now do the same for all the *best* checkpoints bug register them regardless of limit_eps:
    # see i6_core GetBestCheckpointJob() 
    # # I'm not sure what heppens when using this on an unfinished training thoug

import copy

from sisyphus import Job, Task, gs, tk

from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor as sys

# Not suere if this is needed but for now was only way I found to make this sequential
# Sisyphus does some weird things when we pass it the 'system' but by having it in this function contex we don't need to do that
# TODO: this is not very efficient, wrap the system call in a lambda or something!
# But also every experyment hast their own 'system' context so it would prob be fine just passing a pointer somehow
class DispatchFinalSearch(Job): 
    def __init__(self, 
        checkpoint, 
        ep,

        train_job_models = None,
        returnn_train_config = None,
        feature_name = None,
        exp_name = None
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.ep = ep
        self.out_final_wer = self.output_path("final.wer")

        self.train_job_models = train_job_models
        self.returnn_train_config = returnn_train_config
        self.exp_name = exp_name
        self.feature_name = feature_name

    def tasks(self):
        yield Task("run", mini_task=True)

    # TODO: add some intelligent hashy thingi here :D

    def run(self): # Can a job submit additonal jobs? lets try

        print("TBS ") 
        print(self.checkpoint.ckpt_path)
        with open(self.out_final_wer.get_path() , "wt", encoding="utf-8") as f:
            f.write(f"Foool this is just a test, but best ep was: {self.ep}")

        data = "dev-clean"

        system = sys.LibrispeechHybridSystemTim() # We just use another system here ( everything else is wonky with sisyphus )

        # Make a returnn config
        train_corpus_key = 'train-other-960'

        system.create_rasr_am_config(train_corpus_key=train_corpus_key)

        system.init_rasr_am_lm_config_recog( # Can this cause concurrency issues?
            recog_corpus_key=data
        )

        epoch = int(str(self.ep))

        model = self.train_job_models[epoch] # This will always be availabol at this point
        #print(model)
        system_search_for_model(
            model = model,
            system = system,
            returnn_train_config= self.returnn_train_config,
            recog_corpus_key=data,
            feature_name=self.feature_name,
            recog_name = f"{self.exp_name}/{epoch:03}" # This might have run already but that fine
        )

#class FinalBestScoreSummaryJob(Job): 


# Essentialy does the same as make_and_register_returnn_rasr_search,
# but does it only for the best X epochs, this will also make recog on *all* datasets, not just dev-other
def make_and_register_final_rasr_search( 
    train_job = None,
    output_path = None,

    system = None,
    returnn_train_config = None,
    feature_name = None,
    exp_name = None,

    for_best_n = 1, # Only for the best, otherwise for the 'n' best

):
    # This uses 'GetBestCheckpointJob' which I stole from users/zeineldeen
    from ..helpers.returnn_helpers import GetBestCheckpointJob, GetBestEpochJob

    # TBS: final serach registered

    mesasures = ["dev_score_output", "dev_error_output"] # TODO: maybe add more
    best_score_getters = {}
    for m in mesasures:
        best_score_getters[m] = GetBestEpochJob(
            model_dir = train_job.out_model_dir,
            learning_rates = train_job.out_learning_rates, # As far as I unserstand this is only availabol for finished trains
            key=m, # Think this should work as a key
            #index=-1 if "score" in m else 0 TODO: check this is 0 ok for 'score' ?
        )

        best_score_getters[m].add_alias(os.path.join(output_path, f"best_ep_{m}"))

        tk.register_output(f"{output_path}/best_epoch_{m}", best_score_getters[m].out_epoch) # We only really need this

    # Ok using the Final Search Dispatch Works but, causes alot of memory over head,
    # Going for the simpler approach: ( requires sis restart once in the end, after 'all outputs finished' )
    # 1) check if the out file of best_epoch exists,
    # 2) If yes runn all them final searches!

    all_exist = all([os.path.exists(best_score_getters[m].out_epoch.get_path()) for m in mesasures])

    if all_exist: # HINT: to use this rerun the sis job *after* all outputs are finished
        print("TBS: yeah final best epoch was found")

        epochs = [int(str(best_score_getters[m].out_epoch)) for m in mesasures]
        print(f"TBS: {epochs}")

        for epoch in epochs:
            for data in ["dev-other", "dev-clean", "test-other", "test-clean"]:
                model = train_job.out_models[epoch] # This will always be availabol at this point
                #print(model)

                system.init_rasr_am_lm_config_recog(
                    recog_corpus_key=data
                )

                rec_name = f"{exp_name}_{data}"
                system_search_for_model(
                    model = model,
                    system = system,
                    returnn_train_config= returnn_train_config,
                    recog_corpus_key=data,
                    feature_name=feature_name,
                    recog_name = f"{rec_name}/{epoch:03}" # This might have run already but that fine
                )

                # TODO: here add FinalBestScoreSummaryJob

                #self.jobs[corpus]["scorer_%s" % name].out_wer

    #dispatch_final_search = DispatchFinalSearch(
    #    checkpoint=best_checkpoint_job.out_checkpoint,
    #    ep=best_checkpoint_job.out_epoch,
    #    train_job_models = train_job.out_models,
    #    returnn_train_config = returnn_train_config,
    #    feature_name = feature_name,
    #    exp_name = exp_name
    #)

    #tk.register_output(f"{output_path}/final_wer", dispatch_final_search.out_final_wer)
    



def make_and_register_final_rasr_search_manual( 
    train_job = None,
    output_path = None,

    system = None,
    returnn_train_config = None,
    feature_name = None,
    exp_name = None,

    for_best_n = 2, # Only for the best, otherwise for the 'n' best
):


    # 1 - check if the train job is done 
    if os.path.exists(train_job.out_learning_rates):
        print(f"TBS: train job {exp_name} finished performing final recog")
        import glob
        # I suppose this means the train is done
        #print(f"TBS: {train_job.out_model_dir}")
        all_models = sorted([int(x.split("/")[-1].split(".")[1]) for x in glob.glob(str(train_job.out_model_dir) + "/epoch.*.index")])
        #print(f"TBS: found final models: {all_models}")
        for x in range(1, for_best_n + 1): # We could invert but also we can just itterate from the back
            epoch = all_models[-x]
            print(f"Searching for sub ep {epoch}", end=",")
            for data in ["dev-other", "dev-clean", "test-other", "test-clean"]:
                model = train_job.out_models[epoch] # This will always be availabol at this point
                #print(model)

                system.init_rasr_am_lm_config_recog(
                    recog_corpus_key=data
                )

                rec_name = f"{exp_name}_{data}"
                if data == "dev-other":
                    rec_name = exp_name # We dont prefix dev other...

                system_search_for_model(
                    model = model,
                    system = system,
                    returnn_train_config= returnn_train_config,
                    recog_corpus_key=data,
                    feature_name=feature_name,
                    recog_name = f"{rec_name}/{epoch:03}" # This might have run already but that fine
                )


def make_and_register_final_rasr_search_manual_devtrain( 
    train_job = None,
    output_path = None,

    system = None,
    returnn_train_config = None,
    feature_name = None,
    exp_name = None,

    for_best_n = 1, # Only for the best, otherwise for the 'n' best

    use_gpu_and_extra_mem=False
):


    # 1 - check if the train job is done 
    if os.path.exists(train_job.out_learning_rates):
        import glob
        # I suppose this means the train is done
        #print(f"TBS: {train_job.out_model_dir}")
        all_models = sorted([int(x.split("/")[-1].split(".")[1]) for x in glob.glob(str(train_job.out_model_dir) + "/epoch.*.index")])
        #print(f"TBS: found final models: {all_models}")
        for x in range(1, for_best_n + 1): # We could invert but also we can just itterate from the back
            epoch = all_models[-x]
            print(f"devother for ep{epoch}", end=",")
            data = "devtrain2000"
            model = train_job.out_models[epoch] # This will always be availabol at this point
            #print(model)

            system.init_rasr_am_lm_config_recog(
                recog_corpus_key=data
            )

            rec_name = f"{exp_name}_{data}"
            if data == "dev-other":
                rec_name = exp_name # We dont prefix dev other...

            system_search_for_model(
                model = model,
                system = system,
                returnn_train_config= returnn_train_config,
                recog_corpus_key=data,
                feature_name=feature_name,
                recog_name = f"{rec_name}/{epoch:03}", # This might have run already but that fine
                use_gpu_and_extra_mem=use_gpu_and_extra_mem
            )




