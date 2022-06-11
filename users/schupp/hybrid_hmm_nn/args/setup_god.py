
# All possible inputs that 'diversify' a config
# - NAME
# - OUTPUT_PATH 
# - config_base_args
# - network_args:
#   - sampling_func_args
#   - ff1_func_args
#   - ff2_func_args
#   - sa_func_args
#   - conv_func_args
#   - conformer_default_args_00
# - train:
#   - returnn_train_post_config_00
#   - returnn_rasr_args_defaults_00
from typing import OrderedDict
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor as sys
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor_02_devtrain as sys_02_devtrain
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor_02_devtrain_rerecog as sys_02_devtrain_rerecog
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import hybrid_job_dispatcher as job_dispatcher
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_rasr_config_maker as rasr_config_args_maker

# Everything, that is needed to define a full setup
# Conformer modifications can be done either by varring the conformer_create_func or it's args
def create_experiment_world_001(
    name=None,
    output_path=None,
    config_base_args=None,
    extra_returnn_net_creation_args = None,

    # Network stuff
    conformer_create_func=None,
    conformer_func_args=None,
    #   - sampling_func_args
    #   - ff1_func_args
    #   - ff2_func_args
    #   - sa_func_args
    #   - conv_func_args
    #   - conformer_default_args_00

    returnn_train_post_config=None,
    returnn_rasr_args_defaults=None,

    final_recog = False, # Should be changed to 'True' per default at some point
    new_final_recog = False,

    test_construction=False,
    print_net = False,
    write_dummpy_config = None, # String path if given, write the returnn config there
):
    
    system = sys.LibrispeechHybridSystemTim()

    # Make a returnn config
    train_corpus_key = 'train-other-960'

    system.create_rasr_am_config(train_corpus_key=train_corpus_key)

    network = conformer_create_func(
        **conformer_func_args,
        print_net = print_net
    )

    make_train_config_extra = {} if extra_returnn_net_creation_args is None else extra_returnn_net_creation_args

    returnn_train_config = job_dispatcher.make_returnn_train_config_old(
        network = network,
        config_base_args=config_base_args,
        post_config_args=returnn_train_post_config,
        **make_train_config_extra
    )

    if not write_dummpy_config is None:
        # Fuck damit, can actually do this because write always calls _serialize, which always calls  config.update(self.post_config)
        # And this causes an inconsitency error because some post config args are added several times
        # But maybe we can deep copy and then write... waste some memory but should work
        import copy
        assert isinstance(write_dummpy_config, str)
        train_config = copy.deepcopy(returnn_train_config)
        train_config.write(write_dummpy_config)

    # We test the construction now to avaoid error when running on cluster
    if test_construction:
        job_dispatcher.test_net_contruction(returnn_train_config)

    returnn_rasr_config_args : dict = rasr_config_args_maker.get_returnn_rasr_args(
    system, 
    train_corpus_key=train_corpus_key,
    **returnn_rasr_args_defaults
    )

    train_job = job_dispatcher.make_and_register_returnn_rasr_train(
        returnn_train_config,
        returnn_rasr_config_args,
        output_path=f"{name}" # No need for 'output_path' here, its already sub aliased
    )

    rec_corpus = "dev-other" # Other are only done for the best epoch in the end
    # Prepare args for rasr recog
    system.init_rasr_am_lm_config_recog(
    recog_corpus_key=rec_corpus
    )

    job_dispatcher.make_and_register_returnn_rasr_search(
        system=system,
        returnn_train_config=returnn_train_config,
        train_job=train_job,
        recog_corpus_key=rec_corpus,
        feature_name="gammatone",
        limit_eps=returnn_train_post_config["cleanup_old_models"]["keep"],
        exp_name=name
    )

    if final_recog:
        job_dispatcher.make_and_register_final_rasr_search(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )

    if new_final_recog:
        job_dispatcher.make_and_register_final_rasr_search_manual(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )

    return OrderedDict(
        network = network
    )


# + devtrain !!!
# + final_recog = True
def create_experiment_world_002( # New system that adds devtrain dataset
    name=None,
    output_path=None,
    config_base_args=None,
    extra_returnn_net_creation_args = None,

    # Network stuff
    conformer_create_func=None,
    conformer_func_args=None,
    #   - sampling_func_args
    #   - ff1_func_args
    #   - ff2_func_args
    #   - sa_func_args
    #   - conv_func_args
    #   - conformer_default_args_00

    returnn_train_post_config=None,
    returnn_rasr_args_defaults=None,

    extra_recog_epochs = None, # This setup always does *all* recogs for all 'keep' epochs, use this to add more

    final_recog = True,

    test_construction=False,
    print_net = False,
    write_dummpy_config = None, # String path if given, write the returnn config there
):
    
    system = sys_02_devtrain.LibrispeechHybridSystemTim()

    # Make a returnn config
    train_corpus_key = 'train-other-960'

    system.create_rasr_am_config(train_corpus_key=train_corpus_key)

    network = conformer_create_func(
        **conformer_func_args,
        print_net = print_net
    )

    make_train_config_extra = {} if extra_returnn_net_creation_args is None else extra_returnn_net_creation_args

    returnn_train_config = job_dispatcher.make_returnn_train_config_old(
        network = network,
        config_base_args=config_base_args,
        post_config_args=returnn_train_post_config,
        **make_train_config_extra
    )

    if not write_dummpy_config is None:
        # Fuck damit, can actually do this because write always calls _serialize, which always calls  config.update(self.post_config)
        # And this causes an inconsitency error because some post config args are added several times
        # But maybe we can deep copy and then write... waste some memory but should work
        import copy
        assert isinstance(write_dummpy_config, str)
        train_config = copy.deepcopy(returnn_train_config)
        train_config.write(write_dummpy_config)

    # We test the construction now to avaoid error when running on cluster
    if test_construction:
        job_dispatcher.test_net_contruction(returnn_train_config)

    returnn_rasr_config_args : dict = rasr_config_args_maker.get_returnn_rasr_args_02_devtrain(
    system, 
    train_corpus_key=train_corpus_key,
    **returnn_rasr_args_defaults
    )

    train_job = job_dispatcher.make_and_register_returnn_rasr_train_02_devtrain(
        returnn_train_config,
        returnn_rasr_config_args,
        output_path=f"{name}" # No need for 'output_path' here, its already sub aliased
    )

    rec_corpus = "dev-other" # Other are only done for the best epoch in the end
    # Prepare args for rasr recog
    system.init_rasr_am_lm_config_recog(
    recog_corpus_key=rec_corpus
    )

    limit_eps = returnn_train_post_config["cleanup_old_models"]["keep"]
    if extra_recog_epochs:
        limit_eps += extra_recog_epochs

    job_dispatcher.make_and_register_returnn_rasr_search_02( # 15 paralel searches instead of just 10
        system=system,
        returnn_train_config=returnn_train_config,
        train_job=train_job,
        recog_corpus_key=rec_corpus,
        feature_name="gammatone",
        limit_eps=limit_eps,
        exp_name=name
    )

    if final_recog:
        job_dispatcher.make_and_register_final_rasr_search(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )

    return OrderedDict(
        network = network
    )


# + sets defualt mem rqmts for train
def create_experiment_world_003( # New system that adds devtrain dataset
    name=None,
    output_path=None,
    config_base_args=None,
    extra_returnn_net_creation_args = None,

    # Network stuff
    conformer_create_func=None,
    conformer_func_args=None,
    #   - sampling_func_args
    #   - ff1_func_args
    #   - ff2_func_args
    #   - sa_func_args
    #   - conv_func_args
    #   - conformer_default_args_00

    returnn_train_post_config=None,
    returnn_rasr_args_defaults=None,

    extra_recog_epochs = None, # This setup always does *all* recogs for all 'keep' epochs, use this to add more

    final_recog = True,

    extra_recog_devtrain = False,

    test_construction=False,
    print_net = False,
    write_dummpy_config = None, # String path if given, write the returnn config there
):
    
    system = sys_02_devtrain.LibrispeechHybridSystemTim()

    # Make a returnn config
    train_corpus_key = 'train-other-960'

    system.create_rasr_am_config(train_corpus_key=train_corpus_key)

    network = conformer_create_func(
        **conformer_func_args,
        print_net = print_net
    )

    make_train_config_extra = {} if extra_returnn_net_creation_args is None else extra_returnn_net_creation_args

    returnn_train_config = job_dispatcher.make_returnn_train_config_old(
        network = network,
        config_base_args=config_base_args,
        post_config_args=returnn_train_post_config,
        **make_train_config_extra
    )

    if not write_dummpy_config is None:
        # Fuck damit, can actually do this because write always calls _serialize, which always calls  config.update(self.post_config)
        # And this causes an inconsitency error because some post config args are added several times
        # But maybe we can deep copy and then write... waste some memory but should work
        import copy
        assert isinstance(write_dummpy_config, str)
        train_config = copy.deepcopy(returnn_train_config)
        train_config.write(write_dummpy_config)

    # We test the construction now to avaoid error when running on cluster
    if test_construction:
        job_dispatcher.test_net_contruction(returnn_train_config)

    returnn_rasr_config_args : dict = rasr_config_args_maker.get_returnn_rasr_args_02_devtrain(
    system, 
    train_corpus_key=train_corpus_key,
    **returnn_rasr_args_defaults
    )

    train_job = job_dispatcher.make_and_register_returnn_rasr_train_02_devtrain(
        returnn_train_config,
        returnn_rasr_config_args,
        output_path=f"{name}", # No need for 'output_path' here, its already sub aliased
        set_rqmt = True # CHANGED
    )

    rec_corpus = "dev-other" # Other are only done for the best epoch in the end
    # Prepare args for rasr recog
    system.init_rasr_am_lm_config_recog(
    recog_corpus_key=rec_corpus
    )

    limit_eps = returnn_train_post_config["cleanup_old_models"]["keep"]
    if extra_recog_epochs:
        limit_eps += extra_recog_epochs

    job_dispatcher.make_and_register_returnn_rasr_search_02( # 15 paralel searches instead of just 10
        system=system,
        returnn_train_config=returnn_train_config,
        train_job=train_job,
        recog_corpus_key=rec_corpus,
        feature_name="gammatone",
        limit_eps=limit_eps,
        exp_name=name
    )

    if final_recog and False: # THis was still bugg, TODO fixme
        job_dispatcher.make_and_register_final_rasr_search(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )
    

    if final_recog:
        job_dispatcher.make_and_register_final_rasr_search_manual(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )

    if extra_recog_devtrain:

        devsystem = sys_02_devtrain_rerecog.LibrispeechHybridSystemTim()
        # Make a returnn config
        devsystem.create_rasr_am_config(train_corpus_key=train_corpus_key)
        devsystem.init_rasr_am_lm_config_recog(
            recog_corpus_key="devtrain2000"
        )

        job_dispatcher.make_and_register_final_rasr_search_manual_devtrain(
            train_job=train_job,
            output_path=f"{name}",
            system = devsystem,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )


    return OrderedDict(
        network = network
    )


# Allowes to manipulate sequence orders of rasr datasets
def create_experiment_world_004( # New system that adds devtrain dataset
    name=None,
    output_path=None,
    config_base_args=None,
    extra_returnn_net_creation_args = None,

    # Network stuff
    conformer_create_func=None,
    conformer_func_args=None,
    #   - sampling_func_args
    #   - ff1_func_args
    #   - ff2_func_args
    #   - sa_func_args
    #   - conv_func_args
    #   - conformer_default_args_00

    returnn_train_post_config=None,
    returnn_rasr_args_defaults=None,

    extra_recog_epochs = None, # This setup always does *all* recogs for all 'keep' epochs, use this to add more

    final_recog = True,

    extra_recog_devtrain = False,

    test_construction=False,
    print_net = False,
    write_dummpy_config = None, # String path if given, write the returnn config there
):
    
    system = sys_02_devtrain.LibrispeechHybridSystemTim()

    # Make a returnn config
    train_corpus_key = 'train-other-960'

    system.create_rasr_am_config(train_corpus_key=train_corpus_key)

    network = conformer_create_func(
        **conformer_func_args,
        print_net = print_net
    )

    make_train_config_extra = {} if extra_returnn_net_creation_args is None else extra_returnn_net_creation_args

    returnn_train_config = job_dispatcher.make_returnn_train_config_old(
        network = network,
        config_base_args=config_base_args,
        post_config_args=returnn_train_post_config,
        **make_train_config_extra
    )

    if not write_dummpy_config is None:
        # Fuck damit, can actually do this because write always calls _serialize, which always calls  config.update(self.post_config)
        # And this causes an inconsitency error because some post config args are added several times
        # But maybe we can deep copy and then write... waste some memory but should work
        import copy
        assert isinstance(write_dummpy_config, str)
        train_config = copy.deepcopy(returnn_train_config)
        train_config.write(write_dummpy_config)

    # We test the construction now to avaoid error when running on cluster
    if test_construction:
        job_dispatcher.test_net_contruction(returnn_train_config)

    if isinstance(test_construction, str) and test_construction == "advanced":
        job_dispatcher.test_net_construction_advanced(returnn_train_config)

    returnn_rasr_config_args : dict = rasr_config_args_maker.get_returnn_rasr_args_03_overwrite_orders(
        system, 
        train_corpus_key=train_corpus_key,
        **returnn_rasr_args_defaults
    )

    train_job = job_dispatcher.make_and_register_returnn_rasr_train_02_devtrain(
        returnn_train_config,
        returnn_rasr_config_args,
        output_path=f"{name}", # No need for 'output_path' here, its already sub aliased
        set_rqmt = True # CHANGED
    )

    rec_corpus = "dev-other" # Other are only done for the best epoch in the end
    # Prepare args for rasr recog
    system.init_rasr_am_lm_config_recog(
    recog_corpus_key=rec_corpus
    )

    limit_eps = returnn_train_post_config["cleanup_old_models"]["keep"]
    if extra_recog_epochs:
        limit_eps += extra_recog_epochs

    job_dispatcher.make_and_register_returnn_rasr_search_02( # 15 paralel searches instead of just 10
        system=system,
        returnn_train_config=returnn_train_config,
        train_job=train_job,
        recog_corpus_key=rec_corpus,
        feature_name="gammatone",
        limit_eps=limit_eps,
        exp_name=name
    )

    if final_recog and False: # This was still buggy, TODO: fix me 
        job_dispatcher.make_and_register_final_rasr_search(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )

    if final_recog:
        job_dispatcher.make_and_register_final_rasr_search_manual(
            train_job=train_job,
            output_path=f"{name}",
            system = system,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
        )

    # If this flag is set we will also run an additional recog on the 'devother' dataset
    if extra_recog_devtrain:

        devsystem = sys_02_devtrain_rerecog.LibrispeechHybridSystemTim()
        # Make a returnn config
        devsystem.create_rasr_am_config(train_corpus_key=train_corpus_key)
        devsystem.init_rasr_am_lm_config_recog(
            recog_corpus_key="devtrain2000"
        )

        job_dispatcher.make_and_register_final_rasr_search_manual_devtrain(
            train_job=train_job,
            output_path=f"{name}",
            system = devsystem,
            returnn_train_config = returnn_train_config,
            feature_name = "gammatone",
            exp_name = name,
            use_gpu_and_extra_mem=True
        )

    return OrderedDict(
        network = network
    )