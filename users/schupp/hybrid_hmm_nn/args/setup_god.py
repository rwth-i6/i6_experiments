
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
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor as sys
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

    test_construction=False,
    print_net = False
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

    rec_corpus = "dev-other"
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