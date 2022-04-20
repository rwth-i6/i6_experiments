# TODO: package, make imports smaller
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor as system
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_rasr_config_maker as rasr_config_args_maker
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_common_network_generator
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import hybrid_job_dispatcher as job_dispatcher

from recipe.i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob
import inspect


from sisyphus import *

gs.ALIAS_AND_OUTPUT_SUBDIR = "conformer/new_setup_test/"

# Start system:
# - register alignments and features ...
system = system.LibrispeechHybridSystemTim()

# Make a returnn config
config_base_args = experiment_config_args.config_baseline_00

train_corpus_key = 'train-other-960'

system.create_rasr_am_config(train_corpus_key=train_corpus_key)

# Conformer generation code ( should be moved somewhere else )

returnn_train_config = job_dispatcher.make_and_hash_returnn_rtc_config(
  network_func=conformer_returnn_common_network_generator.make_conformer,
  config_base_args=config_base_args
)

returnn_rasr_config_args = rasr_config_args_maker.get_returnn_rasr_args(system, train_corpus_key=train_corpus_key)

job_dispatcher.make_and_register_returnn_rasr_train(
    returnn_train_config,
    returnn_rasr_config_args
)