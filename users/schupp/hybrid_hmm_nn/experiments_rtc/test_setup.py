# TODO: package, make imports smaller
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import librispeech_hybrid_tim_refactor as system
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_rasr_config_maker as rasr_config_args_maker
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_common_network_generator
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline import hybrid_job_dispatcher as job_dispatcher

from recipe.i6_core.returnn import ReturnnConfig, ReturnnRasrTrainingJob

from sisyphus import gs

import inspect

OUTPUT_PATH = "conformer/new_setup_test/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

# Start system:
# - register alignments and features ...
system = system.LibrispeechHybridSystemTim()

# Make a returnn config
config_base_args = experiment_config_args.config_baseline_00

train_corpus_key = 'train-other-960'

system.create_rasr_am_config(train_corpus_key=train_corpus_key)

# Conformer generation code ( should be moved somewhere else )

if False:
  returnn_train_config : ReturnnConfig = job_dispatcher.make_and_hash_returnn_rtc_config(
    network_func=conformer_returnn_common_network_generator.make_conformer,
    config_base_args=config_base_args
  )


job_dispatcher.test_network_contruction(
  network_func=conformer_returnn_common_network_generator.make_conformer,
  config_base_args=config_base_args
)

# Test network contstuction and training on cpu ( saves qsub if this failes )
# TODO: ...

returnn_rasr_config_args : dict = rasr_config_args_maker.get_returnn_rasr_args(system, train_corpus_key=train_corpus_key)


# Create ReturnnRasrTrainJob, register outputs -> submit train
if False:
  train_job : ReturnnRasrTrainingJob = job_dispatcher.make_and_register_returnn_rasr_train(
      returnn_train_config,
      returnn_rasr_config_args,
      output_path=OUTPUT_PATH
  )


# Create Search Jobs for given epochs, *but* will also always make recog for epochs in 'keep_best'
if False:
  search_jobs = job_dispatcher.make_and_register_returnn_rasr_search(
    recog_for_epochs=experiment_config_args.search_job_dispatcher_defaults["epochs"],
    recog_corpus=train_corpus_key,
    train_job=train_job,
    output_path=OUTPUT_PATH
  )