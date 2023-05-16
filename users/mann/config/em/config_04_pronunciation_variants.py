from sisyphus import *

import os

from i6_core.tools import CloneGitRepositoryJob
from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
# s = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

lbs_system = lbs.get_legacy_librispeech_system()

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

#------------------------------ baseline viterbi experiment ---------------------------------------

lbs_system.nn_and_recog(
    name="baseline_viterbi_lstm",
    crnn_config=lbs_system.baselines["viterbi_lstm"](),
    training_args={"keep_epochs": [12, 24, 32, 48, 80, 160]},
    epochs=[]
)

#------------------------------ baseline bw experiment --------------------------------------------

TOTAL_FRAMES = 36107903
baseline_bw = lbs_system.baselines["bw_lstm_fixed_prior_job"](TOTAL_FRAMES)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/25, 0.0)

lbs.init_segment_order_shuffle(lbs_system)
recog_extra_config = lbs.custom_recog_tdps()

baseline_bw.config.config["learning_rate"] = 0.00025
baseline_bw.build_args["static_lr"] = 0.00025
from i6_experiments.users.mann.nn.learning_rates import get_learning_rates
baseline_bw.config.config["learning_rates"] = get_learning_rates(
    inc_min_ratio=0.25, increase=70, decay=70
)

exp_conf = ExpConfig(
    training_args={
        "num_classes": None,
        "alignment": None
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
    },
    epochs=[80, 160],
    recognition_args={"extra_config": recog_extra_config},
    reestimate_prior="CRNN"
)

print(recog_extra_config)

lbs_system.run_exp(
    "baseline_bw_lstm",
    crnn_config=baseline_bw,
    exp_config=exp_conf,
)


#------------------------------ remove pronunciation variants -------------------------------------

from recipe.i6_experiments.users.mann.experimental.lexicon import RemovePronunciationVariantsJob, VariantStatisticsJob

lexicon_file = (lbs_system.crp["train"].lexicon_config.file)
remove_job = RemovePronunciationVariantsJob(lexicon_file)
tk.register_output("out.lexicon.gz", remove_job.out_lexicon)

variants_stats = VariantStatisticsJob(
    lbs_system.crp["train"].corpus_config.file,
    lbs_system.crp["train"].lexicon_config.file
)
tk.register_output("variant_stats/lexicon.stats", variants_stats.out_lexicon_stats)
tk.register_output("variant_stats/corpus.stats", variants_stats.out_corpus_stats)

lbs_system.crp["train"].lexicon_config.file = remove_job.out_lexicon

lbs_system.run_exp(
    "baseline_bw_lstm.no_pronunciation_variants",
    crnn_config=baseline_bw,
    exp_config=exp_conf,
)

# try without prior
import copy
baseline_bw_no_prior = copy.deepcopy(baseline_bw)
baseline_bw_no_prior.warmup.prior_scale = 0.0

lbs_system.run_exp(
    "baseline_bw_lstm.no_pronunciation_variants.no_prior",
    crnn_config=baseline_bw_no_prior,
    exp_config=exp_conf,
)

def clean(gpu=False):
    keep_epochs = exp_conf.epochs
    for name in lbs_system.nn_config_dicts["train"]:
        lbs_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )
