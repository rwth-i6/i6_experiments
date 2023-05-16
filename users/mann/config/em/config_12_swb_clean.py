from sisyphus import *

import os, sys
import copy
import itertools

from i6_core import rasr
from i6_core import tools

from i6_experiments.users.mann.setups.nn_system.base_system import ExpConfig, RecognitionConfig
from i6_experiments.users.mann.setups.tdps import CombinedModel
from i6_experiments.users.mann.setups.nn_system import common

sys.setrecursionlimit(2500)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

swb_system = common.init_system(
    "swb",
    state_tying_args=dict(
        value="monophone-no-tying-dense",
        use_boundary_classes=False,
        use_word_end_classes=True,
    ),
    custom_args=dict(
        legacy=False,
    ),
    extract_features=True,
)


#---------------------------------- tinas baseline ------------------------------------------------

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    swb_system.default_builder
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_transcription_prior()
    .set_specaugment()
)
builder.register("fullsum_w_prior")

builder.set_no_prior().register("fullsum_no_prior")

#--------------------------------- train tdps -----------------------------------------------------

NO_TDP_MODEL = CombinedModel.zeros()
# set up returnn repository
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

PRIOR_MODEL_TINA = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
default_recognition_args = RecognitionConfig(
    tdps=CombinedModel.legacy(),
    beam_pruning=22,
    prior_scale=0.3,
    tdp_scale=0.1,
    lm_scale=3.0,
)
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "returnn_root": RETURNN_TDPS,
        "mem_rqmt": 24,
    },
    fast_bw_args={
        "acoustic_model_extra_config": PRIOR_MODEL_TINA.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
    },
    recognition_args=default_recognition_args.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
)

baselines = {
    "no_prior": swb_system.builders["fullsum_no_prior"].build(),
}

def run_baselines():
    for name, config in baselines.items():
        swb_system.run_exp(
            name="baseline_{}".format(name),
            crnn_config=config,
            exp_config=exp_config,
        )

        if name == "no_prior":
            extra_bw_config = rasr.RasrConfig()
            extra_bw_config[
                "neural-network-trainer"
                ".alignment-fsa-exporter"
                ".model-combination"
                ".acoustic-model"
                ".fix-tdp-leaving-epsilon-arc"
            ] = True
            extra_bw_config[
                "neural-network-trainer"
                ".alignment-fsa-exporter"
                ".alignment-fsa-exporter"
                ".model-combination"
                ".acoustic-model"
            ] = None
            swb_system.run_exp(
                name="baseline_{}.corr_rasr".format(name),
                crnn_config=config,
                exp_config=exp_config.extend(
                    fast_bw_args={
                        "extra_config": extra_bw_config
                    },
                ),
            )

        # train with different tdp scale
        scaled_config = copy.deepcopy(config)
        scaled_config.tdp_scale = 0.3
        swb_system.run_exp(
            name="baseline_{}.tdp-0.3".format(name),
            crnn_config=scaled_config,
            exp_config=exp_config,
        )

        if name == "no_prior":
            swb_system.run_exp(
                name="baseline_{}.tdp-0.3.corr_rasr".format(name),
                crnn_config=scaled_config,
                exp_config=exp_config.extend(
                    fast_bw_args={
                        "extra_config": extra_bw_config
                    },
                ),
            )

def py():
    run_baselines()
