from sisyphus import *

import os, sys
import copy
import itertools

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, ConfigBuilder, RecognitionConfig
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups.nn_system import common
from i6_experiments.users.mann.setups.reports import TableReport
from recipe.i6_experiments.users.mann.setups import prior
from i6_experiments.users.mann.nn import specaugment, learning_rates
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
from i6_core import rasr
from recipe.i6_core import tools

sys.setrecursionlimit(2500)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

swb_system = common.init_system(
    "swb",
    state_tying_args={
        "value": "monophone-no-tying-dense",
        "use_boundary_classes": False,
        "use_word_end_classes": True,
    },
)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr

swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/master-rasr-fsa/src/Tools/Flf/flf-tool.linux-x86_64-standard"
NEW_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard"
CORR_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard"
CORR_AM_TRAINER = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard"

PRIOR_MODEL_TINA = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
default_recognition_args = RecognitionConfig(
    tdps=CombinedModel.legacy(),
    beam_pruning=22,
    prior_scale=0.3,
    tdp_scale=0.1,
    lm_scale=3.0,
)

NO_TDP_MODEL = CombinedModel.zeros()
# set up returnn repository
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

print("Returnn root: ", swb_system.returnn_root)
print("Returnn python exe: ", swb_system.returnn_python_exe)

exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "returnn_root": RETURNN_TDPS,
        # "returnn_python_exe": gs.RETURNN_PYTHON_EXE,
        "mem": 32,
        # **extra_args
    },
    fast_bw_args={
        "acoustic_model_extra_config": NO_TDP_MODEL.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
        "corpus": "train_magic",
    },
    recognition_args=default_recognition_args.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
    # dump_epochs=[12, 300],
    alt_training=True,
)


swb_system.init_dump_system(
    segments=[
        "switchboard-1/sw02001A/sw2001A-ms98-a-0041",
        "switchboard-1/sw02001A/sw2001A-ms98-a-0047",
        "switchboard-1/sw02001B/sw2001B-ms98-a-0004",
        "switchboard-1/sw02001B/sw2001B-ms98-a-0024"
    ],
    occurrence_thresholds=(0.1, 0.05),
)

#-------------------------------------- config builder --------------------------------------------

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    ConfigBuilder(swb_system)
    .set_lstm()
    .set_scales(am=0.3, tdp=0.3)
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_specaugment()
    # .set_oclr()
    .set_no_prior()
)

baseline_fullsum = builder.build()

swb_system.run_exp(
    "baseline_fullsum",
    crnn_config=baseline_fullsum,
    exp_config=exp_config,
    epochs=[],
)

#--------------------------------------- expectation-maximization ---------------------------------

from recipe.i6_experiments.users.mann.nn import preload, tdps
from i6_experiments.users.mann.experimental.em import FindTransitionModelMaximumJob
from i6_experiments.users.mann.setups.em import EmRunner

def maximize_trans():
    pass

viterbi_builder = (
    builder
    .copy()
    .set_loss("viterbi")
    .set_ce_args(focal_loss_factor=None)
    .delete("learning_rates")
    .update(learning_rate=0.0003)
)

runner = EmRunner(
    swb_system,
    "baseline_em",
    baseline_fullsum,
    viterbi_config=viterbi_builder.build(),
    exp_config=exp_config,
    num_epochs_per_maximization=2,
    returnn_root=RETURNN_TDPS,
    returnn_python_exe=common.RETURNN_PYTHON_EXE,
)

def legacy_exp():
    weights = runner.compute_expectation(
        label_pos_model=None,
        trans_model={
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        }
    )

    runner.run(num_iterations=10)

    tk.register_output("weights/label_pos_weights-0", weights["label_pos"])
    tk.register_output("weights/trans_weights-0", weights["trans"])

def main():
    trans_expectation = runner.compute_trans_expectation(
        label_pos_model=None,
        trans_model={
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        }
    )
    trans_weights = runner.maximize_trans(trans_expectation)
    tk.register_output("weights/trans_expectation-0", trans_expectation)
    for k in ["speech_fwd", "silence_fwd"]:
        tk.register_output(f"weights/max_{k}-0", trans_weights[k])

    trans_expectation = runner.compute_trans_expectation(
        label_pos_model=None,
        trans_model={
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        },
        _debug=True
    )
    tk.register_output("weights/trans_expectation-0-debug", trans_expectation)

    label_model = runner.maximize_label_pos(
        name="label_pos-0",
        label_pos_model=None,
        trans_model={
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        },
    )

    runner.run(num_iterations=10)


def legacy_max():
    debug_weights = runner.compute_expectation(
        label_pos_model=None,
        trans_model={
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        },
        _debug=True
    )

    tk.register_output("weights/trans_weights-0-debug", debug_weights["trans"])

    trans_model_updates = runner.maximize_trans(debug_weights["trans"])
    tk.register_output("weights/max_speech_fwd-0-debug", trans_model_updates["speech_fwd"])
    tk.register_output("weights/max_silence_fwd-0-debug", trans_model_updates["silence_fwd"])

    del runner.config.config["batch_size"]
    runner.config.config["forward_batch_size"] = 5000
    new_weights = runner.compute_expectation(
        label_pos_model=None,
        trans_model={
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        },
        trans_model_only=True
    )

    tk.register_output("weights/trans_weights_only-0", new_weights["trans"])

    from i6_core.report import MailJob

    m = MailJob(
        weights["label_pos"],
        subject="Forward job finished",
        mail_address="d.mann95.dm@gmail.com",
    )

    tk.register_output("mail", m.out_status)
