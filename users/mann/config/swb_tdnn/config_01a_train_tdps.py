import copy
from itertools import product

from sisyphus import *

from i6_experiments.users.mann.setups.swb.swbsystem import SWBNNSystem
from i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
from i6_experiments.users.mann.experimental import helpers
from i6_experiments.users.mann.setups import tdps
from i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel

from i6_experiments.users.mann.setups.nn_system.switchboard import get_legacy_switchboard_system, make_cart
from i6_experiments.users.mann.nn.config.tdnn import make_baseline

from i6_core import rasr
from i6_core.lexicon.allophones import DumpStateTyingJob

FNAME = gs.set_alias_and_output_dir()

epochs = [12, 24, 32, 80, 160, 240, 320]
dbg_epochs = [12]

swb_system = get_legacy_switchboard_system()

baseline_viterbi = swb_system.baselines["viterbi_lstm"]()


swb_system.nn_and_recog(
    name="baseline_viterbi_lstm",
    crnn_config=baseline_viterbi,
    epochs=epochs,
    reestimate_prior=False,
)

baseline_tdnn = make_baseline(num_input=40, adam=True)

swb_system.nn_and_recog(
    name="baseline_viterbi_tdnn",
    crnn_config=baseline_tdnn,
    epochs=epochs,
    reestimate_prior=False
)

#----------------------------------- simple state tying -------------------------------------------

from i6_experiments.users.mann.experimental.util import safe_crp
from i6_experiments.users.mann.setups.state_tying import MakeOneStateStateTyingJob, DuplicateStateTyingJob

state_tying_job = DumpStateTyingJob(swb_system.crp["train"])
tk.register_output("state_tying/original.tying", state_tying_job.out_state_tying)  

# state tying for recognition
state_tying_job = MakeOneStateStateTyingJob(state_tying_job.out_state_tying)
tk.register_output("state_tying/simple.1s.tying", state_tying_job.out_state_tying)
SIMPLE_ST_LUT = state_tying_job.out_state_tying

# state tying for training; compatible with 3-state alignment
duplicate_job = DuplicateStateTyingJob(SIMPLE_ST_LUT)
tk.register_output("state_tying/simple.duplicate.tying", duplicate_job.out_state_tying)
SIMPLE_ST_LUT_COMPAT = duplicate_job.out_state_tying


with safe_crp(swb_system):
    for corpus in ["train", "dev", "base"]:
        swb_system.crp[corpus].acoustic_model_config.hmm.states_per_phone = 1 if corpus == "dev" else 3
        swb_system.crp[corpus].acoustic_model_config.state_tying.file = SIMPLE_ST_LUT if corpus == "dev" else SIMPLE_ST_LUT_COMPAT
        swb_system.crp[corpus].acoustic_model_config.state_tying.type = "lookup"
    
    # swb_system.set_num_classes("")

    swb_system.nn_and_recog(
        name="baseline_viterbi_lstm.1s.simple",
        crnn_config=baseline_viterbi,
        epochs=epochs,
        reestimate_prior="CRNN",
        scorer_args={"prior_mixtures": None}
    )

    swb_system.nn_and_recog(
        name="baseline_viterbi_tdnn.1s.simple",
        crnn_config=baseline_tdnn,
        epochs=epochs,
        reestimate_prior="CRNN",
        scorer_args={"prior_mixtures": None}
    )

# quit()

#----------------------------------- one state cart -----------------------------------------------


# make 1-state cart
make_cart(swb_system, hmm_partition=1)

params = {
    "reestimate": dict(reestimate_prior="CRNN"),
    "mixture": dict(scorer_args={"prior_mixtures": ()}),
}

swb_system.nn_and_recog(
    name="baseline_viterbi_lstm.1s",
    crnn_config=baseline_viterbi,
    epochs=epochs,
    reestimate_prior="CRNN"
)

swb_system.nn_and_recog(
    name="baseline_viterbi_tdnn.1s",
    crnn_config=baseline_tdnn,
    epochs=epochs,
    reestimate_prior="CRNN"
)

#---------------------------------- make state-tying ----------------------------------------------


# extra_config = rasr.RasrConfig()
# extra_config.allophone_tool.acoustic_model.hmm.states_per_phone = 1
extra_config = None

state_tying_job = DumpStateTyingJob(swb_system.csp["train"], extra_config)
tk.register_output("state_tying/cart.1s.tying", state_tying_job.out_state_tying)

extra_config = rasr.RasrConfig()
extra_config.allophone_tool.acoustic_model.hmm.states_per_phone = 1
extra_config.allophone_tool.acoustic_model.state_tying.type = "lookup"
extra_config.allophone_tool.acoustic_model.state_tying.file = state_tying_job.out_state_tying

state_tying_job = DumpStateTyingJob(swb_system.csp["train"], extra_config)
tk.register_output("state_tying/lut.1s.tying", state_tying_job.out_state_tying)

LUT_1S = state_tying_job.out_state_tying

#--------------------------------- pooling layers and tuning --------------------------------------

target_hosts = "|".join(f"cluster-cn-{i}" for i in range(10, 26))
recog_args = {
    'extra_rqmts': {'qsub_args': f'-l hostname="{target_hosts}"'}
}
# swb_system.crp["dev"].acoustic_model_config.hmm.states_per_phone = 1
# BASE_COMPILE_CONFIG = copy.deepcopy(baseline_tdnn)
# BASE_COMPILE_CONFIG.python_prolog = ("from returnn.tf.util.data import Dim",)
# BASE_COMPILE_CONFIG.config["a_dim"] = CodeWrapper('Dim(Dim.Types.Spatial, "Window dimesion")')

from i6_core.returnn import CodeWrapper
# pooling by taking fixed position
def output(position, window_size, padding="valid"):
    return {
        'output_': {
            'class': 'subnetwork', 'from': 'output', 'is_output_layer': True,
            'subnetwork': {
                'strided_window': {'class': 'window', 'from': 'data', 'window_size': window_size, 'stride': window_size, 'padding': padding, 'window_dim': CodeWrapper("a_dim")},
                'output': {'class': 'gather', 'from': 'strided_window', 'axis': CodeWrapper("a_dim"), 'position': position}
            } 
        }
    }

# adjust lm scale to downsampling rate
def adjust_lm_scale(lm_scale, downsampling_rate):
    return lm_scale / downsampling_rate

# adjust silence exit penalty to downsampling rate
def adjust_silence_exit_penalty(silence_exit_penalty, downsampling_rate):
    return silence_exit_penalty / downsampling_rate

# set speech and silence forward penalties
def set_fwd_penalties(extra_config: rasr.RasrConfig, speech_loop: float, sil_fwd: float, silence_exit_penalty: float):
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"] \
        = tdps.CombinedModel(
            tdps.SimpleTransitionModel.from_weights([0.0, speech_loop], [sil_fwd, 0.0]),
            silence_exit_penalty
        ).to_acoustic_model_config()

# set skip penalty
def set_skip(extra_config: rasr.RasrConfig, skip_penalty: float):
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"].tdp["*"].skip = skip_penalty

def set_recog_skip(recognition_args: dict, skip_penalty: float):
    set_skip(recognition_args["extra_config"], skip_penalty)

def set_1s_lookup_table(extra_config: rasr.RasrConfig, state_tying_file: str):
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model.hmm.states-per-phone"] = 1
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model.state-tying.type"] = "lookup"
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model.state-tying.file"] = state_tying_file

def set_1s_lookup_table_comp(recognition_args, state_tying_file):
    set_1s_lookup_table(recognition_args["extra_config"], state_tying_file)

import numpy as np
def set_tdps(am: rasr.RasrConfig, drate: int, skip=False, adjust_silence_exit=False, speech_exit=None):
    speech_fwd = 1 / (1 + 7 / drate)
    if skip:
        speech_loop = 1 - speech_fwd
        avg_num_states = 1 + 7 / drate
        skip = speech_loop**(2 * avg_num_states)
        fwd_skip = speech_fwd * skip
        speech_fwd *= 1 - skip
        am["tdp.*"].skip = -np.log(fwd_skip)
    if adjust_silence_exit:
        adjust_silence_exit_penalty(20.0, drate)
    if speech_exit is not None:
        am["tdp.*"].exit = speech_exit
    speech_transition = tdps.Transition.from_fwd_prob(speech_fwd)
    am["tdp.*"]._update(
        speech_transition.to_rasr_config()
    )
    am["tdp.silence"]._update(
        tdps.Transition.from_fwd_prob(1 / (1 + 60 / drate)).to_rasr_config()
    )

class TdpsSetter:
    def __init__(self, drate: int):
        self.drate = drate
    
    def set_tdps_comp(self, recognition_args, tdps):
        spf, spl = tdps
        tdps = CombinedModel.from_weights(
            speech_fwd=spf,
            speech_loop=spl,
            silence_fwd=3.0,
            silence_loop=0.0,
            speech_skip=30.0,
            silence_exit=30.0 / self.drate
        )
        extra_config = recognition_args.get("extra_config", rasr.RasrConfig())
        extra_config["flf-lattice-tool.network.recognizer.acoustic-model"] = tdps.to_acoustic_model_config()
        recognition_args["extra_config"] = extra_config

tdp_params = list(product(
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 0.5, 1.0],
))

from recipe.i6_experiments.users.mann.nn import preload, tdps, bw, learning_rates
from i6_core import tools

swb_system.set_num_classes("lut", swb_system.num_classes())
for crp in swb_system.crp.values():
    crp.acoustic_model_config.hmm.states_per_phone = 1
swb_system.set_state_tying("lut", LUT_1S)

tdp_model = CombinedModel.zeros()
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

NUM_FRAMES = 91026136
BASE_EPOCH = 240
BASE_TRAINING = "baseline_viterbi_tdnn.1s"
BASE_LMS = 12.0
BASE_SILENCE_EXIT = 20.0
SKIP_PARAMETERS = [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0, 30.0, "infinity"]
# SKIP_PARAMETERS = [0.0]
DRATES = [1, 2, 3, 4, 6, 8]
ts = helpers.TuningSystem(swb_system, training_args={})

def make_tdp_training_config(config, drate):
    preload.set_preload(swb_system, config, (BASE_TRAINING, BASE_EPOCH))
    config.config.update(
        gradient_clip = 10,
        learning_rate = 0.1,
        batch_size = 5_000,
        newbob_learning_rate_decay = 0.9,
        learning_rate_control_error_measure = "dev_score_output_tdps"
    )
    # config.config["learning_rates"] = learning_rates.get_learning_rates(increase=10, decay=10, inc_max_ratio=1.0, dec_max_ratio=1.0)
    del config.config["learning_rates"]
    bw.add_bw_layer(
        swb_system.crp["train"], config,
        prior=False, keep_ce=False,
        num_classes=swb_system.num_classes())
    del config.config["network"]["output_bw"]
    tdps.get_model(num_classes=swb_system.num_classes(), arch="label").set_config(config)
    # make encoder untrainable
    for name, layer in config.config["network"].items():
        if any(k in name for k in ["input", "output", "gated"]):
            layer["trainable"] = False
    if drate == 1:
        return
    # connect output to loss
    config.config["network"]["fast_bw"]["from"] = ["output_"]

for drate in [1, 2, 3, 4, 6, 8]:
# for drate in []:
    # add output layer
    compile_args = {}
    extra_recog_args = {}
    base_compile_config = baseline_tdnn
    base_config = copy.deepcopy(baseline_tdnn)
    if drate > 1:
        base_compile_config = copy.deepcopy(baseline_tdnn)
        base_compile_config.python_prolog = ("from returnn.tf.util.data import Dim",)
        base_compile_config.config["a_dim"] = CodeWrapper('Dim(Dim.Types.Spatial, "Window dimension")')
        compile_args["compile_network_extra_config"] = output(0, drate)
        extra_recog_args = {
            'output_tensor_name': 'output_/output_batch_major',
        }
        base_config.config["network"].update(output(0, drate))
        base_config.python_prolog = ("from returnn.tf.util.data import Dim",)
        base_config.config["a_dim"] = CodeWrapper('Dim(Dim.Types.Spatial, "Window dimension")')
    
    
    make_tdp_training_config(base_config, drate)

    # configure silence exit penalty
    extra_config = rasr.RasrConfig()
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"].tdp.silence.exit \
        = adjust_silence_exit_penalty(BASE_SILENCE_EXIT, drate)
    
    # skip_parameters = SKIP_PARAMETERS.copy()
    skip_parameters = []
    # run tuning experiment
    name = "tdp.drate-{}".format(drate)
    swb_system.nn_and_recog(
        name=name,
        crnn_config=base_config,
        # scorer_args={"prior_mixtures": None},
        training_args={
            "num_classes": None,
            "alignment": None,
            "num_epochs": 160,
            "returnn_root": RETURNN_TDPS,
            "mem_rqmt": 24,
        },
        fast_bw_args={"acoustic_model_extra_config": tdp_model.to_acoustic_model_config()},
        epochs=[]
    )

    ts.tune_parameter(
        name=name,
        crnn_config=baseline_tdnn,
        # crnn_config=None,
        parameters=skip_parameters,
        transformation=set_recog_skip,
        procedure=helpers.Recog(BASE_EPOCH, BASE_TRAINING),
        compile_args=compile_args,
        compile_crnn_config=base_compile_config,
        recognition_args={
            **recog_args,
            **extra_recog_args,
            "lm_scale": adjust_lm_scale(BASE_LMS, drate),
            "extra_config": extra_config,
        },
        # scorer_args={"prior_mixtures": None},
        scorer_args={},
        optimize=False,
        reestimate_prior="CRNN" 
    )
