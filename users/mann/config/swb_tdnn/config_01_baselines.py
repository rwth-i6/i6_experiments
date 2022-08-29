import copy

from sisyphus import *

from i6_experiments.users.mann.setups.swb.swbsystem import SWBNNSystem
from i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
from i6_experiments.users.mann.experimental import helpers
from i6_experiments.users.mann.setups import tdps

from i6_experiments.users.mann.setups.nn_system.switchboard import get_legacy_switchboard_system, make_cart
from i6_experiments.users.mann.nn.tdnn import make_baseline

from i6_core import rasr
from i6_core.lexicon.allophones import DumpStateTyingJob

epochs = [12, 24, 32, 80, 160, 240, 320]
dbg_epochs = [12]

swb_system = get_legacy_switchboard_system()

# swb_system.run("baseline_viterbi_lstm")

baseline_viterbi = swb_system.baselines["viterbi_lstm"]()

print(swb_system.crp["train"].nn_trainer_exe)

print(swb_system.returnn_root)

swb_system.nn_and_recog(
    name="baseline_viterbi_lstm",
    crnn_config=baseline_viterbi,
    epochs=epochs,
    reestimate_prior=False,
)

baseline_tdnn = make_baseline(num_input=40)

swb_system.nn_and_recog(
    name="baseline_viterbi_tdnn",
    crnn_config=baseline_tdnn,
    epochs=epochs,
    reestimate_prior=False
)

print(swb_system.crp["train"].acoustic_model_config)
print(swb_system.crp["base"].acoustic_model_config)

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

print(swb_system.crp["train"].acoustic_model_config)
print(swb_system.crp["base"].acoustic_model_config)

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
def output(position, window_size):
    return {
        'output_': {
            'class': 'subnetwork', 'from': 'output', 'is_output_layer': True,
            'subnetwork': {
                'strided_window': {'class': 'window', 'from': 'data', 'window_size': window_size, 'stride': window_size, 'padding': 'valid', 'window_dim': CodeWrapper("a_dim")},
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
def set_tdps(extra_config: rasr.RasrConfig, drate: int, skip=False):
    speech_fwd = 1 / (1 + 7 / drate)
    if skip:
        speech_loop = 1 - speech_fwd
        avg_num_states = 1 + 7 / drate
        skip = speech_loop**(2 * avg_num_states)
        fwd_skip = speech_fwd * skip
        speech_fwd *= 1 - skip
        extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp.*"].skip = -np.log(fwd_skip)
    speech_transition = tdps.Transition.from_fwd_prob(speech_fwd)
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp.*"]._update(
        speech_transition.to_rasr_config()
    )
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp.silence"]._update(
        tdps.Transition.from_fwd_prob(1 / (1 + 60 / drate)).to_rasr_config()
    )

NUM_FRAMES = 91026136
BASE_EPOCH = 240
BASE_TRAINING = "baseline_viterbi_tdnn.1s"
BASE_LMS = 12.0
BASE_SILENCE_EXIT = 20.0
SKIP_PARAMETERS = [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0, 30.0, "infinity"]
# SKIP_PARAMETERS = [0.0]
ts = helpers.TuningSystem(swb_system, training_args={})
wers = {}
rtfs = {}
for drate in [1, 2, 3, 4, 6, 8]:
    wers[drate] = {}
    rtfs[drate] = {}
    for fwds in [True, False, "skip"]:
        # add output layer
        compile_args = {}
        extra_recog_args = {}
        base_compile_config = baseline_tdnn
        if drate > 1:
            base_compile_config = copy.deepcopy(baseline_tdnn)
            base_compile_config.python_prolog = ("from returnn.tf.util.data import Dim",)
            base_compile_config.config["a_dim"] = CodeWrapper('Dim(Dim.Types.Spatial, "Window dimesion")')
            compile_args["compile_network_extra_config"] = output(0, drate)
            extra_recog_args = {
                'output_tensor_name': 'output_/output_batch_major',
            }

        # configure silence exit penalty
        extra_config = rasr.RasrConfig()
        extra_config["flf-lattice-tool.network.recognizer.acoustic-model"].tdp.silence.exit \
            = adjust_silence_exit_penalty(BASE_SILENCE_EXIT, drate)
        
        suffix = ""
        suffix = ".lut"
        set_1s_lookup_table(extra_config, LUT_1S)

        skip_parameters = SKIP_PARAMETERS.copy()
        if fwds:
            suffix += ".fwds"
            set_tdps(extra_config, drate, skip=(fwds == "skip"))
            if fwds == "skip":
                suffix += ".normed"
                skip_parameters = [ts.NoTransformation]
            # skip_parameters.append("infinity")
            
        # run tuning experiment
        name="recog_drate-{}{}.prior.skip".format(drate, suffix)
        ts.tune_parameter(
            name=name,
            crnn_config=baseline_tdnn,
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

        wers[drate][fwds] = {
            skip: ts.get_wer(
                name, BASE_EPOCH, str(skip),
                optlm=False, reestimate_prior=True,
                precise=True
            )
            for skip in skip_parameters
        }
        rtfs[drate][fwds] = {
            skip: ts.get_rtf(
                name, BASE_EPOCH, str(skip),
                optlm=False, reestimate_prior=True,
                num_frames=NUM_FRAMES,
            )
            for skip in skip_parameters
        }

from i6_core import util
class SkipReport:
    def __init__(self, wers, rtfs):
        self.wers = wers
        self.rtfs = rtfs
    
    def __call__(self):
        from pandas import DataFrame
        from tabulate import tabulate
        df_wers = DataFrame(util.instanciate_delayed(self.wers))
        df_wers = df_wers.applymap(lambda x: min(x.values()))
        return tabulate(df_wers, tablefmt="presto")

# tk.register_report("reports/fwds.txt", SkipReport(wers, rtfs))

def dump_summary(name, data):
    import pickle
    print("Dumping {}".format(name))
    with open("output/reports/{}.pkl".format(name), "wb+") as f:
        pickle.dump(util.instanciate_delayed(data), f)

tk.register_callback(dump_summary, "wer", wers)
tk.register_callback(dump_summary, "rtf", rtfs)

#------------------------------------- try interface ----------------------------------------------

def try_stuff():
    from recipe.i6_experiments.users.mann.experimental.tuning import Schedule, Setter
    sched = Schedule()

    @Setter.from_func
    def set_1s_lookup_table_comp(recognition_args, state_tying_file=LUT_1S):
        set_1s_lookup_table(recognition_args["extra_config"], state_tying_file)

    @Setter.from_func
    def set_recog_skip(recognition_args: dict, skip_penalty: float):
        set_skip(recognition_args["extra_config"], skip_penalty)

    print(set_1s_lookup_table_comp.signature)

    sched.add_node("lut", set_1s_lookup_table_comp, yesno=True)
    sched.add_node("skip", set_recog_skip, parent="lut", params=[15.0, 30.0])

    print(sched.root)
    print(sched.root.params[0].children)
    print(sched.root.children)
    print(sched.get_parameters())
    # quit()
