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

tdp_params_drate_1 = list(product(
    [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [-1.0, 0.0, 0.5, 1.0],
))

def set_tdp_scale(recognition_args, scale):
    extra_config = recognition_args.get("extra_config", rasr.RasrConfig())
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp.scale"] = scale
    recognition_args["extra_config"] = extra_config

tdp_scales = [0.1, 0.3, 0.5, 0.7, 1.0]

NUM_FRAMES = 91026136
BASE_EPOCH = 240
BASE_TRAINING = "baseline_viterbi_tdnn.1s"
BASE_LMS = 12.0
BASE_SILENCE_EXIT = 20.0
SKIP_PARAMETERS = [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0, 30.0, "infinity"]
# SKIP_PARAMETERS = [0.0]
DRATES = [1, 2, 3, 4, 6, 8]
ts = helpers.TuningSystem(swb_system, training_args={})
wers = {}
rtfs = {}
for drate in [1, 2, 3, 4, 6, 8]:
# for drate in []:
    wers[drate] = {}
    rtfs[drate] = {}
    # for fwds in [True, False, "skip"]:
    for fwds in ["skip"]:
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
            set_tdps(
                extra_config["flf-lattice-tool.network.recognizer.acoustic-model"],
                drate, skip=(fwds == "skip"),
                speech_exit=0.0 if fwds == "skip" else None
            )
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
    
    if drate in [1, 3]:
        recognition_args={
            **recog_args,
            **extra_recog_args,
            "lm_scale": adjust_lm_scale(BASE_LMS, drate),
            "extra_config": extra_config,
        }
        ts.tune_parameter(
            name="recog_drate-{}.sp-tdps".format(drate),
            crnn_config=baseline_tdnn,
            parameters=tdp_params if drate > 1 else tdp_params_drate_1,
            transformation=TdpsSetter(drate).set_tdps_comp,
            procedure=helpers.Recog(BASE_EPOCH, BASE_TRAINING),
            compile_args=compile_args,
            compile_crnn_config=base_compile_config,
            recognition_args=recognition_args,
            # scorer_args={"prior_mixtures": None},
            scorer_args={},
            optimize=False,
            reestimate_prior="CRNN" 
        )

        sp_tdps = {
            1: (1.0, 0.0),
            3: (0.0, 1.0),
        }

        TdpsSetter(drate).set_tdps_comp(recognition_args, sp_tdps[drate])

        ts.tune_parameter(
            name="recog_drate-{}.tdp_scale".format(drate),
            crnn_config=baseline_tdnn,
            parameters=tdp_scales,
            transformation=set_tdp_scale,
            procedure=helpers.Recog(BASE_EPOCH, BASE_TRAINING),
            compile_args=compile_args,
            compile_crnn_config=base_compile_config,
            recognition_args=recognition_args,
            # scorer_args={"prior_mixtures": None},
            scorer_args={},
            optimize=False,
            reestimate_prior="CRNN" 
        )

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
    fname = "output/{}/reports/{}.pkl".format(FNAME, name)
    tk.dump(util.instanciate_delayed(data), fname)

# tk.register_callback(dump_summary, "wer", wers)
# tk.register_callback(dump_summary, "rtf", rtfs)

#------------------------------------- make downsampled alignments --------------------------------

# set up returnn repository
from i6_core import tools
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-sprint-cache-dataset",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_SPRINT_CACHE = clone_returnn_job.out_repository

swb_system.set_num_classes("lut", swb_system.num_classes())
for crp in swb_system.crp.values():
    crp.acoustic_model_config.hmm.states_per_phone = 1
swb_system.set_state_tying("lut", LUT_1S)

def make_alignment(drate):
    align_returnn_config = baseline_tdnn
    align_compile_args = {}
    if drate > 1:
        align_returnn_config = copy.deepcopy(baseline_tdnn)
        align_returnn_config.python_prolog = ("from returnn.tf.util.data import Dim",)
        align_returnn_config.config["a_dim"] = CodeWrapper('Dim(Dim.Types.Spatial, "Window dimension")')
        align_returnn_config.config["network"].update(output(0, drate, padding="same"))
        align_compile_args = {
            'output_tensor_name': 'output_/output_batch_major',
        }
    
    align_config = rasr.RasrConfig()
    set_tdps(
        align_config[
            "acoustic-model-trainer"
            ".aligning-feature-extractor"
            ".feature-extraction"
            ".alignment"
            ".model-combination"
            ".acoustic-model"
        ],
        drate,
        skip=True, adjust_silence_exit=True, speech_exit=0.0
    )

    align_am = align_config[
        "acoustic-model-trainer"
        ".aligning-feature-extractor"
        ".feature-extraction"
        ".alignment"
        ".model-combination"
        ".acoustic-model"
    ]
    align_am.tdp.silence.exit = 0.0
    align_am.tdp.silence.skip = "infinity"

    swb_system.nn_align(
        nn_name=BASE_TRAINING,
        epoch=BASE_EPOCH,
        extra_config=align_config,
        name="baseline_drate-{}".format(drate),
        compile_args=align_compile_args,
        compile_crnn_config=align_returnn_config,
        evaluate=True,
        scorer_suffix="-prior"
    )

def set_chunking(config, pooled_size):
    chunk_data = 48
    step_data = chunk_data // 2
    chunk_classes = chunk_data // pooled_size
    step_classes = chunk_classes // 2
    config["chunking"] = (
        {"data": chunk_data, "classes": chunk_classes},
        {"data": step_data, "classes": step_classes},
    )


LAYER_SEQ = ["input_conv"] \
    + [x for i in range(6) if (x := f"gated_{i}") in baseline_tdnn.config["network"]] \
    + ["output"]


def pooling_layer(mode, pool_size, sources, padding="same"):
    if isinstance(pool_size, int):
        pool_size=(pool_size,)
    return {
        "class": "pool", "mode": mode, "pool_size": pool_size, "from": sources,
        "padding": padding,
    }

def insert_pooling_layer(network, index=-1, mode="avg", pool_size=3, **_ignored):
    if index < 0:
        index += len(LAYER_SEQ)
    assert index > 0
    assert index < len(LAYER_SEQ)
    if isinstance(pool_size, int):
        pool_size=(pool_size,)
    i = index
    assert LAYER_SEQ[i-1] in network
    assert LAYER_SEQ[i] in network
    network["pooling"] = pooling_layer(mode, pool_size, [LAYER_SEQ[i-1]])
    network[LAYER_SEQ[i]]["from"] = ["pooling"]

from i6_experiments.users.mann.setups.nn_system.trainer import SprintCacheTrainer
swb_system.set_trainer(SprintCacheTrainer(swb_system))

def set_downsampled_recognition(recognition_args, drate):
    extra_config = rasr.RasrConfig()
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"].tdp.silence.exit \
        = adjust_silence_exit_penalty(BASE_SILENCE_EXIT, drate)
    set_tdps(
        extra_config["flf-lattice-tool.network.recognizer.acoustic-model"],
        drate, skip=True,
        speech_exit=0.0
    )
    # adjust recognition args
    recognition_args["lm_scale"] = adjust_lm_scale(BASE_LMS, drate)
    recognition_args["extra_config"] = extra_config

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel
def set_downsampled_recognition_legacy(recognition_args, drate):
    extra_config = rasr.RasrConfig()
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"].tdp.silence.exit \
        = adjust_silence_exit_penalty(BASE_SILENCE_EXIT, drate)
    set_tdps(
        extra_config["flf-lattice-tool.network.recognizer.acoustic-model"],
        drate, skip=True,
        speech_exit=0.0
    )
    model = CombinedModel.from_weights(
        0.0, 3.0,
        3.0, 0.0,
        silence_exit=BASE_SILENCE_EXIT / drate,
        speech_skip=max(40.0 - drate * 10.0, -1.0),
        skip_normed=False,
    )
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"] = model.to_acoustic_model_config()
    # adjust recognition args
    recognition_args["lm_scale"] = adjust_lm_scale(BASE_LMS, drate)
    recognition_args["extra_config"] = extra_config

def set_downsampled_alignment_training(
    config,
    training_args,
    recognition_args,
    scorer_args,
    plugin_args,
    drate
):
    set_downsampled_recognition(recognition_args, drate)

    make_alignment(drate)
    alignment_name = "baseline_drate-{}".format(drate)
    training_args["alignment"] = ("train", alignment_name, -1)
    training_args["seq_ordering"] = "laplace:.1000"
    if drate == 1:
        return 
    
    scorer_args["use_alignment"] = False
    set_chunking(config.config, drate)
    insert_pooling_layer(config.config["network"], pool_size=drate)
    plugin_args["filter_alignment"] = {}

for drate in DRATES[:-1]:
    make_alignment(drate)

ts.tune_parameter(
    name="baseline_downsampled_alignment",
    crnn_config=baseline_tdnn,
    parameters=DRATES[:-1],
    transformation=set_downsampled_alignment_training,
    training_args={
        "returnn_root": RETURNN_SPRINT_CACHE,
    },
    recognition_args={
        **recog_args,
        # **extra_recog_args,
    },
    plugin_args={},
    scorer_args={},
    reestimate_prior="CRNN" ,
    alt_training=True
)

#------------------------------------- recog tuning -----------------------------------------------

# set beam-pruning, beam-pruning-limit and acoustic-lookahead-temporal-approximation-scale in recognition_args
def set_recog_params(recognition_args, params):
    recognition_args["search_parameters"] = sp = {}
    sp["beam-pruning"] = params[0]
    sp["beam-pruning-limit"] = params[1]
    extra_config = recognition_args["extra_config"] # = extra_config = rasr.RasrConfig()
    extra_config["flf-lattice-tool.network.recognizer.recognizer.acoustic-lookahead-temporal-approximation-scale"] = params[2]
    return recognition_args

# possible values for beam-pruning, beam-pruning-limit and acoustic-lookahead-temporal-approximation-scale
from itertools import product
params = list(product([12.0, 14.0, 16.0], [10_000, 20_000, 100_000], [0.0, 1.0, 2.0, 4.0]))
# params = [(14.0, 10_000, 2.0)]

wers = {}
rtfs = {}
gs.GRAPH_WORKER = 4

# for drate in [3]:
for drate in DRATES[:-2]:
    for leg in [False]: #, False]:
        # adjust recognition
        recognition_args = recog_args.copy()
        if leg:
            set_downsampled_recognition_legacy(recognition_args, drate)
            infix = ".legacy"
        else:
            set_downsampled_recognition(recognition_args, drate)
            infix = ""

        base_name = "baseline_downsampled_alignment-{}".format(drate)
        name="drate-{}.recog{}".format(drate, infix)
        ts.tune_parameter(
            name=name,
            crnn_config=swb_system.nn_config_dicts["train"][base_name],
            compile_crnn_config=base_name,
            parameters=params,
            transformation=set_recog_params,
            recognition_args=recognition_args,
            procedure=helpers.Recog(240, base_name),
            scorer_args={
                "prior_file": base_name,
            },
            # reestimate_prior="alt",
            training_args={
                "returnn_root": RETURNN_SPRINT_CACHE,
            },
            # reestimate_prior="alt",
            reestimate_prior=False,
            # alt_training=True,
        )

        print(drate)
        wers[drate] = werp = {}
        rtfs[drate] = rtfp = {}

        for p in params:
            ps = "-".join(map(str, p))
            werp[ps] = ts.get_wer(
                name, 240, ps,
                # reestimate_prior=True,
                precise=True
            )
            rtfp[ps] = ts.get_rtf(
                name, 240, ps,
                # eestimate_prior=True,
                num_frames=NUM_FRAMES,
            )

        tk.register_callback(dump_summary, "wers.{}{}.align".format(drate, infix), werp)
        tk.register_callback(dump_summary, "rtfs.{}{}.align".format(drate, infix), rtfp)


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
