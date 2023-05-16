import copy

from sisyphus import *

from i6_experiments.users.mann.setups.nn_system import (
    ConfigBuilder,
    NNSystem, ExpConfig,
    init_segment_order_shuffle,
)
from i6_experiments.users.mann.experimental import helpers
from i6_experiments.users.mann.setups import tdps

from i6_experiments.users.mann.setups.nn_system.switchboard import (
    get_legacy_switchboard_system,
    make_cart,
    init_prior_system,
)
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from i6_experiments.users.mann.nn.config import make_baseline
from i6_experiments.users.mann.nn import bw, learning_rates, constants

from i6_core import rasr
from i6_core.lexicon.allophones import DumpStateTyingJob

FNAME = gs.set_alias_and_output_dir()

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

#--------------------------------- make baseline bw -----------------------------------------------

# lbs_system = lbs.get_libri_1k_system()
swb_system = swb.get_legacy_switchboard_system()
# for binary in ["rasr_binary_path", "native_ops_path", "returnn_python_exe", "returnn_python_home", "returnn_root"]:
#     setattr(swb_system, binary, getattr(lbs_system, binary))
init_prior_system(swb_system)
init_segment_order_shuffle(swb_system)

TINA_RASR_EXE = rasr.RasrCommand.default_exe("nn-trainer")
print(TINA_RASR_EXE)
def set_tina_rasr(config):
    config.config["network"]["fast_bw"]["sprint_opts"]["sprintExecPath"] = TINA_RASR_EXE

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/60, 0.0)
swb_system.prior_system.lemma_end_probability = 1/4

recog_prior = copy.deepcopy(swb_system.prior_system)
recog_prior.eps = 1e-8
recog_prior.extract_prior()

exp_config = ExpConfig(
    # compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "num_epochs": 320,
    },
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
    },
    epochs=[12, 24, 48, 120, 240, 300, 320],
    scorer_args={
        "prior_mixtures": None,
        "prior_file": recog_prior.prior_xml_file, },
    reestimate_prior=False,
    dump_epochs=[4, 8, 12],
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

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    ConfigBuilder(swb_system)
    .set_tdnn()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_transcription_prior()
    .set_specaugment()
)
baseline_bw_tdnn = builder.build()

def baselines_3s():
    for prior_scale in [0.0, 0.1, 0.3]:
        conf = builder.copy().set_scales(prior_scale=prior_scale).build()
        swb_system.run_exp(
            name="baseline_bw_tdnn.3s.prior_scale-{}".format(prior_scale),
            crnn_config=conf,
            exp_config=exp_config,
        )

    baseline_bw_tdnn_povey = builder.copy().set_povey_prior().build()
    swb_system.run_exp(
        name="baseline_bw_tdnn.3s.scales.povey",
        crnn_config=baseline_bw_tdnn_povey,
        exp_config=exp_config,
    )


#--------------------------------- make 1s baseline -----------------------------------------------

# make 1-state cart
make_cart(swb_system, hmm_partition=1, as_lut=True)

# swb_system.prior_system.hmm_partition = 1
swb_system.prior_system.extract_prior()

tdp_model_one_state = CombinedModel.from_fwd_probs(1/8, 1/40, 0.0)
# exp_config = exp_config.extend(
#     fast_bw_args={"acoustic_model_extra_config": tdp_model_one_state.to_acoustic_model_config()},
# )

baseline_tdnn = make_baseline(num_input=40)
# baseline_bw_tdnn = swb_system.baselines["bw_tina_swb"](baseline_tdnn)
baseline_bw_tdnn = swb_system.baselines["bw_tina_swb_povey"](baseline_tdnn)
# del baseline_bw_tdnn.config["adam"]
assert isinstance(baseline_bw_tdnn, bw.ScaleConfig)
baseline_bw_tdnn.tdp_scale = 0.3

def baselines_1s():
    swb_system.run_exp(
        name="baseline_bw_tdnn.1s",
        crnn_config=baseline_bw_tdnn,
        exp_config=exp_config,
    )

    baseline_bw_tdnn_nadam = copy.deepcopy(baseline_bw_tdnn)
    # del baseline_bw_tdnn_nadam.config["adam"]
    # swb_system.run_exp(
    #     name="baseline_bw_tdnn.1s.nadam",
    #     crnn_config=baseline_bw_tdnn_nadam,
    #     exp_config=exp_config,
    # )

    from recipe.i6_experiments.users.mann.nn.pretrain import PretrainConfigHolder
    baseline_bw_tdnn_1s_pretrain = PretrainConfigHolder(
        config=baseline_bw_tdnn_nadam,
    )
    baseline_bw_tdnn_1s_pretrain.build_args.update(
        static_lr=True, # use static learning rate
    )
    baseline_bw_tdnn_1s_pretrain.config.am_scale = 0.7
    baseline_bw_tdnn_1s_pretrain.warmup.final_am = 0.7
    baseline_bw_tdnn_1s_pretrain.config.tdp_scale = 1.0

    swb_system.run_exp(
        name="baseline_bw_tdnn.1s.pretrain",
        crnn_config=baseline_bw_tdnn_1s_pretrain,
        exp_config=exp_config,
    )

    baseline_bw_tdnn_1s_pretrain_k = copy.deepcopy(baseline_bw_tdnn_1s_pretrain)
    baseline_bw_tdnn_1s_pretrain_k.warmup.absolute_scale = 0.1

    # swb_system.run_exp(
    #     name="baseline_bw_tdnn.1s.pretrain.abs_scale-0.1",
    #     crnn_config=baseline_bw_tdnn_1s_pretrain_k,
    #     exp_config=exp_config,
    # )


#--------------------------------- test downsampling ----------------------------------------------

DRATES = [1, 2, 3, 4, 6]
BASE_LMS = 12.0
BASE_SILENCE_EXIT = 20.0
target_hosts = "|".join(f"cluster-cn-{i}" for i in range(10, 26))
RECOG_ARGS = {
    'extra_rqmts': {'qsub_args': f'-l hostname="{target_hosts}"'}
}

LAYER_SEQ = ["input_conv"] \
    + [x for i in range(6) if (x := f"gated_{i}") in baseline_tdnn.config["network"]] \
    + ["output"]

print(LAYER_SEQ)

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

# adjust silence exit penalty to downsampling rate
def adjust_silence_exit_penalty(silence_exit_penalty, downsampling_rate):
    return silence_exit_penalty / downsampling_rate

def adjust_lm_scale(lm_scale, downsampling_rate):
    return lm_scale / downsampling_rate

def get_downsampled_tdps(
    drate: int,
    skip: bool=False,
    silence_exit: float=0.0,
    avg_speech_loops: int=7,
    avg_silence_loops: int=60
) -> tdps.CombinedModel:
    speech_fwd = 1 / (1 + avg_speech_loops / drate)
    speech_loop = 1 - speech_fwd
    return tdps.CombinedModel.from_fwd_probs(
        speech_fwd,
        silence_fwd = 1 / (1 + avg_silence_loops / drate),
        silence_exit = silence_exit,
        speech_skip = speech_loop**(2 * avg_speech_loops) if skip else None
    )

def set_downsampled_bw_training(config, fast_bw_args, recognition_args, drate, skip=False):
    # scale recognition config
    extra_config = rasr.RasrConfig()
    extra_config["flf-lattice-tool.network.recognizer.acoustic-model"] \
        = get_downsampled_tdps(
            drate, skip=skip, silence_exit=adjust_silence_exit_penalty(BASE_SILENCE_EXIT, drate)
        ).to_acoustic_model_config()
    # adjust recognition args
    recognition_args["lm_scale"] = adjust_lm_scale(BASE_LMS, drate)
    recognition_args["extra_config"] = extra_config

    if drate == 1:
        return 
    # drate > 1
    insert_pooling_layer(config.config["network"], pool_size=drate)
    fast_bw_args["acoustic_model_extra_config"] \
        = get_downsampled_tdps(drate, skip).to_acoustic_model_config()

def downsampling():
    ts = helpers.TuningSystem(swb_system, {})
    ts.tune_parameter(
        name="baseline_downsampled_bw.1s.drate",
        crnn_config=baseline_bw_tdnn,
        parameters=DRATES,
        transformation=set_downsampled_bw_training,
        training_args={
            "num_classes": None,
            "alignment": None
        },
        recognition_args={
            **RECOG_ARGS,
        },
        fast_bw_args={
            "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
            "fix_tdp_leaving_eps_arc": True,
            "normalize_lemma_sequence_scores": False,
        },
        epochs=[12, 24, 48, 120, 240, 300, 320],
        scorer_args={"prior_mixtures": None},
        reestimate_prior="transcription",
        dump_epochs=[4, 8, 12],
    )


# ts.tune_parameter(
#     name="baseline_downsampled_bw.1s.drate",
#     crnn_config=baseline_bw_tdnn,
#     parameters=DRATES,
#     transformation=set_downsampled_bw_training,
#     training_args={
#         "num_classes": None,
#         "alignment": None
#     },
#     recognition_args={
#         **RECOG_ARGS,
#     },
#     fast_bw_args={
#         "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
#         "fix_tdp_leaving_eps_arc": True,
#         "normalize_lemma_sequence_scores": False,
#     },
#     epochs=[12, 24, 48, 120, 240, 300, 320],
#     scorer_args={"prior_mixtures": None},
#     reestimate_prior="transcription",
#     dump_epochs=[4, 8, 12],
# )
