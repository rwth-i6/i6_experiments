from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.common.datasets import librispeech
from recipe.i6_experiments.users.mann.setups.dump import HdfDumpster

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
# s = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

SEGMENTS = [
    "train-clean-100/1034-121119-0089/1",
    "train-clean-100/3607-29116-0015/1",
    "train-clean-100/911-128684-0043/1",
    "train-clean-100/4640-19189-0020/1"   
]

#------------------------- make viterbi baseline ---------------------------------------------------------------------------

import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
lbs_system = lbs.get_legacy_librispeech_system()
hdf_dumper = HdfDumpster(lbs_system, SEGMENTS)

baseline_viterbi = lbs_system.baselines["viterbi_lstm"]()

print(baseline_viterbi.post_config)

print(baseline_viterbi)

print(baseline_viterbi.config)

lbs_system.nn_and_recog(
    name="baseline_viterbi_lstm",
    crnn_config=baseline_viterbi,
    training_args={"keep_epochs": [12, 24, 32, 48, 80, 160]},
    epochs=[]
)

#--------------------------------- make bw baseline -----------------------------------------------

from recipe.i6_experiments.users.mann.experimental.statistics import AlignmentStatisticsJob
from recipe.i6_core.lexicon.allophones import StoreAllophonesJob
s = lbs_system
allophones = StoreAllophonesJob(s.crp["train"])
align_stats = AlignmentStatisticsJob(
    s.alignments["train"]["init_align"],
    allophones.out_allophone_file,
    s.csp["train"].segment_path.hidden_paths,
    s.csp["train"].concurrent
)
TOTAL_FRAMES = 36107903
tk.register_output("stats", align_stats.counts)
baseline_bw = lbs_system.baselines["bw_lstm_fixed_prior_job"](TOTAL_FRAMES)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/25, 0.0)
print(type(tdp_model.to_acoustic_model_extra_config()))
# assert isinstance(tdp_model.to_acoustic_model_config(), rasr.RasrConfig)

"""
Try also the following recognition parameters:
[flf-lattice-tool.network.recognizer.acoustic-model.tdp.*]
exit    = 0.0
forward = 0.105
loop    = 2.303
skip    = infinity

[flf-lattice-tool.network.recognizer.acoustic-model.tdp.silence]
exit    = 0.01
forward = 3.507
loop    = 0.03
skip    = infinity
"""

recog_extra_config = rasr.RasrConfig()
recog_extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp"] = rasr.RasrConfig()
recog_extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp"]["*"] = rasr.ConfigBuilder({})(
    exit    = 0.0,
    forward = 0.105,
    loop    = 2.303,
    skip    = "infinity"
)
recog_extra_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp"]["silence"] = rasr.ConfigBuilder({})(
    exit    = 0.01,
    forward = 3.507,
    loop    = 0.03,
    skip    = "infinity",
)


bw_baseline_exp = ExpConfig(
    baseline_bw,
    training_args={
        "num_classes": lambda s: s.num_classes(),
    },
    # plugin_args={
    #     "tdps": {"model": CombinedModel(SimpleTransitionModel.from_fwd_probs(3/8, 1/25), 0.0)}
    # },
    plugin_args={},
    compile_crnn_config="baseline_viterbi_lstm"
)

import copy
lbs_system.csp["crnn_train"] = copy.deepcopy(s.csp["crnn_train"])
lbs_system.csp["crnn_train"].corpus_config.segment_order_shuffle = True
lbs_system.csp["crnn_train"].corpus_config.segment_order_sort_by_time_length = True
lbs_system.csp["crnn_train"].corpus_config.segment_order_sort_by_time_length_chunk_size = 1000

baseline_bw.config.config["gradient_clip"] = 10.0

baseline_bw_w_lr = copy.deepcopy(baseline_bw)
baseline_bw_w_lr.config.config["learning_rate"] = 0.00025
baseline_bw_w_lr.build_args["static_lr"] = 0.00025

lbs_system.nn_and_recog(
    "baseline_bw_lstm",
    crnn_config=baseline_bw_w_lr,
    training_args={
        "num_classes": None,
        "alignment": None
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args={
        # "acoustic_model_extra_config": tdp_model.to_acoustic_model_config()
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
    },
    epochs=[80, 160],
    recognition_args={"extra_config": recog_extra_config},
    # reestimate_prior="transcription",
    # epochs=[]
)

from i6_experiments.users.mann.nn.learning_rates import get_learning_rates
baseline_bw_higher_lr = copy.deepcopy(baseline_bw_w_lr)
baseline_bw_higher_lr.config.config["learning_rates"] = get_learning_rates(
    inc_min_ratio=0.25, increase=70, decay=70
)

lbs_system.nn_and_recog(
    "baseline_bw_lstm.higher-lr",
    crnn_config=baseline_bw_higher_lr,
    training_args={
        "num_classes": None,
        "alignment": None
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args={
        # "acoustic_model_extra_config": tdp_model.to_acoustic_model_config()
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
    },
    epochs=[80, 160],
    # reestimate_prior='transcription',
    recognition_args={"extra_config": recog_extra_config}
    # epochs=[]
)

default_exp_config = ExpConfig(
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
    reestimate_prior="CRNN",
)

baseline_bw_no_clip = copy.deepcopy(baseline_bw_higher_lr)
baseline_bw_no_clip.config.config["gradient_clip"] = 0
lbs_system.run_exp(
    "baseline_bw_lstm.higher-lr.no-clip",
    crnn_config=baseline_bw_no_clip,
    exp_config=default_exp_config,
)

if False:
    # try diff optimizer
    baseline_bw_neural_opt = copy.deepcopy(baseline_bw_w_lr)
    baseline_bw_neural_opt.config.config["optimizer"] = {"class": "neural_opt"}
    # lbs_system.nn_and_recog(
    #     "baseline_bw_lstm.higher-lr",
    #     crnn_config=baseline_bw_higher_lr,
    #     training_args={
    #         "num_classes": None,
    #         "alignment": None
    #     },
    #     compile_crnn_config="baseline_viterbi_lstm",
    #     fast_bw_args={
    #         # "acoustic_model_extra_config": tdp_model.to_acoustic_model_config()
    #         "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
    #         "fix_tdps_applicator": True,
    #     },
    #     epochs=[80, 160],
    #     reestimate_prior='transcription'
    #     # epochs=[]
    # )

#-------------------------------------- povey prior -----------------------------------------------

baseline_bw_povey = lbs_system.baselines["bw_lstm_povey_prior"]()
baseline_bw_povey.config.config["learning_rate"] = 0.00025
baseline_bw_povey.build_args["static_lr"] = 0.00025

lbs_system.run_exp(
    "baseline_bw_lstm.povey",
    crnn_config=baseline_bw_povey,
    exp_config=default_exp_config,
)


#------------------------------- scale tuning -----------------------------------------------------

from i6_experiments.users.mann.experimental.helpers import TuningSystem
ts = TuningSystem(lbs_system, {})

def set_scales(config, scales):
    config.am_scale, config.prior_scale, config.tdp_scale = scales

scale_config = copy.deepcopy(baseline_bw.config)

# from collections import ChainMap
from itertools import product, chain
scale_configs = product(
    [0.1, 0.3, 0.5], # am
    [0.1, 0.3, 0.5], # prior
    [0.5],      # tdp
)

extra_scale_configs = product(
    [0.07, 0.1],
    [0.01, 0.05, 0.07],
    [0.5],
)

extra_extra_scale_configs = product(
    [0.1],
    [0.001, 0.005,],
    [0.5],
)

extra_config = rasr.RasrConfig()
extra_config["flf-lattice-tool.network.recognizer.recognizer.acoustic-lookahead-temporal-approximation-scale"] = 10

scale_tuning_config = copy.deepcopy(baseline_bw.config)
del scale_tuning_config.config["gradient_clip"]

ts.tune_parameter(
    "scales",
    crnn_config=scale_config,
    parameters=chain(scale_configs, extra_scale_configs, extra_extra_scale_configs),
    # parameters=[next(chain(scale_configs, extra_scale_configs, extra_extra_scale_configs))],
    transformation=set_scales,
    training_args={
        "num_classes": None,
        "alignment": None,
        "num_epochs": 16,
        "time_rqmt": 8
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args=default_exp_config.fast_bw_args,
    recognition_args={
        "extra_config": extra_config,
        "search_parameters": { "beam-pruning": 14.0 }
    },
    epochs=[12, 16],
    # epochs=[],
    optimize=False,
    reestimate_prior='transcription'
)


#----------------------------------------- try scale warmup ---------------------------------------

from i6_experiments.users.mann.nn import pretrain
warmup_config: pretrain.PretrainConfigHolder = copy.deepcopy(baseline_bw)
warmup_config.warmup.initial_am = 0.1
warmup_config.build_args["static_lr"] = False
assert not warmup_config.build_args.get("static_lr", False)
assert not warmup_config.config.config.get("learning_rate", False)

def set_rel_prior(config: pretrain.PretrainConfigHolder, rel_prior: float):
    config.prior_scale = rel_prior

ts.tune_parameter(
    "rel_prior",
    crnn_config=warmup_config,
    parameters=[0.05, 0.1, 0.5, 0.7],
    transformation=set_rel_prior,
    training_args={
        "num_classes": None,
        "alignment": None,
        "num_epochs": 160,
        "time_rqmt": 48
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args=default_exp_config.fast_bw_args,
    reestimate_prior='transcription',
    epochs=[12, 80, 160]
)

ts.tune_parameter(
    "rel_prior",
    crnn_config=warmup_config,
    parameters=[0.05, 0.1, 0.5, 0.7],
    transformation=set_rel_prior,
    training_args={
        "num_classes": None,
        "alignment": None,
        "num_epochs": 160,
        "time_rqmt": 48
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args=default_exp_config.fast_bw_args,
    parameter_representation=lambda x: "{}_recog_tdps".format(x),
    reestimate_prior='transcription',
    epochs=[160],
    recognition_args={"extra_config": recog_extra_config}
)

#------------------------------- feed-forward baseline --------------------------------------------

baseline_ffnn = lbs_system.baselines["bw_ffnn_fixed_prior"]()
baseline_ffnn.build_args["static_lr"] = 0.001

lbs_system.run_exp(
    name="baseline_ffnn.fixed_prior",
    crnn_config=baseline_ffnn,
    exp_config=default_exp_config,
    compile_crnn_config=None,
)

print(default_exp_config.training_args["num_classes"])

acoustic_model_key = ".".join([
    "acoustic-model-trainer",
    "aligning-feature-extractor",
    "feature-extraction",
    "alignment",
    "model-combination",
    "acoustic-model",
])
align_config = rasr.RasrConfig()
align_config[acoustic_model_key] = tdp_model.to_acoustic_model_config()

lbs_system.nn_align(
    name="baseline_ffnn",
    crnn_config=baseline_ffnn.config,
    nn_name="baseline_ffnn.fixed_prior",
    epoch=160,
    extra_config=align_config,
    scorer_suffix="-prior"
)

hdf_dump = hdf_dumper.forward(
    "baseline_ffnn.fixed_prior",
    baseline_ffnn.config,
    160, 
    training_args=default_exp_config.training_args,
    fast_bw_args=default_exp_config.fast_bw_args,
    corpus="crnn_dev"
)
tk.register_output("bw_dumps/ffnn.hdf", hdf_dump["fast_bw"])

from recipe.i6_experiments.users.mann.experimental.plots import PlotSoftAlignmentJob
from recipe.i6_core.lexicon.allophones import StoreAllophonesJob, DumpStateTyingJob

state_tying = DumpStateTyingJob(lbs_system.crp["train"]).out_state_tying

allophones = StoreAllophonesJob(
    lbs_system.crp["train"],
).out_allophone_file


plot_job = PlotSoftAlignmentJob(
    hdf_dump["fast_bw"],
    alignment=lbs_system.alignments["train"]["baseline_ffnn"].alternatives["bundle"],
    allophones=allophones,
    # bliss_corpus=lbs_system.crp["train"].corpus_config.file,
    # bliss_lexicon=lbs_system.crp["train"].lexicon_config.file,
    state_tying=state_tying,
    segments=SEGMENTS,
    occurrence_thresholds=(0.1, 0.05)
)

for seg, path in plot_job.out_plots.items():
    tk.register_output("bw_plots/{}/{}.png".format("ffnn_alignment", seg.replace("/", "\\")), path)


hdf_dumps = hdf_dumper.init_hdf_dataset(
    name="ffnn_align",
    dump_args={
        "num_classes": lbs_system.num_classes(),
        "alignment": ("train", "baseline_ffnn", -1),
        "corpus": "train",
    },
)

# shift align
from i6_experiments.users.mann.experimental.alignment import ShiftAlign, TransformAlignmentJob

j = TransformAlignmentJob(ShiftAlign(20, lbs_system.silence_idx()), hdf_dumps)
tk.register_output("hdf_dumps/ffnn_align.shift-20.hdf", j.out_alignment)



# clean

def clean(gpu=False):
    keep_epochs = default_exp_config.epochs
    for name in lbs_system.nn_config_dicts["train"]:
        lbs_system.clean(
            name, keep_epochs if not name.startswith("scales") else [12, 16],
            cleaner_args={ "gpu": int(gpu), }
        )
