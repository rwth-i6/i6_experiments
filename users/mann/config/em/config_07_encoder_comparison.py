from sisyphus import *

import os
import copy

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import (
    NNSystem,
    ExpConfig,
    RecognitionConfig,
    ConfigBuilder,
    AlignmentConfig
)
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups import prior
from i6_experiments.users.mann.nn.config import make_baseline as make_tdnn_baseline
from i6_experiments.users.mann.nn import specaugment, learning_rates
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
from i6_core import rasr

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

lbs_system = lbs.get_libri_1k_system()
swb_system = swb.get_bw_switchboard_system()
for binary in ["rasr_binary_path", "native_ops_path", "returnn_python_exe", "returnn_python_home", "returnn_root"]:
    setattr(swb_system, binary, getattr(lbs_system, binary))
lbs.init_segment_order_shuffle(swb_system)

baseline_bw = swb_system.baselines['bw_lstm_tina_swb']()
specaugment.set_config(baseline_bw.config)

from i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
# tdp_model = CombinedModel.from_fwd_probs(3/8, 1/60, 0.0)
tdp_model_tina = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
tinas_recog_config = RecognitionConfig(
    lm_scale=3.0,
    beam_pruning=22.0,
    prior_scale=0.3,
    tdp_scale=0.1
)
swb_system.compile_configs["baseline_lstm"] = swb_system.baselines["viterbi_lstm"]()
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None
    },
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model_tina.to_acoustic_model_config(),
        "normalize_lemma_sequence_scores": False,
        # "fix_tdps_applicator": True,
    },
    # recognition_args={"extra_config": lbs.custom_recog_tdps()},
    recognition_args=tinas_recog_config.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
    # reestimate_prior=False,
    dump_epochs=[12, 300],
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

configs = {}

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    ConfigBuilder(swb_system)
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_transcription_prior()
    .set_specaugment()
)

def del_learning_rate(config):
    del config.config["learning_rate"]

builder.transforms.append(del_learning_rate)

#--------------------------------- init configs ---------------------------------------------------

for arch in ["lstm", "tdnn", "ffnn"]:
    config = getattr(builder, f"set_{arch}")().build()
    configs[arch] = config

#---------------------------------- compare with different prior scales ---------------------------

from i6_experiments.users.mann.experimental import helpers

ts = helpers.TuningSystem(swb_system, {})

for arch in ["lstm", "tdnn", "ffnn"]:
    config = copy.deepcopy(configs[arch])
    for prior_scale in [0.0, 0.1, 0.3]:
        tmp_config = copy.deepcopy(config)
        tmp_config.prior_scale = prior_scale
        swb_system.run_exp(
            f"{arch}.prior_scale-{prior_scale}",
            crnn_config=tmp_config,
            exp_config=exp_config.replace(
                compile_crnn_config=None
            ) if arch in ["tdnn", "ffnn"] else exp_config,
            epochs=[300]
        )

#-------------------------------------- recognition tuning ----------------------------------------

from i6_experiments.users.mann.experimental.tuning import RecognitionTuner

TDP_SCALES = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
tuner = RecognitionTuner(swb_system, tdp_scales=TDP_SCALES)
tuner.prior_scales.append(0)
tuner.output_dir = fname

tuning_config = RecognitionConfig(
    lm_scale=3.0,
    tdps=CombinedModel.from_fwd_probs(3/9, 1/40, 20.0)
)

tunings = []
wers = {}

for arch in ["lstm", "tdnn", "ffnn"]:
    config = copy.deepcopy(configs[arch])
    wers[arch] = pwer = {}
    for prior_scale in [0.0, 0.1]:
        tmp_config = copy.deepcopy(config)
        tmp_config.prior_scale = prior_scale
        name = f"{arch}.prior_scale-{prior_scale}"
        tmp_exp_config = exp_config.replace(
            compile_args={"graph": name},
            reestimate_prior="transcription",
            epochs=[300],
        )
        # pwer[prior_scale] = tuner.tune(
        #     f"{arch}.prior_scale-{prior_scale}",
        #     epoch=300,
        #     returnn_config=tmp_config,
        #     recognition_config=tuning_config,
        #     exp_config=tmp_exp_config,
        #     optimum=True,
        # )


#------------------------------------- compare alignments -----------------------------------------

from i6_experiments.users.mann.experimental.statistics.alignment import ComputeTseJob

optima = {
    ("lstm", 0.0): (0.1, 0.3),
    ("lstm", 0.1): (0.1, 0.5),
    ("tdnn", 0.0): (0.1, 0.5),
    ("tdnn", 0.1): (0.1, 0.3),
    ("ffnn", 0.0): (0.1, 0.7),
    ("ffnn", 0.1): (0.1, 0.5),
}

align_eval = {}
tses = {}

viterbi_config = swb_system.baselines["viterbi_lstm"]()
viterbi_exp_config = lambda align: ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": swb_system.num_classes(),
        "alignment": align,
    },
    recognition_args=tinas_recog_config.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
)

from i6_experiments.users.mann.setups.nn_system.plugins import FilterAlignmentPlugin
swb_system.plugins["filter"] = FilterAlignmentPlugin(swb_system, swb.init_nn_args["dev_size"])

for arch in ["lstm", "tdnn", "ffnn"]:
    config = copy.deepcopy(configs[arch])
    for prior_scale in [0.0, 0.1]:
        # if arch == "lstm" and prior_scale == 0.0: continue
        tmp_config = copy.deepcopy(config)
        tmp_config.prior_scale = prior_scale
        name = f"{arch}.prior_scale-{prior_scale}"
        tmp_exp_config = exp_config.replace(
            compile_args={"graph": name}
        )
        opt_tdp, opt_prior = optima[(arch, prior_scale)]
        print(opt_tdp)
        align_config = AlignmentConfig(
            tdp_scale=opt_tdp,
            prior_scale=opt_prior,
            tdps=CombinedModel.from_fwd_probs(3/9, 1/40, 0.0),
        )
        align_name = f"{arch}.prior_scale-{prior_scale}"
        stats = swb_system.nn_align(
            align_name,
            epoch=300,
            graph=name,
            extra_config=align_config.config,
            evaluate=True
        )
        align_eval[(arch, prior_scale)] = stats["total_silence"] / stats["total_states"]

        # if (arch, prior_scale) != ("ffnn", 0.0): continue
        alignment = swb_system.alignments["train"][f"{arch}.prior_scale-{prior_scale}-300"].alternatives["bundle"]
        tse_job = ComputeTseJob(alignment, swb_system.alignments["train"]["init_gmm"], swb_system.get_allophone_file())
        tses[(arch, prior_scale)] = tse_job.out_tse

        # viterbi train
        swb_system.run_exp(
            name=f"viterbi.align-{arch}.prior_scale-{prior_scale}",
            crnn_config=viterbi_config,
            exp_config=viterbi_exp_config(align_name + "-300"),
            plugin_args={"filter": {}},
        )

stats = swb_system.evaluate_alignment("init_gmm", "train", alignment_logs=swb.extra_alignment_logs)
align_eval["gmm"] = stats["total_silence"] / stats["total_states"]

# viterbi train
swb_system.run_exp(
    name=f"viterbi.gmm",
    crnn_config=viterbi_config,
    exp_config=viterbi_exp_config("init_gmm"),
    plugin_args={"filter": {"alignment_logs": swb.extra_alignment_logs}},
)
    
def dump_summary(wers, align_eval, tses):
    print(tses)
    from pylatex import Tabular, MultiRow
    tab = Tabular("l|c|c|c|c")
    tab.add_row(["Encoder", "Prior", "WER [%]", "Silence [%]", "TSE"])
    tab.add_hline()
    tab.add_row(["GMM", "", "", f"{100 * align_eval['gmm'].get():.2f}", ""])
    for arch in ["lstm", "tdnn", "ffnn"]:
        arch_str = MultiRow(2, data=arch.upper())
        for prior_scale in [0.0, 0.1]:
            w = wers[arch].get(prior_scale, "-")
            a = align_eval.get((arch, prior_scale), "-")
            tse = tses[(arch, prior_scale)]
            try:
                a = round(a.get() * 100, 1)
            except Exception:
                pass
            tab.add_row([arch_str if prior_scale == 0.0 else "", prior_scale, w, a, f"{tse.get():.2f}"])
        tab.add_hline()
    print(tab.dumps())
    for (arch, prior_scale), tse in tses.items():
        print(f"{arch} {prior_scale} {tse.get():.2f}")
    tab.generate_tex(os.path.join("output", fname, "summary"))

# tk.register_callback(dump_summary, wers, align_eval, tses)


#-------------------------------------- better viterbi --------------------------------------------

viterbi_builder = builder.copy().set_loss("viterbi")
viterbi_config = (
    viterbi_builder
    .copy()
    .delete("chunking")
    .set_no_prior()
    .set_ce_args(focal_loss_factor=None)
    .build()
)


def all():
    pass

#-------------------------------------- clean up models -------------------------------------------

def clean(gpu=False):
    keep_epochs = sorted(set(
        exp_config.epochs + [4, 8]
    ))
    for name in swb_system.nn_config_dicts["train"]:
        swb_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )
