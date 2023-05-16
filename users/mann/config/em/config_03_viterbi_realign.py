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

RETURNN_REPOSITORY_URL = 'https://github.com/DanEnergetics/returnn.git'
BRANCH_SPRINT_FORCED_ALIGN = 'mann-rasr-fsa-forced-align'
from i6_core.tools import CloneGitRepositoryJob
returnn_root_job = CloneGitRepositoryJob(
    RETURNN_REPOSITORY_URL,
    branch=BRANCH_SPRINT_FORCED_ALIGN,
)
returnn_root_job.add_alias('returnn_forced_align')
RETURNN_FORCED_ALIGN = returnn_root_job.out_repository

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
baseline_bw.config.config["gradient_clip"] = 10.0

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/25, 0.0)

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

import copy
lbs_system.csp["crnn_train"] = copy.deepcopy(lbs_system.csp["crnn_train"])
lbs_system.csp["crnn_train"].corpus_config.segment_order_shuffle = True
lbs_system.csp["crnn_train"].corpus_config.segment_order_sort_by_time_length = True
lbs_system.csp["crnn_train"].corpus_config.segment_order_sort_by_time_length_chunk_size = 1000


baseline_bw_w_lr = copy.deepcopy(baseline_bw)
baseline_bw_w_lr.config.config["learning_rate"] = 0.00025
baseline_bw_w_lr.build_args["static_lr"] = 0.00025

from i6_experiments.users.mann.nn.learning_rates import get_learning_rates
baseline_bw_higher_lr = copy.deepcopy(baseline_bw_w_lr)
baseline_bw_higher_lr.config.config["learning_rates"] = get_learning_rates(
    inc_min_ratio=0.25, increase=70, decay=70
)

lbs_system.nn_and_recog(
    "baseline_bw_lstm",
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

#----------------------------- continued viterbi realign ------------------------------------------

from recipe.i6_experiments.users.mann.nn.preload import Preloader
lbs_system.plugins["preload"] = Preloader(system=lbs_system)

baseline_bw_continued = copy.deepcopy(baseline_bw_w_lr.config)

baseline_bw_continued_viterbi = copy.deepcopy(baseline_bw_continued)
net = baseline_bw_continued_viterbi.config["network"]
net["forced_align"] = {
    "class": "forced_align", "input_type": "log_prob",
    "topology": "sprint",
    "align_target": "sprint",
    "from": ["combine_prior"],
    "sprint_opts": net["fast_bw"]["sprint_opts"].copy()
}
net["output_viterbi"] = {
    "class": "copy", "from": "output",
    # "loss": "via_layer",
    # "loss_opts": {"align_layer": "forced_align", "loss_wrt_to_act_in": "softmax"},
    "loss": "ce", "target": "layer:forced_align"
}
net["output_bw"]["loss_scale"] = 0.0
baseline_bw_continued_viterbi.config["learning_rate_control_error_measure"] = "dev_score_output_viterbi"


baseline_bw_continued_viterbi_no_prior = copy.deepcopy(baseline_bw_continued_viterbi)
baseline_bw_continued_viterbi_no_prior.config["network"]["combine_prior"]["eval_locals"]["prior_scale"] = 0.0

configs = {
    "bw": baseline_bw_continued,
    "viterbi": baseline_bw_continued_viterbi,
    "viterbi.no-prior": baseline_bw_continued_viterbi_no_prior,
}

def set_newbob(config):
    # del config.config["learning_rates"]
    config.config["learning_rate"] = 2e-5

def set_low_oclr(config):
    config.config["learning_rates"] = get_learning_rates(inc_max_ratio=0.04, inc_min_ratio=0.04, increase=70, decay=70)

from i6_experiments.users.mann.experimental.extractors import LearningRateExtractorJob
def set_cont_lr(config):
    config.config["learning_rate"] = LearningRateExtractorJob(
        learning_rate_file=lbs_system.jobs["train"]["train_nn_baseline_bw_lstm"].out_learning_rates
    ).out_learning_rates[-1]

def set_small_warmup(config):
    lr = lambda epoch: 1e-5 + (4e-5 - 1e-5) * (epoch / 10)
    config.config["learning_rates"] = list(map(lr, range(11)))

lrs = {
    "newbob": set_newbob,
    "low-oclr": set_low_oclr,
    "cont-lr": set_cont_lr,
    "small-warmup": set_small_warmup,
}

for name, config in configs.items():
    lbs_system.nn_and_recog(
        name="cont_from_bw-{}".format(name),
        crnn_config=config,
        training_args={
            "num_classes": None,
            "alignment": None,
            "returnn_root": RETURNN_FORCED_ALIGN if "viterbi" in name else None,
        },
        plugin_args={"preload": {"base_training": ("baseline_bw_lstm", 160)}},
        compile_crnn_config="baseline_viterbi_lstm",
        fast_bw_args={
            # "acoustic_model_extra_config": tdp_model.to_acoustic_model_config()
            "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
            "fix_tdps_applicator": True,
            "fix_tdp_leaving_eps_arc": False,
        },
        epochs=[24, 48, 80, 160],
        # reestimate_prior='transcription',
        recognition_args={"extra_config": recog_extra_config}
        # epochs=[]
    )

    if name == "bw":
        for lr_name, lr_setter in lrs.items():
            bw_config = copy.deepcopy(config)
            lr_setter(bw_config)
            lbs_system.nn_and_recog(
                name="cont_from_bw-{}.{}".format(name, lr_name),
                crnn_config=bw_config,
                training_args={
                    "num_classes": None,
                    "alignment": None,
                    "returnn_root": RETURNN_FORCED_ALIGN if "viterbi" in name else None,
                },
                plugin_args={"preload": {"base_training": ("baseline_bw_lstm", 160)}},
                compile_crnn_config="baseline_viterbi_lstm",
                fast_bw_args={
                    # "acoustic_model_extra_config": tdp_model.to_acoustic_model_config()
                    "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
                    "fix_tdps_applicator": True,
                    "fix_tdp_leaving_eps_arc": False,
                },
                epochs=[24, 48, 80, 160],
                # reestimate_prior='transcription',
                recognition_args={"extra_config": recog_extra_config}
                # epochs=[]
            )
    
    lr_config = copy.deepcopy(config)
    set_newbob(lr_config)
    lbs_system.nn_and_recog(
        name="cont_from_bw-{}.newbob".format(name),
        crnn_config=lr_config,
        training_args={
            "num_classes": None,
            "alignment": None,
            "returnn_root": RETURNN_FORCED_ALIGN if "viterbi" in name else None,
        },
        plugin_args={"preload": {"base_training": ("baseline_bw_lstm", 160)}},
        compile_crnn_config="baseline_viterbi_lstm",
        fast_bw_args={
            # "acoustic_model_extra_config": tdp_model.to_acoustic_model_config()
            "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
            "fix_tdps_applicator": True,
            "fix_tdp_leaving_eps_arc": False,
        },
        epochs=[24, 48, 80, 160],
        # reestimate_prior='transcription',
        recognition_args={"extra_config": recog_extra_config}
        # epochs=[]
    )


#------------------------------------- collect scores ---------------------------------------------

lbs_system.init_dump_system(segments=[])
lbs_system.dump_system.init_score_segments()

score_key_map = {
    "viterbi": "dev_score_output_bw",
    "viterbi.no-prior": "dev_score_output_bw",
    "bw": "dev_score",
    "baseline_bw": "dev_score",
}

data = {}

for key in ["bw.newbob", "viterbi.newbob", "viterbi.no-prior.newbob", "baseline_bw_lstm"]:
    if key != "baseline_bw_lstm":
        name = "cont_from_bw-{}".format(key)
        key_reduced = key[:-len(".newbob")]
    else:
        name = key
        key_reduced = key[:-len("_lstm")]
    print(name, key_reduced)
    lbs_system.dump_system.score(
        name, epoch=160,
        returnn_config=None if key != "baseline_bw_lstm" else baseline_bw_higher_lr.config,
        training_args={
            "num_classes": None,
            "alignment": None,
            "returnn_root": RETURNN_FORCED_ALIGN if "viterbi" in name else None,
        },
        fast_bw_args={
            "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
            "fix_tdps_applicator": True,
            "fix_tdp_leaving_eps_arc": False,
        },
    )

    data[key_reduced] = {
        "wer": lbs_system.get_wer(name, epoch=160),
        "score": lbs_system.dump_system.scores[name][score_key_map[key_reduced]],
    }

def dump_summary(data):
    import tabulate as tab
    table = [
        [name, value["wer"], value["score"]] for name, value in data.items()
    ]
    out = tab.tabulate(table, headers=["Name", "WER [%]", "Score"])
    print("Dumped summary")
    with open(f"output/{fname}/summary.txt", "w") as f:
        f.write(out)

tk.register_callback(dump_summary, data)
