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

#--------------------------------------- train tdps -----------------------------------------------

from i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from recipe.i6_experiments.users.mann.nn import preload, tdps
from i6_core import rasr
from recipe.i6_core import tools

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


#--------------------------------------- configurations -------------------------------------------

# word end classes
swb_system.set_state_tying(
    value="monophone-no-tying-dense",
    extra_args={
        "use-boundary-classes": False,
        "use-word-end-classes": True,
    }
)

swb_system.prior_system.eps = 1e-12
swb_system.prior_system.extract_prior()

swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/master-rasr-fsa/src/Tools/Flf/flf-tool.linux-x86_64-standard"
NEW_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard"
CORR_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard"
CORR_AM_TRAINER = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard"

# extended corpus
extra_args = swb.init_extended_train_corpus(swb_system)
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
        "alignment": None,
        "keep_epochs": [12, 24, 48, 120, 240, 300],
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
    alt_training=True,
)

tdp_exp_config = exp_config.extend(
    training_args={
        "returnn_root": RETURNN_TDPS,
    },
    fast_bw_args={
        "acoustic_model_extra_config": NO_TDP_MODEL.to_acoustic_model_config(),
        "corpus": extra_args["feature_corpus"],
    }
)

init_args = {
    # "n_subclasses": 3,
    "speech_fwd": 1/3,
    "silence_fwd": 1/40,
    "silence_idx": swb_system.silence_idx(),
}

arch_args = {
    "n_subclasses": 3,
    "div": 2,
    "silence_idx": swb_system.silence_idx()
}

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
builders = {}

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    ConfigBuilder(swb_system)
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_no_prior()
    .set_specaugment()
)

exp_config.compile_crnn_config = builder.build_compile_config()

def del_learning_rate(config):
    del config.config["learning_rate"]

builder.transforms.append(del_learning_rate)

#--------------------------------- init configs ---------------------------------------------------

ARCHS = ["tdnn", "ffnn"]

TDPS = [
    ("label_speech_silence", "flat"),
    # ("label_substate_and_silence", "flat"),
    ("label", "flat"),
    # ("blstm_large", "random"),
]


for arch in ["tdnn", "ffnn"]:
    builders[arch] = b = getattr(builder.copy(), f"set_{arch}")()
    config = b.build()
    configs[arch] = config


#---------------------------------- compare with different prior scales ---------------------------

def main():
    for arch in ARCHS:
        tmp_config = copy.deepcopy(configs[arch])

        compile_config = builders[arch].build_compile_config()

        for tdp_scale in [0.1, 0.3]:
            tmp_config.tdp_scale = tdp_scale
            swb_system.run_exp(
                f"encoder-{arch}.fixed.tdp-{tdp_scale}",
                crnn_config=tmp_config,
                exp_config=exp_config.replace(
                    compile_crnn_config=compile_config
                ),
                epochs=[240, 300]
            )

        for tdp_arch, tm_init in TDPS:

            # tmp_config.config["gradient_clip_norm"] = 10.0
            tmp_config.config["gradient_clip_norm"] = 100

            tdp_model = tdps.get_model(
                num_classes=swb_system.num_classes(),
                arch=tdp_arch,
                extra_args=arch_args,
                init_args={"type": tm_init, **init_args}
            )
            tdp_model.set_config(tmp_config)

            for tdp_scale in [0.1, 0.3]:
                tmp_config.tdp_scale = tdp_scale
                name = f"encoder-{arch}.arch-{tdp_arch}.tdp-{tdp_scale}"
                swb_system.run_exp(
                    name,
                    crnn_config=tmp_config,
                    exp_config=tdp_exp_config.replace(
                        compile_crnn_config=compile_config,
                    )
                    # epochs=[]
                )

def debug():
    pass

def all():
    pass

def py():
    main()

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
