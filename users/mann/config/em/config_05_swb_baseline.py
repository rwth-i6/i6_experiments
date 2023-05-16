from sisyphus import *

import os
import copy

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, RecognitionConfig, ConfigBuilder
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups import prior
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

baseline_bw = builder.build()
print(type(baseline_bw))
del baseline_bw.config["learning_rate"]


from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/60, 0.0)
swb_system.compile_configs["baseline_lstm"] = swb_system.baselines["viterbi_lstm"]()
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None
    },
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": False,
        "fix_tdps_applicator": True,
    },
    recognition_args={"extra_config": lbs.custom_recog_tdps()},
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="CRNN",
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

tinas_recog_config = RecognitionConfig(
    lm_scale=3.0,
    beam_pruning=22.0,
    prior_scale=0.3,
    tdp_scale=0.1
)

#---------------------------------- tinas baseline ------------------------------------------------

# baseline_tina = copy.deepcopy(baseline_bw.config)
baseline_tina = copy.deepcopy(baseline_bw)
baseline_tina.am_scale = 0.3
baseline_tina.prior_scale = 0.1
baseline_tina.tdp_scale = 0.1

swb_system.run_exp(
    name='baseline_tina',
    crnn_config=baseline_tina,
    exp_config=exp_config,
    dump_epochs=[4, 8, 12, 300]
)

swb_system.run_exp(
    name="baseline_tina.exact_setup",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(
        fast_bw_args={"normalize_lemma_sequence_scores": False},
        training_args={
            "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
            "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
        }
    ),
    dump_epochs=[12, 300],
)

gs.DEFAULT_ENVIRONMENT_SET["CUDA_VISIBLE_DEVICES"] = "0"
swb_system.clean(
    "baseline_tina",
    sorted(set([4, 8, 12, 300] + exp_config.epochs)),
    cleaner_args={
        "gpu": 0,
        # "returnn_root": "/u/mann/dev/returnn"
    }
)

# decode with tinas setup
swb_system.run_exp(
    name="baseline_tina.exact_setup.tina_recog",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(
        fast_bw_args={"normalize_lemma_sequence_scores": False},
        training_args={
            "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
            "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
        },
        recognition_args=tinas_recog_config.to_dict(),
        scorer_args={"prior_scale": 0.3},
    ),
    epochs=[300],
    dump_epochs=[300],
)


tdp_model_tina = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
swb_system.run_exp(
    name="baseline_tina.tdps.recog",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(
        fast_bw_args={
            "normalize_lemma_sequence_scores": False,
            "acoustic_model_extra_config": tdp_model_tina.to_acoustic_model_config(),
        },
        training_args={
            "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
            "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
        },
        recognition_args=tinas_recog_config.replace(
            tdps=CombinedModel.legacy()
        ).to_dict(),
        scorer_args={"prior_scale": 0.3}
    ),
    reestimate_prior="transcription"
)

swb_system.run_exp(
    name="baseline_tina.tdps.recog.new_rasr",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(
        fast_bw_args={
            "normalize_lemma_sequence_scores": False,
            "acoustic_model_extra_config": tdp_model_tina.to_acoustic_model_config(),
        },
        training_args={
            "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
            "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
        },
        recognition_args=tinas_recog_config.replace(
            tdps=CombinedModel.legacy(),
            lm_scale=3.1,
        ).to_dict(),
        scorer_args={"prior_scale": 0.3}
    ),
    reestimate_prior="transcription",
    epochs=[300],
)


# mega exactly the same
extra_bw_config = rasr.RasrConfig()
extra_bw_config[
    "neural-network-trainer"
    ".alignment-fsa-exporter"
    ".alignment-fsa-exporter"
    ".model-combination"
    ".acoustic-model"
    ".fix-tdp-leaving-epsilon-arc"
] = True
more_updates = {
    "min_learning_rate": 1e-5,
    "learning_rate": 0.0003,
}
deletions = ["update_on_device", "window"]
baseline_tina_updates = copy.deepcopy(baseline_tina)
baseline_tina_updates.config.update(more_updates)
for deletion in deletions:
    del baseline_tina_updates.config[deletion]

swb_system.run_exp(
    name="baseline_tina.more_exact_setup",
    crnn_config=baseline_tina_updates,
    exp_config=exp_config.extend(
        fast_bw_args={
            "normalize_lemma_sequence_scores": False,
            "fix_tdps_applicator": None,
            "extra_config": extra_bw_config
        },
        training_args={
            "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
            "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
        }
    ),
    dump_epochs=[12, 300],
)

#---------------------------------- fh decode -----------------------------------------------------

from recipe.i6_experiments.users.raissi.setups.librispeech.search.factored_hybrid_search import FHDecoder, ContextEnum, ContextMapper

decoder = FHDecoder(
    name="baseline_tina.tdps.recog.fh",
    search_crp=swb_system.crp["dev"],
    context_type=ContextEnum.monophone,
    context_mapper=ContextMapper(),
    feature_path=swb_system.feature_flows["dev"]["gt"],
    model_path=swb_system.nn_checkpoints["train"]["baseline_tina.tdps.recog"][300],
    graph=swb_system.jobs["train"]["compile_returnn_{}".format("baseline_tina.tdps.recog")].out_graph,
    mixtures=swb_system.mixtures["train"],
    eval_files=None,
    tensor_mapping=FHDecoder.TensorMapping(center_state_posteriors="output")
)


#---------------------------------- baseline with pretrain ----------------------------------------

# baseline_bw_pretrain = copy.deepcopy(baseline_bw)
# baseline_bw_pretrain.config.config["learning_rate"] = 0.00025
# baseline_bw_pretrain.build_args["static_lr"] = True
# baseline_bw_pretrain.config.config["learning_rates"] = learning_rates.get_learning_rates(
#     inc_min_ratio=0.25, increase=120, decay=120
# )

# baseline_bw_pretrain_conservative = copy.deepcopy(baseline_bw)
# baseline_bw_pretrain_conservative.build_args["static_lr"] = True
# lrs = learning_rates.get_learning_rates(increase=120, decay=120)
# baseline_bw_pretrain_conservative.config.config["learning_rate"] = lrs[0]
# baseline_bw_pretrain_conservative.config.config["learning_rates"] = lrs

def train_w_pretrain(exact_setup=False):
    swb_system.run_exp(
        name='baseline_bw_pretrain',
        crnn_config=baseline_bw_pretrain,
        exp_config=exp_config,
        dump_epochs=[4, 8, 12, 300]
    )

    swb_system.run_exp(
        name='baseline_bw_pretrain_tina_lr',
        crnn_config=baseline_bw_pretrain_conservative,
        exp_config=exp_config,
        dump_epochs=[4, 8, 12, 300]
    )

    if not exact_setup:
        return

    swb_system.run_exp(
        name="baseline_bw_pretrain.exact_setup",
        crnn_config=baseline_bw_pretrain,
        exp_config=exp_config.extend(
            fast_bw_args={"normalize_lemma_sequence_scores": False},
            training_args={
                "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
                "returnn_root": tk.Path("/u/raissi/dev/returnn_packages/returnn")
            }
        ),
    )

# train_w_pretrain()

def clean(gpu=False):
    for name in swb_system.nn_config_dicts["train"]:
        swb_system.clean(
            name,
            sorted(set([4, 8, 12, 300] + exp_config.epochs)),
            cleaner_args={ "gpu": int(gpu), }
        )

#------------------------------------ mono-dense --------------------------------------------------

# mega exactly the same
swb_system.set_state_tying(
    value="monophone-no-tying-dense",
    extra_args={
        "use-boundary-classes": False,
        "use-word-end-classes": True,
    }
)

swb_system.crp["crnn_train"].corpus_config["segment-order-sort-by-time-length-chunk-size"] = 300

assert builder.system is swb_system
print(type(swb_system.num_classes()))
print(swb_system.num_classes())
old_num_classes = swb_system.num_classes()

allo_file = swb_system.get_allophone_file()
for crp in swb_system.crp.values():
    crp.acoustic_model_config.allophones.add_all = False
    crp.acoustic_model_config.allophones.add_from_file = allo_file
    crp.acoustic_model_config.allophones.add_from_lexicon = True

# rebuild because num_outputs changed
assert swb_system.prior_system.eps == 0.0
swb_system.prior_system.eps = 1e-6
swb_system.prior_system.extract_prior()
baseline_tina = builder.build()

print(id(swb_system))
print(id(builder.system))
# print(old_num_classes)
# print(swb_system.num_classes())
# print(builder.system.num_classes())
# print(baseline_tina.config["network"]["output"]["n_out"])
# assert old_num_classes is baseline_tina.config["network"]["output"]["n_out"]

tdp_model_tina = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
swb_system.run_exp(
    name="baseline_tina.tdps.recog.we",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(
        fast_bw_args={
            "normalize_lemma_sequence_scores": False,
            "acoustic_model_extra_config": tdp_model_tina.to_acoustic_model_config(),
            "fix_tdp_leaving_eps_arc": True,
            "fix_tdps_applicator": False,
        },
        training_args={
            # "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
            # "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn"),
            "mem_rqmt": 48,
        },
        recognition_args=tinas_recog_config.replace(
            tdps=CombinedModel.legacy()
        ).to_dict(),
        scorer_args={"prior_scale": 0.3,},
    ),
    reestimate_prior="transcription",
)

