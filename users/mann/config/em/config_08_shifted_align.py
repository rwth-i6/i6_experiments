from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, RecognitionConfig
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

import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
lbs_system = lbs.get_legacy_librispeech_system()
lbs.init_segment_order_shuffle(lbs_system)
hdf_dumper = HdfDumpster(lbs_system, SEGMENTS)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/25, 0.0)
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
    recognition_args={"extra_config": lbs.custom_recog_tdps()},
    # recognition_args=RecognitionConfig(
    #     tdps=CombinedModel.legacy(),
    # ).to_dict(),
    reestimate_prior="CRNN",
)

# ffnn align
baseline_ffnn = lbs_system.baselines["bw_ffnn_fixed_prior"]()
baseline_ffnn.build_args["static_lr"] = 0.001

lbs_system.run_exp(
    name="baseline_ffnn.fixed_prior",
    crnn_config=baseline_ffnn,
    exp_config=default_exp_config,
    compile_crnn_config=None,
)

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

hdf_dumps = hdf_dumper.init_hdf_dataset(
    name="ffnn_align",
    dump_args={
        "num_classes": lbs_system.num_classes(),
        "alignment": ("train", "baseline_ffnn", -1),
        "corpus": "train",
    },
)

# shift align
from i6_experiments.users.mann.experimental.alignment import ShiftAlign, TransformAlignmentJob, MoveSilence
from i6_experiments.users.mann.setups.nn_system.trainer import HdfAlignTrainer

align_transformers = {
    "shift-20": ShiftAlign(20, lbs_system.silence_idx()),
    "shift-10": ShiftAlign(10, lbs_system.silence_idx()),
    "roll-silence": MoveSilence(lbs_system.silence_idx()),
}

hdf_align = {}

for key, transformer in align_transformers.items():
    j = TransformAlignmentJob(transformer, hdf_dumps)
    tk.register_output("hdf_dumps/ffnn_align.{}.hdf".format(key), j.out_alignment)
    hdf_align[key] = j.out_alignment

baseline_viterbi = lbs_system.baselines["viterbi_lstm"]()
del baseline_viterbi.config["chunking"]

default_viterbi_config = ExpConfig(
    training_args={
        "num_classes": lbs_system.num_classes(),
        "alignment": None
    },
    compile_crnn_config="baseline_viterbi_lstm",
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
    },
    epochs=[80, 160],
    recognition_args={"extra_config": lbs.custom_recog_tdps()},
    # recognition_args=RecognitionConfig(
    #     tdps=CombinedModel.legacy(),
    # ).to_dict(),
    reestimate_prior="CRNN",
)

lbs_system.set_trainer(HdfAlignTrainer(lbs_system))


lbs_system.init_dump_system(segments=[])
lbs_system.dump_system.init_score_segments()

wers = {}
scores = {}

lbs_system.nn_and_recog(
    name="viterbi_lstm.base",
    crnn_config=baseline_viterbi, 
    training_args={
        # "hdf_alignment": hdf_dumps,
        # "alignment": None,
        "num_classes": lbs_system.num_classes(),
        # "hdf_classes_key": "classes",
        "alignment": ("train", "baseline_ffnn", -1),
        # "hdf_features": hdf_dumps,
    },
    compile_crnn_config=None,
    epochs=[80, 160],
    # alt_training=True,
)
wers["base"] = lbs_system.get_wer("viterbi_lstm.base", 160)

lbs_system.dump_system.score(
    name="viterbi_lstm.base",
    epoch=160,
    returnn_config=baseline_viterbi,
    training_args={
        "num_classes": lbs_system.num_classes(),
        "alignment": ("train", "baseline_ffnn", -1),
    }
)
scores["base"] = lbs_system.dump_system.scores["viterbi_lstm.base"]["dev_score"]

for key, align in hdf_align.items():
    name="viterbi_lstm.{}".format(key)
    lbs_system.nn_and_recog(
        name=name,
        crnn_config=baseline_viterbi, 
        training_args={
            "hdf_alignment": align,
            "alignment": None,
            "num_classes": lbs_system.num_classes(),
        },
        compile_crnn_config=None,
        epochs=[80, 160],
        alt_training=True,
    )
    lbs_system.dump_system.score(
        name=name,
        epoch=160,
        returnn_config=baseline_viterbi,
        training_args={
            "hdf_alignment": align,
            "alignment": None,
            "num_classes": lbs_system.num_classes(),
        },
        alt_training=True,
    )
    wers[key] = lbs_system.get_wer(name, 160)
    scores[key] = lbs_system.dump_system.scores[name]["dev_score"]

from collections import OrderedDict

key_map = OrderedDict([
    ("base", "None"),
    ("shift-10", "Shift by 10 frames"),
    ("shift-20", "Shift by 20 frames"),
    ("roll-silence", "Move silence to end"),
])

def dump_summary(wers, scores):
    from pylatex import Tabular
    tab = Tabular("l|r|r")
    tab.add_hline()
    tab.add_row(("Align. Trafo", "WER [%]", "Score"))
    tab.add_hline()
    for key, value in key_map.items():
        tab.add_row((value, wers[key].get(), "{:.2f}".format(scores[key].get())))
    tab.add_hline()
    print(tab.dumps())

tk.register_callback(dump_summary, wers, scores)

#------------------------------------- clean up models --------------------------------------------

def clean(gpu=False):
    keep_epochs = [80, 160] 
    for name in lbs_system.nn_config_dicts["train"]:
        lbs_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )
