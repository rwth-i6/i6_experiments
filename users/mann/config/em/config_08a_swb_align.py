from sisyphus import *

import os
import copy

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, RecognitionConfig
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.common.datasets import librispeech
from recipe.i6_experiments.users.mann.setups.dump import HdfDumpster
from recipe.i6_experiments.users.mann.experimental.util import safe_crp

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
# s = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
import recipe.i6_experiments.users.mann.setups.nn_system.common as common
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
lbs_system = lbs.get_libri_1k_system()
swb_system = swb.get_bw_switchboard_system()
for binary in ["rasr_binary_path", "native_ops_path", "returnn_python_exe", "returnn_python_home", "returnn_root"]:
    setattr(swb_system, binary, getattr(lbs_system, binary))
lbs.init_segment_order_shuffle(swb_system)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
default_exp_config = ExpConfig(
    training_args={
        "num_classes": None,
        "alignment": None
    },
    compile_crnn_config="baseline_viterbi_lstm",
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    recognition_args=RecognitionConfig(
        lm_scale=3.0,
        beam_pruning=22,
    ).to_dict(),
    reestimate_prior="CRNN",
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

def gmm_setup(system):
    system.filter_segments(None, swb.extra_alignment_logs)

def tuske_setup(system):
    for crp in system.crp.values():
        crp.acoustic_model_config.allophones.add_all = True
        crp.acoustic_model_config.allophones.add_from_lexicon = False

def setup(system, align):
    return {
        "init_gmm": gmm_setup,
        "tuske": tuske_setup,
    }.get(align, lambda x: None)(system)

swb_system.crp["train"].concurrent = 1

# print(swb_system.crp["train"].segment_path)

# swb_system.set_state_tying(
#     value="monophone-no-tying-dense",
#     extra_args={
#         "use-boundary-classes": False,
#         "use-word-end-classes": True,
#     }
# )
alignments = ["init_gmm", "tuske"]
# alignments = ["init_gmm"] #, "tuske"]

#-------------------------------------- make baseline ---------------------------------------------

viterbi_compile_config = swb_system.baselines["viterbi_lstm"]()

def run_baselines(align, chunking=False):
    baseline_viterbi = swb_system.baselines["viterbi_lstm"]()
    if not chunking:
        del baseline_viterbi.config["chunking"]
    
    with safe_crp(swb_system):
        setup(swb_system, align)
        swb_system.run_exp(
            "baseline_viterbi.chunking-{}.align-{}".format(chunking, align),
            crnn_config=baseline_viterbi,
            exp_config=(
                default_exp_config
                .extend(
                    training_args={
                        "alignment": ("train", align, -1),
                        "num_classes": swb_system.num_classes(),
                    }
                ).replace(
                    compile_crnn_config=viterbi_compile_config,
                )
            )
        )

for align in alignments:
    for chunking in [False, True]:
        run_baselines(align, chunking)

#-------------------------------------- dump hdf --------------------------------------------------

common.rm_segment_order_shuffle(swb_system)

def run_shift(alignment):
    hdf_dumps = swb_system.dump_system.init_hdf_dataset(
        name=alignment,
        dump_args={
            "num_classes": swb_system.num_classes(),
            "alignment": ("train", alignment, -1),
            "corpus": "train",
        },
    )

    # shift align
    from i6_experiments.users.mann.experimental.alignment import (
        ShiftAlign,
        TransformAlignmentJob,
        MoveSilence,
        ReplaceSilenceByLastSpeech,
        SqueezeSpeech,
    )
    from i6_experiments.users.mann.setups.nn_system.trainer import HdfAlignTrainer

    align_transformers = {
        # "shift-20": ShiftAlign(20, swb_system.silence_idx()),
        # "shift-10": ShiftAlign(10, swb_system.silence_idx()),
        "roll-silence": MoveSilence(swb_system.silence_idx()),
        "last-speech": ReplaceSilenceByLastSpeech(swb_system.silence_idx()),
        "squeeze-speech": SqueezeSpeech(swb_system.silence_idx()),
    }

    for repeat in [1, 2]:
        align_transformers["squeeze-rep-{}".format(repeat)] = SqueezeSpeech(swb_system.silence_idx(), repeat=repeat)
    
    for offset in [0, 0.5, 1]:
        align_transformers["squeeze-shift-{}".format(offset)] = SqueezeSpeech(swb_system.silence_idx(), offset=offset)

    hdf_align = {}

    for key, transformer in align_transformers.items():
        j = TransformAlignmentJob(transformer, hdf_dumps)
        tk.register_output("hdf_dumps/{}.{}.hdf".format(alignment, key), j.out_alignment)
        hdf_align[key] = j.out_alignment

    baseline_viterbi_chunk = swb_system.baselines["viterbi_lstm"]()
    baseline_viterbi = copy.deepcopy(baseline_viterbi_chunk)
    del baseline_viterbi.config["chunking"]

    swb_system.set_trainer(HdfAlignTrainer(lbs_system))

    for key, align in hdf_align.items():
        swb_system.run_exp(
            name="viterbi_lstm.align-{}.{}".format(alignment, key),
            crnn_config=baseline_viterbi, 
            exp_config=(
                default_exp_config
                .extend(
                    training_args={
                        "alignment": None,
                        "hdf_alignment": align,
                        "num_classes": swb_system.num_classes(),
                    }
                ).replace(
                    compile_crnn_config=viterbi_compile_config,
                )
            ),
            alt_training=True,
        )

def all():
    alignments = ["init_gmm"]
    for align in alignments:
        with safe_crp(swb_system):
            setup(swb_system, align)
            run_shift(align)

#------------------------------------- clean up models --------------------------------------------

def clean(gpu=False):
    keep_epochs = [80, 160] 
    for name in swb_system.nn_config_dicts["train"]:
        swb_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )

