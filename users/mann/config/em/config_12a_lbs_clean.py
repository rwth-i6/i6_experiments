from sisyphus import *

import os
import copy
import itertools

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, ConfigBuilder, RecognitionConfig
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups.nn_system import common
from recipe.i6_experiments.users.mann.setups import prior
from i6_experiments.users.mann.nn import specaugment, learning_rates
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
from i6_core import rasr
from recipe.i6_core import tools

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

lbs_system = lbs.get_libri_1k_system()

lbs_system = common.init_system(
    "lbs",
    state_tying_args=dict(
        value="monophone-no-tying-dense",
        use_boundary_classes=False,
        use_word_end_classes=True,
    ),
    extract_features=False,
    # extend_train_corpus=True,
)
del lbs_system.crp["dev"].acoustic_model_config.tdp

lbs.init_segment_order_shuffle(lbs_system)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
PRIOR_TDP_MODEL = CombinedModel.from_fwd_probs(3/8, 1/60, 0.0)
lbs_system.compile_configs["baseline_lstm"] = lbs_system.baselines["viterbi_lstm"]()
exp_config = ExpConfig(
    compile_crnn_config=lbs_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None
    },
    fast_bw_args={
        "acoustic_model_extra_config": PRIOR_TDP_MODEL.to_acoustic_model_config(),
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
        "normalize_lemma_sequence_scores": False,
    },
    recognition_args={"extra_config": lbs.custom_recog_tdps()},
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="CRNN",
    # dump_epochs=[12, 300],
)

lbs_system.init_dump_system(
    segments=[
        "librispeech/8465-246943/0014",
        "librispeech/8465-246947/0008",
        "librispeech/8425-292520/0000",
        "librispeech/8425-292520/0005",
    ],
    occurrence_thresholds=(0.1, 0.05),
)

TDP_SCALES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TDP_REDUCED = [0.1, 0.5, 1.0]
PRIOR_SCALES = [0.1, 0.3, 0.5, 0.7]

#---------------------------------- tinas baseline ------------------------------------------------

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    ConfigBuilder(lbs_system)
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_1K)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_transcription_prior()
    .set_specaugment()
)

# configure multi gpu
def multi_gpu(config):
    config.config.update(
        horovod_reduce_type="param",
        horovod_param_sync_time_diff=120,
        horovod_dataset_distribution="random_seed_offset"
    )

builder.transforms.append(multi_gpu)

#--------------------------------- train tdps -----------------------------------------------------

NO_TDP_MODEL = CombinedModel.zeros()
# set up returnn repository
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

print("Returnn root: ", lbs_system.returnn_root)
print("Returnn python exe: ", lbs_system.returnn_python_exe)

from recipe.i6_experiments.users.mann.nn import preload, tdps

#-------------------------------------- prior less training ---------------------------------------

# word end classes
lbs_system.set_state_tying(
    value="monophone-no-tying-dense",
    use_word_end_classes=True,
)

NEW_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard"
CORR_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard"

# extended corpus
extra_args = lbs.init_extended_train_corpus(lbs_system)
PRIOR_MODEL_TINA = CombinedModel.from_fwd_probs(3/9, 1/25, 0.0)
epochs = [24, 48, 128, 256]
epochs += [512 - e for e in epochs[:-1]] + [512]
# num_epochs = 260
# epochs = [20, 40, 80, 160, 240, 260]
default_recognition_config = RecognitionConfig(
    tdps=CombinedModel.legacy(),
    beam_pruning=22,
    prior_scale=0.3,
    tdp_scale=0.1,
    lm_scale=3.0,
)
exp_config = ExpConfig(
    compile_crnn_config=lbs_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "returnn_root": RETURNN_TDPS,
        "mem_rqmt": 32,
        "partition_epoch": {"train": 40},
        "save_interval": 8,
        # "multi_node_slots": 2,
        "horovod_num_processes": 2,
        **extra_args
    },
    fast_bw_args={
        "acoustic_model_extra_config": PRIOR_MODEL_TINA.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
        "corpus": extra_args["feature_corpus"],
    },
    recognition_args=RecognitionConfig(
        tdps=CombinedModel.legacy(),
        beam_pruning=22,
        prior_scale=0.3,
        tdp_scale=0.1,
        lm_scale=3.0,
    ).to_dict(),
    epochs=epochs,
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
    alt_training=True,
)

from i6_experiments.users.mann.setups.nn_system.factored_hybrid import FactoredHybridDecoder, TransitionType
lbs_system.set_decoder("fh", FactoredHybridDecoder())

# baseline_configs
baseline = builder.set_transcription_prior().build()
baseline_no_prior = builder.set_no_prior().build()
# baseline_no_prior.config["gradient_clip"] = 10

baselines = {
    # "with_prior": baseline,
    "no_prior": baseline_no_prior,
}

from collections import OrderedDict
DATA = []

def make_row(arch, tdp_scale, name):
    row = OrderedDict()
    row["arch"] = arch
    row["tdp"] = tdp_scale
    for e in exp_config.epochs:
        row[e] = lbs_system.get_wer(name, e)
    return row

for name, config in baselines.items():
    lbs_system.run_exp(
        name="baseline_{}".format(name),
        crnn_config=config,
        exp_config=exp_config,
    )
    DATA.append(make_row("fixed", 0.1, "baseline_{}".format(name)))
    scaled_config = copy.deepcopy(config)
    scaled_config.tdp_scale = 0.3
    lbs_system.run_exp(
        name="baseline_{}.tdp-0.3".format(name),
        crnn_config=scaled_config,
        exp_config=exp_config.replace(
            recognition_args=default_recognition_config.to_dict(_full_tdp_config=True)
        ),
    )
    DATA.append(make_row("fixed", 0.3, "baseline_{}.tdp-0.3".format(name)))

    extra_config = rasr.RasrConfig()
    extra_config["*"].python_home = "/work/tools/asr/python/3.8.0"
    extra_config["*"].python_program_name = "/work/tools/asr/python/3.8.0/bin/python3.8"
    
    extra_config["*"].fix_allophone_context_at_word_boundaries         = True
    extra_config["*"].transducer_builder_filter_out_invalid_allophones = True
    extra_config["*"].allow_for_silence_repetitions   = False 
    extra_config["*"].applicator_type = "corrected"

    extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"] \
        = PRIOR_MODEL_TINA.to_acoustic_model_config()
    tdp, prior = 0.1, 0.5
    extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.tdp.scale"] = tdp
    extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.mixture-set.priori-scale"] = prior

    from i6_experiments.users.mann.experimental.statistics.alignment import ComputeTseJob
    data = []
    for nn_name in ["baseline_{}".format(name), "baseline_{}.tdp-0.3".format(name)]:
        lbs_system.nn_align(
            nn_name=nn_name,
            epoch=512,
            use_gpu=True,
            feature_corpus="train_magic",
            extra_config=extra_config,
            evaluate=True
        )
        alignment = lbs_system.alignments["train"][nn_name + "-512"].alternatives["bundle"]
        tse_job = ComputeTseJob(
            alignment, lbs_system.alignments["train"]["init_align"], lbs_system.get_allophone_file()
        )
        row = {
            "tdp": 0.1 if "tdp" not in nn_name else 0.3,
            "WER [%]": lbs_system.get_wer(nn_name, 512),
            "TSE": tse_job.out_tse,
        }
        data.append(row)
    lbs_system.report("alignments", data)
    print("Aligned")

reductions = dict(
    # substate_and_silence={"type": "factorize", "n_subclasses": 3, "div": 2, "silence_idx": lbs_system.silence_idx()},
    # substate={"type": "factorize", "n_subclasses": 3, "div": 2},
    speech_silence={"type": "speech_silence", "silence_idx": lbs_system.silence_idx()},
)

init_args = {
    "speech_fwd": 1/3,
    "silence_fwd": 1/25,
    "silence_idx": lbs_system.silence_idx(),
}

arch_args = {
    "n_subclasses": 3,
    "div": 2,
    "silence_idx": lbs_system.silence_idx()
}


# tmp_config = copy.deepcopy(baseline_no_prior)
# tdps.get_model(num_classes=lbs_system.num_classes(), arch="label", reduce=reductions["substate_and_silence"]).set_config(tmp_config)

print("Tdps substate")
from pprint import pprint
# pprint(tmp_config.config["network"]["tdps"])


tdp_exp_config = exp_config.extend(
    training_args={
        "returnn_root": RETURNN_TDPS,
    },
    fast_bw_args={
        "acoustic_model_extra_config": NO_TDP_MODEL.to_acoustic_model_config(),
        "corpus": extra_args["feature_corpus"],
    }
)

from i6_experiments.users.mann.experimental.util import _RelaxedOverwriteConfig
from i6_experiments.users.mann.setups.clean import LatticeCleaner
import tabulate as tab
from collections import defaultdict

cleaner = LatticeCleaner(fname)
name = "baseline_no_prior.tdp-0.3"
epoch = 24
print(lbs_system.jobs["dev"].keys())
j = lbs_system.jobs["dev"]["recog_crnn-{}-{}-prior".format(name, epoch)]
cleaner.clean(j, lbs_system.get_wer(name, 48))

name = "no_prior"
config = baselines[name]
# archs = ["label"]
# archs = []
archs = ["label", "blstm_large", "label_speech_silence", "label_substate_and_silence"]
init_type = "smart"
base_config = copy.deepcopy(config)
dump_epoch_map = defaultdict(
    list,
    {
        ("blstm_large", 0.1): [128, 256],
        ("label", 0.1): [48, 128],
        ("label_substate_and_silence", "single_gpu"): [256, 384],
        ("label_speech_silence", "single_gpu"): [512],
    }
)
base_config.config["gradient_clip"] = 10
base_config.tdp_scale = 0.3

def main(archs=None):
    if archs is None:
        archs = ["label", "blstm_large", "label_speech_silence", "label_substate_and_silence"]
    for arch in archs:
        tmp_config = copy.deepcopy(base_config)
        model = tdps.get_model(
            num_classes=lbs_system.num_classes(),
            arch=arch,
            extra_args=arch_args,
            init_args={
                "type": init_type,
                **init_args
            }
        )
        model.set_config(tmp_config)

        for tdp_scale in [0.1, 0.3]:
            scaled_config = copy.deepcopy(tmp_config)
            scaled_config.tdp_scale = tdp_scale
            dump_epochs = dump_epoch_map[(arch, tdp_scale)]
            name="baseline_tdps.random.{}.arch-{}.tdp-{}".format(init_type, arch, tdp_scale)
            lbs_system.run_exp(
                name="baseline_tdps.random.{}.arch-{}.tdp-{}".format(init_type, arch, tdp_scale),
                crnn_config=scaled_config,
                exp_config=tdp_exp_config.replace(
                    # dump_epochs=dump_epochs,
                    alt_decoding={
                        # "epochs": [240, 300],
                        "epochs": [],
                        # "epochs": dump_epochs,
                        "compile": True,
                        "flf_tool_exe": CORR_FLF_TOOL,
                        "extra_compile_args": {
                            "returnn_root": RETURNN_TDPS if arch == "blstm_large" and init_type == "smart" else None,
                        }
                    },
                ).extend(
                    scorer_args={
                        "prior_scale": 0.3,
                        "fwd_loop_scale": 0.1,
                    },
                ),
            )

            for e in dump_epochs:
                lbs_system.run_decode(
                    name=name,
                    type="fh",
                    epoch=e,
                    exp_config=tdp_exp_config.replace(
                        recognition_args=default_recognition_config.to_dict(_full_tdp_config=True),
                    ),
                    decoding_args={
                        "compile": True,
                        "flf_tool_exe": CORR_FLF_TOOL,
                        "extra_compile_args": {
                            "returnn_root": RETURNN_TDPS if arch == "blstm_large" and init_type == "smart" else None,
                        }
                    },
                    scorer_args={
                        "prior_scale": 0.3,
                        "fwd_loop_scale": 0.1,
                    },
                    extra_suffix="-fh",
                )

            row = OrderedDict()
            row["arch"] = arch
            row["tdp"] = tdp_scale
            for e in tdp_exp_config.epochs:
                row[e] = lbs_system.get_wer("baseline_tdps.random.{}.arch-{}.tdp-{}".format(init_type, arch, tdp_scale), e)
            
            DATA.append(row)

        name="baseline_tdps.random.{}.arch-{}.single_gpu".format(init_type, arch)
        dump_epochs=dump_epoch_map[(arch, "single_gpu")]
        lbs_system.run_exp(
            name="baseline_tdps.random.{}.arch-{}.single_gpu".format(init_type, arch),
            crnn_config=tmp_config,
            exp_config=tdp_exp_config.extend(
                training_args={
                    "horovod_num_processes": None,
                    "partition_epochs": {"train": 20},
                },
            ).replace(
                # dump_epochs=dump_epoch_map[(arch, "single_gpu")],
                alt_decoding={
                    "epochs": [],
                    "compile": True,
                    "flf_tool_exe": CORR_FLF_TOOL,
                    "extra_compile_args": {
                        "returnn_root": RETURNN_TDPS if arch == "blstm_large" and init_type == "smart" else None,
                    }
                },
            ).extend(
                scorer_args={
                    "prior_scale": 0.3,
                    "fwd_loop_scale": 0.1,
                },
            ),
        )
        for e in dump_epochs:
            lbs_system.run_decode(
                name=name,
                type="fh",
                epoch=e,
                exp_config=tdp_exp_config.replace(
                    recognition_args=default_recognition_config.to_dict(_full_tdp_config=True),
                ),
                decoding_args={
                    "compile": True,
                    "flf_tool_exe": CORR_FLF_TOOL,
                    "extra_compile_args": {
                        "returnn_root": RETURNN_TDPS if arch == "blstm_large" and init_type == "smart" else None,
                    }
                },
                scorer_args={
                    "prior_scale": 0.3,
                    "fwd_loop_scale": 0.1,
                },
                extra_suffix="-fh",
            )

from i6_experiments.users.mann.setups.util import P
def test_init():
    baseline_builder = builder.copy()
    del builder.transforms[-1]
    base_config = builder.set_no_prior().build()
    arch = "label_speech_silence"
    inits = [
        # emission model, transition model
        ("pretrained", "smart", 0.3),
        ("random", "smart", 0.3),
        ("random", "flat", 0.3),
        ("random", "flat", 0.1),
        ("random", "random", 0.3),
    ]
    inits_ = P(arch) * P(*inits) \
        + P("label", "ffnn") * P(inits[0][:-1]) * P(0.1, 0.3)
    for arch, em_init, tm_init, tdp_scale in inits_:
        tmp_config = copy.deepcopy(base_config)
        # for k in list(tmp_config.keys()):
        #     if "horovod" in k:
        #         del tmp_config[k]

        if em_init == "pretrained":
            preload.set_preload(lbs_system, tmp_config, (extra_args["feature_corpus"], "baseline_no_prior", 16))
        else:
            assert em_init == "random"


        model = tdps.get_model(
            num_classes=lbs_system.num_classes(),
            arch=arch,
            extra_args=arch_args,
            init_args={
                "type": tm_init,
                **init_args
            }
        )
        model.set_config(tmp_config)

        scaled_config = copy.deepcopy(tmp_config)
        scaled_config.tdp_scale = tdp_scale

        # name="baseline_tdps.random.{}.arch-{}.single_gpu".format(init_type, arch)
        name="baseline_tdps.{}.{}.arch-{}.tdp-{}.single_gpu".format(em_init, tm_init, arch, tdp_scale)
        dump_epochs=dump_epoch_map[(arch, "single_gpu")]
        lbs_system.run_exp(
            name=name,
            crnn_config=scaled_config,
            exp_config=tdp_exp_config.extend(
                training_args={
                    "horovod_num_processes": None,
                    "partition_epochs": {"train": 20},
                },
            ).replace(
                # dump_epochs=dump_epoch_map[(arch, "single_gpu")],
                alt_decoding={
                    "epochs": [],
                    "compile": True,
                    "flf_tool_exe": CORR_FLF_TOOL,
                    "extra_compile_args": {
                        "returnn_root": RETURNN_TDPS if arch == "blstm_large" and tm_init == "smart" else None,
                    }
                },
            ).extend(
                scorer_args={
                    "prior_scale": 0.3,
                    "fwd_loop_scale": 0.1,
                },
            ),
        )


def score():
    main()
    # lbs_system.init_dump_system(
    #     segments=segments
    # )
    lbs_system.init_report_system(fname)
    lbs_system.dump_system.init_score_segments()
    exp_names = [
        ("baseline_no_prior.tdp-0.3", 512),
        ("baseline_tdps.random.smart.arch-blstm_large.tdp-0.1", 128),
        ("baseline_tdps.random.smart.arch-label_speech_silence.single_gpu", 512),
    ]
    targs = copy.deepcopy(tdp_exp_config.training_args)
    del targs["partition_epoch"]
    targs["horovod_num_processes"] = None
    data = []
    for name, epoch in exp_names:

        # epoch = 200

        fast_bw_args = copy.deepcopy(exp_config.fast_bw_args) if "no_prior" in name \
            else copy.deepcopy(tdp_exp_config.fast_bw_args)
        fast_bw_args.update(
            fix_tdps_applicator=True,
            fix_tdp_leaving_eps_arc=False,
        )
        score_config = copy.deepcopy(lbs_system.nn_config_dicts["train"][name])
        score_config.tdp_scale = 0.3

        lbs_system.dump_system.score(
            name=name,
            epoch=epoch,
            returnn_config=score_config,
            training_args=targs,
            fast_bw_args=fast_bw_args,
        )
        data.append(
            {
                "name": name,
                # "wer": lbs_system.get_wer(name, epoch),
                "score": lbs_system.dump_system.scores[name]["train_score"] \
                    if "baseline_no_prior" in name \
                    else lbs_system.dump_system.scores[name]["train_score_output_bw"]
            }
        )
    
    lbs_system.report("tdp_scores", data)


def align():

    # CORR_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard"
    # CORR_ALIGN_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/align.linux-x86_64-standard"
    CORR_ALIGN_TOOL = tk.Path("/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard")
    lbs_system.crp["train"].acoustic_model_trainer_exe = CORR_ALIGN_TOOL
    main(archs=["blstm_large", "label_speech_silence"])
    exp_names = [
        ("baseline_tdps.random.smart.arch-blstm_large.tdp-0.1", 128),
        ("baseline_tdps.random.smart.arch-label_speech_silence.single_gpu", 512),
    ]

    extra_config = rasr.RasrConfig()
    extra_config["*"].python_home = "/work/tools/asr/python/3.8.0"
    extra_config["*"].python_program_name = "/work/tools/asr/python/3.8.0/bin/python3.8"
    
    extra_config["*"].fix_allophone_context_at_word_boundaries         = True
    extra_config["*"].transducer_builder_filter_out_invalid_allophones = True
    extra_config["*"].allow_for_silence_repetitions   = False 
    extra_config["*"].applicator_type = "corrected"

    extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.state-tying"].type = "no-tying-dense"

    extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"] \
        = CombinedModel.zeros(speech_skip="infinity").to_acoustic_model_config()
    
    print(lbs_system.feature_scorers)

    for exp, epoch in exp_names:
        name=exp + "-{}-fh".format(epoch)
        feature_scorer = lbs_system.feature_scorers["train_magic"][name]
        lbs_system.nn_align(
            nn_name=exp,
            name=name,
            epoch=300,
            flow=lbs_system.decoder.decoders[exp][epoch]("train").featureScorerFlow,
            feature_scorer=feature_scorer,
            evaluate=True,
            extra_config=extra_config,
            use_gpu=True,
        )

def debug(train_name, epoch):
    # dump hdfs
    dump_config = copy.deepcopy(lbs_system.nn_config_dicts["train"][train_name])
    dump_config.config["learning_rates"] = dump_config.config["learning_rates"][epoch:]

    net = dump_config.config["network"]
    for name, layer in [
        ("tdp_grads", "fast_bw/tdps"),
        ("tdps_out", "output_tdps")
    ]:
        net[name] = {"class": "expand_dims", "from": layer, "axis": "T",}
        net["dump_" + name] = {"class": "hdf_dump", "filename": name + ".hdf", "from": name, "is_output_layer": True}

    preload.set_preload(lbs_system, dump_config, ("train_magic", train_name, epoch))

    lbs_system.run_exp(
        name="debug.{}".format(train_name),
        crnn_config=dump_config,
        exp_config=tdp_exp_config.extend(
            training_args={
                "num_epochs": 8,
                "horovod_num_processes": None,
                "partition_epochs": {"train": 20},
            },
        ).replace(
            dump_epochs=[],
            epochs=[],
        ),
    )

def run_debug():
    main(archs=["label", "label_substate_and_silence", "blstm_large"])
    align()

    train_name = "baseline_tdps.random.smart.arch-label_substate_and_silence.single_gpu"
    epoch = 336
    debug(train_name, epoch)

    train_name = "baseline_tdps.random.smart.arch-label.tdp-0.1"
    epoch = 112
    last_good_epoch = 48

    # newbob from last good
    cont_config = copy.deepcopy(lbs_system.nn_config_dicts["train"][train_name])
    max_lr = cont_config.config.pop("learning_rates")[last_good_epoch]
    cont_config.config["learning_rate"] = max_lr
    preload.set_preload(lbs_system, cont_config, ("train_magic", train_name, last_good_epoch))

    lbs_system.run_exp(
        name="debug.continued_w_newbob",
        crnn_config=cont_config,
        exp_config=tdp_exp_config
    )

    # diff peak lrs
    get_lrs = lambda peak_lr: learning_rates.get_learning_rates(
        increase=225, decay=225, dec_max_ratio=peak_lr, inc_max_ratio=peak_lr,
    )

    base_lr = 0.001
    print("max_lr: ", max_lr)
    print("max_ratio: ", max_lr / base_lr)

    for peak_lr in [0.05, 0.1, 0.3]:
        lr_config = copy.deepcopy(lbs_system.nn_config_dicts["train"][train_name])
        lr_config.config["learning_rates"] = get_lrs(peak_lr)
        lbs_system.run_exp(
            name="debug.peak_lr-{}".format(peak_lr * base_lr),
            crnn_config=lr_config,
            exp_config=tdp_exp_config
        )
    
    # diff gradient clip
    cont_config = copy.deepcopy(lbs_system.nn_config_dicts["train"][train_name])
    max_lr = cont_config.config.pop("learning_rates")[epoch:]
    cont_config.config["learning_rate"] = max_lr
    preload.set_preload(lbs_system, cont_config, ("train_magic", train_name, epoch))

    del cont_config.config["gradient_clip"]
    cont_config.config["gradient_clip_norm"] = 10
    lbs_system.run_exp(
        name="debug.continued_w_clip_norm",
        crnn_config=cont_config,
        exp_config=tdp_exp_config
    )

    # diff optimizers
    optims = [
        {
            "class": "RMSPropOptimizer",
        },
        {
            "class": "NeuralOptimizer1",
        },
        {
            "class": "Adamax",
        }
    ]

    for optim in optims:
        optim_config = copy.deepcopy(lbs_system.nn_config_dicts["train"][train_name])
        optim_config.config["optimizer"] = optim
        lbs_system.run_exp(
            name="debug.optim-{}".format(optim["class"]),
            crnn_config=optim_config,
            exp_config=tdp_exp_config
        )


    
def dump_tdps():
    from i6_experiments.users.mann.nn.inspect import InspectTFCheckpointJob
    arch = "label_substate_and_silence"
    name="baseline_tdps.random.{}.arch-{}.single_gpu".format(init_type, arch)
    data = []
    step = 24
    epochs = [step * i for i in range(1, 384 // step + 1)]
    for e in epochs:
        print("epoch", e)
        lbs_system.run_decode(
            name=name,
            type="fh",
            epoch=e,
            exp_config=tdp_exp_config.replace(
                recognition_args=default_recognition_config.replace(
                    altas=8.0,
                    beam_pruning=16,
                    beam_pruning_threshold=10000,
                    extra_args={"use_gpu": True},
                ).to_dict(_full_tdp_config=True),
            ),
            decoding_args={
                "compile": True,
                "flf_tool_exe": CORR_FLF_TOOL,
                "extra_compile_args": {
                    "returnn_root": RETURNN_TDPS if arch == "blstm_large" and init_type == "smart" else None,
                }
            },
            scorer_args={
                "prior_scale": 0.3,
                "fwd_loop_scale": 0.1,
            },
            extra_suffix="-fh-quick",
        )

        row = {}
        row["epoch"] = e
        row["wer"] = lbs_system.get_wer(name, e, extra_suffix="-fh-quick")

        j = InspectTFCheckpointJob(
            checkpoint=lbs_system.nn_checkpoints["train_magic"][name][e],
            tensor_name="tdps/base_vars/base_vars",
            returnn_python_exe=None,
            returnn_root=RETURNN_TDPS,
        )
        row["tdps"] = j.out_tensor_file
        data.append(row)
    
    from i6_experiments.users.mann.setups.reports import eval_tree
    def dump(data):
        import pickle
        with open(os.path.join("output", fname, "tdps.pkl"), "wb") as f:
            pickle.dump(eval_tree(data), f)
    
    tk.register_callback(dump, data)


from i6_experiments.users.mann.setups.reports import eval_tree
def make_table():
    return tab.tabulate(eval_tree(DATA), headers="keys", tablefmt="presto")

tk.register_report(
    os.path.join(fname, "summary", "tdps.txt"),
    values=make_table,
)

def all():
    test_init()
    align()
    score()
    dump_tdps()

#-------------------------------------- cleanup ---------------------------------------------------

def clean(gpu=False):
    main()
    main()
    run_debug()
    test_init()
    keep_epochs = sorted(set(
        exp_config.epochs + [4, 8]
    ))
    for name in lbs_system.nn_config_dicts["train"]:
        try:
            lbs_system.clean(
                name, keep_epochs,
                exec_immediately=False,
                cleaner_args={ "gpu": int(gpu), }
            )
        except KeyError:
            lbs_system.clean(
                name, keep_epochs,
                # exec_immediately=True,
                exec_immediately=False,
                cleaner_args={ "gpu": int(gpu), },
                feature_corpus="train_magic"
            )

