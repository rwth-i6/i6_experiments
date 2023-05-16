from sisyphus import *

import os, sys
import copy
import itertools

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, ConfigBuilder, RecognitionConfig
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.common as common
from i6_experiments.users.mann.setups.reports import TableReport
from recipe.i6_experiments.users.mann.setups import prior
from i6_experiments.users.mann.nn import specaugment, learning_rates
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
from i6_core import rasr
from recipe.i6_core import tools

sys.setrecursionlimit(2500)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

swb_system = swb.get_bw_switchboard_system()
swb_system = common.init_system(
    "swb",
    state_tying_args=dict(
        value="monophone-no-tying-dense",
        use_boundary_classes=False,
        use_word_end_classes=True,
    )
)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr

#---------------------------------- tinas baseline ------------------------------------------------

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    swb_system.default_builder
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_transcription_prior()
    .set_specaugment()
)
builder.register("fullsum_w_prior")

builder.set_no_prior().register("fullsum_no_prior")

#--------------------------------- train tdps -----------------------------------------------------

NO_TDP_MODEL = CombinedModel.zeros()
# set up returnn repository
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

from recipe.i6_experiments.users.mann.nn import preload, tdps

#------------------------------------- updated experiments ----------------------------------------

swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/master-rasr-fsa/src/Tools/Flf/flf-tool.linux-x86_64-standard"
NEW_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard"
CORR_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard"
CORR_AM_TRAINER = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard"

PRIOR_MODEL_TINA = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
default_recognition_args = RecognitionConfig(
    tdps=CombinedModel.legacy(),
    beam_pruning=22,
    prior_scale=0.3,
    tdp_scale=0.1,
    lm_scale=3.0,
)
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "returnn_root": RETURNN_TDPS,
        "mem": 32,
        # **extra_args
    },
    fast_bw_args={
        "acoustic_model_extra_config": PRIOR_MODEL_TINA.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
        "corpus": "train_magic",
    },
    recognition_args=default_recognition_args.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
    # dump_epochs=[12, 300],
    alt_training=True,
)

default_dev_align_tina = Path("/work/asr3/raissi/master-thesis/raissi/work/mm/alignment/AlignmentJob.PRSC11kyIGma/output/alignment.cache.bundle", cached=True)
default_dev_align_log_tina = [
    Path(f"/work/asr3/raissi/master-thesis/raissi/work/mm/alignment/AlignmentJob.PRSC11kyIGma/output/alignment.log.{idx}.gz", cached=True)
    for idx in range(1, 2)
]
viterbi_exp_config = lambda align: ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": swb_system.num_classes(),
        "alignment": {
            "train": align,
            "dev": default_dev_align_tina,
        },
        "returnn_root": RETURNN_TDPS,
    },
    recognition_args=default_recognition_args.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
    alt_training=True,
)

viterbi_exp_config_comp = lambda align: ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": swb_system.num_classes(),
        "alignment": align,
    },
    recognition_args=default_recognition_args.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
)

from i6_experiments.users.mann.setups.nn_system.factored_hybrid import FactoredHybridDecoder, TransitionType
swb_system.prior_system.eps = 1e-6
swb_system.prior_system.extract_prior()
swb_system.set_decoder("fh", FactoredHybridDecoder())

# baseline_configs
baseline = builder.set_transcription_prior().build()
baseline_no_prior = builder.set_no_prior().build()
# baseline_no_prior.config["gradient_clip"] = 10

baselines = {
    "with_prior": baseline,
    "no_prior": baseline_no_prior,
}

def run_baselines():
    for name, config in baselines.items():
        swb_system.run_exp(
            name="baseline_{}".format(name),
            crnn_config=config,
            exp_config=exp_config,
        )

        if name == "no_prior":
            extra_bw_config = rasr.RasrConfig()
            extra_bw_config[
                "neural-network-trainer"
                ".alignment-fsa-exporter"
                ".model-combination"
                ".acoustic-model"
                ".fix-tdp-leaving-epsilon-arc"
            ] = True
            extra_bw_config[
                "neural-network-trainer"
                ".alignment-fsa-exporter"
                ".alignment-fsa-exporter"
                ".model-combination"
                ".acoustic-model"
            ] = None
            swb_system.run_exp(
                name="baseline_{}.corr_rasr".format(name),
                crnn_config=config,
                exp_config=exp_config.extend(
                    fast_bw_args={
                        "extra_config": extra_bw_config
                    },
                ),
            )

        # train with different tdp scale
        scaled_config = copy.deepcopy(config)
        scaled_config.tdp_scale = 0.3
        swb_system.run_exp(
            name="baseline_{}.tdp-0.3".format(name),
            crnn_config=scaled_config,
            exp_config=exp_config,
        )

        if name == "no_prior":
            swb_system.run_exp(
                name="baseline_{}.tdp-0.3.corr_rasr".format(name),
                crnn_config=scaled_config,
                exp_config=exp_config.extend(
                    fast_bw_args={
                        "extra_config": extra_bw_config
                    },
                ),
            )

run_baselines()

reductions = dict(
    substate_and_silence={"type": "factorize", "n_subclasses": 3, "div": 2, "silence_idx": swb_system.silence_idx()},
    speech_silence={"type": "speech_silence", "silence_idx": swb_system.silence_idx()},
    substate={"type": "factorize", "n_subclasses": 3, "div": 2},
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

tmp_config = copy.deepcopy(baseline_no_prior)
tdps.get_model(
    num_classes=swb_system.num_classes(),
    arch="label_speech_silence",
    extra_args=arch_args,
    init_args={"type": "smart", **init_args}
).set_config(tmp_config)

tdp_exp_config = exp_config.extend(
    training_args={
        "returnn_root": RETURNN_TDPS,
    },
    fast_bw_args={
        "acoustic_model_extra_config": NO_TDP_MODEL.to_acoustic_model_config(),
        "corpus": "train_magic",
    }
)

def print_report(name, data):
    from i6_experiments.users.mann.setups.reports import eval_tree
    import tabulate as tab
    def make_table():
        return tab.tabulate(eval_tree(data), headers="keys", tablefmt="presto")

    tk.register_report(
        os.path.join(fname, "summary", "{}.txt".format(name)),
        TableReport(data),
    )

from i6_experiments.users.mann.experimental.util import _RelaxedOverwriteConfig
from i6_experiments.users.mann.nn.inspect import InspectTFCheckpointJob

def regsiter_tdp_var_output(
    name,
    var_name="tdps/fwd_prob_var/fwd_prob_var",
    epoch=300
):
    j = InspectTFCheckpointJob(
        checkpoint=swb_system.nn_checkpoints["train_magic"][name][epoch],
        # all_tensors=False,
        tensor_name=var_name,
        returnn_python_exe=None,
        returnn_root=RETURNN_TDPS,
    )
    tk.register_output(
        os.path.join(
            "tdps", name,
        ),
        j.out_tensor_file,
    )

def dump_blstm_tdps():
    baseline_from_init()

    dumps = swb_system.dump_system.forward(
        name="baseline_tdps.no_prior.arch-blstm_large.tdp-0.1",
        returnn_config=None,
        epoch=240,
        # hdf_outputs=["fast_bw", "tdps", "tdps/fwd_prob"],
        hdf_outputs=["fast_bw", "tdps"],
        training_args=tdp_exp_config.training_args,
        fast_bw_args=tdp_exp_config.fast_bw_args,
    )

    tk.register_output("dumps/tdps.blstm_large.tdp-0.1.hdf", dumps["tdps"])

# without preload
def baseline_from_scratch():
    name = "with_prior"
    config = baselines[name]
    for name, config in baselines.items():
        for arch in ["label", "blstm_large"]:
            tmp_config = copy.deepcopy(config)
            # tmp_config.post_config["gradient_clip"] = 10
            tdps.get_model(num_classes=swb_system.num_classes(), arch=arch).set_config(tmp_config)

            swb_system.run_exp(
                name="baseline_tdps.from_scratch.{}.arch-{}".format(name, arch),
                crnn_config=tmp_config,
                exp_config=tdp_exp_config,
            )

def main():
    baseline_from_init()
    baseline_from_scratch()
    dump_blstm_tdps()

def non_word_tdps(corr_infinity=True):
    dev_config = rasr.RasrConfig()
    am = dev_config["flf-lattice-tool.network.recognizer.acoustic-model"]
    tdp = dev_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp"]
    tdp.nonword_phones = ",".join(f"[{p}]" for p in  ["LAUGHTER", "NOISE", "VOCALIZEDNOISE"])
    tdp.tying_type = "global-and-nonword"
    for i in range(2):
        am["tdp.nonword-{}".format(i)].exit = 20.0
        am["tdp.nonword-{}".format(i)].forward = 3.0
        am["tdp.nonword-{}".format(i)].loop = 0.0
        am["tdp.nonword-{}".format(i)].skip = "infinity"
    am["tdp.silence"].exit = 15.0
    return dev_config

from i6_experiments.users.mann.setups.util import P
def compare_init_and_arch(
    archs=None,
    inits=None,
    summary=True,
    align=True,
):
    super_exps = [
        ("label", "random", "flat", 0.3),  
        ("blstm_large", "random", "random", 0.1),
        ("label_speech_silence", "random", "flat", 0.3),
        ("label_substate_and_silence", "pretrained+prior", "smart", 0.3),
    ]
    if archs is None:
        archs = ["label", "blstm_large", "label_speech_silence", "label_substate_and_silence"]
    # archs = ["label", "blstm_large"]
    # archs = ["label"]
    if inits is None:
        inits = [
            # emission model, transition model
            ("pretrained", "smart"),
            ("pretrained+prior", "smart"),
            ("random", "smart"),
            ("random", "flat"),
            ("random", "random"),
        ]
        inits = P(*archs) * P(*inits) \
            + P("ffnn") * P(inits[0])
    else:
        inits = P(*archs) * P(*inits)

    from collections import defaultdict
    tdp_scales = defaultdict(lambda: [0.1, 0.3], {
        # ("random", "smart"): [0.1, 0.3, 0.5],
    })

    from collections import OrderedDict
    from tabulate import tabulate, SEPARATING_LINE
    from i6_experiments.users.mann.setups.reports import eval_tree
    wers = dict.fromkeys(archs)
    epochs = sorted(exp_config.epochs)
    headers = ["em init", "tm init", "tdp scale"] + epochs + ["fh", "fh-corr"]
    empty_row = lambda: OrderedDict.fromkeys(headers, "")

    # base_config = copy.deepcopy(config)
    base_config = builder.copy().set_no_prior().build()
    base_config.config["gradient_clip"] = 10
    # base_config.tdp_scale = 0.3
    wers = defaultdict(list)
    all_init_data = defaultdict(lambda: defaultdict(defaultdict))
    for arch, em_init, tm_init in inits:
        data = wers[arch]
        tmp_config = copy.deepcopy(base_config)

        if em_init == "pretrained":
            preload.set_preload(swb_system, tmp_config, ("train_magic", "baseline_no_prior", 12))
        elif em_init == "pretrained+prior":
            preload.set_preload(swb_system, tmp_config, ("train_magic", "baseline_with_prior.tdp-0.3", 12))
        else:
            assert em_init == "random"

        tdp_model = tdps.get_model(
            num_classes=swb_system.num_classes(),
            arch=arch,
            extra_args=arch_args,
            init_args={"type": tm_init, **init_args}
        )
        tdp_model.set_config(tmp_config)

        for tdp_scale in tdp_scales[(em_init, tm_init)]:
            scaled_config = copy.deepcopy(tmp_config)
            scaled_config.tdp_scale = tdp_scale
            swb_system.run_exp(
                name="baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale),
                crnn_config=scaled_config,
                exp_config=tdp_exp_config.replace(
                    alt_decoding={
                        # "epochs": [300] if arch == "blstm_large" and em_init == "pretrained" and tdp_scale == 0.1 else [],
                        "epochs": [],
                        "compile": True,
                        "flf_tool_exe": NEW_FLF_TOOL if arch == "blstm_large" else None,
                        "extra_compile_args": {
                            "returnn_root": RETURNN_TDPS if arch == "blstm_large" and tm_init == "smart" else None,
                        }
                    },
                ).extend(
                    scorer_args={
                        "prior_scale": 0.5,
                        "fwd_loop_scale": 0.7,
                        "num_label_contexts": 47,
                    },
                ),
            )
        
            fh = True
            # if (arch, em_init, tm_init, tdp_scale) in super_exps and False:
            if True:
                fh = True
                # decode with old rasr
                swb_system.run_decode(
                    name="baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale),
                    type="fh",
                    epoch=300,
                    exp_config=tdp_exp_config.replace(
                        recognition_args=default_recognition_args.replace(beam_pruning=16.0).to_dict(),
                    ),
                    decoding_args={
                        "compile": True,
                        "flf_tool_exe": NEW_FLF_TOOL if arch == "blstm_large" else None,
                        "extra_compile_args": {
                            "returnn_root": RETURNN_TDPS if arch == "blstm_large" and tm_init == "smart" else None,
                        }
                    },
                    extra_suffix="-fh",
                    scorer_args={
                        "prior_scale": 0.5,
                        "fwd_loop_scale": 0.7,
                        "num_label_contexts": 47,
                    },
                )


            if False:
                # new rasr
                extra_config = non_word_tdps()
                extra_config.flf_lattice_tool.lexicon.normalize_pronunciation = True
                extra_config.flf_lattice_tool.network.recognizer.pronunciation_scale = 3.0

                swb_system.run_decode(
                    name="baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale),
                    type="fh",
                    epoch=300,
                    exp_config=tdp_exp_config.replace(
                        recognition_args=default_recognition_args.replace(
                            beam_pruning=22,
                            tdp_scale=1.0,
                            extra_config=extra_config,
                        ).to_dict(),
                    ),
                    decoding_args={
                        "compile": True,
                        "flf_tool_exe": CORR_FLF_TOOL,
                        "extra_compile_args": {
                            "returnn_root": RETURNN_TDPS if arch == "blstm_large" and tm_init == "smart" else None,
                        }
                    },
                    extra_suffix="-fh-super-corr",
                    scorer_args={
                        "prior_scale": 0.5,
                        "fwd_loop_scale": 0.1,
                        "num_label_contexts": 47,
                    },
                )
            swb_system.run_decode(
                name="baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale),
                type="fh",
                epoch=300,
                exp_config=tdp_exp_config.replace(
                    recognition_args=default_recognition_args.replace(
                        altas=8.0,
                        beam_pruning=18.0,
                    ).to_dict(),
                ),
                decoding_args={
                    "compile": True,
                    "flf_tool_exe": CORR_FLF_TOOL,
                    "extra_compile_args": {
                        "returnn_root": RETURNN_TDPS if arch == "blstm_large" and tm_init == "smart" else None,
                    }
                },
                extra_suffix="-fh-corr",
                scorer_args={
                    "prior_scale": 0.5,
                    "fwd_loop_scale": 0.7,
                    "num_label_contexts": 47,
                },
            )

            name="baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale)
            # print(swb_system.jobs["dev"]["recog_crnn-{}-{}-fh-prior".format(name, 240)])

            row = empty_row()
            row["em init"] = em_init
            row["tm init"] = tm_init
            row["tdp scale"] = tdp_scale
            for epoch in epochs:
                row[epoch] = swb_system.get_wer("baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale), epoch)
            if fh:
                row["fh"] = swb_system.get_wer("baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale), 300, extra_suffix="-fh")
                pass
            row["fh-corr"] = swb_system.get_wer("baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale), 300, extra_suffix="-fh-corr")
            data.append(list(row.values()))
            all_init_data[(tm_init, em_init)][arch][tdp_scale] = row["fh"]
        data.append(SEPARATING_LINE)
    
    def make_init_table():
        archs = ["label_speech_silence", "label_substate_and_silence", "label", "blstm_large"]
        data = eval_tree(all_init_data)
        output_template = " \\\\\n".join(
            " & ".join(init) + " & " +  " & ".join(" $|$ ".join(str(arch_values[arch][tdp]) for tdp in [0.1, 0.3]) for arch in archs) for init, arch_values in data.items()
        )
        output_template += " \\\\\n"
        return output_template
    
    tk.register_report(os.path.join(fname, "init_table.tex"), make_init_table)
    
    # make some alignments
    exp_names = [
        "baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em_init, tm_init, arch, tdp_scale)
        for em_init, tm_init, arch, tdp_scale in [
            ("pretrained", "smart", "blstm_large", 0.1),
            ("pretrained", "smart", "label", 0.3),
        ]
    ]

    if align:
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

        for exp in exp_names:
            name=exp + "-300-fh-corr"
            feature_scorer = swb_system.feature_scorers["dev"][name]
            swb_system.nn_align(
                nn_name=exp,
                name=name,
                epoch=300,
                flow=swb_system.decoder.decoders[exp][epoch]("train").featureScorerFlow,
                feature_scorer=feature_scorer,
                evaluate={"ref_alignment": "init_gmm"},
                extra_config=extra_config,
            )
        
    if not summary:
        return
    
    def make_ney_overview():
        epochs = sorted(exp_config.epochs)
        def make_row(param, init, estimated, name):
            return [param, init, "estimated" if estimated else "fixed"] + [swb_system.get_wer(name, epoch) for epoch in epochs]
        
        rows = [
            make_row("speech + silence", "random", True, "baseline_tdps.random.random.arch-label_speech_silence.tdp-0.3"),
            make_row("speech + silence", "guessed", True, "baseline_tdps.random.smart.arch-label_speech_silence.tdp-0.1"),
            make_row("speech + silence", "guessed", False, "baseline_no_prior"),
            make_row("3 substates + silence", "guessed", False, "baseline_tdps.random.smart.arch-label_substate_and_silence.tdp-0.1"),
            make_row("80 phonemes + silence", "guessed", True, "baseline_tdps.random.smart.arch-label.tdp-0.3"),
            make_row("80 phonemes + silence + acoustic input", "random", True, "baseline_tdps.random.random.arch-blstm_large.tdp-0.1"),
        ]

        return tabulate(eval_tree(rows), headers=["parametrization", "initialization", "estimated/fixed"] + epochs, tablefmt="presto")

    def make_tables():
        lines = []
        for arch in archs:
            lines.append("arch: {}".format(arch))
            data = eval_tree(wers[arch])
            lines.append(tabulate(data, headers=headers, tablefmt="presto"))
            lines.append("\n" * 2)
        return "\n".join(lines)
    
    def dump_data():
        import json
        out_data = {
            k: list(filter(lambda x: x is not SEPARATING_LINE, v)) for k, v in wers.items()
        }
        with open(f"output/{fname}/compare_init_and_arch.json", "w") as f:
            json.dump(eval_tree(out_data), f)

    def make_tex_tables():
        lines = []
        for arch in archs:
            lines.append("arch: {}".format(arch))
            data = eval_tree(wers[arch])
            lines.append(tabulate(data, headers=headers, tablefmt="latex"))
            lines.append("\n" * 2)
        return "\n".join(lines)
            
    tk.register_report(fname + "/compare_init_and_arch", make_tables) 
    tk.register_callback(dump_data)
    tk.register_report(fname + "/compare_init_and_arch.tex", make_tex_tables) 
    # tk.register_report(fname + "/ney_overview", make_ney_overview)

def bw_plots():
    segments=[
        "switchboard-1/sw02001A/sw2001A-ms98-a-0041",
        "switchboard-1/sw02001A/sw2001A-ms98-a-0047",
        "switchboard-1/sw02001B/sw2001B-ms98-a-0004",
        "switchboard-1/sw02001B/sw2001B-ms98-a-0024"
    ],
    swb_system.dump_system.segments = [
        "switchboard-1/sw02001A/sw2001A-ms98-a-0019",
    ]
    exp_config.dump_epochs = tdp_exp_config.dump_epochs = [300]
    swb_system.run_exp(
        name="baseline_no_prior.tdp-0.3",
        exp_config=exp_config,
    )
    compare_init_and_arch()


def test_convergence():
    # train after converged model
    tmp_config = copy.deepcopy(swb_system.nn_config_dicts["train"]["baseline_no_prior.tdp-0.3"])
    tdp_model = tdps.get_model(
        num_classes=swb_system.num_classes(),
        arch="label_speech_silence",
        extra_args=arch_args,
        init_args={"type": "smart", **init_args}
    )
    tdp_model.set_config(tmp_config)
    preload.set_preload(swb_system, tmp_config, ("train_magic", "baseline_no_prior.tdp-0.3", 300))

    del tmp_config.config["learning_rates"]
    tmp_config.config["learning_rate"] = 1e-5

    def make_encoder_untrainable(config):
        config = copy.deepcopy(config)
        for key, layer in config.config["network"].items():
            if key[:3] in ("fwd", "bwd") or key == "output":
                layer["trainable"] = False
        return config
    
    exps = {
        "all": lambda config: config,
        "tdp_only": lambda config: make_encoder_untrainable(config),
    }

    cont_exp_config = (
        tdp_exp_config
        .replace(
            epochs=[4, 12, 32, 60],
            dump_epochs=[],
        )
        .extend(
            training_args={
                "num_epochs": 60,
                "time_rqmt": 30,
            }
        )
    )

    for name, make_config in exps.items():
        swb_system.run_exp(
            name="continued_from_converged." + name,
            crnn_config=make_config(tmp_config),
            exp_config=cont_exp_config,
        )


def score():
    compare_init_and_arch(align=False, summary=False)
    test_convergence()

    archs = ["label", "blstm_large", "label_speech_silence", "label_substate_and_silence"]
    inits = [
        # emission model, transition model
        ("pretrained", "smart"),
        ("pretrained+prior", "smart"),
        ("random", "smart"),
        ("random", "flat"),
        ("random", "random"),
    ]
    all_names = [
        "baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em, tm, arch, tdp)
        for em, tm in inits
        for arch in archs
        for tdp in ["0.1", "0.3"]
    ]
    names = [
        "baseline_tdps.random.random.arch-label_speech_silence.tdp-0.3",
        "baseline_tdps.random.smart.arch-label_speech_silence.tdp-0.3",
        "baseline_no_prior",
        "baseline_no_prior.tdp-0.3",
        # "baseline_tdps.random.smart.arch-label_substate_and_silence.tdp-0.3",
        # "baseline_tdps.random.smart.arch-label.tdp-0.3",
        "baseline_tdps.random.random.arch-blstm_large.tdp-0.1",
    ]

    names += [
        (f"continued_from_converged.{sfx}", 60) for sfx in ["all", "tdp_only"]
    ]

    swb_system.dump_system.init_score_segments()
    score_data = []

    targs = copy.deepcopy(tdp_exp_config.training_args)
    del targs["mem"]
    fast_bw_args = copy.deepcopy(tdp_exp_config.fast_bw_args)
    fast_bw_args.update(
        fix_tdps_applicator=True,
        fix_tdp_leaving_eps_arc=False,
    )

    data = []
    for name in names:

        epoch = 300
        if isinstance(name, tuple):
            name, epoch = name

        fast_bw_args = copy.deepcopy(exp_config.fast_bw_args) if "no_prior" in name \
            else copy.deepcopy(tdp_exp_config.fast_bw_args)
        fast_bw_args.update(
            fix_tdps_applicator=True,
            fix_tdp_leaving_eps_arc=False,
        )
        score_config = copy.deepcopy(swb_system.nn_config_dicts["train"][name])
        score_config.tdp_scale = 0.3

        swb_system.dump_system.score(
            name=name,
            epoch=epoch,
            returnn_config=score_config,
            training_args=targs,
            fast_bw_args=fast_bw_args,
        )
        data.append(
            {
                "name": name,
                "wer": swb_system.get_wer(name, epoch),
                "score": swb_system.dump_system.scores[name]["train_score"] \
                    if "baseline_no_prior" in name \
                    else swb_system.dump_system.scores[name]["train_score_output_bw"]
            }
        )
    
    swb_system.report("tdp_scores", data)

    swb_system.set_state_tying(
        value="monophone-no-tying-dense",
        hmm_partition=1,
        use_boundary_classes=False,
        use_word_end_classes=True,
    )

    # dump extra config
    name = "baseline_no_prior"
    score_config = copy.deepcopy(swb_system.nn_config_dicts["train"][name])
    score_config.tdp_scale = 0.1
    score_config.maybe_add_dependencies("import numpy as np")
    from i6_core.returnn.config import CodeWrapper
    fast_bw_args.update({
        "acoustic_model_extra_config": CombinedModel.from_fwd_probs(1/8, 1/40, 0.0).to_acoustic_model_config(),
    })
    net = score_config.config["network"]
    net.update({
        "selector": {
            "class": "constant",
            "value": CodeWrapper("np.reshape(np.transpose(np.reshape(np.arange(282), (47, 3, 2)), (0, 2, 1)), (47 * 2, 3))"),
        },
        "summarize_params": {"class": "gather", "from": ["output"], "position": "selector", "axis": "F"},
        "output_single": {"class": "reduce", "from": "summarize_params", "mode": "sum", "axis": "F"},
    })
    net["fast_bw"]["from"] = ["output_single"]
    net["output_bw"].update({
        "from": ["output_single"],
        "loss_opts": {"error_signal_layer": "fast_bw"},
    })
    swb_system.dump_system.score(
        name=name,
        epoch=300,
        extra_name="baseline_no_prior.1s",
        returnn_config=score_config,
        training_args=targs,
        fast_bw_args=fast_bw_args,
    )

    swb_system.run_decode(
        name="baseline_no_prior",
        recog_name="baseline_no_prior.1s",
        epoch=300,
        exp_config=exp_config.replace(
            compile_crnn_config=score_config,
            recognition_args=default_recognition_args.replace(
                tdp_scale=0.2,
                prior_scale=1.0,
            ).to_dict(use_gpu=True, mem=16),
        ),
        prior_config=score_config,
        reestimate_prior="alt-CRNN"
    )

def dump_all_tdps():
    compare_init_and_arch(align=False, summary=False)
    name = "baseline_tdps.random.smart.arch-label_speech_silence.tdp-0.1"
    names = [
        name,
        "baseline_tdps.pretrained+prior.smart.arch-label_speech_silence.tdp-0.1"
    ]
    epochs = [4, 8] + tdp_exp_config.epochs
    data = []
    for name in names:
        for e in epochs:
            j = InspectTFCheckpointJob(
                checkpoint=swb_system.nn_checkpoints["train_magic"][name][e],
                tensor_name="tdps/base_vars/base_vars",
                returnn_python_exe=None,
                returnn_root=RETURNN_TDPS,
            )
            tk.register_output("tdps/{}.epoch-{}".format(name, e), j.out_tensor_file)

    inits = [
        # emission model, transition model
        ("pretrained", "smart"),
        ("pretrained+prior", "smart"),
        ("random", "smart"),
        ("random", "flat"),
        ("random", "random"),
    ]
    names = [
        "baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em, tm, arch, tdp)
        for em, tm in inits
        for arch in ["label_speech_silence", "label_substate_and_silence"]
        for tdp in ["0.1", "0.3"]
    ]
    for name in names:
        e = 300
        j = InspectTFCheckpointJob(
            checkpoint=swb_system.nn_checkpoints["train_magic"][name][e],
            tensor_name="tdps/base_vars/base_vars",
            returnn_python_exe=None,
            returnn_root=RETURNN_TDPS,
        )
        tk.register_output("tdps/{}.epoch-{}".format(name, e), j.out_tensor_file)


tdp_experiments = [
    ("label", "random", "flat", 0.3, (1.0, 0.5, 0.1)),  
    ("blstm_large", "random", "random", 0.1, (1.0, 0.5, 0.1)),
    ("label_speech_silence", "random", "flat", 0.3, (1.0, 0.3, 0.1)),
    ("label_substate_and_silence", "pretrained+prior", "smart", 0.3, (1.0, 0.5, 0.1)),
]
exp_names = [
    ("baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em, tm, arch, tdp), opt)
    for arch, em, tm, tdp, opt in tdp_experiments
]

    
def run_align():
    from i6_experiments.users.mann.experimental.statistics.alignment import ComputeTseJob
    from i6_experiments.users.mann.experimental.statistics import (
        SilenceAtSegmentBoundaries,
        AlignmentStatisticsJob
    )
    data = []
    
    compare_init_and_arch()
    # make some alignments
    tdp_experiments = [
       ("label", "random", "flat", 0.3, (1.0, 0.5, 0.1)),  
       ("blstm_large", "random", "random", 0.1, (1.0, 0.5, 0.1)),
       ("label_speech_silence", "random", "flat", 0.3, (1.0, 0.3, 0.1)),
       ("label_substate_and_silence", "pretrained+prior", "smart", 0.3, (1.0, 0.5, 0.1)),
    ]
    exp_names = [
        ("baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em, tm, arch, tdp), opt)
        for arch, em, tm, tdp, opt in tdp_experiments
    ]

    extra_config = rasr.RasrConfig()
    extra_config["*"].python_home = "/work/tools/asr/python/3.8.0"
    extra_config["*"].python_program_name = "/work/tools/asr/python/3.8.0/bin/python3.8"
    
    extra_config["*"].fix_allophone_context_at_word_boundaries         = True
    extra_config["*"].transducer_builder_filter_out_invalid_allophones = True
    extra_config["*"].allow_for_silence_repetitions   = False 
    extra_config["*"].applicator_type = "corrected"

    extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"] \
        = CombinedModel.zeros(speech_skip="infinity").to_acoustic_model_config()

    swb_system.crp["train"].acoustic_model_trainer_exe = CORR_AM_TRAINER
    for exp, opt_param in exp_names:
        tdp, prior, fwdloop = opt_param
        epoch=300
        name=exp + "-300-fh-corr"
        feature_scorer = copy.deepcopy(swb_system.feature_scorers["dev"][name])
        tdp_extra_config = extra_config._copy()
        tdp_extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.state-tying"].type = "no-tying-dense"

        # set scales
        feature_scorer.config.center_state_prior_scale = prior
        feature_scorer.config.loop_scale = fwdloop
        feature_scorer.config.forward_scale = fwdloop
        tdp_extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.tdp"] = tdp
        stats = swb_system.nn_align(
            nn_name=exp,
            name=name,
            epoch=300,
            flow=swb_system.decoder.decoders[exp][epoch]("train").featureScorerFlow,
            feature_scorer=feature_scorer,
            evaluate={"ref_alignment": "init_gmm"},
            extra_config=tdp_extra_config,
            use_gpu=True,
        )

        row = {}
        row["Name"] = exp
        row["WER [%]"] = swb_system.get_wer(exp, epoch, extra_suffix="-fh")
        row["Silence [%]"] = stats["total_silence"] / stats["total_states"]

        # get tse
        alignment = swb_system.alignments["train"][name].alternatives["bundle"]
        tse_job = ComputeTseJob(
            alignment, swb_system.alignments["train"]["init_gmm"], swb_system.get_allophone_file()
        )
        row["TSE"] = tse_job.out_tse
        data.append(row)
    
    base_name = "baseline_no_prior"
    tdp, prior = 0.1, 0.7

    for sfx in ["", ".tdp-0.3"]:
        tdp_extra_config = extra_config._copy()
        tdp_extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"] \
            = PRIOR_MODEL_TINA.to_acoustic_model_config()
        tdp_extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.tdp.scale"] = tdp
        tdp_extra_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.mixture-set.priori-scale"] = prior

        # alignment for baseline
        name = base_name + sfx
        # return
        stats = swb_system.nn_align(
            nn_name=name,
            epoch=300,
            extra_config=tdp_extra_config,
            feature_corpus="train_magic",
            evaluate={"ref_alignment": "init_gmm"},
            use_gpu=True,
        )
        row = {}
        row["Name"] = name
        row["WER [%]"] = swb_system.get_wer(name, 300)
        row["Silence [%]"] = stats["total_silence"] / stats["total_states"]

        # get tse
        alignment = swb_system.alignments["train"][f"{name}-{epoch}"].alternatives["bundle"]
        tse_job = ComputeTseJob(
            alignment, swb_system.alignments["train"]["init_gmm"], swb_system.get_allophone_file()
        )
        row["TSE"] = tse_job.out_tse
        data.append(row)


    # train Viterbi
    print(swb_system.alignments)
    job_folder = "/work/asr3/raissi/master-thesis/raissi/work/mm/alignment/AlignmentJob.gcy2oEaOecnF/output"
    swb_system.alignments["train"]["baseline_no_prior.tina"] = tina_align = Path(job_folder, cached=True).join_right("alignment.cache.bundle")
    tse_job = ComputeTseJob(
        tina_align, swb_system.alignments["train"]["init_gmm"], swb_system.get_allophone_file()
    )
    args = (
        tina_align,
        swb_system.get_allophone_file(),
        swb_system.crp["train"].segment_path.hidden_paths,
        swb_system.crp["train"].concurrent
    )
    asj = AlignmentStatisticsJob(*args)
    stats = asj.counts
    row = {}
    row["Name"] = "baseline_no_prior.tina"
    row["WER [%]"] = "?"
    row["Silence [%]"] = stats["total_silence"] / stats["total_states"]
    row["TSE"] = tse_job.out_tse
    data.append(row)

    from i6_experiments.users.mann.setups.reports import TableReport
    make_table = TableReport(data)
    tk.register_report("alignment_table", make_table)

def viterbi():
    # global align
    run_align()

    align = ("train", "baseline_no_prior-300")
    alignments = [
        "baseline_no_prior-300",
    ]
    job_folder = "/work/asr3/raissi/master-thesis/raissi/work/mm/alignment/AlignmentJob.gcy2oEaOecnF/output"
    alignment_logs = [os.path.join(job_folder, f"alignment.log.{idx}.gz") for idx in range(1, 201)] 

    from i6_experiments.users.mann.setups.nn_system.plugins import FilterAlignmentPlugin
    swb_system.plugins["filter"] = FilterAlignmentPlugin(swb_system, swb.init_nn_args["dev_size"])

    baseline_viterbi_comp_fullsum = builder.copy().set_loss("viterbi").set_ce_args(label_smoothing=0.2)

    viterbi_builder = builder.copy().set_loss("viterbi")

    configs = {
        "fullsum_comp": (
            viterbi_builder
            .copy()
            .delete("chunking")
            .set_ce_args(focal_loss_factor=None)
            .build()
        ),
        "good": (
            viterbi_builder
            .copy()
            .set_ce_args(label_smoothing=0.2)
            .set_oclr(dur=120, lrate=8e-4/0.3)
            .build()
        )
    }

    for exp_name, _ in exp_names + [("baseline_no_prior", 0), ("baseline_no_prior.tina", 0)]:
        name = f"{exp_name}-300-fh-corr" if exp_name != "baseline_no_prior" else f"{exp_name}-300"
        epochs = exp_config.epochs
        if "tina" in name:
            name = exp_name
        epochs += [288]
        # baseline_viterbi = swb_system.baselines["viterbi_lstm"]()
        align_logs = {"dev": default_dev_align_log_tina}
        if "tina" in name:
            align_logs["train"] = alignment_logs
        align = ["train", name]

        for config_name, config in configs.items():
            swb_system.run_exp(
                name="viterbi.{}.align-{}".format(config_name, name),
                crnn_config=config,
                exp_config=viterbi_exp_config(align),
                plugin_args={ "filter": { "alignment_logs": align_logs, } },
                epochs=epochs
            )

def viterbi_mono():
    # global align
    run_align()

    swb_system.set_state_tying(
        "monophone",
        use_boundary_classes=None,
        use_word_end_classes=None,
    )

    align = ("train", "baseline_no_prior-300")
    alignments = [
        "baseline_no_prior-300",
    ]
    job_folder = "/work/asr3/raissi/master-thesis/raissi/work/mm/alignment/AlignmentJob.gcy2oEaOecnF/output"
    alignment_logs = [os.path.join(job_folder, f"alignment.log.{idx}.gz") for idx in range(1, 201)] 

    from i6_experiments.users.mann.setups.nn_system.plugins import FilterAlignmentPlugin
    swb_system.plugins["filter"] = FilterAlignmentPlugin(swb_system, swb.init_nn_args["dev_size"])

    viterbi_config = swb_system.baselines["viterbi_lstm"]()

    exp_names = [
        ("baseline_no_prior.tina", None),
        ("baseline_tdps.random.flat.arch-label.tdp-0.3", None),
    ]

    for exp_name, _ in exp_names:
        name = f"{exp_name}-300-fh-corr" if exp_name != "baseline_no_prior" else f"{exp_name}-300"
        epochs = exp_config.epochs
        if "tina" in name:
            name = exp_name
        epochs += [288]
        # baseline_viterbi = swb_system.baselines["viterbi_lstm"]()
        align_logs = None
        if "tina" in name:
            align_logs = alignment_logs
        align = ["train", name]

        swb_system.run_exp(
            name="viterbi.comp_enc.align-{}".format(name),
            crnn_config=viterbi_config,
            exp_config=viterbi_exp_config_comp(align),
            plugin_args={ "filter": { "alignment_logs": align_logs, } },
            epochs=epochs
        )



if __name__ == "__main__":
    baseline_from_init()
    baseline_from_scratch()
    # updated_from_scratch()

class Report:
    def __init__(self, exps):
        self.exps = exps
    
    def __call__(self):
        data = []
        for name, wer in self.wers.items():
            data.append(get_params(name) + [swb_system.get_wer(exp, epoch=300)])

        import tabulate as tab
        from sisyphus.job_path import VariableNotSet
        for row in data:
            try:
                row[-1] = str(row[-1])
            except VariableNotSet:
                row[-1] = "-"
        table = tab.tabulate(data, headers=["arch", "init", "tdp_scale", "prior", "WER"], tablefmt="presto")
        return table

def summary():
    main()
    parameters = [
        ("arch", "label", "blstm_large", "blstm_no_label_large"),
        ("init", "from_init", "from_scratch"),
        ("tdp_scale", "tdp-0.1", "tdp-0.3"),
        ("prior", "with_prior", "no_prior"),
    ]

    decoding_experiments = [
        name for name in swb_system.nn_checkpoints["train_magic"]
    ]

    def get_params(name):
        config = []
        for param in parameters:
            v = None
            for param_value in param[1:]:
                if param_value in exp:
                    v = param_value
            config.append(v)
        return config

    from collections import OrderedDict
    DATA = []
    for exp in decoding_experiments:
        config = []
        row = OrderedDict()
        for param in parameters:
            v = None
            for param_value in param[1:]:
                if param_value in exp:
                    v = param_value
            config.append(v)
            row[param[0]] = v
        print(exp, config)

        # row = config
        # row.append(swb_system.get_wer(exp, epoch=300))
        DATA.append(row)
    
    from sisyphus.job_path import VariableNotSet
    epochs = sorted(exp_config.epochs)
    def try_get(name, epoch):
        try:
            return str(swb_system.get_wer(name, epoch=epoch))
        # except (KeyError, VariableNotSet):
        #     return "-"
        except KeyError:
            return "-"
        except VariableNotSet:
            return "-"
    
    errors = (VariableNotSet, KeyError)
        
    def get_latest(name, epoch, show_before_optlm=False):
        try:
            # if show_before_optlm:
            #     return str(swb_system.get_wer(name, epoch=epoch)) + " (" + str(swb_system.get_wer(name, epoch=epoch, optlm=False)) + ")"
            return str(swb_system.get_wer(name, epoch=epoch))
        except errors:
            pass
        for e in epochs[::-1]:
            try:
                # if show_before_optlm:
                #     s = " (" + str(swb_system.get_wer(name, epoch=e, optlm=False)) + ")"
                return str(swb_system.get_wer(name, epoch=e)) + " ({})".format(e)
            except errors:
                pass
        return "-"
    
    print("Tabulate experiments: ", decoding_experiments)
    
    def tabulate():
        import tabulate as tab
        for exp, row in zip(decoding_experiments, DATA):
            row["WER"] = get_latest(exp, 300)
            row["tuned"] = get_latest(exp + ".transcription.tuned", 300)
            row["crnn-prior"] = get_latest(exp + ".crnn.tuned", 300)
            row["bw-prior"] = get_latest(exp + ".bw.tuned", 300)
            row["fh + transcription"] = get_latest(exp + ".fh.transcription.tuned", 300)
            row["fh + bw"] = get_latest(exp + ".fh.bw.tuned", 300)
            row["fh + crnn"] = get_latest(exp + ".fh.crnn.tuned", 300)
        data = [row.values() for row in DATA]
        table = tab.tabulate(data, headers=list(row.keys()), tablefmt="presto")
        return table

    tk.register_report(f"{fname}/table", tabulate)


#--------------------------------------- decoding -------------------------------------------------

def decode_default(exps):
    from i6_experiments.users.mann.experimental.tuning import RecognitionTuner, FactoredHybridTuner
    tuner = RecognitionTuner(
        swb_system,
        base_config=RecognitionConfig(
            beam_pruning=16,
            beam_pruning_threshold=10000,
            altas=8.0,
        )
    ) #, tdp_scales=[0.1], prior_scales=[0.1])
    tuner.output_dir = fname
    coros = []
    for exp in exps:
        coros.append(tuner.tune_async(
            name=exp,
            epoch=300,
            exp_config=exp_config.replace(
                reestimate_prior=prior
            ),
            # extra_suffix="." + name_map[prior],
            recognition_config=RecognitionConfig(
                tdps=CombinedModel.legacy(),
                beam_pruning=22,
                beam_pruning_threshold=500000,
                lm_scale=3.0,
            ),
            print_report=True,
        ))
    return coros

def get_compile_config(name):
    assert "arch-" in name
    arch = name.split("arch-")[1].split(".")[0]
    assert arch in ["label", "blstm_large", "blstm_no_label_large"]
    compile_config = copy.deepcopy(swb_system.nn_config_dicts["train"][name])
    compile_config.config["network"]["encoder_output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"], "is_output_layer": True}
    del compile_config.config["network"]["fast_bw"], compile_config.config["network"]["output_bw"]
    try:
        del compile_config.config["network"]["combine_prior"]
        del compile_config.config["network"]["accumulate_prior"]
    except KeyError:
        pass
    for key in list(compile_config.config.keys()):
        if key not in ["network", "num_outputs", "extern_data", "target"]:
            del compile_config.config[key]
    decoding_args = {}

    # swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/master-rasr-fsa/src/Tools/Flf/flf-tool.linux-x86_64-standard"
    if "blstm" in arch:
    #     swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard"
        compile_config.config["network"]["tdps"]["subnetwork"]["delta_encoder_output"] = {"class": "copy", "from": ["lstm_fwd", "lstm_bwd"], "is_output_layer": True}
        compile_config.config["network"]["tdps"]["subnetwork"]["fwd_prob"]["activation"] = "sigmoid"
    return arch, compile_config

def non_word_tdps():
    dev_config = rasr.RasrConfig()
    am = dev_config["flf-lattice-tool.network.recognizer.acoustic-model"]
    tdp = dev_config["flf-lattice-tool.network.recognizer.acoustic-model.tdp"]
    tdp.nonword_phones = ",".join(f"[{p}]" for p in  ["LAUGHTER", "NOISE", "VOCALIZEDNOISE"])
    tdp.tying_type = "global-and-nonword"
    for i in range(2):
        am["tdp.nonword-{}".format(i)].exit = 20.0
        am["tdp.nonword-{}".format(i)].forward = 3.0
        am["tdp.nonword-{}".format(i)].loop = 0.0
        am["tdp.nonword-{}".format(i)].skip = "infinity"
    am["tdp.silence"].exit = 15.0
    return dev_config

def decode_fixed_guessed(exps=None):
    if exps is None:
        exps = ["baseline_no_prior.tdp-0.3"]
    from i6_experiments.users.mann.experimental.tuning import RecognitionTuner, FactoredHybridTuner
    tuner = RecognitionTuner(
        swb_system,
        base_config=RecognitionConfig(
            beam_pruning=16,
            beam_pruning_threshold=10000,
            altas=8.0,
        ),
        # tdp_scales=[0.1], prior_scales=[0.1],
    ) 
    tuner.output_dir = fname
    coros = []
    for exp in exps:
        coros.append(tuner.tune_async(
            name=exp,
            epoch=300,
            exp_config=exp_config,
            extra_suffix=".guessed",
            recognition_config=RecognitionConfig(
                tdps=PRIOR_MODEL_TINA.adjust(silence_exit=20.0, speech_skip=30.0),
                beam_pruning=22,
                beam_pruning_threshold=500000,
                lm_scale=3.0,
            ),
            print_report=True,
        ))
    return coros


print("Nonword phones", non_word_tdps())

def decode_tdp(exps):
    from i6_experiments.users.mann.experimental.tuning import FactoredHybridTuner
    tuner = FactoredHybridTuner(
        swb_system,
        base_config=RecognitionConfig(
            beam_pruning=16,
            beam_pruning_threshold=10000,
            altas=8.0,
        ),
        tdp_scales=[0.0, 0.1],
        prior_scales=[0.3, 0.5, 0.7],
    )
    tuner.output_dir = fname
    priors = ["transcription", "alt", "alt-bw"]
    priors = ["transcription"]
    name_map = {
        "transcription": "transcription",
        "alt": "crnn",
        "alt-bw": "bw",
    }
    coros = []
    for prior in priors:
        for exp in exps:
            epoch = 300
            if isinstance(exp, tuple):
                exp, epoch = exp
            # arch, compile_config = get_compile_config(exp)
            coros.append(tuner.tune_async(
                name=exp,
                epoch=epoch,
                extra_suffix=".fh." + name_map[prior],
                # extra_suffix=".fh",
                decoding_args={},
                exp_config=exp_config.replace(
                    compile_crnn_config=None,
                    reestimate_prior=prior,
                ),
                recognition_config=RecognitionConfig(
                    tdps=CombinedModel.legacy(),
                    beam_pruning=22,
                    # beam_pruning=23,
                    beam_pruning_threshold=500000,
                    lm_scale=3.0,
                ),
                flf_tool_exe="/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard" if "blstm" in exp else None,
                print_report=True,
            ))
    return coros

def decode_tina(exps):
    # tune non-word
    from i6_experiments.users.mann.experimental.tuning import FactoredHybridTuner
    scales = P(0.3, 0.5, 0.7, 1.0, 1.3) \
        * P(0.1, 0.3, 0.5, 0.7, 1.0)
    scales += P(0.1) * P(-0.1, 0.0, 0.1, 0.3) \
        + P(0.3, 0.5, 0.7, 1.0, 1.3) * P(-0.1, 0.0)
    scales += P(0.1, 0.3, 0.5) * P(-0.5, -0.3, -0.7)
    scales = P(0.1) * scales
    tuner = FactoredHybridTuner(
        swb_system,
        base_config=RecognitionConfig(
            beam_pruning=16.0,
            beam_pruning_threshold=100000,
            altas=8.0,
        ),
        # tdp_scales = [0.1],
        # prior_scales = [0.3, 0.5, 0.7, 1.0, 1.3],
        # fwd_loop_scales = [0.1, 0.3, 0.5, 0.7, 1.0],
        all_scales = scales,
    )
    tuner.output_dir = fname
    coros = []
    # new rasr
    extra_config = non_word_tdps()
    extra_config.flf_lattice_tool.lexicon.normalize_pronunciation = True
    extra_config.flf_lattice_tool.network.recognizer.pronunciation_scale = 3.0
    for exp in exps:
        coros.append(tuner.tune_async(
            name=exp,
            epoch=300,
            extra_suffix=".fh.tina",
            exp_config=exp_config.replace(
                compile_crnn_config=None,
                reestimate_prior="transcription",
            ),
            decoding_args={},
            recognition_config=RecognitionConfig(
                tdps=CombinedModel.legacy().adjust(silence_exit=20.0),
                beam_pruning=22,
                beam_pruning_threshold=500000,
                lm_scale=3.0,
                extra_config=extra_config, 
            ),
            flf_tool_exe=CORR_FLF_TOOL,
            print_report=True,
        ))
    return coros

    
def decode_non_word(exps):
    # tune non-word
    from i6_experiments.users.mann.experimental.tuning import FactoredHybridTuner
    tuner = FactoredHybridTuner(
        swb_system,
        base_config=RecognitionConfig(
            beam_pruning=16.0,
            beam_pruning_threshold=100000,
            altas=8.0,
        ),
        tdp_scales = [0.1],
        prior_scales = [0.3, 0.5, 0.7, 1.0, 1.3],
        fwd_loop_scales = [0.01, 0.1, 0.3, 0.7, 1.0],
        # prior_scales = [0.3],
        # fwd_loop_scales = [0.1],
    )
    tuner.output_dir = fname
    coros = []
    for exp in exps:
        coros.append(tuner.tune_async(
            name=exp,
            epoch=300,
            extra_suffix=".fh.non_word",
            exp_config=exp_config.replace(
                compile_crnn_config=None,
                reestimate_prior="transcription",
            ),
            decoding_args={},
            recognition_config=RecognitionConfig(
                tdps=CombinedModel.legacy().adjust(silence_exit=20.0),
                # beam_pruning=22,
                beam_pruning=22,
                beam_pruning_threshold=500000,
                lm_scale=3.0,
                extra_config=non_word_tdps(), 
            ),
            flf_tool_exe=CORR_FLF_TOOL,
            print_report=True,
        ))
    return coros

import asyncio

async def decode():
    main()
    summary()
    exclude_infix = ["no-prior", "tina"]
    decoding_experiments = [
        name for name in swb_system.nn_checkpoints["train_magic"]
    ]
    decoding_experiments = [
        "baseline_no_prior",
        "baseline_tdps.no_prior.arch-label.tdp-0.3",
        # ("baseline_tdps.no_prior.arch-blstm_large.tdp-0.1", 240)
    ]

    fix_tdp_exps = [
        exp for exp in decoding_experiments if "arch-" not in exp
    ]

    print("Decoding experiments: ", decoding_experiments)
    print("Fixed tdp decoding experiments: ", fix_tdp_exps)
    coros = decode_default(fix_tdp_exps)
    print("Coroutines: ", coros)

    tdp_experiments = [
        name for name in swb_system.nn_checkpoints["train_magic"]
        if "arch" in name
    ]
    tdp_experiments = [
        "baseline_tdps.no_prior.arch-label.tdp-0.3",
        "baseline_tdps.no_prior.arch-label.tdp-0.1",
        "baseline_tdps.no_prior.arch-blstm_large.tdp-0.3",
        ("baseline_tdps.no_prior.arch-blstm_large.tdp-0.1", 240),
        # ("baseline_tdps.no_prior.arch-blstm_large.tdp-0.1", 120)
        "baseline_tdps.pretrained.smart.arch-label_speech_silence.tdp-0.3",
        "baseline_tdps.pretrained.smart.arch-label_substate_and_silence.tdp-0.3",
    ]
    tdp_coros = decode_tdp(tdp_experiments)
    await asyncio.gather(*(coros + tdp_coros))
    print("Tuning coroutines finished")
    print("tuned experiments: ", [name for name in swb_system.jobs["dev"].keys() if ".tuned" in name])

async def decode_speech_silence():
    compare_init_and_arch(
        archs=["label_speech_silence"],
        inits=[("random", "flat")],
        summary=False,
        align=False,
    )
    exps = [
        "baseline_tdps.random.flat.arch-label_speech_silence.tdp-0.3",
    ]
    await asyncio.gather(
        *decode_tina(exps),
        *decode_fixed_guessed()
    )

async def decode_init_and_arch():
    compare_init_and_arch(
        archs=["label_speech_silence", "label_substate_and_silence"],
        inits=[("pretrained", "smart")],
        summary=False,
        align=False,
    )
    tdp_experiments = [
        "baseline_tdps.pretrained.smart.arch-label_speech_silence.tdp-0.3",
        "baseline_tdps.pretrained.smart.arch-label_substate_and_silence.tdp-0.3",
    ]
    tdp_coros = decode_tdp(tdp_experiments)
    await asyncio.gather(*tdp_coros)

async def decode_correct():
    compare_init_and_arch(
        # inits=[("random", "flat")],
        # archs=["label_speech_silence"],
        summary=False,
        align=False,
    )
    tdp_experiments = [
       ("label", "random", "flat", 0.3),  
       ("blstm_large", "random", "random", 0.1),
       ("label_speech_silence", "random", "flat", 0.3),
       ("label_substate_and_silence", "pretrained+prior", "smart", 0.3),
    ]
    tdp_experiments = [
        "baseline_tdps.{}.{}.arch-{}.tdp-{}".format(em, tm, arch, tdp)
        for arch, em, tm, tdp in tdp_experiments
    ]
    # tdp_coros = decode_non_word(tdp_experiments)
    tdp_coros = decode_tina(tdp_experiments)
    await asyncio.gather(*tdp_coros)

def py():
    main()
    compare_init_and_arch()
    # asyncio.get_event_loop().run_until_complete(decode())
    # align()
    viterbi()
    viterbi_mono()

def all():
    py()

#-------------------------------------- cleanup ---------------------------------------------------

def clean(gpu=True):
    main()
    # updated_from_scratch()
    compare_init_and_arch()
    keep_epochs = sorted(set(
        exp_config.epochs + [4, 8]
    ))
    for name in swb_system.nn_config_dicts["train"]:
        try:
            swb_system.clean(
                name, keep_epochs,
                cleaner_args={ "gpu": int(gpu), }
            )
        except KeyError:
            swb_system.clean(
                name, keep_epochs,
                cleaner_args={ "gpu": int(gpu), },
                feature_corpus="train_magic"
            )


