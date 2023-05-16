from sisyphus import tk, gs

import os

import recipe.i6_experiments.users.mann.setups.nn_system.tedlium as ted
from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, ConfigBuilder, RecognitionConfig
from i6_experiments.users.mann.setups.nn_system.factored_hybrid import FactoredHybridDecoder, TransitionType
from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_experiments.users.mann.setups.util import P

from i6_core import (
    rasr, tools
)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

ted_system = ted.get_tedlium_system(
    full_train_corpus=False,
)

ted_system.set_decoder("fh", FactoredHybridDecoder(use_native_ops=True))

PRIOR_TDP_MODEL = CombinedModel.from_fwd_probs(3/8, 1/40, 0.0)
NO_TDP_MODEL = CombinedModel.zeros()

ted_system.compile_configs["baseline_lstm"] = ted_system.baselines["viterbi_lstm"]()
default_recognition_args = RecognitionConfig(
    tdps=CombinedModel.legacy(),
    beam_pruning=22,
    prior_scale=0.3,
    tdp_scale=0.1,
    lm_scale=3.0,
)

exp_config = ExpConfig(
    compile_crnn_config=ted_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "mem_rqmt": 24,
    },
    fast_bw_args={
        "acoustic_model_extra_config": PRIOR_TDP_MODEL.to_acoustic_model_config(),
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
    },
    compile_args={
        "use_global_binaries": False,
    },
    recognition_args=default_recognition_args.to_dict(),
    epochs=ted_system.default_epochs[-2:],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
)

train_align = tk.Path("/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/mm/alignment/AlignmentJob.xKSl3aodu70n/output/alignment.cache.bundle", cached=True)

ted_system.alignments["train"]["init_align"] = train_align

dev_align = tk.Path("/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/mm/alignment/AlignmentJob.4DDlyOlHnRvr/output/alignment.cache.bundle", cached=True)

job_output_folder = tk.Path("/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/mm/alignment/AlignmentJob.xKSl3aodu70n/output")

align_logs = [
    job_output_folder.join_right(f"alignment.log.{idx}.gz") for idx in range(1, 51)
]

from i6_experiments.users.mann.setups.nn_system.plugins import FilterAlignmentPlugin
ted_system.plugins["filter"] = FilterAlignmentPlugin(ted_system, 1e-4, prefix="returnn")
viterbi_exp_config = ExpConfig(
    compile_crnn_config=ted_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": ted_system.num_classes(),
        # "alignment": {
        #     "train": train_align,
        #     "dev": dev_align,
        # },
        "alignment": train_align,
        # "returnn_root": RETURNN_TDPS,
    },
    plugin_args={ "filter": { "alignment_logs": align_logs, } },
    recognition_args=default_recognition_args.to_dict(),
    epochs=ted_system.default_epochs,
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
)

corr_fsa_config = exp_config.extend(
    fast_bw_args=dict(
        fix_tdps_applicator=True,
        fix_tdp_leaving_eps_arc=False,
    )
)

tk.register_output("state-tying/dev.txt", ted_system.get_state_tying_file())

#-------------------------------------- tdp setup -------------------------------------------------

clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

# NEW_FLF_TOOL = tk.Path("/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard")
NEW_FLF_TOOL = tk.Path("/u/raissi/dev/rasr_tf14py38_private/src/Tools/Flf/flf-tool.linux-x86_64-standard")
CORR_AM_TRAINER = tk.Path("/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard")
ALLOPHONE_TOOL = tk.Path("/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/allophone-tool.linux-x86_64-standard")

tk.register_output("master-rasr-fsa.state-tying", ted_system.get_state_tying_file())
tk.register_output("master-rasr-fsa.allophones", ted_system.get_allophone_file())

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

tdp_exp_config = exp_config.extend(
    training_args={
        "returnn_root": RETURNN_TDPS,
    },
    fast_bw_args={
        "acoustic_model_extra_config": NO_TDP_MODEL.to_acoustic_model_config(),
    },
    scorer_args={
        "prior_scale": 0.3,
        "fwd_loop_scale": 0.1001,
    },
).replace(
    alt_decoding={
        "epochs": ted_system.default_epochs[-3:],
        "compile": True,
        "flf_tool_exe": NEW_FLF_TOOL,
        "extra_compile_args": {
            "returnn_root": RETURNN_TDPS,
        }
    },
    recognition_args=default_recognition_args.replace(pronunciation_scale=3.0).to_dict(use_gpu=True),
    # epochs=ted_system.default_epochs[-3:],
    epochs=[],
)


#---------------------------------------- nn config -----------------------------------------------

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
from recipe.i6_experiments.users.mann.nn import preload, tdps
builder = (
    ConfigBuilder(ted_system)
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_oclr()
    .set_no_prior()
    .set_specaugment()
)

viterbi_builder = (
    builder
    .copy()
    .set_loss("viterbi")
    .delete("chunking")
    .set_ce_args(focal_loss_factor=None)
)

init_args = {
    "speech_fwd": 3/8,
    "silence_fwd": 1/40,
    "silence_idx": ted_system.silence_idx(),
}

arch_args = {
    "n_subclasses": 3,
    "div": 2,
    "silence_idx": ted_system.silence_idx()
}


#---------------------------------------- experiments ---------------------------------------------

ted_system.run_exp(
    "baseline_viterbi",
    crnn_config=viterbi_builder.build(),
    exp_config=viterbi_exp_config,

)

for tdp_scale in [0.1, 0.3]:

    ted_system.run_exp(
        f"baseline_no_prior.tdp-{tdp_scale}",
        crnn_config=builder.build().set_tdp_scale(tdp_scale),
        exp_config=exp_config,
        # epochs=[10]
    )

    ted_system.run_exp(
        f"baseline_no_prior.corr_fsa.tdp-{tdp_scale}",
        crnn_config=builder.build().set_tdp_scale(tdp_scale),
        exp_config=corr_fsa_config,
    )


tdp_experiments = [
    ("blstm_large"                  , "random"          , "random"  , 0.1),
]

extra_exps = [
    ("label"                        , "random"          , "flat" ),  
    ("label_speech_silence"         , "random"          , "flat" ),
    ("label_substate_and_silence"   , "pretrained+prior", "smart"),
]

tdp_experiments = P(*tdp_experiments) \
    + P(*extra_exps) * P(0.1, 0.3)

tdp_experiments = P(*tdp_experiments) \
    + P(("ffnn", "pretrained", "smart")) * P(0.1, 0.3)

for arch, em_init, tm_init, tdp_scale, in tdp_experiments:
    exp_name = f"tdp.{arch}.{em_init}.{tm_init}.tdp-{tdp_scale}"

    tmp_config = builder.build()
    tmp_config.tdp_scale = tdp_scale

    tdp_model = tdps.get_model(
        num_classes=ted_system.num_classes(),
        arch=arch,
        extra_args=arch_args,
        init_args={"type": tm_init, **init_args}
    )
    tdp_model.set_config(tmp_config)

    ted_system.run_exp(
        exp_name,
        crnn_config=tmp_config,
        exp_config=tdp_exp_config,
    )

    if "label" in arch and tdp_scale == 0.1:
        # ted_system.jobs["train"]["train_nn_" + exp_name].rqmt['qsub_args'] = f'-l hostname="*2080*"'
        ted_system.jobs["train"]["train_nn_" + exp_name].rqmt['qsub_args'] = '-l qname=*2080*'


#----------------------------------------- dump tdps ----------------------------------------------

segments = [
    "911Mothers_2010W/1",
    "911Mothers_2010W/35",
    "AJJacobs_2011P/4",
    "AbrahamVerghese_2011G/52",
]

segments = ["TED-LIUM-realease2/" + s for s in segments]

def dump_blstm_tdps():
    ted_system.init_dump_system(
        segments=segments
    )
    dumps = ted_system.dump_system.forward(
        name="tdp.blstm_large.random.random.tdp-0.1",
        returnn_config=None,
        epoch=200,
        # hdf_outputs=["fast_bw", "tdps", "tdps/fwd_prob"],
        hdf_outputs=["fast_bw", "tdps"],
        training_args=tdp_exp_config.training_args,
        fast_bw_args=tdp_exp_config.fast_bw_args,
    )

    tk.register_output("dumps/tdps.blstm_large.tdp-0.1.hdf", dumps["tdps"])
    tk.register_output("dumps/bw.blstm_large.tdp-0.1.hdf", dumps["fast_bw"])


#----------------------------------------- print state-tying -------------------------------------

def state_tying():
    tk.register_output("master-rasr-fsa.state-tying", ted_system.get_state_tying_file())

    ted_system.crp["train"].allophone_tool_exe = ALLOPHONE_TOOL
    tk.register_output("private.state-tying", ted_system.get_state_tying_file())
    tk.register_output("private.allophones", ted_system.get_allophone_file())


#----------------------------------------- scores -------------------------------------------------

def score():
    ted_system.init_dump_system(
        segments=segments
    )
    ted_system.init_report_system(fname)
    ted_system.dump_system.init_score_segments(corpus="returnn_train")
    exp_names = [
        "baseline_no_prior.tdp-0.3",
        "tdp.label_speech_silence.random.flat.tdp-0.3",
        "tdp.blstm_large.random.random.tdp-0.1",
    ]
    targs = copy.deepcopy(tdp_exp_config.training_args)
    data = []
    for name in exp_names:

        epoch = 200

        fast_bw_args = copy.deepcopy(exp_config.fast_bw_args) if "no_prior" in name \
            else copy.deepcopy(tdp_exp_config.fast_bw_args)
        fast_bw_args.update(
            fix_tdps_applicator=True,
            fix_tdp_leaving_eps_arc=False,
        )
        score_config = copy.deepcopy(ted_system.nn_config_dicts["train"][name])
        score_config.tdp_scale = 0.3

        ted_system.dump_system.score(
            name=name,
            epoch=epoch,
            returnn_config=score_config,
            training_args=targs,
            fast_bw_args=fast_bw_args,
        )
        data.append(
            {
                "name": name,
                # "wer": ted_system.get_wer(name, epoch),
                "score": ted_system.dump_system.scores[name]["train_score"] \
                    if "baseline_no_prior" in name \
                    else ted_system.dump_system.scores[name]["train_score_output_bw"]
            }
        )
    
    ted_system.report("tdp_scores", data)



#---------------------------------------- alignment ---------------------------------------------

import copy

ted_system.crp["train"].acoustic_model_trainer_exe = CORR_AM_TRAINER

extra_config = rasr.RasrConfig()
extra_config["*"].python_home = "/work/tools/asr/python/3.8.0"
extra_config["*"].python_program_name = "/work/tools/asr/python/3.8.0/bin/python3.8"

extra_config["*"].fix_allophone_context_at_word_boundaries         = True
extra_config["*"].transducer_builder_filter_out_invalid_allophones = True
extra_config["*"].allow_for_silence_repetitions   = False 
extra_config["*"].applicator_type = "corrected"

def get_zero_config():
    res_config = copy.deepcopy(extra_config)
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"] \
        = CombinedModel.zeros(speech_skip="infinity").to_acoustic_model_config()
    tdp, prior = 0.1, 0.5
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.tdp"] = tdp
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.mixture-set.priori-scale"] = prior
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.state-tying"].type = "no-tying-dense"
    return res_config

def get_default_config():
    res_config = copy.deepcopy(extra_config)
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"] \
        = PRIOR_TDP_MODEL.to_acoustic_model_config()
    tdp, prior = 0.1, 0.5
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.tdp.scale"] = tdp
    res_config["acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model.mixture-set.priori-scale"] = prior
    return res_config


for exp_name in [
    "tdp.blstm_large.random.random.tdp-0.1",
    "baseline_no_prior.tdp-0.3",
]:
    print(exp_name)
    if "baseline_no_prior" in exp_name:
        conf = get_default_config()
    else:
        conf = get_zero_config()

    ted_system.nn_align(
        nn_name=exp_name,
        epoch=200,
        use_gpu=True,
        feature_corpus="train",
        extra_config=conf,
        evaluate=True,
        feature_flow="logmel",
    )

#---------------------------------------- clean ---------------------------------------------------

def clean(gpu=True):
    keep_epochs = sorted(set(
        exp_config.epochs
    ))
    for name in ted_system.nn_config_dicts["train"]:
        ted_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )


def all():
    score()
    state_tying()
    # align()
    clean()


