from sisyphus import *

import os
import copy
import itertools

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, ConfigBuilder, RecognitionConfig
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
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
swb_system = swb.get_bw_switchboard_system()
for binary in ["rasr_binary_path", "native_ops_path", "returnn_python_exe", "returnn_python_home", "returnn_root"]:
    setattr(swb_system, binary, getattr(lbs_system, binary))
lbs.init_segment_order_shuffle(swb_system)

baseline_bw = swb_system.baselines['bw_lstm_tina_swb']()
specaugment.set_config(baseline_bw.config)

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
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
        "normalize_lemma_sequence_scores": False,
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

TDP_SCALES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TDP_REDUCED = [0.1, 0.5, 1.0]
PRIOR_SCALES = [0.1, 0.3, 0.5, 0.7]

from i6_experiments.users.mann.experimental.tuning import RecognitionTuner, FactoredHybridTuner
tuner = RecognitionTuner(swb_system) #, tdp_scales=[0.1], prior_scales=[0.1])

#---------------------------------- tinas baseline ------------------------------------------------

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

baseline_tina = builder.build()

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
    name='baseline_tina',
    crnn_config=baseline_tina_updates,
    exp_config=exp_config, 
)

#--------------------------------- train tdps -----------------------------------------------------

tdp_model = CombinedModel.zeros()
# set up returnn repository
clone_returnn_job = tools.git.CloneGitRepositoryJob(
    url="https://github.com/DanEnergetics/returnn.git",
    branch="mann-fast-bw-tdps",
)

clone_returnn_job.add_alias("returnn_tdp_training")
RETURNN_TDPS = clone_returnn_job.out_repository

print("Returnn root: ", swb_system.returnn_root)
print("Returnn python exe: ", swb_system.returnn_python_exe)

from recipe.i6_experiments.users.mann.nn import preload, tdps

baseline_tdps = copy.deepcopy(baseline_tina_updates)
baseline_tdps.config["gradient_clip"] = 10
preload.set_preload(swb_system, baseline_tdps, ("baseline_tina", 12))

for arch in ["label", "ffnn", "blstm_no_label_sigmoid", "blstm", "blstm_no_label"]:
    tmp_config = copy.deepcopy(baseline_tdps)
    tdps.get_model(num_classes=swb_system.num_classes(), arch=arch).set_config(tmp_config)
    swb_system.run_exp(
        name="baseline_feature_tdps.{}".format(arch) if arch != "label" else "baseline_label_tdps",
        crnn_config=tmp_config,
        exp_config=exp_config.extend(
            training_args={
                "returnn_root": RETURNN_TDPS,
            },
            fast_bw_args={"acoustic_model_extra_config": tdp_model.to_acoustic_model_config()},
        ),
    )

init_args = {
    "type": "simple_tdp_init",
    "silence_idx": swb_system.silence_idx(),
    "speech_fwd": 3/8,
    "silence_fwd": 1/60,
}

from collections import ChainMap

for arch in ["ffnn", "blstm"]:
    for corr in [False, True]:
        tmp_config = copy.deepcopy(baseline_tdps)
        tdps.get_model(
            num_classes=swb_system.num_classes(),
            arch=arch,
            init=dict(as_logit=corr, **init_args),
        ).set_config(tmp_config)
        name="baseline_feature_tdps.{}.init".format(arch)
        if corr:
            name += ".as_logit"
        swb_system.run_exp(
            name=name,
            crnn_config=tmp_config,
            exp_config=exp_config.extend(
                training_args={
                    "returnn_root": RETURNN_TDPS,
                },
                fast_bw_args={"acoustic_model_extra_config": tdp_model.to_acoustic_model_config()},
            ),
        )

#-------------------------------------- prior less training ---------------------------------------

swb_system.set_state_tying(
    value="monophone-no-tying-dense",
    extra_args={
        "use-boundary-classes": False,
        "use-word-end-classes": False,
    }
)

swb_system.prior_system.eps = 5e-6
swb_system.prior_system.extract_prior()

swb_system.run_exp(
    name='baseline_tina.no_tying',
    crnn_config=builder.build(),
    exp_config=exp_config, 
)

for tdp, prior in itertools.product(TDP_SCALES, PRIOR_SCALES):
    swb_system.run_exp(
        name='baseline_tina.no_tying-tune_scales-tdp_{}-prior_{}'.format(tdp, prior),
        crnn_config=builder.build(),
        exp_config=(
            exp_config
                .extend(
                    recognition_args=RecognitionConfig(
                        lm_scale=3.0,
                        tdps=CombinedModel.legacy(),
                        tdp_scale=tdp,
                        prior_scale=prior,
                        altas=2.0,
                        beam_pruning=16).to_dict(),
                ).replace(epochs=[300])
        ), 
        optimize=False,
    )

tuner.tune(
    name='baseline_tina.no_tying',
    epoch=300,
    returnn_config=builder.build(),
    recognition_config=RecognitionConfig(lm_scale=3.0, tdps=CombinedModel.legacy()),
    exp_config=exp_config,
    prior_suffix=False,
)

tdp, prior = 0.2, 0.7
swb_system.run_exp(
    name='baseline_tina.no_tying-tuned',
    crnn_config=builder.build(),
    exp_config=(
        exp_config
            .extend(
                recognition_args=RecognitionConfig(
                    beam_pruning=22.0,
                    beam_pruning_threshold=500000,
                    lm_scale=3.0,
                    tdps=CombinedModel.legacy(),
                    tdp_scale=tdp,
                    prior_scale=prior,
                ).to_dict(),
            ).replace(epochs=[300])
    ), 
)

baseline_prior_less = builder.set_no_prior().build()
baseline_prior_less.config["gradient_clip"] = 10
preload.set_preload(swb_system, baseline_prior_less, ("baseline_tina.no_tying", 12))

new_exp_config = exp_config.extend(
    training_args={
        "returnn_root": RETURNN_TDPS,
    },
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdps_applicator": False,
        "fix_tdp_leaving_eps_arc": True,
        "normalize_lemma_sequence_scores": False,
    },
    recognition_args=RecognitionConfig(
        tdps=CombinedModel.legacy(),
        prior_scale=0.3,
        tdp_scale=0.1
    ).to_dict(),
).replace(
    reestimate_prior="transcription"
)

epoch_mapping = {"blstm": 300, "label": 240}
tunings = {}

for arch in ["label", "ffnn", "blstm", "blstm_no_label"]:
    tmp_config = copy.deepcopy(baseline_prior_less)
    tdps.get_model(num_classes=swb_system.num_classes(), arch=arch).set_config(tmp_config)
    swb_system.run_exp(
        name="baseline_tdps.no-prior-{}".format(arch),
        crnn_config=tmp_config,
        exp_config=new_exp_config,
    )

    if arch not in {"blstm", "label"}:
        continue

    # tuner.tdp_scales = tuner.prior_scales = [0.1]
    if arch == "blstm":
        tune_func = tuner.tune
    else:
        tune_func = tuner.tune_async
        # continue
    tunings[arch] = tune_func(
        name="baseline_tdps.no-prior-{}".format(arch),
        epoch=epoch_mapping[arch],
        returnn_config=tmp_config,
        exp_config=new_exp_config,
        recognition_config=RecognitionConfig(
            tdps=CombinedModel.legacy(),
            beam_pruning=22.0,
            beam_pruning_threshold=500000,
            lm_scale=3.0,
        ),
        optimum=(0.2, 0.5) if arch == "blstm" else "async",
    )

    continue

    for tdp, prior in itertools.product(TDP_SCALES, PRIOR_SCALES):
    # for tdp, prior in itertools.product([0.2], [0.5]):
        swb_system.run_exp(
            name='baseline_tina.no-prior-{}-tune_scales-tdp_{}-prior_{}'.format(arch, tdp, prior),
            crnn_config=tmp_config,
            exp_config=(
                new_exp_config
                    .extend(
                        recognition_args=RecognitionConfig(
                            tdps=CombinedModel.legacy(),
                            lm_scale=3.0,
                            tdp_scale=tdp,
                            prior_scale=prior,
                            altas=2.0,
                            beam_pruning=16).to_dict(),
                    ).replace(epochs=[300 if arch == "blstm" else 240],)
            ), 
            optimize=False,
        )
    
    if arch == "label": continue

    tdp, prior = 0.2, 0.5
    swb_system.run_exp(
        name='baseline_tina.no-prior-{}-tuned'.format(arch),
        crnn_config=tmp_config,
        exp_config=(
            new_exp_config
                .extend(
                    recognition_args=RecognitionConfig(
                        beam_pruning=22.0,
                        beam_pruning_threshold=500000,
                        tdps=CombinedModel.legacy(),
                        lm_scale=3.0,
                        tdp_scale=tdp,
                        prior_scale=prior,
                    ).to_dict(),
                ).replace(epochs=[300],)
        ), 
    )



#-------------------------------------- decoding --------------------------------------------------

from recipe.i6_experiments.users.raissi.setups.librispeech.search.factored_hybrid_search import FHDecoder, ContextEnum, ContextMapper, get_feature_scorer
from i6_core.mm import CreateDummyMixturesJob

EPS = 5e-6
recog_prior = copy.deepcopy(swb_system.prior_system)
recog_prior.eps = EPS
recog_prior.extract_prior()

priorInfo={
    "center-state-prior": {"file": recog_prior.prior_xml_file, "scale": 0.7},
    "left-context-prior": {"file": None},
    "right-context-prior": {"file": None},
}

from i6_experiments.users.mann.setups.nn_system.factored_hybrid import FactoredHybridDecoder, TransitionType

swb_system.set_decoder("fh", FactoredHybridDecoder(default_decoding_args={"prior_info": priorInfo}))

compile_config = swb_system.baselines["viterbi_lstm"]()
compile_config.config["network"]["encoder_output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"], "is_output_layer": True}


base = "baseline_tina.no_tying"
swb_system.run_decode(
    name=base,
    recog_name=base + ".fh.new",
    epoch=120,
    type="fh",
    compile_args={
        "adjust_output_layer": False,
    },
    exp_config=new_exp_config.extend(
        recognition_args={
            # "feature_scorer": feature_scorer,
            # "flow": decoder.featureScorerFlow,
            "lm_scale": 3.0,
            # "extra_config": extra_config,
        },
        scorer_args={"prior_file": recog_prior.prior_xml_file},
    ).replace(
        compile_crnn_config=compile_config,
        reestimate_prior=False,
    )
)

base = "baseline_tdps.no-prior-blstm"
swb_system.run_decode(
    name=base,
    recog_name=base + ".fh.new",
    epoch=48,
    type="fh",
    compile_args={
        "adjust_output_layer": False,
    },
    decoding_args={
        "context_type": ContextEnum.monophone,
    },
    exp_config=new_exp_config.extend(
        recognition_args={
            "lm_scale": 3.0,
            # "extra_config": extra_config,
        },
        scorer_args={"prior_file": recog_prior.prior_xml_file},
    ).replace(
        compile_crnn_config=compile_config,
        reestimate_prior=False,
    )
)


archs = ["blstm", "label"]
pattern = lambda arch: "baseline_{}".format(arch)

type_map = {"blstm": TransitionType.feature, "label": TransitionType.label}

swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/master-rasr-fsa/src/Tools/Flf/flf-tool.linux-x86_64-standard"

fh_tuner = FactoredHybridTuner(
    swb_system,
    base_config=RecognitionConfig(altas=2.0),
)

epoch_selection = {"blstm": 300, "label": 240}

for arch in archs:
    name="baseline_tdps.no-prior-{}".format(arch)
    compile_config = copy.deepcopy(swb_system.nn_config_dicts["train"][name])
    compile_config.config["network"]["encoder_output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"], "is_output_layer": True}
    decoding_args = {}
    if arch == "blstm":
        compile_config.config["network"]["tdps"]["subnetwork"]["delta_encoder_output"] = {"class": "copy", "from": ["lstm_fwd", "lstm_bwd"], "is_output_layer": True}
        compile_config.config["network"]["tdps"]["subnetwork"]["fwd_prob"]["activation"] = "sigmoid"
        decoding_args = {"is_multi_encoder_output": True}
    for epoch in [48, 120, 240, 300]:
        swb_system.decoder.decode(
            name=name,
            recog_name=name + ".fh",
            epoch=epoch,
            decoding_args={
                "context_type": ContextEnum.mono_state_transition,
                "transition_type": type_map[arch],
                **decoding_args,
            },
            **new_exp_config.extend(
                recognition_args={
                    "lm_scale": 3.0,
                    # "extra_config": extra_config,
                },
                scorer_args={"prior_file": recog_prior.prior_xml_file},
            ).replace(
                compile_crnn_config=compile_config,
                reestimate_prior=False,
            ).to_dict()
        )
    
    if arch not in {"blstm", "label"}:
        continue

    tuning = fh_tuner.tune_async(
        name=name,
        epoch=epoch_selection[arch],
        decoding_args={
            "context_type": ContextEnum.mono_state_transition,
            "transition_type": type_map[arch],
            "prior_info": priorInfo,
            **decoding_args,
        },
        recognition_config=RecognitionConfig(
            lm_scale=3.0,
            tdps=CombinedModel.legacy(),
        ),
        exp_config=new_exp_config.replace(
            compile_crnn_config=compile_config,
            reestimate_prior=False,
            scorer_args={"prior_file": recog_prior.prior_xml_file},
        ),
        extra_suffix=".fh",
        optimum=(0.5, 0.2, 0.5) if arch == "blstm" else "async",
    )
    
import asyncio
async def label_tune():
    await asyncio.gather(tuning, tunings["label"])
    # await asyncio.gather(tuning)

#-------------------------------------- cleanup ---------------------------------------------------

def clean(gpu=False):
    keep_epochs = sorted(set(
        exp_config.epochs + [4, 8]
    ))
    for name in swb_system.nn_config_dicts["train"]:
        swb_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )
