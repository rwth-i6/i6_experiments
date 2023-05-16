from sisyphus import *

import os
import copy

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
from recipe.i6_experiments.users.mann.setups.nn_system.librispeech import get_libri_1k_system, custom_recog_tdps, init_segment_order_shuffle
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups import prior
from i6_experiments.users.mann.nn import specaugment, learning_rates
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
# s = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

lbs_system = get_libri_1k_system()
init_segment_order_shuffle(lbs_system)

baseline_bw = lbs_system.baselines['bw_lstm_tina_1k']()
specaugment.set_config(baseline_bw.config)




from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
tdp_model = CombinedModel.from_fwd_probs(3/8, 1/25, 0.0)
epochs = [24, 48, 128, 256]
epochs += [512 - e for e in epochs[:-1]] + [512]
print(epochs)
lbs_system.compile_configs["baseline_lstm"] = lbs_system.baselines["viterbi_lstm"]()
exp_config = ExpConfig(
    compile_crnn_config=lbs_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None
    },
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model.to_acoustic_model_config(),
        "fix_tdps_applicator": True,
        "fix_tdp_leaving_eps_arc": False,
    },
    recognition_args={"extra_config": lbs.custom_recog_tdps()},
    epochs=epochs,
)

from recipe.i6_experiments.users.mann.experimental.lexicon import RemovePronunciationVariantsJob, VariantStatisticsJob
variants_stats = VariantStatisticsJob(
    lbs_system.crp["train"].corpus_config.file,
    lbs_system.crp["train"].lexicon_config.file
)
tk.register_output("variant_stats/lexicon.stats", variants_stats.out_lexicon_stats)
tk.register_output("variant_stats/corpus.stats", variants_stats.out_corpus_stats)

#---------------------------------- tinas baseline ------------------------------------------------

baseline_tina = copy.deepcopy(baseline_bw.config)
baseline_tina.am_scale = 0.3
baseline_tina.prior_scale = 0.1
baseline_tina.tdp_scale = 0.1

lbs_system.run_exp(
    name='baseline_tina',
    crnn_config=baseline_tina,
    exp_config=exp_config, 
)

"""
Try different things to debug:
    * don't normalize lemma sequence scores
    * use Tina's exact setup: returnn_python_exe, returnn_root
    * use epsilon in the prior
"""

# remove lemma normalization
from collections import ChainMap
lbs_system.run_exp(
    name="baseline_tina.no_lemma_norm",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(fast_bw_args={"normalize_lemma_sequence_scores": False}),
)

exp_tinas_binaries = exp_config.extend(
    fast_bw_args={"normalize_lemma_sequence_scores": False},
    training_args={
        "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
        "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
    }
)

# use Tina's exact setup
lbs_system.run_exp(
    name="baseline_tina.exact_setup",
    crnn_config=baseline_tina,
    exp_config=exp_tinas_binaries,
    # exp_config=exp_config.extend_field(
    #     fast_bw_args={"normalize_lemma_sequence_scores": False},
    #     training_args={
    #         "returnn_python_exe": tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh"),
    #         "returnn_root"      : tk.Path("/u/raissi/dev/returnn_packages/returnn")
    #     }
    # ),
)

#---------------------------------- baseline with pretrain ----------------------------------------

baseline_bw_pretrain = copy.deepcopy(baseline_bw)
baseline_bw_pretrain.config.config["learning_rate"] = 0.00025
baseline_bw_pretrain.build_args["static_lr"] = 0.00025
baseline_bw_pretrain.config.config["learning_rates"] = learning_rates.get_learning_rates(
    inc_min_ratio=0.25, increase=225, decay=225
)
lbs_system.run_exp(
    name='baseline_bw_pretrain',
    crnn_config=baseline_bw_pretrain,
    exp_config=exp_config,
)


#----------------------------------- more exactly the same ----------------------------------------

for we in [False, True]:
    lbs_system.set_state_tying(
        value="monophone-no-tying-dense",
        extra_args={
            "use-boundary-classes": False,
            "use-word-end-classes": we,
        }
    )

    EPS = 5e-6
    if we:
        lbs_system.prior_system.eps = EPS
    else:
        lbs_system.prior_system._eps = EPS
    lbs_system.prior_system.extract_prior()
    baseline_bw = lbs_system.baselines['bw_lstm_tina_1k']()
    baseline_tina = copy.deepcopy(baseline_bw.config)
    specaugment.set_config(baseline_tina)
    baseline_tina.am_scale = 0.3
    baseline_tina.prior_scale = 0.1
    baseline_tina.tdp_scale = 0.1

    # mega exactly the same
    extra_bw_config = rasr.RasrConfig()
    extra_bw_config[
        "neural-network-trainer.alignment-fsa-exporter"
        ".alignment-fsa-exporter.model-combination.acoustic-model"
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

    name = "baseline_tina.no-tying"
    if we:
        name += ".we"
        baseline_tina.config["gradient_clip"] = 10
    lbs_system.run_exp(
        name=name,
        crnn_config=baseline_tina_updates,
        exp_config=exp_tinas_binaries.extend(
            training_args={
                "mem_rqmt": 24,
            },
            fast_bw_args={
                "normalize_lemma_sequence_scores": False,
                "fix_tdps_applicator": None,
                "extra_config": extra_bw_config
            },
        ),
        # dump_epochs=[512]
    )

#---------------------------------------- fh decode -----------------------------------------------

from recipe.i6_experiments.users.raissi.setups.librispeech.search.factored_hybrid_search import FHDecoder, ContextEnum, ContextMapper, get_feature_scorer

recog_prior = copy.deepcopy(lbs_system.prior_system)
recog_prior.eps = EPS
recog_prior.extract_prior()

base = "baseline_tina.no-tying"
decoder = FHDecoder(
    name="baseline_tina.no-tying.fh",
    search_crp=lbs_system.crp["dev"],
    context_type=ContextEnum.monophone,
    context_mapper=ContextMapper(),
    feature_path=lbs_system.feature_flows["dev"]["gt"],
    model_path=lbs_system.nn_checkpoints["train"][base][128],
    graph=lbs_system.jobs["train"]["compile_returnn_{}".format(base)].out_graph,
    mixtures=lbs_system.mixtures["train"],
    eval_files=lbs_system.scorer_args["dev"],
    tensor_mapping=FHDecoder.TensorMapping(center_state_posteriors="output")
)

priorInfo={
    "center-state-prior": {"file": recog_prior.prior_xml_file, "scale": 0.1},
    "left-context-prior": {"file": None},
    "right-context-prior": {"file": None},
}
decoder.recognize_count_lm(
    priorInfo=priorInfo,
    lmScale=5.0,
)
recog_args = copy.deepcopy(exp_tinas_binaries.recognition_args)
recog_args["feature_scorer"] = get_feature_scorer(
    context_type=ContextEnum.monophone,
    context_mapper=ContextMapper(),
    featureScorerConfig=decoder.featureScorerConfig,
    mixtures=lbs_system.mixtures["train"],
    silence_id=lbs_system.silence_idx(),
    prior_info=priorInfo,
)

# lbs_system.decode(
#     name="baseline_tina.no-tying.fh",
#     crnn_config=None,
#     training_args=exp_tinas_binaries.training_args,
#     scorer_args=exp_tinas_binaries.scorer_args,
#     recognition_args=exp_tinas_binaries.recognition_args,
#     compile_crnn_config=exp_tinas_binaries.compile_crnn_config,
# )


# cleanup

keep_epochs = sorted(set(
    [8, 12, 24] + exp_config.epochs
))

def clean(gpu=False):
    for name in lbs_system.nn_config_dicts["train"]:
        lbs_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )
