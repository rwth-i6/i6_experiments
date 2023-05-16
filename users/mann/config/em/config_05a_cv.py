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
# print(swb_system.num_classes())
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
            "mem_rqmt": 48,
        },
        recognition_args=tinas_recog_config.replace(
            tdps=CombinedModel.legacy()
        ).to_dict(),
        scorer_args={"prior_scale": 0.3,},
    ),
    reestimate_prior="transcription",
)

#-------------------------------------- full corpus training --------------------------------------

overlay_name = "train_magic"
swb_system.add_overlay("train", overlay_name)

from recipe.i6_core import features
from recipe.i6_core import corpus

# cv_feature_bundle = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/features/gammatones/FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle"
# # swb_system.feature_bundles[overlay_name]["gt"] = tk.Path(cv_feature_bundle, cached=True)
# # swb_system.feature_flows[overlay_name]["gt"] = features.basic_cache_flow([
# #     swb_system.feature_bundles["train"]["gt"],
# #     tk.Path(cv_feature_bundle, cached=True),
# # ])
# # swb_system.crp[overlay_name].corpus_file = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/merged_corpora/train-dev.corpus.gz"
# # swb_system.crp[overlay_name].segment_path = ""

# overlay_name = "returnn_train_magic"
# swb_system.add_overlay("train_magic", overlay_name)
# swb_system.crp[overlay_name].concurrent = 1
# swb_system.crp[overlay_name].corpus_config = corpus_config = swb_system.crp[overlay_name].corpus_config._copy()
# swb_system.crp[overlay_name].segment_path = corpus.SegmentCorpusJob(corpus_config.file, num_segments=1).out_single_segment_files[1]

# overlay_name = "returnn_cv_magic"
# swb_system.add_overlay("dev", overlay_name)
# swb_system.crp[overlay_name].concurrent = 1
# swb_system.crp[overlay_name].segment_path = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/zhou-files-dev/segments"
# swb_system.crp[overlay_name].corpus_config = corpus_config = swb_system.crp[overlay_name].corpus_config._copy()
# swb_system.crp[overlay_name].corpus_config.file = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/zhou-files-dev/hub5_00.corpus.cleaned.gz"
# swb_system.crp[overlay_name].acoustic_model_config = swb_system.crp[overlay_name].acoustic_model_config._copy()
# del swb_system.crp[overlay_name].acoustic_model_config.tdp
# swb_system.feature_bundles[overlay_name]["gt"] = tk.Path(cv_feature_bundle, cached=True)
# swb_system.feature_flows[overlay_name]["gt"] = flow = features.basic_cache_flow(tk.Path(cv_feature_bundle, cached=True))

# # from recipe.i6_experiments.users.mann.experimental.write import WriteFlowNetworkJob
# # feature_flow_file = WriteFlowNetworkJob(flow).out_file
# # dev_extra_config = rasr.RasrConfig()
# # dev_extra_config.neural_network_trainer.feature_extraction.file = feature_flow_file

# merged_corpus = corpus.MergeCorporaJob(
#     [swb_system.crp[f"returnn_{k}_magic"].corpus_config.file for k in ["train", "cv"]],
#     name="switchboard-1",
#     merge_strategy=corpus.MergeStrategy.FLAT,
# ).out_merged_corpus
# swb_system.crp["train_magic"].corpus_config.file = merged_corpus
# # swb_system.crp["train_magic"].segment_path = corpus.SegmentCorpusJob(merged_corpus, num_segments=1).out_single_segment_files[1]

swb.init_extended_train_corpus(system=swb_system, reinit_shuffle=False)

from i6_experiments.users.mann.setups.nn_system.trainer import RasrTrainer

swb_system.set_trainer(RasrTrainer())

swb_system.run_exp(
    name="baseline_tina.tdps.recog.we.ext_corpus",
    crnn_config=baseline_tina,
    exp_config=exp_config.extend(
        fast_bw_args={
            "normalize_lemma_sequence_scores": False,
            "acoustic_model_extra_config": tdp_model_tina.to_acoustic_model_config(),
            "fix_tdp_leaving_eps_arc": True,
            "fix_tdps_applicator": False,
            "corpus": "train_magic",
        },
        training_args={
            "mem_rqmt": 48,
            "feature_corpus": "train_magic",
            "train_corpus": "returnn_train_magic",
            "dev_corpus": "returnn_cv_magic",
        },
        recognition_args=tinas_recog_config.replace(
            tdps=CombinedModel.legacy()
        ).to_dict(),
        scorer_args={"prior_scale": 0.3,},
    ),
    reestimate_prior="transcription",
    alt_training=True,
)

#-------------------------------------- clean up --------------------------------------------------

def clean(gpu=False):
    for name in swb_system.nn_config_dicts["train"]:
        swb_system.clean(
            name,
            sorted(set([4, 8, 12, 300] + exp_config.epochs)),
            cleaner_args={ "gpu": int(gpu), }
        )