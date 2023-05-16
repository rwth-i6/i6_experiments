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
PRIOR_TDP_MODEL = CombinedModel.from_fwd_probs(3/8, 1/60, 0.0)
swb_system.compile_configs["baseline_lstm"] = swb_system.baselines["viterbi_lstm"]()
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
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

from i6_experiments.users.mann.nn.util import DelayedCodeWrapper

def make_constant_gamma_net(net, value):
    net = copy.deepcopy(net)
    net["gamma_init"] = {
        "class": "constant",
        "value": value,
        "dtype": "float32",
    }
    net["gamma_broadcast"] = {
        "class": "eval",
        "eval": "source(0) * 0 + source(1)",
        "from": ["output", "gamma_init"]
    }
    net["fast_bw"]["from"] = ["gamma_broadcast"]
    return net

def set_constant_gamma(config, type, ):
    epochs = 12

class Initializer:
    def __init__(self, prior):
        self.prior = prior

    @classmethod
    def set_uniform_init(cls, config):
        out_layer = config.config["network"]["output"]
        out_layer["forward_weights_init"] = 0.0
        out_layer["bias_init"] = 1e-6

    def set_smart_init(self, config):
        config.maybe_add_dependencies("import numpy as np")
        init_code = DelayedCodeWrapper(
            "np.log(np.loadtxt('{}'))", self.prior
        )
        out_layer = config.config["network"]["output"]
        out_layer["forward_weights_init"] = 0.0
        out_layer["bias_init"] = init_code
    
    def set_init(self, config, init):
        return {
            "smart": self.set_smart_init,
            "uniform": self.set_uniform_init
        }[init](config)
    
    def set_constant_gammas(self, config, init, epoch=12):
        value = {
            "uniform": lambda: 0.0,
            "smart": lambda: DelayedCodeWrapper("np.loadtxt('{}')", self.prior)
        }[init]()
        net_orig = config.config.pop("network")
        config.config["network"] = make_constant_gamma_net(net_orig, value)
        if init == "smart":
            config.maybe_add_dependencies("import numpy as np")
        # net_orig = config.config.pop("network")
        # config.staged_network_dict = {
        #     1: make_constant_gamma_net(net_orig, value),
        #     epoch: net_orig,
        # }

#--------------------------------- train tdps -----------------------------------------------------

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

from recipe.i6_experiments.users.mann.nn import preload, tdps

#------------------------------------- updated experiments ----------------------------------------

# word end classes
swb_system.set_state_tying(
    value="monophone-no-tying-dense",
    hmm_partition=1,
    extra_args={
        "use-boundary-classes": False,
        "use-word-end-classes": True,
    }
)

from i6_experiments.users.mann.setups.state_tying import DuplicateStateTyingJob
pseudo_3s_tying = DuplicateStateTyingJob(swb_system.get_state_tying_file()).out_state_tying

fbw_extra_config = rasr.RasrConfig()
am = fbw_extra_config["neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model"]
am.state_tying.type = "lut"
am.state_tying.file = pseudo_3s_tying
am.hmm.states_per_phone = 3

print(fbw_extra_config)

tk.register_output("state_tying_1s_we", swb_system.get_state_tying_file())
tk.register_output("state_tying_pseudo_3s_we", pseudo_3s_tying)

CORR_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_private/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard"
NEW_FLF_TOOL = "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/Flf/flf-tool.linux-x86_64-standard"

swb_system.crp["dev"].flf_tool_exe = "/u/raissi/dev/master-rasr-fsa/src/Tools/Flf/flf-tool.linux-x86_64-standard"
# extended corpus
extra_args = swb.init_extended_train_corpus(swb_system)
PRIOR_MODEL_TINA = CombinedModel.from_fwd_probs(1/8, 1/40, 0.0)
default_recognition_config = RecognitionConfig(
    tdps=CombinedModel.legacy(),
    beam_pruning=22,
    prior_scale=0.3,
    tdp_scale=0.1,
    lm_scale=3.0,
    extra_args={"mem": 16},
)
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None,
        "returnn_root": RETURNN_TDPS,
        # "mem": 32,
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
    ).to_dict(
        mem=16,
    ),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription",
    dump_epochs=[12, 300],
    alt_training=True,
)



swb_system.prior_system.eps = 1e-6
swb_system.prior_system.extract_prior()

# baseline_configs
baseline = builder.set_transcription_prior().build()
baseline_no_prior = builder.set_no_prior().build()
# baseline_no_prior.config["gradient_clip"] = 10

baselines = {
    # "with_prior": baseline,
    "no_prior": baseline_no_prior,
}

def print_report(name, data):
    from i6_experiments.users.mann.setups.reports import eval_tree
    import tabulate as tab
    def make_table():
        return tab.tabulate(eval_tree(data), headers="keys", tablefmt="presto")

    tk.register_report(
        os.path.join(fname, "summary", "{}.txt".format(name)),
        make_table,
    )

from collections import OrderedDict
DATA = []
def make_row(init, arch, name):
    row = OrderedDict()
    row["em init"] = init
    row["arch"] = arch
    for e in exp_config.epochs:
        row[e] = swb_system.get_wer(name, e)
    row["tuned"] = swb_system.get_wer(name, 300, extra_suffix="-tuned")
    return row

class RowMaker:
    def __init__(self, *keys, tuned=False):
        self.keys = keys
        self.tuned = tuned
    
    def __call__(self, name, *values):
        row = OrderedDict()
        for k, v in zip(self.keys, values):
            row[k] = v
        for e in exp_config.epochs:
            row[e] = swb_system.get_wer(name, e)
        if self.tuned:
            row["tuned"] = swb_system.get_wer(name, 300, extra_suffix="-tuned", optlm=True, prior=True)
        return row

initializer = Initializer(swb_system.prior_system.prior_txt_file)
routines = {
    "fixed": initializer.set_constant_gammas,
    "init": initializer.set_init
}

def decode_tuned(name):
    swb_system.run_decode(
        name=name,
        epoch=300,
        extra_suffix="-tuned",
        exp_config=exp_config.replace(
            recognition_args=default_recognition_config.replace(
                prior_scale=1.0,
                tdp_scale=0.2,
                beam_pruning_threshold=500000,
            ).to_dict(),
            reestimate_prior="alt-CRNN",
        )
    )

def run_baselines():
    make_row = RowMaker("em init", "guess", tuned=True)
    data = []
    for name, config in baselines.items():
        swb_system.run_exp(
            name="baseline_{}".format(name),
            crnn_config=config,
            exp_config=exp_config,
        )
        decode_tuned("baseline_{}".format(name))
        data.append(make_row(
            "baseline_{}".format(name),
            "random",
            "none"
        ))

        for rname, routine in routines.items():
            for init in ["uniform", "smart"]:
                if rname == "init":
                    init_config = copy.deepcopy(config)
                    routine(init_config, init)
                    train_name="baseline_{}.gammas-{}.{}".format(name, rname, init)
                    swb_system.run_exp(
                        name="baseline_{}.gammas-{}.{}".format(name, rname, init),
                        crnn_config=init_config,
                        exp_config=exp_config,
                    )
                    decode_tuned(name=train_name)
                    data.append(make_row(train_name, "linear init", init))
                elif rname == "fixed":
                    init_config = copy.deepcopy(config)
                    routine(init_config, init)
                    init_name = "fixed_gammas.{}".format(init)
                    swb_system.run_exp(
                        name=init_name,
                        crnn_config=init_config,
                        exp_config=exp_config.replace(
                            epochs=[12],
                            dump_epochs=[12],
                        ).extend(
                            training_args={
                                "num_epochs": 18,
                            }
                        ),
                    )
                    preload_config = copy.deepcopy(config)
                    preload.set_preload(
                        swb_system,
                        preload_config,
                        (extra_args["feature_corpus"], init_name, 12)
                    )
                    preload_config.config["learning_rates"] = preload_config.config["learning_rates"][12:]
                    train_name="baseline_{}.init_fixed_gammas-{}".format(name, init)
                    swb_system.run_exp(
                        name="baseline_{}.init_fixed_gammas-{}".format(name, init),
                        crnn_config=preload_config,
                        exp_config=exp_config,
                    )
                    decode_tuned(name=train_name)
                    data.append(make_row(train_name, "fixed gammas", init))

        swb_system.run_exp(
            name="baseline_{}.pseudo_3s".format(name),
            crnn_config=config,
            exp_config=exp_config.extend(
                fast_bw_args={
                    "extra_config": fbw_extra_config,
                }
            ),
        )
        # DATA.append(make_row("random", "pseudo_3s", "baseline_{}.pseudo_3s".format(name)))
    
    score_names = [
        "baseline_no_prior",
        "baseline_{}.pseudo_3s".format(name)
    ]
    targs = copy.deepcopy(exp_config.training_args)
    # targs["feature_corpus"] = "train"
    # del targs["mem"]
    swb_system.dump_system.init_score_segments()
    score_data = []
    for name in score_names:
        swb_system.dump_system.score(
            name=name,
            epoch=300,
            returnn_config=None,
            training_args=targs,
            fast_bw_args=exp_config.fast_bw_args,
        )
        score_data.append(
            {
                "name": name,
                "wer": swb_system.get_wer(name, 300),
                "score": swb_system.dump_system.scores[name]["dev_score"]
            }
        )

    print_report("fixed_tdps", data)
    print_report("fixed_tdps_scores", score_data)

    name = "baseline_no_prior"
    fast_bw_args = copy.deepcopy(exp_config.fast_bw_args)
    fast_bw_args["extra_config"] = fbw_extra_config
    fast_bw_args["acoustic_model_extra_config"] = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0).to_acoustic_model_config()
    swb_system.dump_system.score(
        name=name,
        epoch=300,
        extra_name="baseline_no_prior.3s",
        returnn_config=None,
        training_args=targs,
        fast_bw_args=fast_bw_args,
    )
    # tk.register_output("scores/{}.epoch-300".format(name), swb_system.dump_system.scores[name])

    # recognition
    recog_extra_config = rasr.RasrConfig()
    am = recog_extra_config["flf-lattice-tool.network.recognizer.acoustic-model"]
    am.state_tying.type = "lut"
    am.state_tying.file = pseudo_3s_tying
    am.hmm.states_per_phone = 3

    for prior in ["transcription", "alt-CRNN"]:
        swb_system.run_decode(
            name="baseline_no_prior",
            epoch=300,
            recog_name="baseline_no_prior.3s.{}".format(prior),
            exp_config=exp_config.replace(
                recognition_args=default_recognition_config.replace(
                    prior_scale=0.7,
                    extra_config=recog_extra_config,
                ).to_dict(use_gpu=True),
                reestimate_prior=prior.split("-")[0],
            ),
        )


reductions = dict(
    substate_and_silence={"type": "factorize", "n_subclasses": 3, "div": 2, "silence_idx": swb_system.silence_idx()},
    speech_silence={"type": "speech_silence", "silence_idx": swb_system.silence_idx()},
    substate={"type": "factorize", "n_subclasses": 3, "div": 2},
)

init_args = {
    # "n_subclasses": 3,
    "speech_fwd": 1/8,
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

print("Tdps substate")
from pprint import pprint
# pprint(tmp_config.config["network"]["tdps"])

# quit()



tdp_exp_config = exp_config.extend(
    training_args={
        "returnn_root": RETURNN_TDPS,
    },
    fast_bw_args={
        "acoustic_model_extra_config": NO_TDP_MODEL.to_acoustic_model_config(),
        "corpus": extra_args["feature_corpus"],
    }
)


def compare_init_and_arch():
    # main()
    name = "no_prior"
    config = baselines[name]

    # archs = ["label", "blstm_large", "label_speech_silence", "label_substate_and_silence"]
    archs = ["label", "label_speech_silence"]
    inits = [
        # emission model, transition model
        ("pretrained", "smart"),
        ("random", "smart"),
        # ("random", "flat"),
        # ("random", "random"),
    ]

    data = []
    make_row = RowMaker("em init", "arch")

    base_config = copy.deepcopy(config)
    base_config.config["gradient_clip"] = 10
    base_config.tdp_scale = 0.3
    for arch in archs:
        for em_init, tm_init in inits:
            tmp_config = copy.deepcopy(base_config)

            if em_init == "pretrained":
                preload.set_preload(swb_system, tmp_config, (extra_args["feature_corpus"], "baseline_no_prior.pseudo_3s", 12))
            else:
                assert em_init == "random"

            tdp_model = tdps.get_model(
                num_classes=swb_system.num_classes(),
                arch=arch,
                extra_args=arch_args,
                init_args={"type": tm_init, **init_args}
            )
            tdp_model.set_config(tmp_config)
            swb_system.run_exp(
                name="baseline_tdps.{}.{}.arch-{}".format(em_init, tm_init, arch),
                crnn_config=tmp_config,
                exp_config=tdp_exp_config,
            )

            data.append(make_row(
                "baseline_tdps.{}.{}.arch-{}".format(em_init, tm_init, arch),
                "pseudo_3s" if em_init == "pretrained" else "random",
                arch,
            ))
    
    print_report("tdps", data)

def quick_summary():
    from i6_experiments.users.mann.setups.reports import eval_tree
    import tabulate as tab

    def make_table():
        return tab.tabulate(eval_tree(DATA), headers="keys", tablefmt="presto")

    tk.register_report(
        os.path.join(fname, "summary", "tdps.txt"),
        make_table,
    )

def py():
    run_baselines()
    compare_init_and_arch()
    # quick_summary()

def all():
    py()

#--------------------------------------- decoding -------------------------------------------------

from i6_experiments.users.mann.setups.nn_system.factored_hybrid import FactoredHybridDecoder, TransitionType
swb_system.set_decoder("fh", FactoredHybridDecoder())

def decode_default(exps):
    from i6_experiments.users.mann.experimental.tuning import RecognitionTuner, FactoredHybridTuner
    tuner = RecognitionTuner(
        swb_system,
        base_config=RecognitionConfig(
            beam_pruning=16,
            beam_pruning_threshold=10000,
            altas=8.0,
        ),
        tdp_scales=[0.1, 0.2, 0.3],
        # tdp_scales=[0.1], prior_scales=[0.1])
    )
    tuner.prior_scales += [1.0, 1.3]
    print(tuner.prior_scales)
    coros = []
    tuner.output_dir = fname
    priors = ["transcription", "CRNN", "bw"]
    f = lambda s: "alt-{}".format(s) if s != "transcription" else s
    for exp in exps:
        for prior in priors:
            coros.append(tuner.tune_async(
                name=exp,
                epoch=300,
                exp_config=exp_config.replace(
                    reestimate_prior=f(prior)
                ),
                extra_suffix="." + prior,
                recognition_config=RecognitionConfig(
                    tdps=CombinedModel.legacy(),
                    beam_pruning=22,
                    beam_pruning_threshold=500000,
                    lm_scale=3.0,
                    extra_args={"use_gpu": True}
                ),
                print_report=True,
            ))
    return coros

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
        prior_scales = [0.3, 0.5, 0.7],
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
            extra_suffix=".fh.bw.non_word",
            exp_config=exp_config.replace(
                compile_crnn_config=None,
                reestimate_prior="alt-bw",
            ).extend(
                scorer_args={ "num_label_contexts": 47, }
            ),
            decoding_args={},
            recognition_config=RecognitionConfig(
                tdps=CombinedModel.legacy().adjust(silence_exit=20.0),
                # beam_pruning=22,
                beam_pruning=22,
                beam_pruning_threshold=500000,
                lm_scale=3.0,
                extra_config=non_word_tdps(), 
                extra_args={"use_gpu": True}
            ),
            flf_tool_exe=CORR_FLF_TOOL,
            # flf_tool_exe=NEW_FLF_TOOL,
            print_report=True,
        ))
    return coros

import asyncio

async def decode():
    run_baselines()
    compare_init_and_arch()
    fix_tdp_exps = [
        "baseline_no_prior",
        "baseline_no_prior.pseudo_3s"
    ]

    coros = decode_default(fix_tdp_exps)

    archs = ["label", "label_speech_silence"]
    inits = [
        # emission model, transition model
        ("pretrained", "smart"),
    ]
    tdp_experiments = [
        "baseline_tdps.{}.{}.arch-{}".format(em_init, tm_init, arch)
        for em_init, tm_init in inits
        for arch in archs
    ]
    # tdp_experiments = []
    tdp_coros = decode_non_word(tdp_experiments)
    await asyncio.gather(*(coros + tdp_coros))
    print("Tuning coroutines finished")
    print("tuned experiments: ", [name for name in swb_system.jobs["dev"].keys() if ".tuned" in name])

#-------------------------------------- cleanup ---------------------------------------------------

def clean(gpu=False):
    main()
    updated_from_scratch()
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


