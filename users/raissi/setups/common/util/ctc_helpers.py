__all__ = ["get_loss_opts_ctc_rasr_loss_config_dense"]


from IPython import embed
import copy
from typing import Dict, List, Optional

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFunction, DelayedFormat

from i6_core.am.config import acoustic_model_config
from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
import i6_core.rasr as rasr

from i6_experiments.users.berger.corpus.switchboard import data
from i6_experiments.users.berger.corpus.general import CTCSetupData, build_feature_hdf_dataset_config
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.berger.recipe.lexicon.modification import EnsureSilenceFirstJob
from i6_experiments.users.raissi.setups.common.data.factored_label import RasrStateTying


def get_switchboard_data(
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    **kwargs,
) -> CTCSetupData:
    if cv_keys is None:
        cv_keys = ["hub5e00"]
    if dev_keys is None:
        dev_keys = ["hub5e00"]
    if test_keys is None:
        test_keys = ["hub5e01", "rt03s"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        add_all_allophones=True,
        **kwargs,
    )

    train_lexicon = tk.Path("/work/asr4/berger/dependencies/switchboard/lexicon/wei_train_ctc.lexicon.v2.xml")

    # ********** Train data **********
    config_file = tk.Path("/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/ctc_data/rasr.train.config")
    feature_flow_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/train.feature.flow")
    train_data_config = {
        "class": "ExternSprintDataset",
        "partitionEpoch": 6,
        "sprintConfigStr": f"--config={config_file} --*.LOGFILE=nn-trainer.train.log --*.TASK=1 "
        f"--*.corpus.segment-order-shuffle=true --*.segment-order-sort-by-time-length=true "
        f"--*.segment-order-sort-by-time-length-chunk-size=300 --feature-extraction.file={feature_flow_file}",
        "sprintTrainerExecPath": rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}"),
    }

    # ********** CV data **********
    config_file = tk.Path("/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/ctc_data/rasr.dev.config")
    feature_flow_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/dev.feature.flow")
    cv_data_config = {
        "class": "ExternSprintDataset",
        "partitionEpoch": 1,
        "sprintConfigStr": f"--config={config_file} --*.LOGFILE=nn-trainer.dev.log --*.TASK=1 "
        f"--*.corpus.segment-order-shuffle=true --*.segment-order-sort-by-time-length=true "
        f"--*.segment-order-sort-by-time-length-chunk-size=50 --feature-extraction.file={feature_flow_file}",
        "sprintTrainerExecPath": rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}"),
    }

    # ********** Loss corpus **********

    loss_corpus = tk.Path("/work/asr4/berger/dependencies/switchboard/corpus/wei_train-dev.corpus.gz")
    loss_lexicon = train_lexicon

    # ********** Recog lexicon **********

    recog_lexicon = EnsureSilenceFirstJob(train_lexicon).out_lexicon
    recog_lexicon = AddEowPhonemesToLexiconJob(
        recog_lexicon, nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]
    ).out_lexicon

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    align_lexicon = EnsureSilenceFirstJob(train_lexicon).out_lexicon
    align_lexicon = AddEowPhonemesToLexiconJob(
        align_lexicon, nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]
    ).out_lexicon

    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = align_lexicon

    return CTCSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        loss_corpus=loss_corpus,
        loss_lexicon=loss_lexicon,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )


def format_func(s, *args):
    return s % args


def get_loss_opts_ctc_rasr_loss_config_dense(
    rasr_binary_path: tk.Path,
    loss_corpus_path: tk.Path,
    loss_lexicon_path: tk.Path,
    am_args: Dict,
    state_tying: RasrStateTying = RasrStateTying.monophone,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
    remove_prefix: str = "loss-corpus/",
):
    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = loss_corpus_path  # type: ignore
    loss_crp.corpus_config.remove_corpus_name_prefix = remove_prefix  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = loss_lexicon_path  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_all = False  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_from_lexicon = True  # type: ignore
    loss_crp.acoustic_model_config.state_tying.type = state_tying
    loss_crp.acoustic_model_config.state_tying.use_boundary_classes = False
    loss_crp.acoustic_model_config.state_tying.use_word_end_classes = True

    # Leave consistent HCLG
    del loss_crp.acoustic_model_config.phonology

    loss_crp.allophone_tool_exe = tk.Path(
        "/work/tools/users/raissi/rasr/rasr_tf2/arch/linux-x86_64-standard/allophone-tool.linux-x86_64-standard"
    )

    from i6_core.lexicon import DumpStateTyingJob

    st = DumpStateTyingJob(loss_crp).out_state_tying
    tk.register_output("dense_ctc.state_tying", st)

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        loss_crp,
        mapping,
        parallelize=False,
    )
    config.action = "python-control"
    config.python_control_loop_type = "python-control-loop"
    config.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True  # type: ignore
    config["*"].fix_allophone_context_at_word_boundaries = True  # type: ignore

    # Automaton manipulation
    if allow_label_loop:
        topology = "ctc"
    else:
        topology = "rna"
    config["*"].allophone_state_graph_builder.topology = topology  # type: ignore

    if min_duration > 1:
        config["*"].allophone_state_graph_builder.label_min_duration = min_duration  # type: ignore

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False  # type: ignore

    config._update(extra_config)
    post_config._update(extra_post_config)

    post_config["*"].output_channel.file = "fastbw.log"

    automaton_config = rasr.WriteRasrConfigJob(config, post_config).out_config
    tk.register_output("train/bw.config", automaton_config)

    loss_opts = {
        "sprint_opts": {
            "sprintExecPath": rasr_binary_path.join_right(f"nn-trainer.linux-x86_64-standard"),
            "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
            "numInstances": 1,
            "usePythonSegmentOrder": False,
            "sprintControlConfig": {"verbose": True},
        },
        "tdp_scale": 0.0,
    }

    return loss_opts
