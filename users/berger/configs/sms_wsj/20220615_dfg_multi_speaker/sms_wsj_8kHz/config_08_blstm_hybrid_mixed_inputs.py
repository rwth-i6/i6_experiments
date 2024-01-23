import copy
from functools import partial
import os
from i6_core.am.config import acoustic_model_config
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.features.common import samples_flow
from i6_core.mm.mixtures import CreateDummyMixturesJob

import i6_core.rasr as rasr
from i6_core.rasr.feature_scorer import PrecomputedHybridFeatureScorer
from i6_core.rasr.flow import FlowNetwork
from i6_core.recognition.advanced_tree_search import AdvancedTreeSearchJob
from i6_core.recognition.optimize_parameters import OptimizeAMandLMScaleJob
from i6_core.recognition.conversion import LatticeToCtmJob
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.compile import CompileNativeOpJob, CompileTFGraphJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.extract_prior import ReturnnComputePriorJob
import i6_core.text as text
from i6_core.returnn.training import ReturnnTrainingJob
from i6_private.users.vieting.helpers.returnn import (
    prune_network_dict,
    serialize_dim_tags,
)
from i6_private.users.vieting.jobs.scoring import (
    MinimumPermutationCtmJob,
    JointLatticeCacheJob,
    CtmRepeatForSpeakersJob,
)
from i6_experiments.common.datasets.sms_wsj.returnn_datasets import (
    SequenceBuffer,
    ZipAudioReader,
    SmsWsjBase,
    SmsWsjBaseWithHdfClasses,
    SmsWsjWrapper,
    SmsWsjMixtureEarlyDataset,
    SmsWsjMixtureEarlyAlignmentDataset,
)
from i6_experiments.users.berger.args.returnn.config import (
    get_network_config,
    get_returnn_config,
)
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_data_inputs,
    process_string,
    lm_cleaning,
)
from i6_experiments.users.berger.network.models.blstm_hybrid_dual_output import (
    make_blstm_hybrid_dual_output_combine_enc_model,
    make_blstm_hybrid_dual_output_combine_enc_recog_model,
    make_blstm_hybrid_dual_output_soft_context_recog_model,
    make_blstm_hybrid_dual_output_model,
    make_blstm_hybrid_dual_output_soft_context_model,
    make_blstm_hybrid_dual_output_recog_model,
)
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.transducer_system import SummaryKey
from returnn.tf.util.data import batch_dim
from sisyphus import gs, tk


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

# rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")
rasr_binary_path = tk.Path("/u/berger/rasr_github/arch/linux-x86_64-standard")

frequency = 8

num_inputs = 40

num_outputs = 9001

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

recog_dev_key = "sms_cv_dev93_mixed"
recog_test_key = "sms_test_eval92_mixed"

scoring_dev_key = "sms_cv_dev93"
scoring_test_key = "sms_test_eval92"

alignments = {
    train_key: tk.Path("/work/asr4/berger/dependencies/sms_wsj/hdf/alignment/train_si284.cart-9001.hdf"),
    dev_key: tk.Path("/work/asr4/berger/dependencies/sms_wsj/hdf/alignment/cv_dev93.cart-9001.hdf"),
}

json_paths = {
    train_key: tk.Path("/work/asr4/berger/dependencies/sms_wsj/json/sms_wsj_remove_invalid.json"),
    dev_key: tk.Path("/work/asr4/berger/dependencies/sms_wsj/json/sms_wsj_remove_invalid.json"),
}

zip_cache_path = tk.Path("/work/asr3/converse/data/sms_wsj_original_and_rir.zip")

speech_sep_checkpoint = tk.Path(
    "/u/vieting/testing/20210714_dfg_kickoff/import-padertorch/mask_estimator.checkpoint.index"
)

am_checkpoints = {
    1: {
        (6, 4, 2, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-2_lr-Newbob-4e-04.epoch.060.index"
        ),  # 21.7 cv_dev93; 15.3 test_eval92
        (6, 4, 2, 1, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-2_lr-Newbob-4e-04.epoch.060.index"
        ),  # 21.7 cv_dev93; 15.3 test_eval92
        (6, 4, 1, 1, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-2_lr-Newbob-4e-04.epoch.060.index"
        ),  # 21.7 cv_dev93; 15.3 test_eval92
        (6, 4, 0, 1, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-2_lr-Newbob-4e-04.epoch.060.index"
        ),  # 21.7 cv_dev93; 15.3 test_eval92
        (4, 4, 2, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-4_enc-mix-4_enc-01+mix-2_lr-Newbob-4e-04.epoch.060.index"
        ),  # 21.9 cv_dev93; 15.8 test_eval92
        (4, 2, 2, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-4_enc-mix-2_enc-01+mix-2_lr-Newbob-4e-04.epoch.060.index"
        ),  # 22.1 cv_dev93; 15.7 test_eval92
        (6, 0, 0, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-0_enc-01+mix-0_lr-Newbob-4e-04.epoch.060.index"
        ),  # 22.7 cv_dev93; 16.2 test_eval92
    },
    2: {
        (6, 4, 2, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-2_lr-Newbob-4e-04.epoch.080.index"
        ),  #
        (6, 4, 0, 1, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_enc-01-6_enc-mix-4_enc-01+mix-0_enc-combine-1.epoch.080.index"
        ),  #
        (6, 4, 1, 1, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_enc-01-6_enc-mix-4_enc-01+mix-1_enc-combine-1.epoch.080.index"
        ),  #
        (6, 4, 2, 1, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-2_enc-combine-1_lr-Newbob-4e-04.epoch.080.index"
        ),  #
        (6, 0, 0, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-0_enc-01+mix-0_lr-Newbob-4e-04.epoch.080.index"
        ),  # 22.1 cv_dev93; 15.6 test_eval92
        (6, 0, 4, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-0_enc-01+mix-4_lr-Newbob-4e-04.epoch.080.index"
        ),  # 21.3 cv_dev93; 15.0 test_eval92
        (6, 4, 0, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-0_lr-Newbob-4e-04.epoch.080.index"
        ),  # 20.5 cv_dev93; 14.8 test_eval92
        (0, 6, 4, 0, 0): tk.Path(
            "/work/asr4/berger/dependencies/sms_wsj/models/blstm_Hybrid_dual_no-chunk_enc-01-6_enc-mix-4_enc-01+mix-0_lr-Newbob-4e-04.epoch.080.index"
        ),  # 21.8 cv_dev93; 15.4 test_eval92
    },
}

am_args = {
    "state_tying": "cart",
    "state_tying_file": tk.Path(
        # "/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/work/i6_core/cart/estimate/EstimateCartJob.tAyTg5fbuEMF/output/cart.tree.xml.gz"
        "/work/asr4/berger/dependencies/sms_wsj/cart/cart.tree.wsj_8k.xml.gz"
    ),
}


def segment_mapping_fn_train(seg_name: str) -> str:
    segs = str(seg_name).split("_")[1:]
    return [f"train_si284/{seg}/0000" for seg in segs]


def segment_mapping_fn_cv(seg_name: str) -> str:
    segs = str(seg_name).split("_")[1:]
    return [f"cv_dev93/{seg}/0000" for seg in segs]


def run_exp(**kwargs) -> SummaryReport:

    lm_cleaning = kwargs.get("lm_cleaning", False)

    # ********** Summary Report **********

    summary_report = SummaryReport(
        [
            key.value
            for key in [
                SummaryKey.NAME,
                SummaryKey.CORPUS,
                SummaryKey.EPOCH,
                SummaryKey.PRIOR,
                SummaryKey.LM,
                SummaryKey.WER,
                SummaryKey.SUB,
                SummaryKey.DEL,
                SummaryKey.INS,
                SummaryKey.ERR,
            ]
        ],
        col_sort_key=SummaryKey.WER.value,
    )

    # ********** Extern data **********

    _, dev_data_inputs, test_data_inputs, _ = get_data_inputs(
        train_keys=[],
        dev_keys=[recog_dev_key],
        test_keys=[recog_test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=False,
        lm_cleaning=lm_cleaning,
    )

    _, scoring_dev_data_inputs, scoring_test_data_inputs, _ = get_data_inputs(
        train_keys=[],
        dev_keys=[scoring_dev_key],
        test_keys=[scoring_test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=False,
        preprocessing=False,
        lm_cleaning=lm_cleaning,
    )

    datasets = {
        key: {
            "class": CodeWrapper("SmsWsjMixtureEarlyAlignmentDataset"),
            "num_outputs": {
                "data": {"dim": 1, "shape": (None, 1)},
                "target_signals": {"dim": 2, "shape": (None, 2)},
                "target_classes": {
                    "dim": num_outputs,
                    "shape": (None, 2),
                    "sparse": True,
                },
            },
            "seq_ordering": "default",
            "partition_epoch": {train_key: 3, dev_key: 1}[key],
            "sms_wsj_kwargs": {
                "dataset_name": key,
                "json_path": json_paths[key],
                "zip_cache": zip_cache_path,
                "zip_prefix": "/work/asr3/converse/data/",
                "shuffle": True,
                "hdf_file": alignments[key],
                "segment_mapping_fn": {
                    train_key: CodeWrapper("segment_mapping_fn_train"),
                    dev_key: CodeWrapper("segment_mapping_fn_cv"),
                }[key],
                "pad_label": num_outputs - 1,
                "hdf_data_key": "data",
            },
        }
        for key in [train_key, dev_key]
    }

    extern_data_config = {"data_time_tag": CodeWrapper('Dim(kind=Dim.Types.Time, description="time")')}

    # ********** Training setup **********

    name = "_".join(filter(None, ["blstm_Hybrid_dual", kwargs.get("name_suffix", "")]))

    num_01_layers = kwargs.get("num_01_layers", 4)
    num_mix_layers = kwargs.get("num_mix_layers", 4)
    num_01_mix_layers = kwargs.get("num_01_mix_layers", 2)
    num_combine_layers = kwargs.get("num_combine_layers", 0)
    num_context_layers = kwargs.get("num_context_layers", 0)

    if num_combine_layers:
        (train_blstm_net, train_python_code, dim_tags,) = make_blstm_hybrid_dual_output_combine_enc_model(
            num_outputs=num_outputs,
            gt_args={
                "sample_rate": frequency * 1000,
                "specaug_after_dct": kwargs.get("use_specaug", True),
            },
            blstm_01_args={"num_layers": num_01_layers, "size": 400},
            blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
            blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
            blstm_combine_args={"num_layers": num_combine_layers, "size": 800},
            aux_loss_01_layers=[(num_01_layers, 0.3)] if num_mix_layers else [],
            aux_loss_01_mix_layers=[(num_01_mix_layers, 0.3)] if num_01_mix_layers else [],
            freeze_separator=kwargs.get("freeze_separator", True),
        )
    elif num_context_layers:
        (train_blstm_net, train_python_code, dim_tags,) = make_blstm_hybrid_dual_output_soft_context_model(
            num_outputs=num_outputs,
            gt_args={
                "sample_rate": frequency * 1000,
                "specaug_after_dct": kwargs.get("use_specaug", True),
            },
            blstm_01_args={"num_layers": num_01_layers, "size": 400},
            blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
            blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
            blstm_context_args={"num_layers": num_context_layers, "size": 400},
            aux_loss_01_layers=[(num_01_layers, 0.3)] if num_mix_layers else [],
            aux_loss_01_mix_layers=[],
            pre_context_loss_scale=0.3,
            use_logits=kwargs.get("use_logits", True),
            freeze_separator=kwargs.get("freeze_separator", True),
        )
    else:
        (train_blstm_net, train_python_code, dim_tags,) = make_blstm_hybrid_dual_output_model(
            num_outputs=num_outputs,
            gt_args={
                "sample_rate": frequency * 1000,
                "specaug_after_dct": kwargs.get("use_specaug", True),
            },
            blstm_01_args={"num_layers": num_01_layers, "size": 400},
            blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
            blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
            aux_loss_01_layers=[(num_01_layers, 0.3)] if num_01_layers and num_01_mix_layers else [],
            aux_loss_01_mix_layers=[],
            freeze_separator=kwargs.get("freeze_separator", True),
        )

    extern_data_config = {
        "data": {
            "dim": 1,
            "dim_tags": [
                batch_dim,
                dim_tags["waveform_time"],
                dim_tags["waveform_feature"],
            ],
        },
        "target_signals": {
            "dim": 2,
            "shape": (None, 2),
            "dim_tags": [batch_dim, dim_tags["waveform_time"], dim_tags["speaker"]],
        },
        "target_classes": {
            "dim": num_outputs,
            "sparse": True,
            "shape": (None, 2),
            "dim_tags": [batch_dim, dim_tags["target_time"], dim_tags["speaker"]],
        },
    }

    python_prolog = {
        "modules": [
            "import functools",
            "import json",
            "import numpy as np",
            "import os.path",
            "import tensorflow as tf",
            "import subprocess as sp",
            "import sys",
            "sys.setrecursionlimit(3000)",
            "sys.path.append(os.path.dirname('/u/berger/asr-exps/recipe/i6_core'))",
            "from typing import Dict, Tuple, Any, Optional",
            "from returnn.datasets.basic import DatasetSeq",
            "from returnn.datasets.hdf import HDFDataset",
            "from returnn.datasets.map import MapDatasetBase, MapDatasetWrapper",
            "from returnn.log import log as returnn_log",
            "from returnn.util.basic import OptionalNotImplementedError, NumbersDict",
            "from sms_wsj.database import SmsWsj, AudioReader, scenario_map_fn",
        ],
        "dataset": [
            SequenceBuffer,
            ZipAudioReader,
            SmsWsjBase,
            SmsWsjBaseWithHdfClasses,
            SmsWsjWrapper,
            SmsWsjMixtureEarlyDataset,
            SmsWsjMixtureEarlyAlignmentDataset,
            segment_mapping_fn_train,
            segment_mapping_fn_cv,
        ],
        "extra_python": train_python_code,
    }

    model_preload = {
        "mask_estimator": {
            "filename": speech_sep_checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
            "prefix": "speech_separator/",
        },
    }
    init_am_version = kwargs.get("init_am", 0)
    if init_am_version:
        model_preload["am_model"] = {
            "filename": am_checkpoints[init_am_version][
                (
                    num_01_layers,
                    num_mix_layers,
                    num_01_mix_layers,
                    num_combine_layers,
                    num_context_layers,
                )
            ],
            "init_for_train": True,
            "ignore_missing": True,
        }

    num_subepochs = kwargs.get("num_subepochs", 120)

    extra_config = {
        "train": datasets[train_key],
        "dev": datasets[dev_key],
        "preload_from_files": model_preload,
        "extern_data": extern_data_config,
    }
    if num_context_layers:
        extra_config["forward_output_layer"] = "final_output"
    if kwargs.get("chunking", True):
        extra_config["chunking"] = (
            {
                "data": 128 * 10 * frequency,
                "target_signals": 128 * 10 * frequency,
                "target_classes": 128,
            },
            {
                "data": 64 * 10 * frequency,
                "target_signals": 64 * 10 * frequency,
                "target_classes": 64,
            },
        )

    train_config = get_returnn_config(
        train_blstm_net,
        target="target_classes",
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_epochs=num_subepochs,
        batch_size=kwargs.get("batch_size", 400_000),
        accum_grad=kwargs.get("accum_grad", 1),
        schedule=kwargs.get("schedule", LearningRateSchedules.Newbob),
        peak_lr=kwargs.get("peak_lr", 1e-03),
        learning_rate=kwargs.get("learning_rate", 4e-04),
        min_learning_rate=1e-06,
        n_steps_per_epoch=1550,
        use_chunking=False,
        python_prolog=python_prolog,
        extra_config=extra_config,
    )

    train_config = serialize_dim_tags(train_config)

    train_job = ReturnnTrainingJob(
        train_config,
        log_verbosity=4,
        num_epochs=num_subepochs,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=168,
        mem_rqmt=16,
    )
    train_job.update_rqmt("run", {"file_size": 100})

    train_job.set_vis_name(f"Train {name}")
    train_job.add_alias(f"train_{name}")

    tk.register_output(f"train_nn/{name}", train_job.out_learning_rates)

    ## ********** Recognition **********

    # *** Create graphs ***

    recog_graphs = {}
    for speaker_idx in [0, 1]:
        if num_combine_layers:
            (recog_blstm_net, recog_python_code, dim_tags,) = make_blstm_hybrid_dual_output_combine_enc_recog_model(
                num_outputs=num_outputs,
                speaker_idx=speaker_idx,
                gt_args={"sample_rate": frequency * 1000},
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                blstm_combine_args={"num_layers": num_combine_layers, "size": 800},
            )
        elif num_context_layers:
            (recog_blstm_net, recog_python_code, dim_tags,) = make_blstm_hybrid_dual_output_soft_context_recog_model(
                num_outputs=num_outputs,
                speaker_idx=speaker_idx,
                gt_args={"sample_rate": frequency * 1000},
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                blstm_context_args={"num_layers": num_context_layers, "size": 400},
                use_logits=kwargs.get("use_logits", True),
            )
        else:
            (recog_blstm_net, recog_python_code, dim_tags,) = make_blstm_hybrid_dual_output_recog_model(
                num_outputs=num_outputs,
                speaker_idx=speaker_idx,
                gt_args={"sample_rate": frequency * 1000},
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
            )

        extern_data_config = {
            "data": {
                "dim": 1,
                "dim_tags": [
                    batch_dim,
                    dim_tags["waveform_time"],
                    dim_tags["waveform_feature"],
                ],
            },
        }
        python_prolog["extra_python"] = recog_python_code
        recog_config = get_returnn_config(
            recog_blstm_net,
            target=None,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_epochs=num_subepochs,
            use_chunking=False,
            python_prolog=python_prolog,
            extra_config={
                "train": datasets[train_key],
                "dev": datasets[dev_key],
                "extern_data": extern_data_config,
            },
        )
        recog_config = serialize_dim_tags(recog_config)

        graph = CompileTFGraphJob(recog_config, returnn_root=tk.Path(gs.RETURNN_ROOT)).out_graph
        recog_graphs[speaker_idx] = graph

    # *** Compile nativelstm2 ***

    native_lstm_path = CompileNativeOpJob(
        native_op="NativeLstm2",
        returnn_python_exe=tk.Path(gs.RETURNN_PYTHON_EXE),
        returnn_root=tk.Path(gs.RETURNN_ROOT),
        blas_lib=(tk.Path(gs.BLAS_LIB) if hasattr(gs, "BLAS_LIB") else None),
    ).out_op

    lm_scale = kwargs.get("lm_scale", 16.0)
    prior_scale = kwargs.get("prior_scale", 0.7)

    for corpus_key, data_input, scoring_corpus_key, scoring_data_input in [
        (
            recog_dev_key,
            dev_data_inputs[recog_dev_key],
            scoring_dev_key,
            scoring_dev_data_inputs[scoring_dev_key],
        ),
        (
            recog_test_key,
            test_data_inputs[recog_test_key],
            scoring_test_key,
            scoring_test_data_inputs[scoring_test_key],
        ),
    ]:

        # *** Create base flow ***

        base_flow = samples_flow(
            dc_detection=False,
            audio_format=data_input.corpus_object.audio_format,
            input_options={"block-size": 1},
            scale_input=2**-15,
        )

        # *** Create crp for corpus ***

        crp = rasr.CommonRasrParameters()

        rasr.crp_add_default_output(crp)
        crp.set_executables(rasr_binary_path=rasr_binary_path, rasr_arch="linux-x86_64-standard")

        rasr.crp_set_corpus(crp, data_input.corpus_object)
        crp.concurrent = data_input.concurrent
        crp.segment_path = SegmentCorpusJob(
            data_input.corpus_object.corpus_file, data_input.concurrent
        ).out_segment_path

        crp.language_model_config = rasr.RasrConfig()
        crp.language_model_config.type = data_input.lm["type"]
        crp.language_model_config.file = data_input.lm["filename"]
        crp.language_model_config.scale = lm_scale

        crp.lexicon_config = rasr.RasrConfig()
        crp.lexicon_config.file = data_input.lexicon["filename"]
        crp.lexicon_config.normalize_pronunciation = data_input.lexicon["normalize_pronunciation"]

        crp.acoustic_model_config = acoustic_model_config(**am_args)
        crp.acoustic_model_config.allophones.add_all = data_input.lexicon["add_all"]
        crp.acoustic_model_config.allophones.add_from_lexicon = data_input.lexicon["add_from_lexicon"]

        model_combination_config = rasr.RasrConfig()
        model_combination_config.pronunciation_scale = 0.0

        # *** STM file for scoring

        stm_path = CorpusToStmJob(
            scoring_data_input.corpus_object.corpus_file, non_speech_tokens=["<NOISE>"]
        ).out_stm_path

        # *** Create tf flow ***

        input_name = "tf-fwd-input"

        tf_flow = FlowNetwork()
        tf_flow.add_input(input_name)
        tf_flow.add_output("features")
        tf_flow.add_param("id")

        tf_fwd_name = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tf_flow.link(f"network:{input_name}", f"{tf_fwd_name}:input")
        tf_flow.link(f"{tf_fwd_name}:log-posteriors", "network:features")

        tf_flow.config = rasr.RasrConfig()
        tf_flow.config[tf_fwd_name].input_map.info_0.param_name = "input"
        tf_flow.config[tf_fwd_name].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
        tf_flow.config[
            tf_fwd_name
        ].input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"

        tf_flow.config[tf_fwd_name].output_map.info_0.param_name = "log-posteriors"
        tf_flow.config[
            tf_fwd_name
        ].output_map.info_0.tensor_name = f"{train_config.get('forward_output_layer', 'output')}/output_batch_major"

        tf_flow.config[tf_fwd_name].loader.type = "meta"
        tf_flow.config[tf_fwd_name].loader.required_libraries = native_lstm_path

        for epoch in kwargs.get("recog_epochs", [20, 60, 80, 100, 120]):

            prefix = f"nn_recog/{train_key}_{name}_recog/"
            exp_name = f"{corpus_key}-e{epoch:03d}-prior{prior_scale:02.2f}-lm{lm_scale:02.2f}"

            ## ********** Prior computation **********

            prior_job = ReturnnComputePriorJob(
                model_checkpoint=train_job.out_checkpoints[epoch],
                returnn_config=train_config,
                log_verbosity=4,
                mem_rqmt=16,
            )
            prior_job.update_rqmt("run", {"file_size": 100})
            prior_file = prior_job.out_prior_xml_file

            # *** Create feature scorer ***

            acoustic_mixture_path = CreateDummyMixturesJob(num_outputs, num_inputs).out_mixtures
            feature_scorer = PrecomputedHybridFeatureScorer(
                prior_mixtures=acoustic_mixture_path,
                priori_scale=prior_scale,
                prior_file=prior_file,
            )

            # *** Generate two lattices ***

            scoring_reports = {}
            ctm_files = {}
            lattices = {}

            for speaker_idx in [0, 1]:

                # *** Finalize tf flow and interconnect with base flow ***

                tf_flow.config[tf_fwd_name].loader.meta_graph_file = recog_graphs[speaker_idx]
                tf_flow.config[tf_fwd_name].loader.saved_model_file = train_job.out_checkpoints[epoch]

                ext_flow = FlowNetwork()
                base_mapping = ext_flow.add_net(base_flow)
                tf_mapping = ext_flow.add_net(tf_flow)
                ext_flow.interconnect_inputs(base_flow, base_mapping)
                ext_flow.interconnect(
                    base_flow,
                    base_mapping,
                    tf_flow,
                    tf_mapping,
                    {list(base_flow.outputs)[0]: input_name},
                )
                ext_flow.interconnect_outputs(tf_flow, tf_mapping)

                # *** Search jobs ***

                recog_job = AdvancedTreeSearchJob(
                    crp=crp,
                    feature_flow=ext_flow,
                    feature_scorer=feature_scorer,
                    model_combination_config=model_combination_config,
                    search_parameters={
                        "beam-pruning": 16.0,
                        "beam-pruning-limit": 200_000,
                        "word-end-pruning": 0.5,
                        "word-end-pruning-limit": 25_000,
                    },
                    use_gpu=True,
                    mem=8,
                )
                recog_job.set_vis_name(f"Recog {prefix}{exp_name}_speaker-{speaker_idx}")
                recog_job.add_alias(f"{prefix}{exp_name}_speaker-{speaker_idx}")

                lattice_job = LatticeToCtmJob(
                    crp=crp,
                    lattice_cache=recog_job.out_lattice_bundle,
                    fill_empty_segments=True,
                    best_path_algo="bellman-ford",
                )

                lattices[speaker_idx] = recog_job.out_lattice_bundle
                ctm_files[speaker_idx] = lattice_job.out_ctm_file
                ctm_file = CtmRepeatForSpeakersJob(lattice_job.out_ctm_file, 2).out_ctm_file

                scoring_reports[speaker_idx] = ScliteJob(ref=stm_path, hyp=ctm_file).out_report_dir

            # *** Score minimum permutation ctm

            min_perm_ctm_job = MinimumPermutationCtmJob(scoring_files=scoring_reports, ctms=ctm_files, stm=stm_path)

            scorer_minimum_job = ScliteJob(ref=stm_path, hyp=min_perm_ctm_job.out_ctm_file)

            tk.register_output(f"{prefix}recog_{exp_name}.reports", scorer_minimum_job.out_report_dir)

            summary_report.add_row(
                {
                    SummaryKey.NAME.value: name,
                    SummaryKey.CORPUS.value: corpus_key,
                    SummaryKey.EPOCH.value: epoch,
                    SummaryKey.PRIOR.value: prior_scale,
                    SummaryKey.LM.value: lm_scale,
                    SummaryKey.WER.value: scorer_minimum_job.out_wer,
                    SummaryKey.SUB.value: scorer_minimum_job.out_percent_substitution,
                    SummaryKey.DEL.value: scorer_minimum_job.out_percent_deletions,
                    SummaryKey.INS.value: scorer_minimum_job.out_percent_insertions,
                    SummaryKey.ERR.value: scorer_minimum_job.out_num_errors,
                }
            )

            # *** Optimize LM scale ***
            if kwargs.get("optlm", False):
                join_lattices = JointLatticeCacheJob(
                    list(lattices.values()),
                    list(min_perm_ctm_job.out_segment_files.values()),
                    scoring_corpus_key,
                )
                crp_scoring_corpus = copy.deepcopy(crp)
                crp_scoring_corpus.corpus_config.file = scoring_data_input.corpus_object.corpus_file
                opt_job = OptimizeAMandLMScaleJob(
                    crp=crp_scoring_corpus,
                    lattice_cache=join_lattices.out_lattice_cache,
                    initial_am_scale=0.0,
                    initial_lm_scale=lm_scale,
                    scorer_cls=ScliteJob,
                    scorer_kwargs={"ref": stm_path},
                    opt_only_lm_scale=True,
                )

                opt_crp = copy.deepcopy(crp)
                opt_crp.language_model_config.scale = opt_job.out_best_lm_score

                for speaker_idx in [0, 1]:

                    # *** Finalize tf flow and interconnect with base flow ***

                    tf_flow.config[tf_fwd_name].loader.meta_graph_file = recog_graphs[speaker_idx]
                    tf_flow.config[tf_fwd_name].loader.saved_model_file = train_job.out_checkpoints[epoch]

                    ext_flow = FlowNetwork()
                    base_mapping = ext_flow.add_net(base_flow)
                    tf_mapping = ext_flow.add_net(tf_flow)
                    ext_flow.interconnect_inputs(base_flow, base_mapping)
                    ext_flow.interconnect(
                        base_flow,
                        base_mapping,
                        tf_flow,
                        tf_mapping,
                        {list(base_flow.outputs)[0]: input_name},
                    )
                    ext_flow.interconnect_outputs(tf_flow, tf_mapping)

                    # *** Search jobs ***

                    recog_job = AdvancedTreeSearchJob(
                        crp=opt_crp,
                        feature_flow=ext_flow,
                        feature_scorer=feature_scorer,
                        model_combination_config=model_combination_config,
                        search_parameters={
                            "beam-pruning": 16.0,
                            "beam-pruning-limit": 200_000,
                            "word-end-pruning": 0.5,
                            "word-end-pruning-limit": 25_000,
                        },
                        use_gpu=True,
                        mem=8,
                    )
                    recog_job.set_vis_name(f"Recog {prefix}{exp_name}_speaker-{speaker_idx}_optlm")
                    recog_job.add_alias(f"{prefix}{exp_name}_speaker-{speaker_idx}_optlm")

                    lattice_job = LatticeToCtmJob(
                        crp=opt_crp,
                        lattice_cache=recog_job.out_lattice_bundle,
                        fill_empty_segments=True,
                        best_path_algo="bellman-ford",
                    )

                    lattices[speaker_idx] = recog_job.out_lattice_bundle
                    ctm_files[speaker_idx] = lattice_job.out_ctm_file
                    ctm_file = CtmRepeatForSpeakersJob(lattice_job.out_ctm_file, 2).out_ctm_file

                    scoring_reports[speaker_idx] = ScliteJob(ref=stm_path, hyp=ctm_file).out_report_dir

                # *** Score minimum permutation ctm

                min_perm_ctm_job = MinimumPermutationCtmJob(scoring_files=scoring_reports, ctms=ctm_files, stm=stm_path)

                scorer_minimum_job = ScliteJob(ref=stm_path, hyp=min_perm_ctm_job.out_ctm_file)

                tk.register_output(
                    f"{prefix}recog_{exp_name}_optlm.reports",
                    scorer_minimum_job.out_report_dir,
                )

                summary_report.add_row(
                    {
                        SummaryKey.NAME.value: name,
                        SummaryKey.CORPUS.value: corpus_key,
                        SummaryKey.EPOCH.value: epoch,
                        SummaryKey.PRIOR.value: prior_scale,
                        SummaryKey.LM.value: opt_job.out_best_lm_score,
                        SummaryKey.WER.value: scorer_minimum_job.out_wer,
                        SummaryKey.SUB.value: scorer_minimum_job.out_percent_substitution,
                        SummaryKey.DEL.value: scorer_minimum_job.out_percent_deletions,
                        SummaryKey.INS.value: scorer_minimum_job.out_percent_insertions,
                        SummaryKey.ERR.value: scorer_minimum_job.out_num_errors,
                    }
                )

    return summary_report


def py() -> SummaryReport:

    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    summary_report = SummaryReport()

    # AM-only training
    if True:
        for enc01, encmix, enc01mix in [
            (6, 4, 2),
            (6, 6, 2),
            (6, 0, 0),
            (6, 0, 4),
            (6, 4, 0),
            (0, 6, 4),
        ]:
            summary_report.merge_report(
                run_exp(
                    name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}",
                    num_01_layers=enc01,
                    num_mix_layers=encmix,
                    num_01_mix_layers=enc01mix,
                    schedule=LearningRateSchedules.Newbob,
                    learning_rate=4e-04,
                    chunking=False,
                    num_subepochs=80,
                    recog_epochs=[20, 60, 80],
                    optlm=True,
                ),
                update_structure=True,
                collapse_rows=False,
            )

    # Joint training
    if False:
        for enc01, encmix, enc01mix in [
            # (4, 2, 2),
            # (4, 4, 2),
            (6, 4, 2),
            (6, 0, 0),
        ]:
            for lr in [1e-04]:
                summary_report.merge_report(
                    run_exp(
                        name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_lr-Newbob-{lr}_joint",
                        num_01_layers=enc01,
                        num_mix_layers=encmix,
                        num_01_mix_layers=enc01mix,
                        schedule=LearningRateSchedules.Newbob,
                        learning_rate=lr,
                        chunking=False,
                        init_am=0,
                        freeze_separator=False,
                    ),
                    update_structure=True,
                    collapse_rows=False,
                )

    # Better LR and batch size
    if False:
        for enc01, encmix, enc01mix in [
            (6, 4, 2),
            (6, 0, 0),
        ]:
            if encmix:
                bs = 400_000
                accum_grad = 2
            else:
                bs = 800_000
                accum_grad = 1
            summary_report.merge_report(
                run_exp(
                    name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_joint",
                    num_01_layers=enc01,
                    num_mix_layers=encmix,
                    num_01_mix_layers=enc01mix,
                    schedule=LearningRateSchedules.Newbob,
                    learning_rate=3e-05,
                    batch_size=bs,
                    accum_grad=accum_grad,
                    chunking=False,
                    init_am=1,
                    freeze_separator=False,
                ),
                collapse_rows=False,
            )

    # Combine output layer
    if True:
        for enc01, encmix, enc01mix, combine in [
            (6, 4, 0, 1),
            (6, 4, 1, 1),
            (6, 4, 2, 1),
        ]:
            summary_report.merge_report(
                run_exp(
                    name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_enc-combine-{combine}",
                    num_01_layers=enc01,
                    num_mix_layers=encmix,
                    num_01_mix_layers=enc01mix,
                    num_combine_layers=combine,
                    schedule=LearningRateSchedules.Newbob,
                    learning_rate=4e-04,
                    batch_size=400_000,
                    accum_grad=2,
                    chunking=False,
                    init_am=1,
                    freeze_separator=True,
                    optlm=True,
                    num_subepochs=80,
                    recog_epochs=[20, 60, 80],
                ),
                update_structure=True,
                collapse_rows=False,
            )

    # Soft context model
    if False:
        for enc01, encmix, enc01mix, enccon in [
            (4, 4, 2, 1),
        ]:
            for logits in [True]:
                for lr in [4e-04]:
                    summary_report.merge_report(
                        run_exp(
                            name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_enc-con-{enccon}_logit-{logits}_lr-Newbob-{lr}",
                            num_01_layers=enc01,
                            num_mix_layers=encmix,
                            num_01_mix_layers=enc01mix,
                            num_context_layers=enccon,
                            use_logits=logits,
                            schedule=LearningRateSchedules.Newbob,
                            learning_rate=lr,
                            batch_size=400_000,
                            accum_grad=2,
                            chunking=False,
                            init_am=0,
                            freeze_separator=False,
                        ),
                        collapse_rows=False,
                    )

    # No specaugment
    if False:
        for enc01, encmix, enc01mix in [
            (6, 4, 2),
            (6, 0, 0),
        ]:
            if encmix:
                bs = 400_000
                accum_grad = 2
            else:
                bs = 800_000
                accum_grad = 1
            summary_report.merge_report(
                run_exp(
                    name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_no-specaug_joint",
                    num_01_layers=enc01,
                    num_mix_layers=encmix,
                    num_01_mix_layers=enc01mix,
                    schedule=LearningRateSchedules.Newbob,
                    learning_rate=3e-05,
                    use_specaug=False,
                    batch_size=bs,
                    accum_grad=accum_grad,
                    chunking=False,
                    init_am=1,
                    freeze_separator=False,
                ),
                collapse_rows=False,
            )

    # New initializing
    for enc01, encmix, enc01mix, combine in [
        (6, 0, 0, 0),
        (6, 4, 0, 0),
        (0, 6, 4, 0),
        (6, 0, 4, 0),
        (6, 4, 2, 0),
        (6, 4, 0, 1),
        (6, 4, 1, 1),
        (6, 4, 2, 1),
    ]:
        if encmix or enc01mix or combine:
            bs = 400_000
            accum_grad = 2
        else:
            bs = 800_000
            accum_grad = 1
        summary_report.merge_report(
            run_exp(
                name_suffix=f"enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_combine-{combine}_no-specaug_joint_v2",
                num_01_layers=enc01,
                num_mix_layers=encmix,
                num_01_mix_layers=enc01mix,
                num_combine_layers=combine,
                schedule=LearningRateSchedules.Newbob,
                learning_rate=3e-05,
                use_specaug=False,
                batch_size=bs,
                accum_grad=accum_grad,
                chunking=False,
                init_am=2,
                freeze_separator=False,
                optlm=True,
                recog_epochs=[20, 40, 60, 80],
            ),
            collapse_rows=False,
        )

    tk.register_report(
        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report",
        summary_report,
    )
    return summary_report
