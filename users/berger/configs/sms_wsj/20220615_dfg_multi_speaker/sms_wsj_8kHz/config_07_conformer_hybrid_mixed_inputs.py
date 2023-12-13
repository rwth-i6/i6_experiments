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
from i6_experiments.users.berger.network.models.conformer_hybrid_dual_output import (
    make_conformer_hybrid_dual_output_model,
    make_conformer_hybrid_dual_output_recog_model,
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
        col_sort_key=SummaryKey.ERR.value,
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

    name = "_".join(filter(None, ["Conformer_Hybrid_dual", kwargs.get("name_suffix", "")]))

    num_01_blocks = kwargs.get("num_01_blocks", 12)
    num_mix_blocks = kwargs.get("num_mix_blocks", 8)
    num_01_mix_blocks = kwargs.get("num_01_mix_blocks", 4)

    aux_loss_01_blocks = [(num_01_blocks // 2, 0.3)]
    if num_01_mix_blocks:
        aux_loss_01_blocks.append((num_01_blocks, 0.3))

    aux_loss_01_mix_blocks = []
    if num_01_mix_blocks > 4:
        aux_loss_01_mix_blocks.append((num_01_mix_blocks // 2, 0.3))

    (train_conformer_net, train_python_code, dim_tags,) = make_conformer_hybrid_dual_output_model(
        num_outputs=num_outputs,
        gt_args={"sample_rate": frequency * 1000},
        conformer_01_args={"num_blocks": num_01_blocks},
        conformer_mix_args={"num_blocks": num_mix_blocks},
        conformer_01_mix_args={"num_blocks": num_01_mix_blocks},
        aux_loss_01_blocks=aux_loss_01_blocks,
        aux_loss_01_mix_blocks=aux_loss_01_mix_blocks,
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
        }
    }

    num_subepochs = kwargs.get("num_subepochs", 120)

    extra_config = {
        "train": datasets[train_key],
        "dev": datasets[dev_key],
        "preload_from_files": model_preload,
        "extern_data": extern_data_config,
    }

    use_chunking = kwargs.get("use_chunking", False)
    if use_chunking:
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
        train_conformer_net,
        target="target_classes",
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_epochs=num_subepochs,
        batch_size=kwargs.get("batch_size", 300000),
        accum_grad=kwargs.get("accum_grad", 2),
        schedule=kwargs.get("schedule", LearningRateSchedules.Newbob),
        learning_rate=kwargs.get("learning_rate", 4e-04),
        min_learning_rate=1e-06,
        use_chunking=use_chunking,
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
        (recog_conformer_net, recog_python_code, dim_tags,) = make_conformer_hybrid_dual_output_recog_model(
            num_outputs=num_outputs,
            speaker_idx=speaker_idx,
            gt_args={"sample_rate": frequency * 1000},
            conformer_01_args={"num_blocks": num_01_blocks},
            conformer_mix_args={"num_blocks": num_mix_blocks},
            conformer_01_mix_args={"num_blocks": num_01_mix_blocks},
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
            recog_conformer_net,
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

    for corpus_key, data_input, scoring_data_input in [
        (
            recog_dev_key,
            dev_data_inputs[recog_dev_key],
            scoring_dev_data_inputs[scoring_dev_key],
        ),
        (
            recog_test_key,
            test_data_inputs[recog_test_key],
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

        for epoch in kwargs.get("recog_epochs", [20, 40, 60, 80, 100, 110, 120]):
            prefix = f"nn_recog/{train_key}_{name}_recog/"
            exp_name = f"{corpus_key}-e{epoch:03d}-prior{prior_scale:02.2f}-lm{lm_scale:02.2f}"

            ## ********** Prior computation **********

            prior_file = ReturnnComputePriorJob(
                model_checkpoint=train_job.out_checkpoints[epoch],
                returnn_config=train_config,
                log_verbosity=4,
                mem_rqmt=16,
            ).out_prior_xml_file

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

    return summary_report


def py() -> SummaryReport:

    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    summary_report = SummaryReport()

    # Without chunking
    for enc01, encmix, enc01mix in [(12, 0, 0), (8, 4, 4), (6, 6, 4), (6, 4, 4)]:
        summary_report.merge_report(
            run_exp(
                name_suffix=f"no-chunk_enc-01-{enc01}_enc-mix-{encmix}_enc-01+mix-{enc01mix}_lr-Newbob-4e-04",
                num_01_blocks=enc01,
                num_mix_blocks=encmix,
                num_01_mix_blocks=enc01mix,
                schedule=LearningRateSchedules.Newbob,
                learning_rate=4e-04,
                chunking=False,
            ),
            update_structure=True,
        )

    tk.register_report(
        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report",
        summary_report,
    )
    return summary_report
