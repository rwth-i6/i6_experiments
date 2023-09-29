__all__ = ["run", "run_single"]

import copy
import dataclasses
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import corpus, lexicon, rasr, returnn
import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.analysis import PlotPhonemeDurationsJob, PlotViterbiAlignmentsJob
from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import PriorConfig, PriorInfo
from ...setups.fh.decoder.search import DecodingTensorMap
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    BLSTM_FH_DECODING_TENSOR_CONFIG,
    BLSTM_FH_TINA_DECODING_TENSOR_CONFIG,
    CONF_CHUNKING_10MS,
    RASR_ARCH,
    RASR_ROOT_NO_TF,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_NO_TF, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RASR_BINARY_PATH_TF = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True, eq=True)
class Experiment:
    feature_time_shift: float
    import_checkpoint: returnn.Checkpoint
    import_epoch: int
    import_graph: tk.Path
    import_priors: tk.Path
    name: str
    t_step: float
    tensor_config: DecodingTensorMap


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    configs = [
        Experiment(
            feature_time_shift=10 / 1000,
            import_checkpoint=returnn.Checkpoint(
                tk.Path(
                    "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/tina-blstm-30ms-fs/epoch.493.index",
                )
            ),
            import_epoch=493,
            import_graph=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/tina-blstm-30ms-fs/graph.meta"
            ),
            import_priors=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/tina-blstm-30ms-fs/prior.xml"
            ),
            name="30ms-fs",
            t_step=30 / 1000,
            tensor_config=BLSTM_FH_TINA_DECODING_TENSOR_CONFIG,
        ),
        Experiment(
            feature_time_shift=7.5 / 1000,
            import_checkpoint=returnn.Checkpoint(
                tk.Path(
                    "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/blstm-30ms-mp/epoch.600.index",
                )
            ),
            import_epoch=600,
            import_graph=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/blstm-30ms-mp/graph.meta"
            ),
            import_priors=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/blstm-30ms-mp/prior.xml"
            ),
            name="30ms-mp",
            t_step=30 / 1000,
            tensor_config=BLSTM_FH_DECODING_TENSOR_CONFIG,
        ),
        Experiment(
            feature_time_shift=10 / 1000,
            import_checkpoint=returnn.Checkpoint(
                tk.Path(
                    "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/blstm-40ms-mp/epoch.600.index",
                )
            ),
            import_epoch=600,
            import_graph=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/blstm-40ms-mp/graph.meta"
            ),
            import_priors=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/blstm-40ms-mp/priors.xml"
            ),
            name="40ms-mp",
            t_step=40 / 1000,
            tensor_config=BLSTM_FH_DECODING_TENSOR_CONFIG,
        ),
    ]
    experiments = {
        exp: run_single(
            exp_name=exp.name,
            feature_time_shift=exp.feature_time_shift,
            import_checkpoint=exp.import_checkpoint,
            import_epoch=exp.import_epoch,
            import_graph=exp.import_graph,
            import_priors=exp.import_priors,
            returnn_root=returnn_root,
            tensor_config=exp.tensor_config,
        )
        for exp in configs
    }

    return experiments


def run_single(
    *,
    exp_name: str,
    feature_time_shift: float,
    import_checkpoint: returnn.Checkpoint,
    import_epoch: int,
    import_graph: tk.Path,
    import_priors: tk.Path,
    returnn_root: tk.Path,
    tensor_config: DecodingTensorMap,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"blstm-1-previous-{exp_name}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()

    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=False)
    rasr_init_args.feature_extraction_args["gt"]["parallel"] = 50
    rasr_init_args.feature_extraction_args["gt"]["rtf"] = 0.5
    rasr_init_args.feature_extraction_args["gt"]["gt_options"]["tempint_shift"] = feature_time_shift

    data_preparation_args = gmm_setups.get_final_output(name="data_preparation")
    # *********** System Instantiation *****************
    steps = rasr_util.RasrSteps()
    steps.add_step("init", None)  # you can create the label_info and pass here
    s = fh_system.FactoredHybridSystem(
        rasr_binary_path=RASR_BINARY_PATH,
        rasr_init_args=rasr_init_args,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )

    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1, state_tying=RasrStateTying.triphone)
    s.lexicon_args["norm_pronunciation"] = False
    s.train_key = train_key

    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING_10MS,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="heuristic",
        eval_tdp_type="heuristic",
        add_base_allophones=False,
    )

    s.set_experiment_dict("fh", "scratch", "mono", postfix_name=name)

    class FakeReturnnJob:
        def __init__(self, epoch: int, ckpt: returnn.Checkpoint):
            self.out_checkpoints = {epoch: ckpt}

    s.experiments["fh"]["train_job"] = FakeReturnnJob(import_epoch, import_checkpoint)

    s.experiments["fh"]["graph"]["inference"] = import_graph
    s.experiments["fh"]["priors"] = PriorInfo(PriorConfig(file=import_priors, scale=0.0))

    s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.triphone)
    s._update_crp_am_setting(crp_key="dev-other", tdp_type="default", add_base_allophones=False)
    s.set_binaries_for_crp("train-other-960.train", RASR_BINARY_PATH_TF)
    s.create_stm_from_corpus("train-other-960.train")
    s._set_scorer_for_corpus("train-other-960.train")
    s._init_lm("train-other-960.train", **next(iter(dev_data_inputs.values())).lm)
    s._update_crp_am_setting("train-other-960.train", tdp_type="default", add_base_allophones=False)
    recognizer, recog_args = s.get_recognizer_and_args(
        key="fh",
        context_type=PhoneticContext.monophone,
        crp_corpus="train-other-960.train",
        epoch=import_epoch,
        gpu=False,
        tensor_map=tensor_config,
        set_batch_major_for_feature_scorer=False,
        tf_library=s.native_lstm2_job.out_op,
        lm_gc_simple_hash=True,
    )
    sil_tdp = (*recog_args.tdp_silence[:3], 0.0)
    align_cfg = (
        recog_args.with_prior_scale(0.0)
        .with_tdp_scale(1.0)
        .with_tdp_silence(sil_tdp)
        .with_tdp_non_word(sil_tdp)
        .with_lm_scale(0.8)
    )
    align_search_jobs = recognizer.recognize_count_lm(
        label_info=s.label_info,
        search_parameters=align_cfg,
        num_encoder_output=2 * 512,
        rerun_after_opt_lm=False,
        opt_lm_am=False,
        add_sis_alias_and_output=False,
        calculate_stats=True,
        rtf_cpu=12,
    )
    crp = copy.deepcopy(align_search_jobs.search_crp)
    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    crp.acoustic_model_config.allophones.add_all = False
    crp.acoustic_model_config.allophones.add_from_lexicon = True
    crp.concurrent = 300
    crp.segment_path = corpus.SegmentCorpusJob(s.corpora[s.train_key].corpus_file, crp.concurrent).out_segment_path

    a_name = f"{name}-pC{align_cfg.prior_info.center_state_prior.scale}-tdp{align_cfg.tdp_scale}-silE{align_cfg.tdp_silence[-1]}"

    a_job = recognizer.align(
        a_name,
        crp=crp,
        feature_scorer=align_search_jobs.search_feature_scorer,
        default_tdp=True,
    )

    allophones = lexicon.StoreAllophonesJob(crp)
    tk.register_output(f"allophones/{a_name}/allophones", allophones.out_allophone_file)

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        segments=["train-other-960/2920-156224-0013/2920-156224-0013"],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/{a_name}/alignment-plots", plots.out_plot_folder)

    phoneme_durs = PlotPhonemeDurationsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        time_step_s=40 / 1000,
    )
    tk.register_output(f"alignments/{a_name}/statistics/plots", phoneme_durs.out_plot_folder)
    tk.register_output(f"alignments/{a_name}/statistics/means", phoneme_durs.out_means)
    tk.register_output(f"alignments/{a_name}/statistics/variances", phoneme_durs.out_vars)

    s.experiments["fh"]["alignment_job"] = a_job

    return s
