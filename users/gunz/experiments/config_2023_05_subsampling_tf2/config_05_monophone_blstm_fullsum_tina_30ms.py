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
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    BLSTM_FH_TINA_DECODING_TENSOR_CONFIG,
    CONF_CHUNKING_10MS,
    CONF_FOCAL_LOSS,
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
    alignment_name: str
    bw_label_scale: float
    feature_time_shift: float
    import_checkpoint: returnn.Checkpoint
    lr: str
    multitask: bool
    dc_detection: bool
    subsampling_factor: int

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    configs = [
        Experiment(
            alignment_name="scratch",
            bw_label_scale=0.3,
            dc_detection=False,
            feature_time_shift=7.5 / 1000,
            import_checkpoint=returnn.Checkpoint(
                tk.Path(
                    "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/tina-blstm-7.5ms-ss-4/epoch.493.index"
                )
            ),
            lr="v6",
            multitask=False,
            subsampling_factor=4,
        ),
    ]
    experiments = {
        exp: run_single(
            alignment_name=exp.alignment_name,
            bw_label_scale=exp.bw_label_scale,
            dc_detection=exp.dc_detection,
            feature_time_shift=exp.feature_time_shift,
            focal_loss=exp.focal_loss,
            import_checkpoint=exp.import_checkpoint,
            lr=exp.lr,
            multitask=exp.multitask,
            returnn_root=returnn_root,
            subsampling_factor=exp.subsampling_factor,
        )
        for exp in configs
    }

    return experiments


def run_single(
    *,
    alignment_name: str,
    bw_label_scale: float,
    dc_detection: bool,
    feature_time_shift: float,
    focal_loss: float,
    import_checkpoint: returnn.Checkpoint,
    lr: str,
    multitask: bool,
    returnn_root: tk.Path,
    subsampling_factor: int,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"blstm-1-fs_tina-ss:{subsampling_factor}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()

    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=dc_detection)
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

    s.experiments["fh"]["train_job"] = FakeReturnnJob(493, import_checkpoint)

    s.experiments["fh"]["graph"]["inference"] = tk.Path(
        "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/tina-blstm-7.5ms-ss-4/graph.meta"
    )
    s.experiments["fh"]["priors"] = PriorInfo(
        PriorConfig(
            file=tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/train/tina-blstm-7.5ms-ss-4/prior.xml",
            ),
            scale=0.0,
        )
    )

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
        epoch=493,
        gpu=False,
        tensor_map=BLSTM_FH_TINA_DECODING_TENSOR_CONFIG,
        set_batch_major_for_feature_scorer=False,
        tf_library=s.native_lstm2_job.out_op,
        lm_gc_simple_hash=True,
    )
    sil_tdp = (*recog_args.tdp_silence[:3], 3.0)
    align_cfg = (
        recog_args.with_prior_scale(0.5).with_tdp_scale(1.0).with_tdp_silence(sil_tdp).with_tdp_non_word(sil_tdp)
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

    a_job = recognizer.align(
        f"{name}-pC{align_cfg.prior_info.center_state_prior.scale}-tdp{align_cfg.tdp_scale}",
        crp=crp,
        feature_scorer=align_search_jobs.search_feature_scorer,
        default_tdp=True,
    )

    allophones = lexicon.StoreAllophonesJob(crp)
    tk.register_output(f"allophones/{name}/allophones", allophones.out_allophone_file)

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        segments=[
            "train-other-960/2920-156224-0013/2920-156224-0013",
            "train-other-960/2498-134786-0003/2498-134786-0003",
            "train-other-960/6178-86034-0008/6178-86034-0008",
            "train-other-960/5983-39669-0034/5983-39669-0034",
        ],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/{name}/alignment-plots", plots.out_plot_folder)

    phoneme_durs = PlotPhonemeDurationsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        time_step_s=40 / 1000,
    )
    tk.register_output(f"alignments/{name}/statistics/plots", phoneme_durs.out_plot_folder)
    tk.register_output(f"alignments/{name}/statistics/means", phoneme_durs.out_means)
    tk.register_output(f"alignments/{name}/statistics/variances", phoneme_durs.out_vars)

    return s
