"""
Config for pre-training experiments on LibriSpeech using wav2vec 2.0.
"""
import os.path

from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from .fairseq import SetupFairseqJob
from .config_01_fairseq_main import get_fairseq_args


def get_alignment_hdf():
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="1a6554570058dd632c65b6d3328cadeab018d2d5",
    ).out_repository
    returnn_root.hash_overwrite = "TEDLIUM_HYBRID_RETURNN_ROOT"
    from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
    alignment_caches = [tk.Path(
        "/work/asr4/vieting/setups/librispeech/dependencies/alignments/monophone_10ms_gmm_fix/output/"
        f"alignment.cache.{idx}",
        hash_overwrite=f"ls960_monophone_10ms_gmm_fix_alignment_{idx}",
    ) for idx in range(1, 201)]
    allophone_file = tk.Path(
        "/work/asr4/vieting/setups/librispeech/dependencies/alignments/monophone_10ms_gmm_fix/dependencies/"
        "StoreAllophonesJob.68JjXthmrl8y/output/allophones",
        hash_overwrite="ls960_monophone_10ms_gmm_fix_allophones",
    )
    # import i6_core.am as am
    # import i6_core.rasr as rasr
    # from i6_core.lexicon.allophones import DumpStateTyingJob
    # from i6_experiments.common.datasets.librispeech import get_bliss_lexicon
    # from i6_experiments.common.baselines.tedlium2.default_tools import RASR_BINARY_PATH
    # crp = rasr.CommonRasrParameters()
    # rasr.crp_add_default_output(crp)
    # am_args = {
    #     "state_tying": "monophone",
    #     "states_per_phone": 1,
    #     "state_repetitions": 1,
    #     "across_word_model": True,
    #     "early_recombination": False,
    #     "tdp_scale": 1.0,
    #     "tdp_transition": (3.0, 0.0, "infinity", 0.0),  # loop, forward, skip, exit
    #     "tdp_silence": (0.0, 3.0, "infinity", 20.0),
    #     "tying_type": "global",
    #     "nonword_phones": "",
    # }
    # crp.acoustic_model_config = am.acoustic_model_config(**am_args)
    # crp.acoustic_model_config.allophones.add_from_lexicon = False
    # crp.acoustic_model_config.allophones.add_from_file = allophone_file
    # crp.allophone_tool_exe = RASR_BINARY_PATH + "/allophone-tool.linux-x86_64-standard"
    # crp.lexicon_config = rasr.RasrConfig()
    # crp.lexicon_config.file = get_bliss_lexicon()
    # crp.lexicon_config.normalize_pronunciation = False
    # state_tying_file = DumpStateTyingJob(crp).out_state_tying
    state_tying_file = tk.Path(
        "/work/asr4/vieting/setups/librispeech/dependencies/alignments/monophone_10ms_gmm_fix/dependencies/"
        "state-tying-map-to-single-state",
        hash_overwrite="ls960_monophone_10ms_gmm_state_tying_monophone_1",
    )
    job = RasrAlignmentDumpHDFJob(
        alignment_caches=alignment_caches,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file,
        returnn_root=returnn_root,
        sparse=True,
    )
    return job.out_hdf_files


def get_fairseq_root():
    fairseq_root = CloneGitRepositoryJob(
        "https://github.com/facebookresearch/fairseq",
        checkout_folder_name="fairseq",
        commit="176cd934982212a4f75e0669ee81b834ee71dbb0").out_repository
    fairseq_root = SetupFairseqJob(fairseq_root).out_fairseq_root
    return fairseq_root


def run_fairseq_pretraining_informed():
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    # run pre-training
    exp_name = "monophone1"
    alignment = get_alignment_hdf()
    fairseq_args = get_fairseq_args(num_gpus=8)
    fairseq_args["task"]["alignment"] = alignment
    fairseq_config = FairseqHydraConfig(fairseq_args)
    fairseq_root = get_fairseq_root()
    job = FairseqHydraTrainingJob(
        fairseq_config,
        save_interval=25,
        max_epoch=300,
        max_update=400000,
        fairseq_root=fairseq_root,
        rqmt={"time": 120, "mem": 8, "cpu": 2, "gpu": 8},
    )
    job.add_alias(os.path.join(prefix_name, exp_name, "pretraining"))
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
    return job


def py():
    run_fairseq_pretraining_informed()
