"""
Config for pre-training experiments on LibriSpeech using wav2vec 2.0.
"""
import os.path

from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
from .fairseq import SetupFairseqJob
from .config_01_fairseq_main import get_fairseq_args


def get_alignment_hdf():
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="1a6554570058dd632c65b6d3328cadeab018d2d5",
    ).out_repository
    returnn_root.hash_overwrite = "TEDLIUM_HYBRID_RETURNN_ROOT"

    dependency_dir = None
    for dep_dir in [
        "/work/asr4/vieting/setups/librispeech/dependencies/",
        "/work/pv653172/data/librispeech/",
    ]:
        if os.path.exists(dep_dir):
            dependency_dir = dep_dir
    assert dependency_dir is not None
    alignment_caches = [tk.Path(
        os.path.join(dependency_dir, f"alignments/monophone_10ms_gmm_fix/output/alignment.cache.{idx}"),
        hash_overwrite=f"ls960_monophone_10ms_gmm_fix_alignment_{idx}",
    ) for idx in range(1, 201)]
    allophone_file = tk.Path(
        os.path.join(
            dependency_dir,
            f"alignments/monophone_10ms_gmm_fix/dependencies/StoreAllophonesJob.68JjXthmrl8y/output/allophones"
        ),
        hash_overwrite="ls960_monophone_10ms_gmm_fix_allophones",
    )
    state_tying_file = tk.Path(
        os.path.join(
            dependency_dir,
            "alignments/monophone_10ms_gmm_fix/dependencies/state-tying-map-to-single-state"
        ),
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


def get_fairseq_root(commit="e4a2e4e93efbcbaaae52a17ae6600beb2083fb33", fairseq_exe=None):
    fairseq_root = CloneGitRepositoryJob(
        "git@github.com:vieting/fairseq_phoneme.git",
        checkout_folder_name="fairseq",
        commit=commit).out_repository
    fairseq_root = SetupFairseqJob(fairseq_root, python_exe=fairseq_exe).out_fairseq_root
    return fairseq_root


def run_fairseq_pretraining_negatives_other_target():
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    alignment = get_alignment_hdf()
    num_gpus = 8
    fairseq_python_exe = tk.Path(
        "/home/pv653172/setups/librispeech/20230328_wav2vec2/dependencies/python_launcher.sh",
        hash_overwrite="itc_python_launcher_py310_torch",
    )
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe)
    fairseq_training_args = dict(
        save_interval=25,
        max_epoch=300,
        max_update=400000,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_python_exe,
        rqmt={"time": 120, "mem": 12, "cpu": 2, "gpu": num_gpus},
    )

    # run pre-training
    exp_name = "monophone_negatives_other_target_v1"
    fairseq_args = get_fairseq_args(num_gpus=num_gpus)
    fairseq_args["task"]["alignment"] = alignment
    fairseq_args["model"]["negative_sampling_strategy"] = "other_target"
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe, commit="1397363c5c0e3c4e3ab620be562730399c852493")
    fairseq_training_args["fairseq_root"] = fairseq_root
    fairseq_config = FairseqHydraConfig(fairseq_args)
    job = FairseqHydraTrainingJob(fairseq_config, **fairseq_training_args)
    job.add_alias(os.path.join(prefix_name, exp_name, "pretraining"))
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
    return job


def run_fairseq_pretraining_phoneme_boundary_masking():
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    alignment = get_alignment_hdf()
    num_gpus = 8
    fairseq_python_exe = tk.Path(
        "/home/pv653172/setups/librispeech/20230328_wav2vec2/dependencies/python_launcher.sh",
        hash_overwrite="itc_python_launcher_py310_torch",
    )
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe)
    fairseq_training_args = dict(
        save_interval=25,
        max_epoch=600,
        max_update=420000,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_python_exe,
        rqmt={"time": 120, "mem": 12, "cpu": 2, "gpu": num_gpus},
    )

    # run pre-training
    exp_name = "monophone_boundary_masking_v1"
    fairseq_args = get_fairseq_args(num_gpus=num_gpus)
    fairseq_args["task"]["alignment"] = alignment
    fairseq_args["model"]["mask_strategy"] = "phoneme"
    fairseq_args["model"]["mask_length"] = 1
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe, commit="b768be5b81987364d39a07d1caad2bfe1e956896")
    fairseq_training_args["fairseq_root"] = fairseq_root
    fairseq_config = FairseqHydraConfig(fairseq_args)
    job = FairseqHydraTrainingJob(fairseq_config, **fairseq_training_args)
    job.add_alias(os.path.join(prefix_name, exp_name, "pretraining"))
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
    return job


def run_fairseq_pretraining_phoneme_negatives_other_target_boundary_masking():
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    alignment = get_alignment_hdf()
    num_gpus = 8
    fairseq_python_exe = tk.Path(
        "/home/pv653172/setups/librispeech/20230328_wav2vec2/dependencies/python_launcher.sh",
        hash_overwrite="itc_python_launcher_py310_torch",
    )
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe)
    fairseq_training_args = dict(
        save_interval=25,
        max_epoch=600,
        max_update=420000,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_python_exe,
        rqmt={"time": 120, "mem": 12, "cpu": 2, "gpu": num_gpus},
    )

    # run pre-training
    exp_name = "monophone_negatives_other_target_boundary_masking_v1"
    fairseq_args = get_fairseq_args(num_gpus=num_gpus)
    fairseq_args["task"]["alignment"] = alignment
    fairseq_args["model"]["negative_sampling_strategy"] = "other_target"
    fairseq_args["model"]["mask_strategy"] = "phoneme"
    fairseq_args["model"]["mask_length"] = 1
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe, commit="b768be5b81987364d39a07d1caad2bfe1e956896")
    fairseq_training_args["fairseq_root"] = fairseq_root
    fairseq_config = FairseqHydraConfig(fairseq_args)
    job = FairseqHydraTrainingJob(fairseq_config, **fairseq_training_args)
    job.add_alias(os.path.join(prefix_name, exp_name, "pretraining"))
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
    return job


def py():
    run_fairseq_pretraining_negatives_other_target()
    run_fairseq_pretraining_phoneme_boundary_masking()
    run_fairseq_pretraining_phoneme_negatives_other_target_boundary_masking()
