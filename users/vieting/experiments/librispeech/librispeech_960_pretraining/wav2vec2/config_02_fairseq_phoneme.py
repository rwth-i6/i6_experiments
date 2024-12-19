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
    alignment_caches = [
        tk.Path(
            os.path.join(dependency_dir, f"alignments/monophone_10ms_gmm_fix/output/alignment.cache.{idx}"),
            hash_overwrite=f"ls960_monophone_10ms_gmm_fix_alignment_{idx}",
        )
        for idx in range(1, 201)
    ]
    allophone_file = tk.Path(
        os.path.join(
            dependency_dir,
            f"alignments/monophone_10ms_gmm_fix/dependencies/StoreAllophonesJob.68JjXthmrl8y/output/allophones",
        ),
        hash_overwrite="ls960_monophone_10ms_gmm_fix_allophones",
    )
    state_tying_file = tk.Path(
        os.path.join(dependency_dir, "alignments/monophone_10ms_gmm_fix/dependencies/state-tying-map-to-single-state"),
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
        "git@github.com:vieting/fairseq_phoneme.git", checkout_folder_name="fairseq", commit=commit
    ).out_repository
    fairseq_root = SetupFairseqJob(fairseq_root, python_exe=fairseq_exe).out_fairseq_root
    return fairseq_root


def run_fairseq_pretraining(exp_name, commit, python_exe_hash_overwrite=None, checkpoint=None, **kwargs):
    """
    Runs a FairseqHydraTrainingJob to pretrain a wav2vec 2.0 model.

    Args:
        exp_name (str): The name of the experiment, used for output and alias folder.
        commit (str): The commit ID of the fairseq_phoneme repository to use.
        python_exe_hash_overwrite (Optional[str]): The hash overwrite for the fairseq_python_exe to use.
            It should only be used to achieve compatibility with the previous setup structure and should be ignored
            in all other cases.
        checkpoint (Optional[str]): The path to the checkpoint to start from. If None, the training will start
            from scratch.
        **kwargs: Additional arguments to pass to the job. These will be used to overwrite the model configuration.
    """
    # job requirements
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    alignment = get_alignment_hdf()
    num_gpus = 8
    fairseq_python_exe = tk.Path("/usr/bin/python3", hash_overwrite=python_exe_hash_overwrite)
    fairseq_root = get_fairseq_root(fairseq_exe=fairseq_python_exe, commit=commit)
    fairseq_training_args = dict(
        save_interval=25,
        max_epoch=600,
        max_update=420000,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_python_exe,
        rqmt={"time": 336, "mem": 16, "cpu": 2, "gpu": num_gpus},
    )

    # generate config
    fairseq_args = get_fairseq_args(num_gpus=num_gpus)
    fairseq_args["task"]["alignment"] = alignment
    if checkpoint is not None:
        fairseq_args["checkpoint"]["continue_once"] = checkpoint
    for k, v in kwargs.items():
        fairseq_args["model"][k] = v
    fairseq_config = FairseqHydraConfig(fairseq_args)

    # run pretraining
    job = FairseqHydraTrainingJob(fairseq_config, **fairseq_training_args)
    job.add_alias(os.path.join(prefix_name, exp_name, "pretraining"))
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
    return job


def py():
    # negatives other
    run_fairseq_pretraining(
        exp_name="monophone_negatives_other_target_v1",
        commit="1397363c5c0e3c4e3ab620be562730399c852493",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        negative_sampling_strategy="other_target",
    )
    # negatives hard
    run_fairseq_pretraining(
        exp_name="monophone_negatives_hard_v1",
        commit="be51394d876428ad531e0786d80de43d6a8818af",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        negative_sampling_strategy="hard_negatives",
    )
    # boundary_masking
    run_fairseq_pretraining(
        exp_name="monophone_boundary_masking_v1",
        commit="b768be5b81987364d39a07d1caad2bfe1e956896",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        mask_strategy="phoneme",
        mask_length=1,
    )
    # negatives other + boundary masking
    run_fairseq_pretraining(
        exp_name="monophone_negatives_other_target_boundary_masking_v1",
        commit="b768be5b81987364d39a07d1caad2bfe1e956896",
        negative_sampling_strategy="other_target",
        mask_strategy="phoneme",
        mask_length=1,
    )
    # positive sampling
    for num_positives in [5, 10, 15]:
        run_fairseq_pretraining(
            exp_name=f"monophone_positive_sampling_{num_positives}_v1",
            commit="91b936231c5ebbfd639ccd7e78869d3df45a12bb",
            num_positives=num_positives,
        )
