"""
Config for finetune experiments on LibriSpeech using wav2vec 2.0.
"""
import os.path
from typing import List

from sisyphus import tk, gs
import recipe.i6_core.datasets.librispeech as librispeech
from recipe.i6_core.audio.encoding import BlissChangeEncodingJob
from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from recipe.i6_experiments.users.engler.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from recipe.i6_experiments.users.vieting.jobs.fairseq import CreateFairseqLabeledDataJob
from recipe.i6_experiments.users.vieting.experiments.librispeech.librispeech_960_pretraining.wav2vec2.fairseq import SetupFairseqJob

#python_path = tk.Path("/work/asr3/vieting/hiwis/pletschko/miniconda3/envs/fairseq_python38/bin/python")


def get_task(
    valid_percent: float = 0.01, 
    audio_format: str = "ogg", 
    output_prefix: str = "datasets", 
    corpus_names: List[str] = ["train-clean-100", "train-clean-360", "train-other-500"]
):
    """
    :param valid_percent: percentage of the training data to be used as validation set. 
        If <= 0, the validation set is taken from the dev corpus.
    :param audio_format: audio format of the output files
    :param output_prefix: prefix of the output files
    :param corpus_names: list of names of the corpora to be used for training
    """
    assert audio_format in ["ogg", "wav", "flac"], f"audio format not implemented: '{audio_format}'"
    assert set(corpus_names).issubset(
        {"train-clean-100", "train-clean-360", "train-clean-460", "train-other-500", "train-other-960", "dev-clean", "dev-other", "test-clean", "test-other"}
    ), f"unknown corpus names: {corpus_names}"

    if valid_percent <= 0:
        assert "dev-clean" in corpus_names or "dev-other" in corpus_names, "validation set is required if valid_percent <= 0"

    corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)
    # filter out corpora that are not in corpus_names
    corpus_dict = {corpus: path for corpus, path in corpus_dict.items() if corpus in corpus_names}

    task_creation_job = CreateFairseqLabeledDataJob(
        corpus_paths=list(corpus_dict.values()),
        file_extension=audio_format,
        valid_percent=valid_percent,
    )
    task_creation_job.rqmt["time"] = 4
    task = task_creation_job.out_task_path
    return task

def get_fairseq_root():
    fairseq_root = CloneGitRepositoryJob(
        "https://github.com/facebookresearch/fairseq",
        checkout_folder_name="fairseq",
        commit="91c364b7ceef8032099363cb10ba19a85b050c1c").out_repository
    #fairseq_root = SetupFairseqJob(fairseq_root, python_path).out_fairseq_root
    fairseq_root = SetupFairseqJob(fairseq_root).out_fairseq_root
    return fairseq_root


def get_fairseq_args(w2v_path, corpus_names, num_gpus=1):
    # create wav2vec manifest for training
    task = get_task(corpus_names=corpus_names)

    # Set training and model parameters
    fairseq_args = {
        "common": {
            "fp16": True,
            "log_format": "json",
            "log_interval": 200,
        },
        "checkpoint": {
            "no_epoch_checkpoints": True,
            "best_checkpoint_metric": "wer",
        },
        "task": {
            "_name": "audio_finetuning",
            "data": task,
            "normalize": False,
            "labels": "ltr"
        },
        "dataset": {
            "num_workers": 6,
            "max_tokens": 3200000,  # length of tokens in one batch
            "skip_invalid_size_inputs_valid_test": True,
            "valid_subset": "valid",
        },
        "distributed_training": {
            "distributed_world_size": num_gpus,
            "ddp_backend": "legacy_ddp"
        },
        "criterion": {
            "_name": "ctc",
            "zero_infinity": True,
        },
        "optimization": {
            "sentence_avg": True,
            "max_update": 80000,
            "lr": [0.00003],
            "update_freq": [8 // num_gpus],
        },
        "optimizer": {
            "_name": "adam",
            "adam_betas": "(0.9,0.98)",
            "adam_eps": "1e-08",
        },
        "lr_scheduler": {
            "_name": "tri_stage",
            "phase_ratio": [0.1, 0.4, 0.5],
            "final_lr_scale": 0.05,
        },
        "model": {
            "_name": "wav2vec_ctc",
            "w2v_path": w2v_path,
            "apply_mask": True,
            "mask_prob": 0.65,
            "mask_channel_prob": 0.5,
            "mask_channel_length": 64,
            "layerdrop": 0.1,
            "activation_dropout": 0.1,
            "feature_grad_mult": 0.0,
            "freeze_finetune_updates": 0,
        }
    }
    return fairseq_args


def main():
    prefix_name = "experiments/librispeech/librispeech_100_ctc/fairseq/"
    # run finetuning
    exp_name = "base"
    num_gpus = 2
    gpu_mem_rqmt = 24
    corpus_names = ["train-clean-100"]

    fairseq_args = get_fairseq_args(corpus_names=corpus_names, num_gpus=num_gpus, w2v_path="/work/asr3/vieting/hiwis/pletschko/fairseq/models/wav2vec_small.pt")
    fairseq_config = FairseqHydraConfig(fairseq_args)
    fairseq_root = get_fairseq_root()
    job = FairseqHydraTrainingJob(
        fairseq_config,
        max_epoch=300,
        save_interval=25,
        time_rqmt=1,
        mem_rqmt=8,
        cpu_rqmt=2,
        gpu_rqmt=num_gpus,
        gpu_mem_rqmt=gpu_mem_rqmt,
        fairseq_root=fairseq_root,
        use_cache_manager=False,
    )
    
    job.add_alias(os.path.join(prefix_name, exp_name, "finetune"))
    tk.register_output(f"{prefix_name}/{exp_name}/finetune/scores.png", job.out_plot_se)

main()
