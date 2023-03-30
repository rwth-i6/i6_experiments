"""
Config for pre-training experiments on LibriSpeech using wav2vec 2.0.
"""
from sisyphus import tk
from i6_core.datasets.librispeech import DownloadLibriSpeechCorpusJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.engler.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from i6_experiments.users.dierkes.preprocessing.wav2vec import FairseqAudioManifestCreationJob


def get_fairseq_args(num_gpus=1):
    # create wav2vec manifest for training
    audio_dirs = [
      DownloadLibriSpeechCorpusJob(corpus_key=key).out_corpus_folder for key in [
        "train-clean-100", "train-clean-360", "train-other-500"]
    ]
    job = FairseqAudioManifestCreationJob(
        audio_dir_path=audio_dirs,
        file_extension="flac",
        valid_percent=0.001,
    )
    job.rqmt["time"] = 8
    manifest = job.out_manifest_path

    # Set training and model parameters
    sample_rate = 16000
    fairseq_args = {
        "common": {
            "fp16": True,
            "log_format": "json",
            "log_interval": 200,
        },
        "checkpoint": {
            "no_epoch_checkpoints": False,
            "save_interval": 25,
            "save_dir": "output/checkpoints",
            "restore_file": "checkpoint_last.pt",
        },
        "task": {
            "_name": "audio_pretraining",
            "data": manifest,
            "max_sample_size": 20 * sample_rate,  # max 20s, length of one crop in batch
            "min_sample_size": 2 * sample_rate,  # 2s of minimal length
            "sample_rate": sample_rate,
            "normalize": True,  # TODO: False in fairseq
        },
        "dataset": {
            "num_workers": 6,
            "max_tokens": 1400000,  # length of tokens in one batch
            "skip_invalid_size_inputs_valid_test": True,
        },
        "distributed_training": {
            "distributed_world_size": num_gpus,
            "ddp_backend": "legacy_ddp"
        },
        "criterion": {
            "_name": "wav2vec",
            "infonce": True,
            "log_keys": ["prob_perplexity", "code_perplexity", "temp"],
            "loss_weights": [0.1, 10]
        },
        "optimization": {
            "max_epoch": 1000,
            "max_update": 400000,
            "lr": [0.0005],
            "update_freq": [64 / num_gpus]
        },
        "optimizer": {
            "_name": "adam",
            "adam_betas": "(0.9,0.98)",
            "adam_eps": "1e-06",
            "weight_decay": 0.01,
        },
        "lr_scheduler": {
            "_name": "polynomial_decay",
            "warmup_updates": 32000,
        },
        "model": {
            "_name": "wav2vec2",
            "quantize_targets": True,
            "final_dim": 256,
            "encoder_layerdrop": 0.05,
            "dropout_input": 0.1,
            "dropout_features": 0.1,
            "feature_grad_mult": 0.1,
            "encoder_embed_dim": 768,
        }
    }
    return fairseq_args


def main():
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    # run pre-training
    exp_name = "base"
    fairseq_args = get_fairseq_args(num_gpus=8)
    fairseq_config = FairseqHydraConfig(fairseq_args, yaml_prefix="# @package _group_")
    fairseq_root = CloneGitRepositoryJob(
        "https://github.com/facebookresearch/fairseq", commit="176cd934982212a4f75e0669ee81b834ee71dbb0").out_repository
    fairseq_python = tk.Path(
        "/home/oh751555/python_env/env3.6fairseq20220421/bin/python3", hash_overwrite="FAIRSEQ_PYTHON")
    job = FairseqHydraTrainingJob(
        fairseq_config,
        max_epoch=300,
        save_interval=25,
        time_rqmt=120,
        mem_rqmt=8,
        cpu_rqmt=2,
        gpu_rqmt=8,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_python,
        use_cache_manager=False,
    )
    job.add_alias(f"{prefix_name}/{exp_name}/pretraining")
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
