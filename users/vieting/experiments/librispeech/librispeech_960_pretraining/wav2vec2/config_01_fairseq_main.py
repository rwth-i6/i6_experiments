"""
Config for pre-training experiments on LibriSpeech using wav2vec 2.0.
"""
import os.path

from sisyphus import tk
import i6_core.datasets.librispeech as librispeech
from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from i6_core.fairseq.manifest import CreateManifestJob, SplitTrainCvDataJob
from .fairseq import SetupFairseqJob


def get_manifest(valid_percent=0.001, audio_format="ogg", output_prefix="datasets"):
    assert audio_format in ["ogg", "wav"], f"audio format not implemented: '{audio_format}'"
    audio_dirs = {}

    output_prefix = os.path.join(output_prefix, "LibriSpeech")

    download_metadata_job = librispeech.DownloadLibriSpeechMetadataJob()
    download_metadata_job.add_alias(os.path.join(output_prefix, "download", "metadata_job"))
    for corpus_name in ["train-clean-100", "train-clean-360", "train-other-500"]:
        download_corpus_job = librispeech.DownloadLibriSpeechCorpusJob(corpus_key=corpus_name)
        download_corpus_job.add_alias(os.path.join(output_prefix, "download", corpus_name))
        create_bliss_corpus_job = librispeech.LibriSpeechCreateBlissCorpusJob(
            corpus_folder=download_corpus_job.out_corpus_folder,
            speaker_metadata=download_metadata_job.out_speakers,
        )
        create_bliss_corpus_job.add_alias(os.path.join(output_prefix, "create_bliss", corpus_name))
        audio_format_options = {
            "wav": {
                "output_format": "wav",
                "codec": "pcm_s16le",
            },
            "ogg": {"output_format": "ogg", "codec": "libvorbis"},
        }
        bliss_change_encoding_job = BlissChangeEncodingJob(
            corpus_file=create_bliss_corpus_job.out_corpus,
            sample_rate=16000,
            **audio_format_options[audio_format],
        )
        bliss_change_encoding_job.add_alias(
            os.path.join(
                output_prefix,
                "%s_conversion" % audio_format,
                corpus_name,
            )
        )
        audio_dirs[corpus_name] = bliss_change_encoding_job.out_audio_folder

    manifest_creation_job = CreateManifestJob(
        audio_dir_paths=list(audio_dirs.values()),
    )
    manifest_split_job = SplitTrainCvDataJob(manifest_creation_job.out_tsv_file, valid_portion=0.001)
    return manifest_split_job.out_manifest_path


def get_fairseq_root(commit="da8fb630880d529ab47e53381c30ddc8ad235216", fairseq_exe=None):
    fairseq_root = CloneGitRepositoryJob(
        "https://github.com/facebookresearch/fairseq",
        checkout_folder_name="fairseq",
        commit=commit).out_repository
    fairseq_root = SetupFairseqJob(fairseq_root, fairseq_exe).out_fairseq_root
    return fairseq_root


def get_fairseq_args(num_gpus=1):
    # create wav2vec manifest for training
    manifest = get_manifest()

    # Set training and model parameters
    fairseq_args = {
        "common": {
            "fp16": True,
            "log_format": "json",
            "log_interval": 200,
        },
        "checkpoint": {
            "no_epoch_checkpoints": False,
            "save_dir": "output/checkpoints",
            "restore_file": "checkpoint_last.pt",
        },
        "task": {
            "_name": "audio_pretraining",
            "data": manifest,
            "max_sample_size": 250000,
            "min_sample_size": 32000,
            "normalize": False,
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
            "lr": [0.0005],
            "update_freq": [64 // num_gpus]
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


def run_fairseq_pretraining():
    prefix_name = "experiments/librispeech/librispeech_960_pretraining/wav2vec2/"
    # run pre-training
    exp_name = "base"
    num_gpus = 8
    fairseq_args = get_fairseq_args(num_gpus=num_gpus)
    fairseq_config = FairseqHydraConfig(fairseq_args)
    fairseq_root = get_fairseq_root()
    itc_python_launcher = "/home/pv653172/setups/librispeech/20230328_wav2vec2/dependencies/python_launcher.sh"
    if os.path.exists(itc_python_launcher):
        fairseq_exe = tk.Path(itc_python_launcher, hash_overwrite="python_launcher")
    else:
        fairseq_exe = tk.Path("/usr/bin/python3", hash_overwrite="python_launcher")
    job = FairseqHydraTrainingJob(
        fairseq_config,
        save_interval=25,
        max_epoch=600,
        max_update=420000,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_exe,
        rqmt={"time": 120, "mem": 12, "cpu": 2, "gpu": num_gpus},
    )
    job.add_alias(os.path.join(prefix_name, exp_name, "pretraining"))
    tk.register_output(f"{prefix_name}/{exp_name}/pretraining/scores.png", job.out_plot_se)
    return job


def py():
    run_fairseq_pretraining()
