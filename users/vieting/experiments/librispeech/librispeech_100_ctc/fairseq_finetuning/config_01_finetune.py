"""
Config for finetune experiments on LibriSpeech using wav2vec 2.0.
"""
import os.path

from sisyphus import tk, gs
import recipe.i6_core.datasets.librispeech as librispeech
from recipe.i6_core.audio.encoding import BlissChangeEncodingJob
from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_experiments.users.engler.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from recipe.i6_experiments.users.vieting.jobs.fairseq import CreateFairseqLabeledDataJob
from recipe.i6_experiments.users.vieting.experiments.librispeech.librispeech_960_pretraining.wav2vec2.fairseq import SetupFairseqJob

def get_task(valid_percent=0.001, audio_format="ogg", output_prefix="datasets", corpus_names=["train-clean-100", "train-clean-360", "train-other-500"]):
    assert audio_format in ["ogg", "wav"], f"audio format not implemented: '{audio_format}'"
    corpus_dirs = {}

    output_prefix = os.path.join(output_prefix, "LibriSpeech")

    download_metadata_job = librispeech.DownloadLibriSpeechMetadataJob()
    download_metadata_job.add_alias(os.path.join(output_prefix, "download", "metadata_job"))
    
    for corpus_name in corpus_names:
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

        corpus_dirs[corpus_name] = bliss_change_encoding_job.out_corpus
    

    task_creation_job = CreateFairseqLabeledDataJob(
        corpus_paths=list(corpus_dirs.values()),
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
        commit="176cd934982212a4f75e0669ee81b834ee71dbb0").out_repository
    fairseq_root = SetupFairseqJob(fairseq_root).out_fairseq_root
    return fairseq_root


def get_fairseq_args(w2v_path, corpus_names, num_gpus=1):
    # create wav2vec manifest for training
    task = get_task(corpus_names=corpus_names)

    # Set training and model parameters
    sample_rate = 16000
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
            # TODO: check if this is needed
            #"max_sample_size": 20 * sample_rate,  # max 20s, length of one crop in batch
            #"min_sample_size": 2 * sample_rate,  # 2s of minimal length
            #"sample_rate": sample_rate,
            "normalize": False,  # TODO: False in fairseq
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
            # TODO: check if this is needed
            #"infonce": True,
            #"log_keys": ["prob_perplexity", "code_perplexity", "temp"],
            #"loss_weights": [0.1, 10]
        },
        "optimization": {
            "sentence_avg": True,
            "max_update": 320000,
            "lr": [0.0001],
            "update_freq": [8 // num_gpus],
            # TODO check if this is needed
            #"max_epoch": 1000,
        },
        "optimizer": {
            "_name": "adam",
            "adam_betas": "(0.9,0.98)",
            "adam_eps": "1e-08",
            # TODO check if this is needed
            #"weight_decay": 0.01,
        },
        "lr_scheduler": {
            "_name": "tri_stage",
            "phase_ratio": [0.1, 0.4, 0.5],
            "final_lr_scale": 0.05,
        },
        "model": {
            "_name": "wav2vec2_ctc",
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
    # run pre-training
    exp_name = "base"
    num_gpus = 2
    corpus_names = ["train-clean-100"]
    python_path = "work/asr3/vieting/hiwis/pletschko/miniconda3/envs/fairseq_python38/bin/python"
    setattr(gs, "FAIRSEQ_PYTHON_EXE", python_path)
    fairseq_args = get_fairseq_args(corpus_names=corpus_names, num_gpus=num_gpus, w2v_path="/work/asr3/vieting/hiwis/pletschko/fairseq/models/wav2vec_small.pt")
    fairseq_config = FairseqHydraConfig(fairseq_args)
    fairseq_root = get_fairseq_root()
    job = FairseqHydraTrainingJob(
        fairseq_config,
        max_epoch=300,
        save_interval=25,
        time_rqmt=120,
        mem_rqmt=8,
        cpu_rqmt=2,
        gpu_rqmt=num_gpus,
        fairseq_root=fairseq_root,
        #fairseq_python_exe=python_path,
        use_cache_manager=False,
    )
    #job.add_alias(os.path.join(prefix_name, exp_name, "finetune"))
    tk.register_output(f"{prefix_name}/{exp_name}/finetune/scores.png", job.out_plot_se)

main()
