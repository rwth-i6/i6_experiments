"""
Config for finetune experiments on LibriSpeech using wav2vec 2.0.
"""
# ------------------- IMPORTS ------------------- #
import os.path
from typing import List, Optional, Union

from sisyphus import tk, gs
import recipe.i6_core.datasets.librispeech as librispeech
from recipe.i6_core.audio.encoding import BlissChangeEncodingJob
from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.tools.download import DownloadJob 
from recipe.i6_core.corpus.segments import ShuffleAndSplitSegmentsJob, SegmentCorpusJob
from recipe.i6_core.corpus.filter import FilterCorpusBySegmentsJob
from recipe.i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from recipe.i6_experiments.users.engler.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob
from recipe.i6_experiments.users.vieting.jobs.fairseq import (
    CreateFairseqLabeledDataJob,
    MergeLabeledFairseqDataJob,
    FairseqDecodingJob,
)
from recipe.i6_experiments.users.vieting.experiments.librispeech.librispeech_960_pretraining.wav2vec2.fairseq \
    import SetupFairseqJob


# ------------------- GENERAL ------------------- #

def get_fairseq_root(fairseq_python_exe: Optional[tk.Path] = None):
    """
    :param fairseq_python_exe: path to the python executable of the fairseq environment
    """
    fairseq_root = CloneGitRepositoryJob(
        "https://github.com/facebookresearch/fairseq",
        checkout_folder_name="fairseq",
        commit="91c364b7ceef8032099363cb10ba19a85b050c1c").out_repository
    fairseq_root = SetupFairseqJob(fairseq_root, fairseq_python_exe).out_fairseq_root
    return fairseq_root


def get_labels(
    dest_name: str,
    corpus_paths: Union[List[tk.Path], tk.Path],
):
    """
    :param dest_name: name of the output file
    :param corpus_paths: path to the corpora
    """
    if isinstance(corpus_paths, tk.Path):
        corpus_paths = [corpus_paths]

    label_data_job = CreateFairseqLabeledDataJob(
        corpus_paths=corpus_paths,
        dest_name=dest_name,
    )
    
    return label_data_job.out_labels_path


# ------------------- FINETUNING ------------------- #

def get_task_dev_sampled(
    corpus_name: str,
    valid_percent: float = 0.01, 
    audio_format: str = "ogg", 
    output_prefix: str = "datasets",
):
    """
    :param corpus_name: name of the corpora to be used for training and to sample the dev set from
    :param valid_percent: percentage of the training data to be used as validation set. 
    :param audio_format: audio format of the output files
    :param output_prefix: prefix of the output files
    """
    # check input
    assert audio_format in ["ogg", "wav", "flac"], f"audio format not implemented: '{audio_format}'"
    assert corpus_name in {
        "train-clean-100", 
        "train-clean-360", 
        "train-clean-460", 
        "train-other-500", 
        "train-other-960",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    }, f"unknown corpus name: {corpus_name}"
    assert 0 <= valid_percent <= 1, f"invalid percentage: {valid_percent}"

    # get corpus
    corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)
    corpus_path = corpus_dict[corpus_name]

    # split corpus into train and dev set
    segment_corpus_job = SegmentCorpusJob(corpus_path, 1)

    split = {"train": 1 - valid_percent, "dev": valid_percent}
    shuffle_job = ShuffleAndSplitSegmentsJob(
        segment_file = segment_corpus_job.out_single_segment_files[1],
        split=split,
    )   

    train_filter_corpus_job = FilterCorpusBySegmentsJob(
        bliss_corpus=corpus_path,
        segment_file=shuffle_job.out_segments["train"], 
        compressed=True,
        delete_empty_recordings=True
    )
    dev_filter_corpus_job = FilterCorpusBySegmentsJob(
        bliss_corpus=corpus_path,
        segment_file=shuffle_job.out_segments["dev"],
        compressed=True,
        delete_empty_recordings=True
    )

    # create labeled data for training and dev set
    train_labels = get_labels(
        dest_name="train",
        corpus_paths=train_filter_corpus_job.out_corpus,
    )
    dev_labels = get_labels(
        dest_name="dev",
        corpus_paths=dev_filter_corpus_job.out_corpus,
    )

    # merge labeled data into single task directory
    merge_job = MergeLabeledFairseqDataJob(
        labeled_data_paths=[train_labels, dev_labels],
        create_letter_dict=True,
    )

    task = merge_job.out_task_path
    return task


def get_task_dev_separate(
    train_corpus_name: str,
    dev_corpus_name: str,
    audio_format: str = "ogg",
    output_prefix: str = "datasets",
):
    """
    :param train_corpus_name: name of the corpus to be used for training set
    :param dev_corpus_name: name of the corpus to be used for validation set
    :param audio_format: audio format of the output files
    :param output_prefix: prefix of the output files
    """
    assert audio_format in ["ogg", "wav", "flac"], f"audio format not implemented: '{audio_format}'"
    assert train_corpus_name in {
        "train-clean-100", 
        "train-clean-360", 
        "train-clean-460", 
        "train-other-500", 
        "train-other-960",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    }, f"unknown train corpus name: {train_corpus_name}"

    assert dev_corpus_name in {"dev-clean", "dev-other"}, f"unknown dev corpus name: {dev_corpus_name}"

    corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)
    train_corpus_path = corpus_dict[train_corpus_name]
    dev_corpus_path = corpus_dict[dev_corpus_name]

    train_labels = get_labels(
        dest_name="train",
        corpus_paths=train_corpus_path,
    )
    dev_labels = get_labels(
        dest_name=dev_corpus_name,
        corpus_paths=dev_corpus_path,
    )

    merge_job = MergeLabeledFairseqDataJob(
        labeled_data_paths=[train_labels, dev_labels],
        create_letter_dict=True,
    )
    task = merge_job.out_task_path
    return task


def get_fairseq_args(
    labeled_data: tk.Path, 
    w2v_path: tk.Path, 
    dev_file_name: str = "dev", 
    num_gpus: int = 1
    ):
    """
    :param labeled_data: path to the labeled data including the following files:
        - train.tsv
        - train.wrd
        - train.ltr
        - <dev_file_name>.tsv
        - <dev_file_name>.wrd
        - <dev_file_name>.ltr
        - letter_dict.txt
    :param dev_file_name: name of the dev label files
    :param w2v_path: path to the (pretrained) wav2vec model
    :param corpus_names: list of names of the corpora to be used for training
    :param num_gpus: number of gpus to be used for training
    """
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
            "data": labeled_data,
            "normalize": False,
            "labels": "ltr"
        },
        "dataset": {
            "num_workers": 2 * num_gpus,
            "max_tokens": 3200000,  # length of tokens in one batch
            "skip_invalid_size_inputs_valid_test": True,
            "valid_subset": dev_file_name,
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

def get_pretrained_model(model_path: Optional[Union[str, tk.Path]] = None):
    """
    :param model_path: path to the pretrained wav2vec model if available
    :return: path to the pretrained wav2vec model. 
        If model_path is None, the pretrained model is downloaded from fairseq repository.
    """
    if model_path is not None:
        pretrained_model = tk.input_path(model_path)
    else:
        url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        pretrained_model = DownloadJob(url=url, target_filename="wav2vec_small.pt").out_file
    return pretrained_model


def finetune():
    prefix_name = "experiments/librispeech/librispeech_100_ctc/fairseq/"
    exp_name = "base"
    num_gpus = 1
    gpu_mem_rqmt = 24
    corpus_name = "train-clean-100"
    fairseq_python_exe = tk.Path("/work/asr3/vieting/hiwis/pletschko/miniconda3/envs/fairseq_python38/bin/python")
    use_sampled_dev = False

    # get pretrained model
    w2v_path = get_pretrained_model()

    # create wav2vec labeled data
    if use_sampled_dev:
        dev_set = "dev"
        labeled_data = get_task_dev_sampled(corpus_name=corpus_name, valid_percent=0.01)
    else:
        dev_set = "dev-other"
        labeled_data = get_task_dev_separate(train_corpus_name=corpus_name, dev_corpus_name=dev_set)

    # create fairseq config for finetuning
    fairseq_args = get_fairseq_args(
        labeled_data=labeled_data,
        w2v_path=w2v_path,
        dev_file_name=dev_set,
        num_gpus=num_gpus,
    )
    fairseq_config = FairseqHydraConfig(fairseq_args)
    fairseq_root = get_fairseq_root(fairseq_python_exe=fairseq_python_exe)

    # run finetuning
    job = FairseqHydraTrainingJob(
        fairseq_config,
        max_epoch=300,
        save_interval=25,
        time_rqmt=100,
        mem_rqmt=16,
        cpu_rqmt=2,
        gpu_rqmt=num_gpus,
        gpu_mem_rqmt=gpu_mem_rqmt,
        fairseq_root=fairseq_root,
        fairseq_python_exe=fairseq_python_exe,
        use_cache_manager=False,
    )
    
    job.add_alias(os.path.join(prefix_name, exp_name, "finetune"))
    tk.register_output(os.path.join(prefix_name, exp_name, "finetune", "scores.png"), job.out_plot_se)
    tk.register_output(os.path.join(prefix_name, exp_name, "finetune", "checkpoints"), job.out_checkpoint_dir)

    return job.out_checkpoint_dir


# ------------------- DECODING ------------------- #

def get_dev_labels(
    audio_format: str = "ogg",
    output_prefix: str = "datasets",
    corpus_name: str = "dev-other"
):
    """
    :param audio_format: audio format of the output files
    :param output_prefix: prefix of the output files
    :param corpus_names: list of names of the corpora to be used for decoding
    """
    assert audio_format in ["ogg", "wav", "flac"], f"audio format not implemented: '{audio_format}'"
    assert corpus_name in ({
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    }), f"unknown corpus names: {corpus_name}"

    corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)
    # select corpus given by corpus_name
    corpus_path = corpus_dict[corpus_name]

    return get_labels(
        dest_name=corpus_name,
        corpus_paths=corpus_path,
    )


def decode(model_path: tk.Path):
    """
    :param model_path: path to the model to be used for decoding
    """
    # defines
    fairseq_python_exe = tk.Path(
        "/work/asr3/vieting/hiwis/pletschko/miniconda3/envs/fairseq_python38/bin/python",
        hash_overwrite="ls100_ctc_fairseq_python_exe",
    )
    lm_url = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
    lexicon_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst"

    prefix_name = "experiments/librispeech/librispeech_100_ctc/fairseq/"
    exp_name = "base"

    # prepare labels and language model
    fairseq_root = get_fairseq_root(fairseq_python_exe=fairseq_python_exe)

    dev_clean = get_dev_labels(corpus_name="dev-clean")
    dev_other = get_dev_labels(corpus_name="dev-other")

    decoder = "kenlm"
    lm_path = DownloadJob(url=lm_url).out_file
    lexicon_path = DownloadJob(url=lexicon_url).out_file

    # run decoding
    dev_clean_decoding = FairseqDecodingJob(
        fairseq_python_exe=fairseq_python_exe,
        fairseq_root=fairseq_root,
        model_path=model_path,
        data_path=dev_clean,
        gen_subset="dev-clean",
        w2l_decoder=decoder,
        lm_path=lm_path,
        lm_lexicon=lexicon_path,
    )

    dev_other_decoding = FairseqDecodingJob(
        fairseq_python_exe=fairseq_python_exe,
        fairseq_root=fairseq_root,
        model_path=model_path,
        data_path=dev_other,
        gen_subset="dev-other",
        w2l_decoder=decoder,
        lm_path=lm_path,
        lm_lexicon=lexicon_path,
    )

    tk.register_output(
        os.path.join(prefix_name, exp_name, "decode", "dev-clean", "results"),
        dev_clean_decoding.out_results
    )
    tk.register_output(
        os.path.join(prefix_name, exp_name, "decode", "dev-other", "results"),
        dev_other_decoding.out_results
    )


# ------------------- MAIN ------------------- #

def main():
    out_checkpoint_dir = finetune()
    checkpoint_best = tk.Path(os.path.join(out_checkpoint_dir.get(), "checkpoint_best.pt"))
    decode(checkpoint_best)

main()
