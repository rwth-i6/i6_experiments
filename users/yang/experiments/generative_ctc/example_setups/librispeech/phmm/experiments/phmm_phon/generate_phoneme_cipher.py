import copy
from dataclasses import asdict
from pathlib import Path

from sisyphus import tk

from ...cipher_jobs import EvaluateCipherTableTransferJob, GeneratePhonemeCipherHDFJob
from ...data.phmm_common import TrainingDatasets
from ...lm_score_dump_jobs import DumpNgramConvLmScoreTableJob
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_pipeline import training
from ...phmm_rasr import CreateLibrasrVenvJob
from ...pytorch_networks.phmm.table_transfer_fullsum_cfg import ModelConfig


class _ReturnnDataset:
    def __init__(self, opts):
        self.opts = opts

    def as_returnn_opts(self):
        return copy.deepcopy(self.opts)


def _count_vocab_entries(vocab_file):
    return len(Path(vocab_file).read_text().splitlines())


def _make_hdf_dataset(*, hdf_file, partition_epoch, seq_ordering, segment_file=None):
    opts = {
        "class": "HDFDataset",
        "files": [hdf_file],
        "partition_epoch": partition_epoch,
        "seq_ordering": seq_ordering,
    }
    if segment_file is not None:
        opts["seq_list_filter_file"] = segment_file
    return _ReturnnDataset(opts)


def _constant_lr(num_epochs, lr):
    return [lr] * num_epochs


def _warmup_plateau_decay_lr(num_epochs, init_lr, peak_lr, final_lr):
    peak_epoch = round(0.2 * num_epochs)
    decay_start_epoch = round(0.7 * num_epochs)
    lrs = []
    for epoch in range(1, num_epochs + 1):
        if epoch <= peak_epoch:
            if peak_epoch <= 1:
                lr = peak_lr
            else:
                lr = init_lr + (peak_lr - init_lr) * (epoch - 1) / (peak_epoch - 1)
        elif epoch <= decay_start_epoch:
            lr = peak_lr
        else:
            denom = max(1, num_epochs - decay_start_epoch)
            lr = peak_lr + (final_lr - peak_lr) * (epoch - decay_start_epoch) / denom
        lrs.append(lr)
    return lrs


def eow_phon_phmm_ls960_generate_phoneme_cipher():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phoneme_cipher"

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    phoneme_base = (
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_oggzip_to_phoneme_audio_hdf"
    )
    phoneme_hdf = tk.Path(f"{phoneme_base}/train_960_phoneme_text_no_eow.hdf")
    phoneme_vocab = tk.Path(f"{phoneme_base}/train_960_phoneme_vocab.txt")
    train_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/train.segments"
    )
    cv_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/cv.segments"
    )

    cipher_job = GeneratePhonemeCipherHDFJob(
        phoneme_hdf=phoneme_hdf,
        phoneme_vocab=phoneme_vocab,
        num_cipher_labels_per_phoneme=3,
        random_seed=1,
        output_filename="train_960_cipher_m3.hdf",
        distribution_filename="train_960_cipher_m3_distribution.npz",
        empirical_distribution_filename="train_960_cipher_m3_empirical_distribution.npz",
        hdf_format_version=2,
        mem_rqmt=24,
        time_rqmt=4,
    )
    cipher_job.add_alias(prefix_name + "/train_960_m3")
    tk.register_output(prefix_name + "/train_960_cipher_m3.hdf", cipher_job.out_hdf)
    tk.register_output(prefix_name + "/train_960_cipher_m3_distribution.npz", cipher_job.out_distribution)
    tk.register_output(
        prefix_name + "/train_960_cipher_m3_empirical_distribution.npz",
        cipher_job.out_empirical_distribution,
    )
    tk.register_output(prefix_name + "/train_960_cipher_m3_stats.txt", cipher_job.out_stats)

    lm_checkpoint = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/returnn/training/"
        "ReturnnTrainingJob.06kj3ZpsHmSC/output/models/epoch.200.pt"
    )
    lm_score_job = DumpNgramConvLmScoreTableJob(
        lm_checkpoint=lm_checkpoint,
        python_exe=returnn_exe,
        recipe_root=tk.Path("/u/zyang/setups/mini/recipe"),
        context_length=3,
        vocab_size=41,
        model_config_dict={
            "vocab_size": 41,
            "embedding_dim": 128,
            "conv_channels": 256,
            "conv_kernel_size": 3,
            "projection_dim": 256,
            "dropout": 0.3,
            "pad_token_id": 0,
            "bos_token_id": 40,
        },
        chunk_size=8192,
        output_filename="phoneme_ngram_conv_lm_v2_context3_ep200_log_probs.pt",
        mem_rqmt=8,
        time_rqmt=2,
    )
    lm_score_job.add_alias(prefix_name + "/dump_phoneme_ngram_conv_lm_v2_context3_ep200")
    tk.register_output(
        prefix_name + "/phoneme_ngram_conv_lm_v2_context3_ep200_log_probs.pt",
        lm_score_job.out_scores,
    )
    tk.register_output(
        prefix_name + "/phoneme_ngram_conv_lm_v2_context3_ep200_log_probs_stats.txt",
        lm_score_job.out_stats,
    )

    num_phonemes = _count_vocab_entries(str(phoneme_vocab))
    num_cipher_labels = num_phonemes * 3
    num_epochs = 200
    table_transfer_model_config = ModelConfig(
        input_vocab_size=num_cipher_labels,
        output_vocab_size=num_phonemes,
        lm_table_path=lm_score_job.out_scores,
        lm_vocab_size=41,
        lm_context_length=3,
        beam_size=200,
        softmax_temperature=1.0,
        use_lm_silence_score=False,
        lm_scale=0.6,
        am_scale=1.0,
        table_init_scale=0.02,
    )
    cipher_dataset = _make_hdf_dataset(
        hdf_file=cipher_job.out_hdf,
        partition_epoch=10,
        seq_ordering="laplace:.100",
        segment_file=train_segments,
    )
    cipher_dev_dataset = _make_hdf_dataset(
        hdf_file=cipher_job.out_hdf,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=cv_segments,
    )
    cipher_training_datasets = TrainingDatasets(
        train=cipher_dataset,
        cv=cipher_dev_dataset,
        devtrain=cipher_dev_dataset,
        datastreams={},
        prior=None,
    )
    lr_variants = [
        ("const_lr1e-3", _constant_lr(num_epochs, 1e-3)),
        ("const_lr1e-4", _constant_lr(num_epochs, 1e-4)),
        ("const_lr1e-5", _constant_lr(num_epochs, 1e-5)),
        (
            "warmup1e-4_peak1e-3_final1e-5",
            _warmup_plateau_decay_lr(num_epochs, init_lr=1e-4, peak_lr=1e-3, final_lr=1e-5),
        ),
        (
            "warmup1e-5_peak5e-4_final1e-5",
            _warmup_plateau_decay_lr(num_epochs, init_lr=1e-5, peak_lr=5e-4, final_lr=1e-5),
        ),
    ]
    for lr_name, learning_rates in lr_variants:
        table_transfer_train_config = {
            "optimizer": {"class": "Adam"},
            "learning_rates": learning_rates,
            "batch_size": 80_000,
            "max_seqs": 200,
            "num_workers_per_gpu": 0,
            "accum_grad_multiple_step": 4,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": [num_epochs],
            },
        }
        table_transfer_train_name = (
            prefix_name
            + f"/table_transfer_fullsum.cipher_m3_phoneme_lm_context3_adam_{lr_name}_ep200"
        )
        table_transfer_train_job = training(
            training_name=table_transfer_train_name,
            datasets=cipher_training_datasets,
            train_args={
                "network_module": "phmm.table_transfer_fullsum",
                "config": table_transfer_train_config,
                "net_args": {"model_config_dict": asdict(table_transfer_model_config)},
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        table_transfer_train_job.rqmt["gpu_mem"] = 24
        final_checkpoint = table_transfer_train_job.out_checkpoints[num_epochs].path
        tk.register_output(
            table_transfer_train_name + "/epoch.200.pt",
            final_checkpoint,
        )

        eval_job = EvaluateCipherTableTransferJob(
            model_checkpoint=final_checkpoint,
            cipher_hdf=cipher_job.out_hdf,
            phoneme_hdf=phoneme_hdf,
            segment_file=cv_segments,
            python_exe=returnn_exe,
            output_filename=f"cipher_table_eval_{lr_name}.txt",
            mem_rqmt=8,
            time_rqmt=2,
        )
        eval_job.add_alias(table_transfer_train_name + "/eval_cv_1pct_argmax")
        tk.register_output(
            table_transfer_train_name + "/eval_cv_1pct_argmax/report.txt",
            eval_job.out_report,
        )


py = eow_phon_phmm_ls960_generate_phoneme_cipher
