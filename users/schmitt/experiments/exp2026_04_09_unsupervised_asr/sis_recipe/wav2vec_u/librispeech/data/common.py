from typing import Optional

from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob

from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.schmitt.datasets.multi_proc import MultiProcDataset
from i6_experiments.users.schmitt.datasets.utils.extract_seq_list import FilterSeqListByHdfSeqTagsJob
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset
from i6_experiments.common.setups.returnn.datastreams.base import FeatureDatastream

from ....data.librispeech import audio, text
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings


# `torch_dataloader_opts.num_workers` must stay 1 for these setups: RETURNN's
# `ReturnnDatasetIterDataPipe` iterates `seq_index = 0..num_seqs-1` with no worker sharding (there is
# no `sharding_filter` in returnn/torch/data/pipeline.py, and torch's `apply_sharding` only touches
# datapipes that implement it), so every torch dataloader worker yields the *whole* sub-epoch -- with
# the same epoch and the same `random_seed_offset`, hence the same seq order. With num_workers=6 that
# fed each batch 6 times in a row (measured: the run-length histogram of identical
# `(num_seqs, data_T, phon_T)` over consecutive steps was exactly `{6: 1669}`), i.e. 6 consecutive
# optimizer updates on identical data. Parallel loading has to come from RETURNN's own
# `MultiProcDataset` instead, which shards properly ("one epoch (or subepoch) is exactly as in the
# original dataset").
#
# Only the audio branch is wrapped: it is 85 GB of 512-dim float features, whereas the text branch is
# either 108 MB (train-960 transcripts) or already read in a separate process by
# `DistributeFilesDataset`'s per-sub-epoch worker.
_AUDIO_LOADER_NUM_WORKERS = 6
_AUDIO_LOADER_BUFFER_SIZE = 10


def _parallel_audio_dataset(dataset: HdfDataset) -> MultiProcDataset:
    """Wrap the (expensive) audio feature branch in a `MultiProcDataset` -- see the comment above."""
    return MultiProcDataset(
        dataset=dataset,
        num_workers=_AUDIO_LOADER_NUM_WORKERS,
        buffer_size=_AUDIO_LOADER_BUFFER_SIZE,
    )


def build_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    max_abs_value: Optional[float] = None,
):
    features_960_hdfs, clusters_960, pca_960, _ = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )
    features_dev_other_hdfs, _, _, _ = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )
    features_dev_clean_hdfs, _, _, _ = audio.get_featurized_audio(
        librispeech_key="dev-clean",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )

    # we don't pass sil_prob here, because we just want to get the lexicon here
    # we don't use the text-only data for training here
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_clean_hdfs, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    dev_seq_tags = ConcatenateJob([dev_clean_seq_tags, dev_other_seq_tags], zip_out=False).out

    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out
    dev_seq_tags = TakeNRandomLinesJob(text_file=dev_seq_tags, num_lines=3000).out

    return TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "features": _parallel_audio_dataset(
                    HdfDataset(
                        files=features_960_hdfs,
                        segment_file=train_seq_tags,
                        partition_epoch=settings.train_partition_epoch,
                        seq_ordering=settings.train_seq_ordering,
                    )
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                ("phon_indices", "data"): "phon_indices",
                ("features", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "features": HdfDataset(
                        files=features_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("features", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
                datasets={
                    "features": HdfDataset(
                        files=features_dev_other_hdfs + features_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("features", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": FeatureDatastream(
                available_for_inference=True,
                feature_size=512,
            ),
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


# Size of the phonemized `lm_minus_librivox` LM corpus vs. the `train-other-960` transcripts
# (`wc -l` on the two `PhonemizeTextDataJob.*/output/text.phonemes.txt`). Used to derive the text
# sub-dataset's `partition_epoch` so that one sub-epoch draws roughly as many text seqs as audio
# seqs -- otherwise `CombinedDataset(seq_ordering="interleave")` mixes proportionally to the
# dataset sizes and a batch would be ~99% text rows, starving the GAN's speech side.
_NUM_LM_MINUS_LIBRIVOX_LINES = 33_013_647
_NUM_TRAIN_960_LINES = 266_927


def build_training_datasets_w_lm_text(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    max_abs_value: Optional[float] = None,
    text_partition_epoch: Optional[int] = None,
):
    """
    Same as :func:`build_training_datasets`, but the *training* text comes from the
    `lm_minus_librivox` LM corpus instead of the `train-other-960` transcripts (the audio side and
    the eval sets are unchanged). This makes the text truly unpaired: it shares no utterances with
    the speech data.

    :param text_partition_epoch: `partition_epoch` of the LM-text sub-dataset. Defaults to the
        LM-corpus / train-960 line ratio (~124), so a sub-epoch draws about as much text as audio
        (same audio:text ratio per batch as :func:`build_training_datasets`) -- without it
        `CombinedDataset(seq_ordering="interleave")` mixes proportionally to the sub-dataset sizes and
        a batch would be ~99% text rows, starving the GAN's speech side.
    """
    features_960_hdfs, clusters_960, pca_960, _ = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )
    features_dev_other_hdfs, _, _, _ = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )
    features_dev_clean_hdfs, _, _, _ = audio.get_featurized_audio(
        librispeech_key="dev-clean",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )

    # Same call every other setup uses to obtain the lexicon/vocab -- here we additionally keep its
    # phoneme HDFs, which ARE the training text. It runs with the default sil_prob/surround_w_sil,
    # so refuse silently diverging from what the rest of the config asks for (we cannot pass them
    # through without rehashing the shared lexicon job).
    assert (sil_prob, surround_w_sil) == (0.25, True), (
        "the lm_minus_librivox phonemization is shared with the lexicon lookup and runs with the"
        " default sil_prob=0.25 / surround_w_sil=True"
    )
    lm_text_hdfs, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text(
        "lm_minus_librivox", dump_hdf_concurrent=100
    )
    # NB: no seq tags for the LM corpus (`PhonemizeTextDataJob.out_seq_tags` is None without an
    # input seq-tag file), so the dump job invents "lm-data-<task_id>-<i>" tags -- unique across the
    # 100 shards, but there is no seq-tag *file* to filter by, hence no segment file below.
    # only used for the eval sets (devtrain) + the audio branch's segment list
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_clean_hdfs, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    dev_seq_tags = ConcatenateJob([dev_clean_seq_tags, dev_other_seq_tags], zip_out=False).out

    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out
    dev_seq_tags = TakeNRandomLinesJob(text_file=dev_seq_tags, num_lines=3000).out

    # NB the LM text is 8.5 GB over `len(lm_text_hdfs)` (=100) shards against 108 MB over 10 for the
    # train-960 transcripts, so this branch has a real *startup* cost: the cache manager copies all of
    # it to node-local /var/tmp and every process then `HDFDataset.add_file`-parses all 100 shards
    # (~2.4 s each, measured). On a cold node that was ~20 min of main-process init plus a 16 min
    # "time to get first batch data" -- but that was with `num_workers: 6`, i.e. 7 processes each
    # parsing all 100 shards; with `num_workers=1` (now enforced, see `_parallel_audio_dataset`) it is
    # 2. It is also one-time: the copy is per node and reused, and epoch 2's first batch took 43 s.
    # A `DistributeFilesDataset` here was tried and reverted -- it works, but `partition_epoch` is then
    # capped at the shard count (100 < the 124 the line ratio wants), which skews the audio:text ratio
    # for no steady-state gain. Only revisit it if node-local disk becomes the binding constraint.
    if text_partition_epoch is None:
        text_partition_epoch = round(_NUM_LM_MINUS_LIBRIVOX_LINES / _NUM_TRAIN_960_LINES)

    return TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "features": _parallel_audio_dataset(
                    HdfDataset(
                        files=features_960_hdfs,
                        segment_file=train_seq_tags,
                        partition_epoch=settings.train_partition_epoch,
                        seq_ordering=settings.train_seq_ordering,
                    )
                ),
                "phon_indices": HdfDataset(
                    files=lm_text_hdfs,
                    segment_file=None,  # use the whole LM corpus (and it has no usable seq tags)
                    partition_epoch=text_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                ("phon_indices", "data"): "phon_indices",
                ("features", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        # eval sets unchanged w.r.t. `build_training_datasets` (LibriSpeech dev/devtrain), so the
        # scores stay comparable to the 960h-text baseline.
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "features": HdfDataset(
                        files=features_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("features", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
                datasets={
                    "features": HdfDataset(
                        files=features_dev_other_hdfs + features_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("features", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": FeatureDatastream(
                available_for_inference=True,
                feature_size=512,
            ),
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


def build_text_only_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    """
    Text-only (single-task) training data: just the phoneme indices (exposed under the
    ``phon_indices`` key), no audio clusters and no alternate batching. Used as a single-task
    reference for :func:`build_training_datasets` (multi-task text+audio). Reuses the exact same
    phoneme HDFs as the multi-task setup (same Sisyphus jobs), so the text data is identical.
    """
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_clean_hdfs, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    dev_seq_tags = ConcatenateJob([dev_clean_seq_tags, dev_other_seq_tags], zip_out=False).out
    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out
    dev_seq_tags = TakeNRandomLinesJob(text_file=dev_seq_tags, num_lines=3000).out

    def _phon_dataset(hdfs, seq_tags, partition_epoch=None, seq_ordering=None):
        # wrap in a MetaDataset only to expose the phoneme HDF under the "phon_indices" key
        # (a raw HdfDataset exposes "data"), so train + recog use the same key.
        return MetaDataset(
            datasets={
                "phon_indices": HdfDataset(
                    files=hdfs,
                    segment_file=seq_tags,
                    partition_epoch=partition_epoch,
                    seq_ordering=seq_ordering,
                ),
            },
            data_map={"phon_indices": ("phon_indices", "data")},
            seq_order_control_dataset="phon_indices",
        )

    return TrainingDatasets(
        train=_phon_dataset(
            phoneme_960_hdfs,
            train_seq_tags,
            partition_epoch=settings.train_partition_epoch,
            seq_ordering=settings.train_seq_ordering,
        ),
        eval_datasets={
            "devtrain": _phon_dataset(phoneme_960_hdfs, devtrain_seq_tags, seq_ordering="sorted"),
            "dev": _phon_dataset(phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs, dev_seq_tags, seq_ordering="sorted"),
        },
        datastreams={
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


def build_test_datasets(max_abs_value: Optional[float] = None, keep_all_seqs: bool = True):
    """
    :param max_abs_value:
    :param keep_all_seqs: phonemize the full corpus instead of dropping the sequences that the language-ID
        filter and the lexicon-OOV filter of ``PhonemizeTextDataJob`` remove. Without it, dev-other is scored
        on 2712 of 2864 utterances (see the "Lost eval sequences" section in CLAUDE.md). Only affects
        forward/scoring jobs, never a training.
    """
    assert max_abs_value is None, "We must not filter seqs for testing! Otherwise not comparable."

    _, all_dev_other_seq_tags = text.get_text("dev-other")

    _, clusters_960, pca_960, _ = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    features_dev_other_hdfs, _, _, _ = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
        max_abs_value=max_abs_value,
    )

    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_dev_hdfs, _, _, _ = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        # never drop eval seqs: the reference must cover the whole corpus, otherwise WER/PER is not
        # comparable (LID filter: 5 seqs, lexicon OOV: 147 seqs on dev-other)
        apply_lid_filter=not keep_all_seqs,
        extend_lexicon_w_g2p=keep_all_seqs,
    )

    return {
        "dev-other": MetaDataset(
            datasets={
                "features": HdfDataset(
                    files=features_dev_other_hdfs,
                    segment_file=all_dev_other_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_dev_hdfs,
                    segment_file=all_dev_other_seq_tags,
                ),
            },
            data_map={
                "data": ("features", "data"),
                "phon_indices": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }
