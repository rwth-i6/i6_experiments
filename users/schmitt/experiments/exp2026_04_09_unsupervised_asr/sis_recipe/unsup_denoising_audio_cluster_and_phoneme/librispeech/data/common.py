from typing import Optional

from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob
from i6_core.serialization import CallImport

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datasets import OggZipDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.schmitt.datasets.postprocessing import PostprocessingDataset
from i6_experiments.users.schmitt.datasets.lm import LmDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset
from i6_experiments.users.schmitt.datasets.utils.hdf import DumpCorpusTextAsUtf8ToHdfJob
from i6_experiments.users.schmitt.datasets.utils.extract_seq_list import FilterSeqListByHdfSeqTagsJob

from ....data.librispeech import audio, text
from ....data.librispeech.text import PhonemeLexicon
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings, _wrap_in_post_proc

from sisyphus import tk


def _get_phonemize_post_proc_func(
    sil_prob: float,
    surround_w_sil: bool,
    min_num_sil: int,
    max_num_sil: int,
    lexicon_file: tk.Path,
    phoneme_datastream,
    min_num_surround_sil: int = 1,
    max_num_surround_sil: int = 1,
):
    hashed_args = {
        "sil_prob": sil_prob,
        "surround_w_sil": surround_w_sil,
        "lexicon_file": lexicon_file,
        "target_key": "data",
        "new_target_key": "data_w_sil",
        "vocab_opts": phoneme_datastream.as_returnn_targets_opts(),
        "min_num_sil": min_num_sil,
        "max_num_sil": max_num_sil,
    }
    if min_num_surround_sil != 1:
        hashed_args["min_num_surround_sil"] = min_num_surround_sil
    if max_num_surround_sil != 1:
        hashed_args["max_num_surround_sil"] = max_num_surround_sil
    return CallImport(
        code_object_path="i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.models.post_proc.phonemize.PhonemizeAndInsertSilence",
        hashed_arguments=hashed_args,
        unhashed_package_root=None,
        unhashed_arguments={},
    )


def _get_cheating_train_clusters(num_clusters: int):
    assert num_clusters in (512, 1024, 2048, 8192)
    return [
        tk.Path(
            f"/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/ls960_gmm_oracle_segment_clustering/clustering/k{num_clusters}/compat/cluster_labels_k{num_clusters}.{idx:03d}.returnn_compat.hdf"
        )
        for idx in range(20)
    ]


def _get_train_960_phonemes(sil_prob: float, surround_w_sil: bool, lexicon: Optional[PhonemeLexicon]):
    """
    Phonemized train-other-960 transcripts: ``(hdfs, vocab_file, vocab_size, seq_tags)``.

    ``lexicon=None`` = the historical fairseq/g2p_en lexicon derived from the LM corpus (41-symbol vocab incl.
    ``'`` and ``<SIL>``; LID-filtered + OOV-dropped -> 266,927 utts); otherwise the given
    :class:`PhonemeLexicon` (e.g. ``text.get_lbs_lexicon()``: the 39-phoneme LibriSpeech GMM set, all 281,241
    utts kept). Only the cheating-cluster builders take the lexicon so far.
    """
    if lexicon is None:
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
        return phoneme_960_hdfs, phoneme_vocab, 41, train_seq_tags

    phoneme_960_hdfs, phoneme_vocab, _, train_seq_tags = text.get_phonemized_text_w_lexicon(
        "train-other-960",
        lexicon=lexicon,
        dump_hdf_concurrent=10,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    return phoneme_960_hdfs, phoneme_vocab, lexicon.vocab_size, train_seq_tags


def build_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    _, clusters_960, pca_960, clusters_960_hdfs = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_other_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_clean_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-clean",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
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
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
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
                ("feature_clusters", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_dev_other_hdfs + clusters_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=128,
            ),
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


def build_training_datasets_w_cheating_clusters(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    num_audio_clusters: int = 512,
    lexicon: Optional[PhonemeLexicon] = None,
):
    """
    :param lexicon: phonemize the (unpaired) text with this lexicon / phoneme set instead of the historical
        g2p_en one, see :func:`_get_train_960_phonemes`. ``text.get_lbs_lexicon()`` gives the 39-phoneme set of
        the GMM alignment the cheating clusters come from (``[SILENCE]`` placeholder at index 0, vocab size 40).
        None keeps the existing job hashes.
    """
    clusters_960_hdfs = _get_cheating_train_clusters(num_audio_clusters)

    phoneme_960_hdfs, phoneme_vocab, phoneme_vocab_size, train_seq_tags = _get_train_960_phonemes(
        sil_prob=sil_prob, surround_w_sil=surround_w_sil, lexicon=lexicon
    )

    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out

    return TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
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
                ("feature_clusters", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "phon_indices",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=num_audio_clusters,
            ),
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=phoneme_vocab_size,
            ),
        },
    )


def build_training_datasets_w_silence_in_input(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    min_num_sil: int = 1,
    max_num_sil: int = 3,
    min_num_surround_sil: int = 1,
    max_num_surround_sil: int = 1,
):
    _, clusters_960, pca_960, clusters_960_hdfs = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_other_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_clean_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-clean",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )

    # we use these functions only to get the correct seq tags because some seqs are filtered out in the process
    # because the words cannot be phonemized by the lexicon
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    _, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    _, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    _, _, _, dev_other_seq_tags = text.get_phonemized_text(
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

    phoneme_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=phoneme_vocab,
        vocab_size=41,
    )
    bliss_corpus_dict = get_bliss_corpus_dict("ogg")

    train_bytes_hdfs = DumpCorpusTextAsUtf8ToHdfJob(
        bliss_corpus=bliss_corpus_dict["train-other-960"], concurrent=10
    ).out_hdfs
    dev_other_hdf = DumpCorpusTextAsUtf8ToHdfJob(bliss_corpus=bliss_corpus_dict["dev-other"], concurrent=1).out_hdfs[0]
    dev_clean_hdf = DumpCorpusTextAsUtf8ToHdfJob(bliss_corpus=bliss_corpus_dict["dev-clean"], concurrent=1).out_hdfs[0]

    phonemize_func = _get_phonemize_post_proc_func(
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
        lexicon_file=lexicon_file,
        min_num_sil=min_num_sil,
        max_num_sil=max_num_sil,
        min_num_surround_sil=min_num_surround_sil,
        max_num_surround_sil=max_num_surround_sil,
        phoneme_datastream=phoneme_datastream,
    )

    data_map = {
        ("phon_indices", "data"): "phon_indices",
        ("phon_indices", "data_w_sil"): "phon_indices_w_sil",
        ("feature_clusters", "data"): "data",
    }
    map_outputs = {
        "data": phoneme_datastream.as_returnn_extern_data_opts(),
        "data_w_sil": phoneme_datastream.as_returnn_extern_data_opts(),
    }

    return TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
                "phon_indices": _wrap_in_post_proc(
                    dataset=HdfDataset(
                        files=list(train_bytes_hdfs.values()),
                        segment_file=train_seq_tags,
                        partition_epoch=settings.train_partition_epoch,
                        seq_ordering=settings.train_seq_ordering,
                    ),
                    map_seq=phonemize_func,
                    map_outputs=map_outputs,
                    settings=settings,
                ),
            },
            data_map=data_map,
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": _wrap_in_post_proc(
                        dataset=HdfDataset(
                            files=list(train_bytes_hdfs.values()),
                            segment_file=devtrain_seq_tags,
                            partition_epoch=1,
                        ),
                        map_seq=phonemize_func,
                        map_outputs=map_outputs,
                        settings=settings,
                    ),
                },
                data_map=data_map,
                # sorted does not work/ is not implemented
                seq_ordering="interleave",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_dev_other_hdfs + clusters_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": _wrap_in_post_proc(
                        dataset=HdfDataset(
                            files=[dev_clean_hdf, dev_other_hdf],
                            segment_file=dev_seq_tags,
                            partition_epoch=1,
                        ),
                        map_seq=phonemize_func,
                        map_outputs=map_outputs,
                        settings=settings,
                    ),
                },
                data_map=data_map,
                seq_ordering="interleave",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=128,
            ),
            "phon_indices": phoneme_datastream,
            "phon_indices_w_sil": phoneme_datastream,
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


def build_test_datasets(
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    # TODO: set to True to score the *full* 2864-utt dev-other, as the wav2vec-U setup now does.
    #  Kept False here so the existing recog/scoring job hashes -- and thus every PER/WER number
    #  measured so far -- stay untouched; flipping it re-runs all recog/analysis/PPL jobs of this
    #  setup and makes the new numbers incomparable to the old ones.
    #  See "Eval-set sequence coverage (dev-other = 2864 utts)" in CLAUDE.md.
    keep_all_seqs: bool = False,
):
    """
    :param keep_all_seqs: phonemize the full corpus instead of dropping the sequences that the language-ID
        filter and the lexicon-OOV filter of ``PhonemizeTextDataJob`` remove. With the default False,
        dev-other is scored on 2712 of 2864 utterances only (see "Eval-set sequence coverage" in CLAUDE.md).
        Only affects forward/scoring jobs, never a training.
    """
    _, clusters_960, pca_960, _ = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_other_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )

    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_dev_hdfs, _, _, dev_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
        # never drop eval seqs: the reference must cover the whole corpus, otherwise WER/PER is not
        # comparable (LID filter: 5 seqs, lexicon OOV: 147 seqs on dev-other)
        apply_lid_filter=not keep_all_seqs,
        extend_lexicon_w_g2p=keep_all_seqs,
    )

    return {
        "dev-other": MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_dev_other_hdfs,
                    segment_file=dev_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_dev_hdfs,
                    segment_file=dev_seq_tags,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "phon_indices": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }


def build_test_datasets_w_cheating_clusters(
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    num_audio_clusters: int = 512,
    lexicon: Optional[PhonemeLexicon] = None,
):
    """
    NB unlike `build_test_datasets`, this one deliberately keeps the sequence-dropping phonemization
    (no `keep_all_seqs`): its "dev-other" is an arbitrary 3000-utterance sample of *train*-other-960
    (the cheating clusters only exist there), so full corpus coverage buys nothing, while un-filtering
    would enlarge the pool that `TakeNRandomLinesJob` samples from and thus silently re-draw the eval
    set, invalidating all cheat-seg numbers measured so far. It also shares the phonemization job with
    the training data, so the un-filtered variant would mean re-phonemizing all 281k train utterances.

    :param lexicon: see :func:`build_training_datasets_w_cheating_clusters`. With a lexicon the text is
        phonemized without dropping seqs, so the 3000-utt sample is drawn from a different pool -- the eval
        set of a lexicon variant is NOT the same 3000 utterances as the historical one (nor is the phoneme set,
        so the PERs are not comparable anyway).
    """
    clusters_960_hdfs = _get_cheating_train_clusters(num_audio_clusters)

    phoneme_960_hdfs, phoneme_vocab, _, train_seq_tags = _get_train_960_phonemes(
        sil_prob=sil_prob, surround_w_sil=surround_w_sil, lexicon=lexicon
    )
    # the cheating clusters and the phoneme HDFs do not cover the exact same set of utterances
    # (the clusters are missing ~1% of the phonemized seqs). The MetaDataset below hands the seq list of its
    # control dataset ("phon_indices") to the cluster dataset, which raises a KeyError for the seqs it does not
    # have, so restrict the seq list to the intersection before sampling from it.
    train_seq_tags = FilterSeqListByHdfSeqTagsJob(
        seq_list=train_seq_tags, hdf_files=clusters_960_hdfs
    ).out_seq_list
    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out

    return {
        "dev-other": MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=devtrain_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=devtrain_seq_tags,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "phon_indices": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }


def build_test_datasets_w_silence_in_input(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    min_num_sil: int = 1,
    max_num_sil: int = 3,
    min_num_surround_sil: int = 1,
    max_num_surround_sil: int = 1,
):
    _, clusters_960, pca_960, _ = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_other_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )

    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    # NB no `keep_all_seqs` here (unlike `build_test_datasets`): the reference of this test set is produced at
    # runtime by `PhonemizeAndInsertSilence`, which raises on an OOV word instead of skipping the seq, so
    # recovering the 152 dropped seqs would also require handing the G2P-extended lexicon to that post-proc
    # func. Left as-is since this legacy setup's trainings ran with `keep=[1000]` and re-running its recogs
    # would hit the deleted-checkpoint problem (see the Gotchas section in CLAUDE.md).
    _, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    phoneme_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=phoneme_vocab,
        vocab_size=41,
    )
    bliss_corpus_dict = get_bliss_corpus_dict("ogg")

    dev_other_bytes_hdf = DumpCorpusTextAsUtf8ToHdfJob(
        bliss_corpus=bliss_corpus_dict["dev-other"], concurrent=1
    ).out_hdfs[0]

    phonemize_func = _get_phonemize_post_proc_func(
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
        lexicon_file=lexicon_file,
        min_num_sil=min_num_sil,
        max_num_sil=max_num_sil,
        min_num_surround_sil=min_num_surround_sil,
        max_num_surround_sil=max_num_surround_sil,
        phoneme_datastream=phoneme_datastream,
    )

    map_outputs = {
        "data": phoneme_datastream.as_returnn_extern_data_opts(available_for_inference=True),
        "data_w_sil": phoneme_datastream.as_returnn_extern_data_opts(available_for_inference=True),
    }

    return {
        "dev-other": MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_dev_other_hdfs,
                    segment_file=dev_other_seq_tags,
                ),
                "phon_indices": _wrap_in_post_proc(
                    dataset=HdfDataset(
                        files=[dev_other_bytes_hdf],
                        segment_file=dev_other_seq_tags,
                        partition_epoch=1,
                    ),
                    map_seq=phonemize_func,
                    map_outputs=map_outputs,
                    settings=settings,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "phon_indices": ("phon_indices", "data_w_sil"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }
