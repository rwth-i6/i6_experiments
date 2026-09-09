from typing import Optional

from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob

from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset

from ....data.librispeech import audio, text
from ....data.librispeech.text import PhonemeLexicon
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings


def _phonemized(
    data_name: str,
    dump_hdf_concurrent: int,
    sil_prob: float,
    surround_w_sil: bool,
    lexicon: Optional[PhonemeLexicon],
    keep_all_seqs: bool = False,
):
    """
    Phonemized LibriSpeech text: ``(hdfs, vocab_file, vocab_size, seq_tags)``. ``lexicon=None`` = the
    historical fairseq/g2p_en lexicon from the LM corpus (41-symbol vocab); otherwise the given
    :class:`PhonemeLexicon` (every seq kept, OOVs G2P'd), e.g. ``text.get_lbs_lexicon()`` for the 39-phoneme
    LibriSpeech GMM set (vocab size 40 with the ``[SILENCE]`` placeholder).
    """
    if lexicon is None:
        # we don't pass sil_prob here, because we just want to get the lexicon here
        # we don't use the text-only data for training here
        _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
        hdfs, _, _, seq_tags = text.get_phonemized_text(
            data_name,
            lexicon_file=lexicon_file,
            dump_hdf_concurrent=dump_hdf_concurrent,
            vocab_file=phoneme_vocab,
            sil_prob=sil_prob,
            surround_w_sil=surround_w_sil,
            apply_lid_filter=not keep_all_seqs,
            extend_lexicon_w_g2p=keep_all_seqs,
        )
        return hdfs, phoneme_vocab, 41, seq_tags
    hdfs, phoneme_vocab, _, seq_tags = text.get_phonemized_text_w_lexicon(
        data_name,
        lexicon=lexicon,
        dump_hdf_concurrent=dump_hdf_concurrent,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    return hdfs, phoneme_vocab, lexicon.vocab_size, seq_tags


def build_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    lexicon: Optional[PhonemeLexicon] = None,
):
    """
    :param lexicon: phonemize with this lexicon / phoneme set instead of the historical g2p_en one, see
        :func:`_phonemized`. None keeps the existing job hashes.
    """
    phoneme_960_hdfs, phoneme_vocab, phoneme_vocab_size, train_seq_tags = _phonemized(
        "train-other-960", 10, sil_prob, surround_w_sil, lexicon
    )
    phoneme_dev_clean_hdfs, _, _, dev_clean_seq_tags = _phonemized("dev-clean", 1, sil_prob, surround_w_sil, lexicon)
    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = _phonemized("dev-other", 1, sil_prob, surround_w_sil, lexicon)

    dev_seq_tags = ConcatenateJob([dev_clean_seq_tags, dev_other_seq_tags], zip_out=False).out

    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out
    dev_seq_tags = TakeNRandomLinesJob(text_file=dev_seq_tags, num_lines=3000).out

    return TrainingDatasets(
        # HDFDataset alone does not work with partition epoch...
        train=MetaDataset(
            datasets={
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=train_seq_tags,
                    # set here because this controls which seqs are loaded
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                "data": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
        eval_datasets={
            "devtrain": HdfDataset(
                files=phoneme_960_hdfs,
                segment_file=devtrain_seq_tags,
                seq_ordering="sorted_reverse",
            ),
            "dev": HdfDataset(
                files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                segment_file=dev_seq_tags,
                seq_ordering="sorted_reverse",
            ),
        },
        datastreams={
            "data": LabelDatastream(
                available_for_inference=True,
                vocab=phoneme_vocab,
                vocab_size=phoneme_vocab_size,
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
    lexicon: Optional[PhonemeLexicon] = None,
):
    """
    :param keep_all_seqs: phonemize the full corpus instead of dropping the sequences that the language-ID
        filter and the lexicon-OOV filter of ``PhonemizeTextDataJob`` remove. With the default False,
        dev-other is scored on 2712 of 2864 utterances only (see "Eval-set sequence coverage" in CLAUDE.md).
        Only affects forward/scoring jobs, never a training.
    :param lexicon: see :func:`build_training_datasets`; a lexicon always keeps all seqs (2864 utts).
    """
    # never drop eval seqs: the reference must cover the whole corpus, otherwise WER/PER is not
    # comparable (LID filter: 5 seqs, lexicon OOV: 147 seqs on dev-other)
    phoneme_dev_hdfs, _, _, dev_seq_tags = _phonemized(
        "dev-other", 1, sil_prob, surround_w_sil, lexicon, keep_all_seqs=keep_all_seqs
    )

    return {
        "dev-other": HdfDataset(
            files=phoneme_dev_hdfs,
            segment_file=dev_seq_tags,
            seq_ordering="sorted_reverse",
        )
    }
