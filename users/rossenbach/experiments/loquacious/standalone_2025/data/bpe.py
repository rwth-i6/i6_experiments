"""
Dataset helpers for the BPE-based training
"""
from sisyphus import tk
from typing import List, Optional

from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob

from i6_experiments.common.datasets.loquacious.corpus import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.datasets.loquacious.vocab import get_subword_nmt_bpe
from i6_experiments.common.datasets.loquacious.lexicon import get_bliss_lexicon
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.rossenbach.datasets.loquacious import get_dev_segments

from .common import DatasetSettings, TrainingDatasets, build_training_datasets
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE, SUBWORD_NMT_REPO


def get_bpe_datastream(loquacious_key: str, bpe_size: int, is_recog: bool, use_postfix: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    :param use_postfix: True for RNN-T or Attention, False for CTC
    """
    bpe_settings = get_subword_nmt_bpe(corpus_key=loquacious_key, bpe_size=bpe_size, unk_label="<unk>")

    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog,
        seq_postfix=0 if use_postfix else None,
    )
    return bpe_targets


def get_bpe_lexicon(loquacious_key: str, bpe_size: int, variant=1) -> tk.Path:
    """
    Create BPE lexicon without unknown and silence

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon bliss xml file
    """
    bpe_settings = get_subword_nmt_bpe(corpus_key=loquacious_key, bpe_size=bpe_size, unk_label="<unk>")
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False, variant=variant),
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon

    return bpe_lexicon


def get_text_lexicon(prefix: str, loquacious_key: str, bpe_size: int, variant=1) -> tk.Path:
    """
    Get a bpe lexicon in line-based text format to be used for torchaudio/Flashlight decoding

    :param prefix:
    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon text file
    """
    bliss_lex = get_bpe_lexicon(loquacious_key=loquacious_key, bpe_size=bpe_size, variant=1)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_custom_bpe_lexicon(bliss_lexicon, bpe_codes, bpe_vocab):
    """

    :param bliss_lexicon:
    :param bpe_codes:
    :param bpe_vocab:
    :return:
    """
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=bliss_lexicon,
        bpe_codes=bpe_codes,
        bpe_vocab=bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bpe_lexicon,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon



def build_bpe_training_datasets(
    prefix: str,
    loquacious_key: str,
    bpe_size: int,
    settings: DatasetSettings,
    use_postfix: bool,
    extra_train_ogg_zips: Optional[List[tk.Path]] = None,
    data_repetition_factors: Optional[List[int]] = None,
    explicit_devtrain=False,
    extra_train_bliss: Optional[List[tk.Path]] = None,
) -> TrainingDatasets:
    """

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :param settings: configuration object for the dataset pipeline
    :param use_postfix: True for RNN-T or Attention, False for CTC
    :param extra_train_ogg_zips: add additional ogg zips for training, e.g. created by `synthetic_librispeech_bliss_to_ogg_zip`
    :param data_repetition_factors: list if integers, first entry is for the original librispeech data
    """
    label_datastream = get_bpe_datastream(
        loquacious_key=loquacious_key, bpe_size=bpe_size, is_recog=True, use_postfix=use_postfix
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[loquacious_key]

    if extra_train_ogg_zips is None:
        ogg_zips = train_ogg
    else:
        assert data_repetition_factors, "please provide repetition factors if you provide extra ogg zips"
        assert len(extra_train_ogg_zips) + 1 == len(data_repetition_factors)
        ogg_zips = [train_ogg] * data_repetition_factors[0]
        for ogg_zip, repetition in zip(extra_train_ogg_zips, data_repetition_factors[1:]):
            ogg_zips += [ogg_zip] * repetition

    dev_ogg = ogg_zip_dict["dev.all"]
    dev_segments = get_dev_segments()

    explicit_devtrain_ogg = None
    if explicit_devtrain:
        train_bliss = get_bliss_corpus_dict()[loquacious_key]
        if extra_train_bliss:
            from i6_core.corpus.transform import MergeCorporaJob
            merge_corpora = [train_bliss] + extra_train_bliss
            devtrain_ref_bliss = MergeCorporaJob(merge_corpora, "devtrain").out_merged_corpus
        else:
            devtrain_ref_bliss = train_bliss

        from i6_core.corpus.filter import FilterCorpusBySegmentsJob
        from i6_core.corpus.segments import SegmentCorpusJob
        from i6_core.returnn.oggzip import BlissToOggZipJob
        # still 1 based nightmare
        all_segments = SegmentCorpusJob(devtrain_ref_bliss, 1).out_single_segment_files[1]
        from i6_experiments.users.rossenbach.segments.helper import shuffle_and_head
        devtrain_segments = shuffle_and_head(all_segments, num_lines=3000)
        devtrain_corpus = FilterCorpusBySegmentsJob(
            devtrain_ref_bliss,
            devtrain_segments,
            compressed=True,
            delete_empty_recordings=True
        ).out_corpus
        zip_dataset_job = BlissToOggZipJob(
            bliss_corpus=devtrain_corpus,
            no_conversion=True,
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        zip_dataset_job.rqmt = {"cpu": 16, "mem": 16, "time": 24}
        explicit_devtrain_ogg = zip_dataset_job.out_ogg_zip


    return build_training_datasets(
        train_ogg=ogg_zips,
        dev_ogg=dev_ogg,
        dev_segments=dev_segments,
        settings=settings,
        label_datastream=label_datastream,
        explicit_devtrain_ogg=explicit_devtrain_ogg,
    )
