from dataclasses import dataclass
from typing import List, Optional, Tuple

from sisyphus import tk, Path

from i6_experiments.users.schmitt.text.normalize import NormalizeLBSLMDataJob
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.schmitt.corpus.seq_tags import GetSeqTagsFromCorpusJob
from i6_experiments.users.schmitt.datasets.utils.phonemize import (
    LexiconTxtToLineBasedLexiconJob,
    CountLexiconVariantsInSegmentPhonemesJob,
)

from i6_core.tools.download import DownloadJob
from i6_core.corpus.convert import CorpusToTextDictJob, CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob, PipelineJob

from ..text import get_phonemized_data
from ...default_tools import get_wav2letter_root


def get_lm_minus_librivox() -> Tuple[tk.Path, Optional[tk.Path]]:
    lm_minus_librivox = DownloadJob(
        url="https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt",
        target_filename="lm_corpus_minus_librivox",
    ).out_file

    return NormalizeLBSLMDataJob(
        wav2letter_root=get_wav2letter_root(),
        wav2letter_python_exe=tk.Path(""),
        librispeech_lm_corpus=lm_minus_librivox,
    ).out_corpus_norm, None


def _get_corpus_text_dict(key: str) -> Tuple[tk.Path, tk.Path]:
    corpus = librispeech.get_bliss_corpus_dict()[key]
    text_dict = CorpusToTextDictJob(corpus, gzip=True).out_dictionary
    seq_tags = GetSeqTagsFromCorpusJob(corpus, gzip=False).out_txt
    return text_dict, seq_tags


def get_corpus_text(key: str, gzip=False) -> Tuple[tk.Path, tk.Path]:
    """train corpus text (used for LM training)"""
    corpus = librispeech.get_bliss_corpus_dict()[key]
    seq_tags = GetSeqTagsFromCorpusJob(corpus, gzip=gzip).out_txt
    text_lines = CorpusToTxtJob(corpus, gzip=gzip).out_txt
    return text_lines, seq_tags


def get_dev_text() -> Tuple[tk.Path, tk.Path]:
    text_dev_other, seq_tags_dev_other = get_corpus_text("dev-other")
    text_dev_clean, seq_tags_dev_clean = get_corpus_text("dev-clean")
    concat_text = ConcatenateJob([text_dev_clean, text_dev_other], zip_out=False).out
    lowercase_text = PipelineJob(concat_text, pipeline=["tr A-Z a-z"]).out

    concat_seq_tags = ConcatenateJob([seq_tags_dev_clean, seq_tags_dev_other], zip_out=False).out
    return lowercase_text, concat_seq_tags


def get_text(librispeech_key: str) -> Tuple[tk.Path, Optional[tk.Path]]:
    if librispeech_key == "dev":
        lowercase_text, seq_tags = get_dev_text()
    elif librispeech_key == "lm_minus_librivox":
        lowercase_text, seq_tags = get_lm_minus_librivox()
    else:
        text, seq_tags = get_corpus_text(librispeech_key)
        lowercase_text = PipelineJob(text, pipeline=["tr A-Z a-z"]).out

    return lowercase_text, seq_tags


def get_phonemized_text(
    data_name: str,
    dump_hdf_concurrent: int,
    lexicon_file: Optional[Path] = None,
    vocab_file: Optional[Path] = None,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    apply_lid_filter: bool = True,
    extend_lexicon_w_g2p: bool = False,
    collapse_repeats: bool = False,
    output_subdir: Optional[str] = None,
):
    """
    :param apply_lid_filter: drop lines not confidently classified as English (see `get_phonemized_data`)
    :param extend_lexicon_w_g2p: keep lines with OOV words by G2P-extending the lexicon (see
        `get_phonemized_data`). Both default to the historical behavior; enable them for eval sets only.
    :param collapse_repeats: merge adjacent identical phonemes (see `get_phonemized_data`)
    :param output_subdir: extra output dir level for the registered text (see `get_phonemized_data`)
    """
    text_data, seq_tags = get_text(data_name)

    return get_phonemized_data(
        dataset_name=data_name,
        corpus_name="librispeech",
        text_file=text_data,
        dump_hdf_concurrent=dump_hdf_concurrent,
        lexicon_file=lexicon_file,
        seq_tag_file=seq_tags,
        vocab_file=vocab_file,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
        apply_lid_filter=apply_lid_filter,
        extend_lexicon_w_g2p=extend_lexicon_w_g2p,
        collapse_repeats=collapse_repeats,
        output_subdir=output_subdir,
    )


# ---------------------------------------------------------------------------------------------------------
# Alternative phoneme set / lexicon: the official LibriSpeech lexicon (i6 GMM inventory)
# ---------------------------------------------------------------------------------------------------------

#: Segment-level phoneme labels of zyang's LibriSpeech-960 GMM alignment (the alignment the "cheating"
#: k512 segment clusters of ``unsup...data.common._get_cheating_train_clusters`` are cut from): one integer per
#: non-silence phoneme segment, same 20 shards / 278,400 seqs / identical per-seq lengths as the cluster HDFs
#: (verified), ``1=AA, ..., 39=ZH`` in alphabetical order, silence (0) already removed. Also listed as
#: ``GMM_SEGMENT_PHONEMES_LS960`` in lkleppel's ``guided_kmeans/setup/constants.py``.
GMM_SEGMENT_PHONEME_HDFS: List[tk.Path] = [
    tk.Path(
        "/work/asr4/zyang/mini/alias/example_setups/librispeech/phmm_standalone_2024"
        f"/ls960_gmm_oracle_segment_clustering/segment_phonemes/shard_{i:03d}/output/gmm_segment_phonemes.{i:03d}.hdf"
    )
    for i in range(20)
]


@dataclass(frozen=True)
class PhonemeLexicon:
    """
    A word->phoneme lexicon plus the phoneme vocab it induces, for phonemizing text with a phoneme set other
    than the historical fairseq/g2p_en one (39 ARPAbet + ``'`` + ``<SIL>`` = 41). See :func:`get_lbs_lexicon`.
    """

    name: str  # short id, used for output dir names
    lexicon_file: tk.Path  # ``word<TAB>phonemes`` per line, one pronunciation per word
    vocab_file: tk.Path  # RETURNN vocab dict literal
    vocab_size: int
    collapse_repeats: bool  # merge adjacent identical phonemes when phonemizing


def get_lbs_lexicon(variant_selection: str = "aligner", collapse_repeats: bool = True) -> PhonemeLexicon:
    """
    The official LibriSpeech lexicon (openslr.org/11, 200k words), stress markers stripped -> the 39 ARPAbet
    phonemes of the i6 LibriSpeech GMM setups, i.e. the phoneme set of zyang's GMM alignment / segmentation and
    of lkleppel's ``phoneme.lex.xml.gz``: no silence phoneme and no ``'`` (our g2p_en-derived set has both,
    ``'`` from words with a trailing apostrophe). The vocab is ``[SILENCE]=0, AA=1, ..., ZH=39`` (size 40) so
    that its indices coincide with :data:`GMM_SEGMENT_PHONEME_HDFS`; index 0 never occurs in phonemized text.

    :param variant_selection: which pronunciation to use for the 3% of words with several variants:
        ``"first"`` = the first one listed; ``"aligner"`` = the one zyang's GMM alignment realized most often
        (:class:`CountLexiconVariantsInSegmentPhonemesJob` over the train-other-960 transcripts vs
        :data:`GMM_SEGMENT_PHONEME_HDFS`). Measured against those segment phonemes on held-out utterances:
        first + collapse 5.7% token disagreement / 29% utterances of equal length; aligner + collapse 3.8% /
        38%; the historical g2p_en text 6.6% / 25%. The aligner variant uses the alignment only for
        corpus-level pronunciation statistics (no utterance pairing).
    :param collapse_repeats: merge adjacent identical phonemes (``S AH M M IH S`` -> ``S AH M IH S``), as a
        segmentation cut from a frame alignment does (90% of the GMM segment sequences have no repeats).
    """
    assert variant_selection in ("first", "aligner"), variant_selection
    from i6_core.tools.download import DownloadJob

    # same job as i6_experiments.common.datasets.librispeech.lexicon._get_raw_bliss_lexicon
    lexicon_txt = DownloadJob(
        url="https://www.openslr.org/resources/11/librispeech-lexicon.txt",
        target_filename="librispeech-lexicon.txt",
        checksum="d722bc29908cd338ae738edd70f61826a6fca29aaa704a9493f0006773f79d71",
    ).out_file

    variant_counts_file = None
    if variant_selection == "aligner":
        train_text, train_seq_tags = get_text("train-other-960")
        count_job = CountLexiconVariantsInSegmentPhonemesJob(
            lexicon_txt=lexicon_txt,
            text_file=train_text,
            seq_tag_file=train_seq_tags,
            segment_phoneme_hdfs=GMM_SEGMENT_PHONEME_HDFS,
            collapse_repeats=collapse_repeats,
        )
        tk.register_output("data/librispeech/text/lexicon/lbs_aligner_variant_counts.txt", count_job.out_variant_counts)
        tk.register_output("data/librispeech/text/lexicon/lbs_aligner_variant_counts.stats.txt", count_job.out_stats)
        variant_counts_file = count_job.out_variant_counts

    lexicon_job = LexiconTxtToLineBasedLexiconJob(
        lexicon_txt=lexicon_txt,
        strip_stress=True,
        variant_counts_file=variant_counts_file,
        special_symbols_first=("[SILENCE]",),
    )
    name = f"lbs_{variant_selection}" + ("_collapse" if collapse_repeats else "")
    tk.register_output(f"data/librispeech/text/lexicon/{name}.lst", lexicon_job.out_lexicon_file)
    tk.register_output(f"data/librispeech/text/lexicon/{name}.vocab.txt", lexicon_job.out_phoneme_vocab)
    tk.register_output(f"data/librispeech/text/lexicon/{name}.stats.txt", lexicon_job.out_stats)
    return PhonemeLexicon(
        name=name,
        lexicon_file=lexicon_job.out_lexicon_file,
        vocab_file=lexicon_job.out_phoneme_vocab,
        # 39 phonemes + [SILENCE]; the job writes the actual count to out_num_phonemes -- check the stats file
        # if the lexicon file ever changes
        vocab_size=40,
        collapse_repeats=collapse_repeats,
    )


def get_phonemized_text_w_lexicon(
    data_name: str,
    lexicon: PhonemeLexicon,
    dump_hdf_concurrent: int,
    sil_prob: float = 0.0,
    surround_w_sil: bool = False,
):
    """
    :func:`get_phonemized_text` with an explicit :class:`PhonemeLexicon` instead of the historical g2p_en one.
    Every sequence is kept (no language-ID filter -- LibriSpeech is English -- and OOV words are G2P'd into the
    lexicon's inventory via :class:`ExtendLexiconWithG2PJob`; g2p_en's ARPAbet is the same inventory), so e.g.
    all 281,241 train-other-960 and all 2864 dev-other utterances are phonemized. Returns
    ``(hdfs, vocab_file, lexicon_file, seq_tags)`` like :func:`get_phonemized_text`.

    NB ``sil_prob``/``surround_w_sil`` default to *no* silence here: the lexicon has no silence phoneme
    (``<SIL>`` would not be in the vocab), the ``[SILENCE]`` vocab entry is only an index placeholder.
    """
    assert sil_prob == 0.0 and not surround_w_sil, "the LBS phoneme set has no silence symbol in the vocab"
    return get_phonemized_text(
        data_name,
        dump_hdf_concurrent=dump_hdf_concurrent,
        lexicon_file=lexicon.lexicon_file,
        vocab_file=lexicon.vocab_file,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
        apply_lid_filter=False,
        extend_lexicon_w_g2p=True,
        collapse_repeats=lexicon.collapse_repeats,
        output_subdir=lexicon.name,
    )
