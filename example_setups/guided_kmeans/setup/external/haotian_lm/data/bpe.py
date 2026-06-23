"""
Dataset helpers for the BPE-based training
"""
import os

from sisyphus import tk

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream, LabelDatastream

from .common import DatasetSettings, TrainingDatasets, build_training_datasets
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE, SUBWORD_NMT_REPO
from ..rasr import (
    CreateCorpusBpePhmmLexiconJob,
    CreateCorpusBpeFsaLexiconJob,
    DEFAULT_CTC_BPE_RASR_CONFIG,
)


def get_bpe_datastream(librispeech_key: str, bpe_size: int, is_recog: bool, use_postfix: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    :param use_postfix: True for RNN-T or Attention, False for CTC
    """
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="<unk>")

    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog,
        seq_postfix=0 if use_postfix else None,
    )
    return bpe_targets


def get_bpe_lexicon(librispeech_key: str, bpe_size: int) -> tk.Path:
    """
    Create BPE lexicon without unknown and silence

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon bliss xml file
    """
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="<unk>")
    corpus_text = CorpusToTxtJob(get_bliss_corpus_dict(audio_format="ogg")[librispeech_key], gzip=True).out_txt

    return CreateCorpusBpePhmmLexiconJob(
        corpus_text=corpus_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon


def get_bpe_ctc_lexicon(librispeech_key: str, bpe_size: int) -> tk.Path:
    """
    Create a BPE Bliss lexicon for **CTC** training (RASR FSA, ``topology="ctc"``) and recognition.

    Analogous to :func:`get_bpe_lexicon` but with the CTC blank instead of silence: ``[BLANK]`` at
    index 0 (``special="blank"``, with NO empty ``""`` orth -- see :data:`...rasr.DEFAULT_CTC_BPE_RASR_CONFIG`
    and ``[[ctc-fsa-overcounts-vs-torch]]``) followed by the BPE subword phonemes, with every corpus word
    mapped to its subword-nmt pronunciation. Used both by the ``topology="ctc"`` FSA (the ``fbw2`` full-sum
    CTC loss) and by the LibRASR CTC search. The BPE counterpart of :func:`...data.phon.get_ctc_eow_lexicon`.

    :param librispeech_key: which librispeech corpus to use for bpe training / lexicon coverage
    :param bpe_size: number of BPE merge operations
    :return: path to the CTC BPE bliss lexicon (without sentence-boundary lemmata)
    """
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="<unk>")
    corpus_text = CorpusToTxtJob(get_bliss_corpus_dict(audio_format="ogg")[librispeech_key], gzip=True).out_txt

    return CreateCorpusBpeFsaLexiconJob(
        corpus_text=corpus_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        rasr_config=DEFAULT_CTC_BPE_RASR_CONFIG,
        unk_label="<unk>",
    ).out_lexicon


def get_bpe_ctc_vocab_datastream(prefix: str, librispeech_key: str, bpe_size: int) -> LabelDatastream:
    """
    BPE-subword LabelDatastream from the CTC BPE lexicon (``[BLANK]`` at index 0).

    Only its ``vocab_size`` (= #BPE subwords + 1, including ``[BLANK]``) is consumed for the AM
    ``label_target_size``; the FSA full-sum loss reads the raw orthography, not an encoded label stream.
    Derived from the lexicon phoneme inventory (NOT the ``BpeDatastream``) so the count exactly matches
    the ``topology="ctc"`` FSA emission width. The BPE counterpart of
    :func:`...data.phon.get_ctc_eow_vocab_datastream`.

    :param prefix: alias prefix
    :param librispeech_key: baseline librispeech dataset
    :param bpe_size: number of BPE merge operations
    """
    lexicon = get_bpe_ctc_lexicon(librispeech_key=librispeech_key, bpe_size=bpe_size)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{librispeech_key}_bpe{bpe_size}", "ctc_bpe_returnn_vocab_job"))

    return LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )


def build_bpe_ctc_fsa_training_datasets(
    prefix: str,
    librispeech_key: str,
    bpe_size: int,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """
    Build training datasets for BPE CTC training via the RASR FSA (``fbw2`` full-sum) loss.

    Mirrors :func:`...data.phon.build_eow_phon_ctc_fsa_training_datasets`: raw-orthography targets (the
    FSA builder looks up words in the BPE lexicon at training time, ``use_raw_text_labels=True``), but the
    label datastream comes from the CTC BPE lexicon (``[BLANK]`` at index 0) so the reported ``vocab_size``
    matches the CTC FSA / AM inventory (= #BPE subwords + 1).

    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param bpe_size: number of BPE merge operations
    :param settings: configuration object for the dataset pipeline
    """
    label_datastream = get_bpe_ctc_vocab_datastream(
        prefix=prefix, librispeech_key=librispeech_key, bpe_size=bpe_size
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict["dev-clean"]
    dev_other_ogg = ogg_zip_dict["dev-other"]

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
        use_raw_text_labels=True,
    )


def get_text_lexicon(prefix: str, librispeech_key: str, bpe_size: int) -> tk.Path:
    """
    Get a bpe lexicon in line-based text format to be used for torchaudio/Flashlight decoding

    :param prefix:
    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon text file
    """
    bliss_lex = get_bpe_lexicon(librispeech_key=librispeech_key, bpe_size=bpe_size)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_bpe_training_datasets(
    prefix: str,
    librispeech_key: str,
    bpe_size: int,
    settings: DatasetSettings,
    use_postfix: bool,
    use_raw_text_labels: bool = False,
) -> TrainingDatasets:
    """

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :param settings: configuration object for the dataset pipeline
    :param use_postfix: True for RNN-T or Attention, False for CTC
    :param use_raw_text_labels: if True, do not apply BPE encoding on the target side.
        Needed for pHMM training where the FSA builder requires orthography text.
    """
    label_datastream = get_bpe_datastream(
        librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=False, use_postfix=use_postfix
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict["dev-clean"]
    dev_other_ogg = ogg_zip_dict["dev-other"]

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
        use_raw_text_labels=use_raw_text_labels,
    )
