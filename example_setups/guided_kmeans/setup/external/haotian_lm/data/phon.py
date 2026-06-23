"""
Dataset helpers for the EOW-augmented phoneme training
"""
from sisyphus import tk

import os
from typing import Optional

import pickle

from sisyphus import Job, Task

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.util import uopen

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_corpus_dict,
    get_bliss_lexicon,
)
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from .common import get_zip, build_training_datasets, TrainingDatasets, DatasetSettings
from ..rasr import NormalizePhonForPhmmLexiconJob, BuildEowPhonCtcLexiconJob


def get_eow_lexicon(g2p_librispeech_key: Optional[str], with_g2p: bool) -> tk.Path:
    """
    get the g2p bliss lexicon with EOW tokens added

    :param g2p_librispeech_key: which librispeech to use as baseline data
    :param with_g2p:
    :return: phoneme based bliss lexicon
    """
    if with_g2p:
        assert g2p_librispeech_key is not None
        lex = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=False)[g2p_librispeech_key]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=False)

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_phon_lexicon(g2p_librispeech_key: Optional[str], with_g2p: bool) -> tk.Path:
    """
    get the g2p bliss lexicon WITHOUT EOW tokens (plain monophone phonemes)

    Same as :func:`get_eow_lexicon` but without the :class:`AddEowPhonemesToLexiconJob` step.

    :param g2p_librispeech_key: which librispeech to use as baseline data
    :param with_g2p:
    :return: phoneme based bliss lexicon
    """
    if with_g2p:
        assert g2p_librispeech_key is not None
        lex = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=False)[g2p_librispeech_key]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=False)

    return lex


def get_eow_bliss(librispeech_key: str, g2p_librispeech_key: str, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param librispeech_key: which bliss dataset to "get"
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="ogg")[librispeech_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            # cv may include words from g2p
            bliss_lexicon=get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True),
            all_unknown=False,
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(librispeech_key: str, g2p_librispeech_key: str, remove_unk_seqs=False):
    """
    :param librispeech_key: which bliss dataset to "get"
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(
        librispeech_key=librispeech_key, g2p_librispeech_key=g2p_librispeech_key, remove_unk_seqs=remove_unk_seqs
    )
    zip_dataset = get_zip(f"{g2p_librispeech_key}_{librispeech_key}_filtered_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream

    :param prefix:
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    """
    lexicon = get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_text_lexicon() -> tk.Path:
    """
    :return: Text lexicon for the Flashlight decoder
    """
    bliss_lex = get_eow_lexicon(g2p_librispeech_key=None, with_g2p=False)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_eow_phon_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: DatasetSettings,
    lexicon_librispeech_key: Optional[str] = None,
) -> TrainingDatasets:
    """
    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    :param lexicon_librispeech_key: if we are using extra synthetic data, we might want a lexicon with the OOV coverage of that data as well
    """
    label_datastream = get_eow_vocab_datastream(
        prefix=prefix, g2p_librispeech_key=lexicon_librispeech_key or librispeech_key
    )

    _, train_ogg = get_eow_bliss_and_zip(
        librispeech_key=librispeech_key, g2p_librispeech_key=librispeech_key, remove_unk_seqs=False
    )
    _, dev_clean_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-clean", g2p_librispeech_key=librispeech_key, remove_unk_seqs=True
    )
    _, dev_other_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-other", g2p_librispeech_key=librispeech_key, remove_unk_seqs=True
    )

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
    )


def get_phmm_eow_lexicon(g2p_librispeech_key: str) -> tk.Path:
    """
    Get an EOW phoneme lexicon normalized for PHMM training:
    [SILENCE] at index 0, silence lemma with empty orth, no sentence-begin/end.

    :param g2p_librispeech_key: which librispeech corpus for G2P augmentation
    :return: path to the normalized PHMM phoneme lexicon
    """
    eow_lex = get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    return NormalizePhonForPhmmLexiconJob(eow_lex).out_lexicon


def get_phmm_phon_lexicon(g2p_librispeech_key: str) -> tk.Path:
    """
    Non-EOW counterpart of :func:`get_phmm_eow_lexicon`: a plain monophone phoneme lexicon
    normalized for PHMM training ([SILENCE] at index 0, silence lemma with empty orth, no
    sentence-begin/end), built without the EOW phoneme augmentation.

    :param g2p_librispeech_key: which librispeech corpus for G2P augmentation
    :return: path to the normalized non-EOW PHMM phoneme lexicon
    """
    phon_lex = get_phon_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    return NormalizePhonForPhmmLexiconJob(phon_lex).out_lexicon


def get_ctc_eow_lexicon(g2p_librispeech_key: str) -> tk.Path:
    """
    Get an EOW phoneme lexicon for CTC training (RASR FSA, ``topology="ctc"``) and recognition.

    Analogous to :func:`get_phmm_eow_lexicon` but with the CTC blank instead of silence:
    ``[BLANK]`` at index 0 (``special="blank"``) followed by the EOW phonemes. Used both by the
    FSA exporter (the ``fbw2`` full-sum CTC loss) and by the LibRASR CTC search.

    :param g2p_librispeech_key: which librispeech corpus for G2P augmentation
    :return: path to the CTC EOW phoneme lexicon (without sentence-boundary lemmata)
    """
    eow_lex = get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    return BuildEowPhonCtcLexiconJob(eow_lex).out_lexicon


def get_ctc_eow_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    EOW phoneme LabelDatastream from the CTC lexicon (``[BLANK]`` at index 0).

    Only its ``vocab_size`` (= #EOW phonemes + 1) is consumed for the AM ``label_target_size``;
    the FSA full-sum loss reads the raw orthography, not an encoded label stream.

    :param prefix:
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    """
    lexicon = get_ctc_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "ctc_eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_phmm_eow_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream from the PHMM-normalized lexicon (includes [SILENCE]).

    :param prefix:
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    """
    lexicon = get_phmm_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "phmm_eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_phmm_phon_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    Non-EOW counterpart of :func:`get_phmm_eow_vocab_datastream`: plain monophone phoneme
    LabelDatastream from the non-EOW PHMM-normalized lexicon (includes [SILENCE]).

    :param prefix:
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    """
    lexicon = get_phmm_phon_lexicon(g2p_librispeech_key=g2p_librispeech_key)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "phmm_phon_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


class ExtendPhmmVocabWithBosEosJob(Job):
    """
    Append `<s>` and `</s>` to the end of a pHMM phoneme vocab pickle so it can be
    used to train a phoneme LM with explicit sentence boundaries while staying
    index-compatible with the AM's label inventory at positions 0..N-1.
    """

    def __init__(self, vocab: tk.Path, bos_token: str = "<s>", eos_token: str = "</s>"):
        super().__init__()
        self.vocab = vocab
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.out_vocab = self.output_path("vocab.pkl")
        self.out_vocab_size = self.output_var("vocab_size")
        self.out_bos_index = self.output_var("bos_index")
        self.out_eos_index = self.output_var("eos_index")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.vocab.get_path(), "rb") as f:
            vocab = pickle.load(f)
        assert self.bos_token not in vocab and self.eos_token not in vocab
        bos_index = len(vocab)
        eos_index = bos_index + 1
        vocab[self.bos_token] = bos_index
        vocab[self.eos_token] = eos_index
        with uopen(self.out_vocab.get_path(), "wb") as f:
            pickle.dump(vocab, f)
        self.out_vocab_size.set(len(vocab))
        self.out_bos_index.set(bos_index)
        self.out_eos_index.set(eos_index)


def get_phmm_eow_lm_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream for LM training: pHMM phoneme inventory
    extended with `<s>` and `</s>` appended at the end.
    """
    base = get_phmm_eow_vocab_datastream(prefix=prefix, g2p_librispeech_key=g2p_librispeech_key)
    extend = ExtendPhmmVocabWithBosEosJob(vocab=base.vocab)
    extend.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "phmm_eow_lm_vocab_job"))
    return LabelDatastream(
        available_for_inference=True,
        vocab=extend.out_vocab,
        vocab_size=extend.out_vocab_size,
    )


def get_phmm_phon_lm_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    Non-EOW phoneme LabelDatastream for LM training: plain pHMM phoneme inventory
    extended with `<s>` and `</s>` appended at the end.
    """
    base = get_phmm_phon_vocab_datastream(prefix=prefix, g2p_librispeech_key=g2p_librispeech_key)
    extend = ExtendPhmmVocabWithBosEosJob(vocab=base.vocab)
    extend.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "phmm_phon_lm_vocab_job"))
    return LabelDatastream(
        available_for_inference=True,
        vocab=extend.out_vocab,
        vocab_size=extend.out_vocab_size,
    )


def build_eow_phon_phmm_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """
    Build training datasets for phoneme PHMM training with raw text labels.

    Uses ORIGINAL corpus OGG zips (English word orth), not the phoneme-converted ones.
    The FSA builder looks up words in the PHMM phoneme lexicon at training time.

    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    """
    from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
    from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

    label_datastream = get_phmm_eow_vocab_datastream(
        prefix=prefix, g2p_librispeech_key=librispeech_key
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


def build_phon_phmm_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """
    Non-EOW counterpart of :func:`build_eow_phon_phmm_training_datasets`.

    Identical pipeline (ORIGINAL corpus OGG zips with raw word orth; the FSA builder looks up
    words in the PHMM phoneme lexicon at training time) but the label datastream comes from the
    non-EOW PHMM lexicon (plain monophones), so the reported ``vocab_size`` matches the non-EOW
    FSA / AM inventory.

    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    """
    from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
    from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

    label_datastream = get_phmm_phon_vocab_datastream(
        prefix=prefix, g2p_librispeech_key=librispeech_key
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


def build_eow_phon_ctc_fsa_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """
    Build training datasets for phoneme CTC training via the RASR FSA (``fbw2`` full-sum) loss.

    Identical pipeline to :func:`build_eow_phon_phmm_training_datasets` (raw orthography targets;
    the FSA builder looks up words in the phoneme lexicon at training time), but the label
    datastream comes from the CTC lexicon (``[BLANK]`` at index 0) so the reported ``vocab_size``
    matches the CTC FSA / AM inventory.

    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    """
    from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
    from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

    label_datastream = get_ctc_eow_vocab_datastream(
        prefix=prefix, g2p_librispeech_key=librispeech_key
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
