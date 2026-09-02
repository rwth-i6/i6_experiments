from typing import Optional

from sisyphus import Path, tk

from i6_experiments.users.schmitt.datasets.utils.phonemize import (
    PhonemizeTextDataJob,
    DumpPhonemeIndicesToHdfJob,
    ExtendLexiconWithG2PJob,
)

from ..default_tools import (
    get_fairseq_root,
    get_lid_model,
    get_fasttext_python_exe,
    get_g2p_python_exe,
    get_nltk_data,
)


def get_phonemized_data(
    dataset_name: str,
    corpus_name: str,
    text_file: Path,
    dump_hdf_concurrent: int,
    fixed_random_subset: Optional[int] = None,
    lexicon_file: Optional[Path] = None,
    language: str = "en",
    sil_prob: float = 0.25,
    seq_tag_file: Optional[Path] = None,
    vocab_file: Optional[Path] = None,
    surround_w_sil: bool = True,
    apply_lid_filter: bool = True,
    extend_lexicon_w_g2p: bool = False,
):
    """
    :param apply_lid_filter: see :class:`PhonemizeTextDataJob`. Drops lines that fasttext does not confidently
        classify as ``language`` (5 of 2864 on dev-other).
    :param extend_lexicon_w_g2p: G2P the words of ``text_file`` that are missing from ``lexicon_file`` and
        phonemize against the extended lexicon, instead of dropping every line containing such a word
        (147 of 2864 on dev-other). See :class:`ExtendLexiconWithG2PJob`.

    Both default to the historical behavior. Set both for eval sets, where dropping sequences makes the
    reported WER/PER incomparable; do NOT set them for training text, which would rehash every training.
    """
    if extend_lexicon_w_g2p:
        assert lexicon_file is not None, (
            "extend_lexicon_w_g2p needs an existing lexicon to extend"
            " (with lexicon_file=None the lexicon is built from this very text by G2P anyway)"
        )
        lexicon_file = ExtendLexiconWithG2PJob(
            text_file=text_file,
            lexicon_file=lexicon_file,
            python_exe=get_g2p_python_exe(),
            nltk_data=get_nltk_data(),
        ).out_lexicon_file

    prepare_text_job_training = PhonemizeTextDataJob(
        text_file=text_file,
        fairseq_root=get_fairseq_root(),
        python_exe=get_fasttext_python_exe(),
        lid_path=get_lid_model(),
        language=language,
        sil_prob=sil_prob,
        lexicon_file=lexicon_file,
        seq_tag_file=seq_tag_file,
        min_phoneme_occurrence=1000,
        surround_w_sil=surround_w_sil,
        apply_lid_filter=apply_lid_filter,
    )
    text_file = prepare_text_job_training.out_phoneme_text
    # distinct output name per variant, otherwise the filtered and the full-coverage version of the same
    # dataset would register the same output path (register_output is not hashed, so this is free)
    out_name_suffix = ("" if apply_lid_filter else ".no_lid") + (".g2p_ext" if extend_lexicon_w_g2p else "")
    tk.register_output(f"data/{corpus_name}/text/phonemized/{dataset_name}{out_name_suffix}.txt", text_file)
    lexicon_file = prepare_text_job_training.out_lexicon_file
    vocab_file = prepare_text_job_training.out_phoneme_vocab if vocab_file is None else vocab_file
    seq_tags_after_phonemize = prepare_text_job_training.out_seq_tags

    dump_phoneme_indices_job = DumpPhonemeIndicesToHdfJob(
        text_file=text_file,
        phoneme_vocab=vocab_file,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        seq_tag_file=seq_tags_after_phonemize,
    )

    return (
        list(dump_phoneme_indices_job.out_hdfs.values()),
        vocab_file,
        lexicon_file,
        seq_tags_after_phonemize,
    )
