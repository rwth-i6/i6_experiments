__all__ = ["get_bpe_lexicon"]

import os
from typing import Optional
from sisyphus import setup_path, tk

import i6_core.returnn as returnn
import i6_core.lib.lexicon as lexicon
import i6_core.util as util

from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.bpe.train import ReturnnTrainBpeJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()


def get_special_lemma_lexicon(
        add_unknown_phoneme_and_mapping: bool = True, add_silence: bool = True
) -> lexicon.Lexicon:
    """
    Generate the special lemmas for Voxpopuli

    Voxpopuli uses silence, sentence begin/end and unknown, but no other special tokens.

    :param bool add_unknown_phoneme_and_mapping: add [UNKNOWN] as phoneme, otherwise add only the lemma without it
    :param bool add_silence: add [SILENCE] phoneme and lemma, otherwise do not include it at all (e.g. for CTC setups)
    :return: the lexicon with special lemmas and phonemes
    :rtype: lexicon.Lexicon
    """
    lex = lexicon.Lexicon()
    if add_silence:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SILENCE]", ""],
                phon=["[SILENCE]"],
                synt=[],
                special="silence",
                eval=[[]],
            )
        )
    # lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE-BEGIN]"], synt=["<s>"], special="sentence-begin"))
    # lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE-END]"], synt=["</s>"], special="sentence-end"))
    if add_unknown_phoneme_and_mapping:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[UNKNOWN]"],
                phon=["[UNKNOWN]"],
                synt=["<UNK>"],
                special="unknown",
            )
        )
    else:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[UNKNOWN]"],
                synt=["<UNK>"],
                special="unknown",
            )
        )

    if add_silence:
        lex.add_phoneme("[SILENCE]", variation="none")
    if add_unknown_phoneme_and_mapping:
        lex.add_phoneme("[UNKNOWN]", variation="none")
    return lex


def get_bliss_lexicon(
        raw_lexicon_path: tk.Path,
        full_text_path: tk.Path,
        bpe_size: int,
        subdir_prefix: str = "vox_legacy") -> tk.Path:
    """
    Creates Voxpopuli bliss xml lexicon. The special lemmas are added via `get_special_lemma_lexicon` function.

    :param subdir_prefix: alias prefix name
    :param raw_lexicon_path: tk.Path to .txt lexicon in the format  <WORD> <PHONEME1> <PHONEME2> ...
    :param bpe_size: number of bpe tokens
    :return: Path to voxpopuli bliss lexicon
    """
    static_lexicon = get_special_lemma_lexicon(add_unknown_phoneme_and_mapping=True)
    static_lexicon_job = WriteLexiconJob(static_lexicon=static_lexicon, sort_phonemes=True, sort_lemmata=False)
    static_lexicon_job.add_alias(os.path.join(subdir_prefix, "write_special_lemma_lexicon_legacy_job"))

    mapped_raw_lexicon_file = raw_lexicon_path

    bliss_lexicon = LexiconFromTextFileJob(mapped_raw_lexicon_file, compressed=True)
    bliss_lexicon.add_alias(os.path.join(subdir_prefix, "create_lexicon_legacy_job"))

    bpe_train_job = ReturnnTrainBpeJob(text_file=full_text_path, bpe_size=bpe_size,
                                       subword_nmt_repo=SUBWORD_NMT_REPO)

    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=bliss_lexicon.out_bliss_lexicon,
        bpe_codes=bpe_train_job.out_bpe_codes,
        bpe_vocab=bpe_train_job.out_bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label=bpe_train_job.unk_label,
    )

    bpe_train_job.add_alias(os.path.join(subdir_prefix, "train_bpe_legacy_job"))
    tk.register_output(
        f"voxpopuli/joint_bpe/bpe_{bpe_size}",
        bpe_train_job.out_bpe_vocab,
    )

    bpe_lexicon.add_alias(os.path.join(subdir_prefix, "create_bpe_lexicon_legacy_job"))

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            bpe_lexicon.out_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
        # keep_duplicate=False,
    )
    merge_lexicon_job.add_alias(os.path.join(subdir_prefix, "merge_lexicon_legacy_job"))
    tk.register_output(
        f"voxpopuli/joint_bpe/bpe_{bpe_size}_lexicon",
        merge_lexicon_job.out_bliss_lexicon,
    )

    return merge_lexicon_job.out_bliss_lexicon, bpe_train_job.out_bpe_vocab


def get_bliss_lang_lexicons(
        raw_lexicon_path: str,
        full_text_path: str,
        bpe_size: dict[str, int],
        add_prefix: bool,
        subdir_prefix: str) -> tk.Path:
    """
    Creates Voxpopuli bliss xml lexicon. The special lemmas are added via `get_special_lemma_lexicon` function.

    :param subdir_prefix: alias prefix name
    :param raw_lexicon_path: tk.Path to .txt lexicon in the format  <WORD> <PHONEME1> <PHONEME2> ...
    :param bpe_size: number of bpe tokens
    :return: Path to voxpopuli bliss lexicon
    """
    static_lexicon = get_special_lemma_lexicon(add_unknown_phoneme_and_mapping=True)
    static_lexicon_job = WriteLexiconJob(static_lexicon=static_lexicon, sort_phonemes=True, sort_lemmata=False)
    static_lexicon_job.add_alias(os.path.join(subdir_prefix, "write_special_lemma_lexicon_job"))

    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    bpe_lexicons = {}

    for lang in langs:
        mapped_raw_lexicon_file = tk.Path(raw_lexicon_path + f"/lex_{lang}.txt")
        mapped_text_file = tk.Path(full_text_path + f"/text_{lang}.txt")

        bliss_lexicon = LexiconFromTextFileJob(mapped_raw_lexicon_file, compressed=True)
        bliss_lexicon.add_alias(os.path.join(subdir_prefix, f"create_lexicon_{lang}_job"))

        if add_prefix:
            prefix = lang.upper() + "_"
        else:
            prefix = ""

        bpe_train_job = ReturnnTrainBpeJob(text_file=mapped_text_file, bpe_size=bpe_size['base'], prefix=prefix,
                                           subword_nmt_repo=SUBWORD_NMT_REPO)

        bpe_train_job.add_alias(os.path.join(subdir_prefix, f"train_bpe_{lang}_lexicon_job"))

        bpe_lexicons[lang] = CreateBPELexiconJob(
            base_lexicon_path=bliss_lexicon.out_bliss_lexicon,
            bpe_codes=bpe_train_job.out_bpe_codes,
            bpe_vocab=bpe_train_job.out_bpe_vocab,
            prefix=prefix,
            subword_nmt_repo=SUBWORD_NMT_REPO,
            unk_label=bpe_train_job.unk_label,
        )
        bpe_lexicons[lang].add_alias(os.path.join(subdir_prefix, f"create_bpe_{lang}_lexicon_job"))

    bpe_lexicons_list = [bpe_lexicon.out_lexicon for bpe_lexicon in bpe_lexicons.values()]

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            *[x for x in bpe_lexicons_list],
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
        # keep_duplicate=False,
    )
    merge_lexicon_job.add_alias(os.path.join(subdir_prefix, "merge_lexicon_job"))
    tk.register_output(
        f"voxpopuli_asr/lexicon_{bpe_size['base']}",
        merge_lexicon_job.out_bliss_lexicon,
    )

    return merge_lexicon_job.out_bliss_lexicon


def get_unique_bpe_bliss_lexicons(
        raw_lexicon_path: str,
        full_text_path: str,
        bpe_size: dict[str, int],
        version: str,
        subdir_prefix: str,
        add_prefix: bool = False) -> tk.Path:
    """
    Creates Voxpopuli bliss xml lexicon. The special lemmas are added via `get_special_lemma_lexicon` function.

    :param subdir_prefix: alias prefix name
    :param raw_lexicon_path: tk.Path to .txt lexicon in the format  <WORD> <PHONEME1> <PHONEME2> ...
    :param bpe_size: number of bpe tokens
    :return: Path to voxpopuli bliss lexicon
    """
    static_lexicon = get_special_lemma_lexicon(add_unknown_phoneme_and_mapping=True)
    static_lexicon_job = WriteLexiconJob(static_lexicon=static_lexicon, sort_phonemes=True, sort_lemmata=False)
    static_lexicon_job.add_alias(os.path.join(subdir_prefix, "write_special_lemma_lexicon_job"))

    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    bpe_lexicons = {}

    for lang in langs:
        mapped_raw_lexicon_file = tk.Path(raw_lexicon_path + f"/lex_{lang}.txt")
        mapped_text_file = tk.Path(full_text_path + f"/text_{lang}.txt")

        bliss_lexicon = LexiconFromTextFileJob(mapped_raw_lexicon_file, compressed=True)
        bliss_lexicon.add_alias(os.path.join(subdir_prefix, f"create_lexicon_{version}_{lang}_job"))

        if add_prefix:
            prefix = lang.upper() + "_"
        else:
            prefix = ""

        bpe_train_job = ReturnnTrainBpeJob(text_file=mapped_text_file, bpe_size=bpe_size[lang], prefix=prefix,
                                           subword_nmt_repo=SUBWORD_NMT_REPO)

        bpe_train_job.add_alias(os.path.join(subdir_prefix, f"train_{version}_{lang}_lexicon_job"))

        bpe_lexicons[lang] = CreateBPELexiconJob(
            base_lexicon_path=bliss_lexicon.out_bliss_lexicon,
            bpe_codes=bpe_train_job.out_bpe_codes,
            bpe_vocab=bpe_train_job.out_bpe_vocab,
            prefix=prefix,
            subword_nmt_repo=SUBWORD_NMT_REPO,
            unk_label=bpe_train_job.unk_label,
        )
        bpe_lexicons[lang].add_alias(os.path.join(subdir_prefix, f"create_{version}_{lang}_lexicon_job"))

    bpe_lexicons_list = [bpe_lexicon.out_lexicon for bpe_lexicon in bpe_lexicons.values()]

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            *[x for x in bpe_lexicons_list],
        ],
        sort_phonemes=False,
        sort_lemmata=False,
        compressed=True,
        # keep_duplicate=False,
    )
    merge_lexicon_job.add_alias(os.path.join(subdir_prefix, f"merge_{version}_lexicon_job"))
    tk.register_output(
        f"voxpopuli_asr/lexicon_{version}",
        merge_lexicon_job.out_bliss_lexicon,
    )

    return merge_lexicon_job.out_bliss_lexicon


def get_text_lexicon(prefix: str, bpe_size: dict[str, int], add_prefix: bool, lexicon_path: str = None) -> tk.Path:
    """
    Get a bpe lexicon in line-based text format to be used for torchaudio/Flashlight decoding

    :param prefix:
    :param bpe_size: number of BPE splits
    :return: path to a lexicon text file
    """
    if 'base' in bpe_size:
        bliss_lex = get_bliss_lang_lexicons(
            subdir_prefix=prefix,
            raw_lexicon_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            full_text_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            bpe_size=bpe_size,
            add_prefix=add_prefix)
    else:
        bliss_lex = get_unique_bpe_bliss_lexicons(
            subdir_prefix=prefix,
            raw_lexicon_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            full_text_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            version="_".join(lexicon_path.split('/')[-1].split('_')[1:]),
            bpe_size=bpe_size,
            add_prefix=add_prefix)

    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon
