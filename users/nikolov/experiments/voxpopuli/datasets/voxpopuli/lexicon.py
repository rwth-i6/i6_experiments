__all__ = ["get_bpe_lexicon"]
import os
from typing import Optional
from sisyphus import setup_path,gs,tk

import i6_core.returnn as returnn
from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.bpe.train import ReturnnTrainBpeJob
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
import i6_core.lib.lexicon as lexicon


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
    #lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE-BEGIN]"], synt=["<s>"], special="sentence-begin"))
    #lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE-END]"], synt=["</s>"], special="sentence-end"))
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
    subdir_prefix: str = "vox_lex") -> tk.Path:
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

    mapped_raw_lexicon_file = raw_lexicon_path

    bliss_lexicon = LexiconFromTextFileJob(mapped_raw_lexicon_file, compressed=True)
    bliss_lexicon.add_alias(os.path.join(subdir_prefix, "create_lexicon_job"))

    bpe_train_job = ReturnnTrainBpeJob(text_file=full_text_path, bpe_size=bpe_size, subword_nmt_repo=SUBWORD_NMT_REPO)

    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=bliss_lexicon.out_bliss_lexicon,
        bpe_codes=bpe_train_job.out_bpe_codes,
        bpe_vocab=bpe_train_job.out_bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label=bpe_train_job.unk_label,
    )


    bpe_lexicon.add_alias(os.path.join(subdir_prefix, "create_bpe_lexicon_job"))

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            bpe_lexicon.out_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
    )
    merge_lexicon_job.add_alias(os.path.join(subdir_prefix, "merge_lexicon_job"))
    tk.register_output(
                            f"voxpopuli_asr/lexicon",
                            merge_lexicon_job.out_bliss_lexicon,
    ) 

    return merge_lexicon_job.out_bliss_lexicon