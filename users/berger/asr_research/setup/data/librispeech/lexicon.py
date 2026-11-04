from typing import Optional

from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob, MergeLexiconJob, WriteLexiconJob
from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.util import write_xml
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from sisyphus import Job, Task, tk

from ...tools import subword_nmt_repo
from ..base import RemovePhonemeVariationsFromLexiconJob, RemoveSpecialLemmasFromLexiconJob


class AddBpeOovFallbackLemmasJob(Job):
    """Add the per-piece special lemmata used by RASR's open-vocabulary BPE subtree."""

    def __init__(self, bliss_lexicon: tk.Path, unknown_lm_token: str = "<UNK>") -> None:
        self.bliss_lexicon = bliss_lexicon
        self.unknown_lm_token = unknown_lm_token
        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())

        # Special acoustic labels such as blank, sentence-end, and the legacy
        # unknown label are not BPE pieces and must not enter the OOV subtree.
        special_pronunciation_labels = {
            label
            for lemma in lexicon.lemmata
            if lemma.special is not None
            for pronunciation in lemma.phon
            for label in pronunciation.split()
        }
        bpe_labels = [
            label
            for label in lexicon.phonemes
            if label not in special_pronunciation_labels
            and not (label.startswith("<") and label.endswith(">"))
        ]
        continuation_labels = [label for label in bpe_labels if label.endswith("@@")]
        final_labels = [label for label in bpe_labels if not label.endswith("@@")]
        if not continuation_labels or not final_labels:
            raise ValueError(
                "Cannot construct BPE OOV fallback without both continuation and word-final labels "
                f"(found {len(continuation_labels)} continuation and {len(final_labels)} final labels)"
            )

        # Make the operation idempotent in case it is applied to an already
        # decorated lexicon. The conventional singleton unknown lemma remains
        # available to Bliss's orthographic parser, but RASR excludes it from
        # the tree whenever the multi-lemma fallback is present.
        lexicon.lemmata = [
            lemma for lemma in lexicon.lemmata if lemma.special not in {"unknown-continuation", "unknown-final"}
        ]
        for label in continuation_labels:
            lexicon.add_lemma(
                Lemma(
                    orth=[label],
                    phon=[label],
                    synt=[],
                    special="unknown-continuation",
                )
            )
        for label in final_labels:
            lexicon.add_lemma(
                Lemma(
                    # Keep a visible marker in traceback output so that an
                    # unknown-word-final path can be distinguished from the
                    # ordinary lemma with the same BPE orthography.  The
                    # marker is removed by the traceback formatters before
                    # producing transcriptions/CTMs.
                    orth=[f"{label}&&"],
                    phon=[label],
                    synt=[self.unknown_lm_token],
                    special="unknown-final",
                )
            )

        write_xml(self.out_lexicon.get_path(), lexicon.to_xml())


def get_bpe_bliss_lexicon(
    bpe_size: int,
    add_blank: bool,
    add_sentence_end_pron: bool,
    base_lexicon_path: Optional[tk.Path] = None,
    add_oov_fallback: bool = False,
) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    lexicon = base_lexicon_path if base_lexicon_path is not None else get_bliss_lexicon()
    bpe_lexicon_file = CreateBPELexiconJob(
        base_lexicon_path=lexicon,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        unk_label="<unk>",
        vocab_blacklist=["</s>"],
        subword_nmt_repo=subword_nmt_repo,
        keep_special_lemmas=False,
    ).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=[""], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(
        Lemma(
            orth=["[SENTENCE-END]"],
            phon=["<s>" if add_sentence_end_pron else ""],
            synt=["</s>"],
            special="sentence-end",
        )
    )
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<UNK>"], special="unknown"))

    if add_blank:
        lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], special="blank"))
        lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    merged_lexicon = MergeLexiconJob([bpe_lexicon_file, lexicon_ext_file]).out_bliss_lexicon
    lexicon_file = RemovePhonemeVariationsFromLexiconJob(merged_lexicon).out_lexicon
    if add_oov_fallback:
        lexicon_file = AddBpeOovFallbackLemmasJob(lexicon_file).out_lexicon
    return lexicon_file


def get_bliss_phoneme_lexicon() -> tk.Path:
    lexicon_file = get_bliss_lexicon(add_silence=False, add_unknown_phoneme_and_mapping=False)
    cleaned_lexicon_file = RemoveSpecialLemmasFromLexiconJob(lexicon_file).out_lexicon
    eow_lexicon_file = AddEowPhonemesToLexiconJob(bliss_lexicon=cleaned_lexicon_file).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_phoneme("<unk>", variation="none")
    lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=[""], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=[""], synt=["</s>"], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<UNK>"], special="unknown"))
    lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], special="blank"))

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    merged_lexicon = MergeLexiconJob([eow_lexicon_file, lexicon_ext_file]).out_bliss_lexicon
    return RemovePhonemeVariationsFromLexiconJob(merged_lexicon).out_lexicon
