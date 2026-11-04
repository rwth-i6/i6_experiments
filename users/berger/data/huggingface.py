"""
Data preparation for externally trained HuggingFace CTC models.

Such models bring their own subword inventory, so neither the corpus BPE codes nor the corpus BPE
lexicon apply. The jobs here derive the RASR search inventories directly from the tokenizer shipped
with the HuggingFace repository:

* :class:`HuggingFaceCtcVocabJob` writes the label inventory for lexicon-free search.
* :class:`HuggingFaceBpeBlissLexiconJob` writes a Bliss lexicon for tree search, mapping the words of
  a corpus lexicon onto tokenizer pieces.

Both use the same label order, namely the tokenizer id order with the CTC blank taking over one id
(index 0 for the Granite CTC models), so the RASR label indices line up with the model outputs.
"""

__all__ = [
    "DownloadHuggingFaceRepoJob",
    "HuggingFaceBpeBlissLexiconJob",
    "HuggingFaceCtcVocabJob",
    "get_hf_bpe_bliss_lexicon",
    "get_hf_ctc_vocab_file",
    "get_huggingface_repo_dir",
]

import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.util import uopen, write_xml
from sisyphus import Job, Task, tk

# SentencePiece-style word boundary marker. The first piece of every word carries it, which is what
# makes the decoded token sequence split into words again.
WORD_BOUNDARY_MARKER = "▁"

# Casing variants a word is tokenized in. The acoustic model was trained on true-cased text, so the
# uppercase corpus orthography alone would tokenize into pieces the model never predicts.
DEFAULT_CASINGS: Tuple[str, ...] = ("lower", "capitalize", "upper")


_RASR_LABEL_REPLACEMENTS = {".": "_dot_", "{": "_lbrace_", "}": "_rbrace_", "+": "_plus_"}

# Candidate acoustic spellings for normalized model output. An expansion is
# only attached to the original lemma if its exact token path is not already
# covered by ordinary pronunciations (e.g. DO -> NOT instead of DON'T).
_TEXT_NORMALIZATION_EXPANSIONS = {
    "I'M": "I AM",
    "I'LL": "I WILL",
    "I'D": "I WOULD",
    "I'VE": "I HAVE",
    "YOU'RE": "YOU ARE",
    "YOU'LL": "YOU WILL",
    "YOU'D": "YOU WOULD",
    "YOU'VE": "YOU HAVE",
    "HE'S": "HE IS",
    "SHE'S": "SHE IS",
    "IT'S": "IT IS",
    "THAT'S": "THAT IS",
    "THERE'S": "THERE IS",
    "WE'RE": "WE ARE",
    "WE'LL": "WE WILL",
    "WE'D": "WE WOULD",
    "WE'VE": "WE HAVE",
    "THEY'RE": "THEY ARE",
    "THEY'LL": "THEY WILL",
    "THEY'D": "THEY WOULD",
    "THEY'VE": "THEY HAVE",
    "DON'T": "DO NOT",
    "DOESN'T": "DOES NOT",
    "DIDN'T": "DID NOT",
    "ISN'T": "IS NOT",
    "AREN'T": "ARE NOT",
    "WASN'T": "WAS NOT",
    "WEREN'T": "WERE NOT",
    "HAVEN'T": "HAVE NOT",
    "HASN'T": "HAS NOT",
    "HADN'T": "HAD NOT",
    "WON'T": "WILL NOT",
    "WOULDN'T": "WOULD NOT",
    "CAN'T": "CAN NOT",
    "COULDN'T": "COULD NOT",
    "SHOULDN'T": "SHOULD NOT",
    "MUSTN'T": "MUST NOT",
    "SHAN'T": "SHALL NOT",
}


def _text_normalization_variants(text: str) -> List[str]:
    """Return candidate alternatives, subject to exact token-path coverage checks."""

    variants: List[str] = []
    expansion = _TEXT_NORMALIZATION_EXPANSIONS.get(text.upper())
    if expansion is not None:
        if text.islower():
            expansion = expansion.lower()
        elif text.istitle():
            expansion = expansion.title()
        variants.append(expansion)

    # Consider this fallback only when the ordinary WORD -> IS path is absent;
    # do not give that existing path a competing possessive interpretation.
    if re.search(r"['’]S$", text, flags=re.IGNORECASE):
        variants.append(re.sub(r"['’]S$", " IS", text, flags=re.IGNORECASE))

    # A few punctuation-attached forms (e.g. O'CLOCK) are emitted with the
    # apostrophe as a word separator.
    if "'" in text and not variants:
        variants.append(text.replace("'", " "))
    return variants


def _add_uncovered_normalization_aliases(
    lexicon: Lexicon, candidates: List[Tuple[Lemma, List[str]]]
) -> Tuple[int, int]:
    """Keep aliases only when no concatenation of ordinary pronunciations covers them.

    Compare complete label sequences, not just word membership: tokenization can
    differ across word boundaries. Use a fixed inventory without any candidate
    aliases so removal cannot depend on lemma order or on another removed alias.
    Casing/numeric pronunciations and empty-syntactic punctuation remain intact.
    Returns the numbers of omitted and retained candidate pronunciations.
    """
    ordinary = {
        tuple(pronunciation.split())
        for lemma in lexicon.lemmata
        if lemma.special is None
        for pronunciation in lemma.phon
        if pronunciation.strip()
    }
    lengths = sorted({len(pronunciation) for pronunciation in ordinary})

    def covered(tokens: Tuple[str, ...]) -> bool:
        reachable = [False] * (len(tokens) + 1)
        reachable[0] = True
        for start in range(len(tokens)):
            if not reachable[start]:
                continue
            for length in lengths:
                end = start + length
                if end > len(tokens):
                    break
                if tokens[start:end] in ordinary:
                    reachable[end] = True
            if reachable[-1]:
                return True
        return False

    omitted = retained = 0
    for lemma, pronunciations in candidates:
        for pronunciation in pronunciations:
            if pronunciation in lemma.phon:
                continue
            if covered(tuple(pronunciation.split())):
                omitted += 1
            else:
                lemma.phon.append(pronunciation)
                retained += 1
    return omitted, retained


def _sanitize_rasr_label(label: str) -> str:
    """Escape characters which RASR rejects in phoneme/label symbols."""

    return "".join(_RASR_LABEL_REPLACEMENTS.get(char, char) for char in label)


def _get_rasr_label_mapping(labels: Sequence[str]) -> Dict[str, str]:
    """Return a collision-checked mapping from tokenizer labels to RASR labels."""

    mapping: Dict[str, str] = {}
    for label in labels:
        rasr_label = _sanitize_rasr_label(label)
        previous = mapping.get(rasr_label)
        if previous is not None and previous != label:
            raise ValueError(
                f"Tokenizer labels {previous!r} and {label!r} map to the same RASR label {rasr_label!r}"
            )
        mapping[rasr_label] = label
    return mapping


class DownloadHuggingFaceRepoJob(Job):
    """
    Materialize a HuggingFace repository as a plain directory usable with ``from_pretrained``.

    Unlike ``DownloadHuggingFaceRepoJob`` in ``users/zeyer/external_models``, this downloads into the
    job output directly instead of into a hub cache, so no snapshot resolution is needed afterwards.

    Token auth follows the standard HuggingFace mechanisms, i.e. the ``HF_TOKEN`` env var or the token
    stored in the HuggingFace home dir. Gated or license-restricted repositories need such a token.
    """

    def __init__(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        :param repo_id: e.g. "ibm-granite/granite-speech-5.0-470m-turboctc-nc"
        :param revision: optional git revision, branch or tag. Pin it for reproducibility.
        :param allow_patterns: optional glob patterns limiting which files are downloaded.
        """
        self.repo_id = repo_id
        self.revision = revision
        self.allow_patterns = allow_patterns

        self.out_repo_dir = self.output_path("repo", directory=True)

        self.rqmt = {"cpu": 2, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.revision,
            allow_patterns=self.allow_patterns,
            local_dir=self.out_repo_dir.get_path(),
        )


@lru_cache
def get_huggingface_repo_dir(repo_id: str, revision: Optional[str] = None) -> tk.Path:
    job = DownloadHuggingFaceRepoJob(repo_id=repo_id, revision=revision)
    job.add_alias(f"datasets/huggingface/{repo_id.replace('/', '_')}")
    return job.out_repo_dir


def _load_tokenizer(huggingface_repo_dir: tk.Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(huggingface_repo_dir.get_path())


def _get_model_vocab_size(huggingface_repo_dir: tk.Path) -> Optional[int]:
    """Vocabulary size declared by the model config, i.e. the number of CTC output labels."""

    config_file = os.path.join(huggingface_repo_dir.get_path(), "config.json")
    if not os.path.exists(config_file):
        return None
    with uopen(config_file, "rt") as file:
        config = json.load(file)
    vocab_size = config.get("vocab_size")
    return int(vocab_size) if vocab_size is not None else None


def _get_labels(huggingface_repo_dir: tk.Path, blank_index: int, blank_label: str) -> List[str]:
    """
    Label inventory in model output order.

    The tokenizer ids map one-to-one onto the model outputs, with the blank taking over one of them
    (the ``<unk>`` id for the Granite CTC models). Ids without a token are filled with placeholders so
    that the list stays dense and the indices keep matching the model outputs.
    """

    tokenizer = _load_tokenizer(huggingface_repo_dir)
    vocab: Dict[str, int] = tokenizer.get_vocab()
    if not vocab:
        raise ValueError(f"Tokenizer in {huggingface_repo_dir.get_path()} has an empty vocabulary")

    vocab_size = _get_model_vocab_size(huggingface_repo_dir) or (max(vocab.values()) + 1)
    if not 0 <= blank_index < vocab_size:
        raise ValueError(f"Blank index {blank_index} is outside the label inventory of size {vocab_size}")

    labels: List[str] = [f"<unused-{index}>" for index in range(vocab_size)]
    for token, token_id in vocab.items():
        if token_id < vocab_size:
            labels[token_id] = _sanitize_rasr_label(token)

    # The blank shares its id with a tokenizer token that the model never emits as text.
    labels[blank_index] = blank_label

    # Fail early rather than producing an ambiguous label-to-token mapping.
    _get_rasr_label_mapping(vocab.keys())

    if len(set(labels)) != len(labels):
        raise ValueError("Label inventory contains duplicates, cannot build a consistent RASR inventory")

    return labels


class HuggingFaceCtcVocabJob(Job):
    """Write the label inventory of a HuggingFace CTC model for lexicon-free RASR search."""

    def __init__(
        self,
        huggingface_repo_dir: tk.Path,
        blank_index: int = 0,
        blank_label: str = "<blank>",
    ) -> None:
        self.huggingface_repo_dir = huggingface_repo_dir
        self.blank_index = blank_index
        self.blank_label = blank_label

        self.out_vocab_file = self.output_path("vocab.txt")
        self.out_vocab_size = self.output_var("vocab_size")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        labels = _get_labels(
            huggingface_repo_dir=self.huggingface_repo_dir,
            blank_index=self.blank_index,
            blank_label=self.blank_label,
        )

        with uopen(self.out_vocab_file, "wt") as file:
            for label in labels:
                if any(char.isspace() for char in label):
                    raise ValueError(f"Label {label!r} contains whitespace and cannot be written as vocab text")
                file.write(f"{label}\n")

        self.out_vocab_size.set(len(labels))


def get_hf_ctc_vocab_file(
    huggingface_repo_dir: tk.Path,
    blank_index: int = 0,
) -> tk.Path:
    return HuggingFaceCtcVocabJob(
        huggingface_repo_dir=huggingface_repo_dir,
        blank_index=blank_index,
    ).out_vocab_file


class HuggingFaceBpeBlissLexiconJob(Job):
    """
    Build a Bliss lexicon whose acoustic units are the subword pieces of a HuggingFace tokenizer.

    The lemma orthographies are taken over unchanged from ``bliss_lexicon``, so scoring and the word
    level LM keep operating on the corpus vocabulary. Only the pronunciations change: every word is
    tokenized in several casing variants, because the tokenizer was trained on true-cased text while
    the corpus orthography is uppercase.

    Expanded contractions/possessives and apostrophe-spacing aliases are added
    only if ordinary lemma paths cannot already represent their token sequence.
    Thus DO NOT uses the word-LM tokens DO -> NOT, not a second DON'T lemma path.
    Scoring normalization handles equivalent output spellings after decoding.

    Words that cannot be represented (unknown pieces, or pieces outside the model label inventory) are
    dropped and reported in ``out_report``. Since that silently shrinks the search vocabulary, the job
    fails when the coverage drops below ``min_coverage``.
    """

    # Invalidate previously built lexica (and dependent searches/tree caches).
    __sis_version__ = 2

    def __init__(
        self,
        bliss_lexicon: tk.Path,
        huggingface_repo_dir: tk.Path,
        blank_index: int = 0,
        blank_label: str = "<blank>",
        casings: Sequence[str] = DEFAULT_CASINGS,
        add_unknown_lemma: bool = False,
        min_coverage: float = 0.95,
    ) -> None:
        """
        :param bliss_lexicon: corpus lexicon providing the word orthographies.
        :param huggingface_repo_dir: repository providing the tokenizer.
        :param blank_index: model output index of the CTC blank.
        :param casings: casing variants each word is tokenized in, one pronunciation per variant.
        :param add_unknown_lemma: add an unknown lemma. It gets no pronunciation, so it is never
            emitted acoustically and only serves as a mapping target for the LM unknown token.
        :param min_coverage: minimum fraction of words that must receive a pronunciation.
        """
        for casing in casings:
            if casing not in {"lower", "capitalize", "upper", "none"}:
                raise ValueError(f"Unsupported casing: {casing!r}")

        self.bliss_lexicon = bliss_lexicon
        self.huggingface_repo_dir = huggingface_repo_dir
        self.blank_index = blank_index
        self.blank_label = blank_label
        self.casings = tuple(casings)
        self.add_unknown_lemma = add_unknown_lemma
        self.min_coverage = min_coverage

        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("lexicon_report.json")
        self.out_skipped_words = self.output_path("skipped_words.txt")
        self.out_num_lemmata = self.output_var("num_lemmata")
        self.out_coverage = self.output_var("coverage")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def _apply_casing(word: str, casing: str) -> str:
        if casing == "lower":
            return word.lower()
        if casing == "capitalize":
            return word.capitalize()
        if casing == "upper":
            return word.upper()
        return word

    def run(self) -> None:
        # Granite was trained with text normalization which may spell a number as
        # digits/ordinals (e.g. ``20`` or ``1ST``), while the Bliss lexicon uses
        # words (``TWENTY``/``FIRST``). Add those tokenizations as pronunciation
        # alternatives without changing the lemma orthography or LM symbol.
        numeric_aliases = {
            **dict(zip(
                "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety".split(),
                [str(i) for i in range(0, 21)] + ["30", "40", "50", "60", "70", "80", "90"],
            )),
            **dict(zip(
                "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(),
                [f"{i}{'th' if i % 100 in (11, 12, 13) else {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')}" for i in range(1, 21)],
            )),
        }

        tokenizer = _load_tokenizer(self.huggingface_repo_dir)
        vocab: Dict[str, int] = tokenizer.get_vocab()
        labels = _get_labels(
            huggingface_repo_dir=self.huggingface_repo_dir,
            blank_index=self.blank_index,
            blank_label=self.blank_label,
        )
        vocab_size = len(labels)
        unk_token = getattr(tokenizer, "unk_token", None)

        def tokenize(text: str) -> Optional[List[str]]:
            """Tokenize a single word into pieces usable as a pronunciation, or None."""

            # The word is tokenized on its own, without a leading space: the tokenizer prepends the
            # word boundary marker itself, whereas an explicit space would additionally produce a
            # standalone marker piece. Tokenizing this way reproduces exactly the pieces the word
            # gets inside a sentence.
            tokens = tokenizer.tokenize(text)
            if not tokens:
                return None
            if unk_token is not None and unk_token in tokens:
                return None
            if not tokens[0].startswith(WORD_BOUNDARY_MARKER):
                # Without the word boundary marker the decoded hypothesis would glue this word onto
                # the previous one.
                return None
            for token in tokens:
                token_id = vocab.get(token)
                if token_id is None or token_id >= vocab_size or token_id == self.blank_index:
                    return None
            if tokenizer.convert_tokens_to_string(tokens).strip() != text:
                # The pieces do not reproduce the word, so this pronunciation would decode into
                # something else than the lemma it is attached to.
                return None
            return tokens

        base_lexicon = Lexicon()
        base_lexicon.load(self.bliss_lexicon.get_path())

        lexicon = Lexicon()
        for label in labels:
            lexicon.add_phoneme(label, variation="none")

        num_words = 0
        word_lemmas: List[Lemma] = []
        normalization_aliases: List[Tuple[Lemma, List[str]]] = []

        for lemma in base_lexicon.lemmata:
            if lemma.special is not None:
                continue
            orths = [orth for orth in lemma.orth if orth]
            if not orths:
                continue
            num_words += 1

            pronunciations: List[str] = []
            alias_pronunciations: List[str] = []
            for orth in orths:
                for casing in self.casings:
                    text = self._apply_casing(orth, casing)
                    texts = [text]
                    alias = numeric_aliases.get(text.lower())
                    if alias is not None:
                        texts.append(alias)
                    for token_text in texts:
                        tokens = tokenize(token_text)
                        if tokens is None:
                            continue
                        pronunciation = " ".join(labels[vocab[token]] for token in tokens)
                        if pronunciation not in pronunciations:
                            pronunciations.append(pronunciation)
                    for token_text in _text_normalization_variants(text):
                        tokens = tokenize(token_text)
                        if tokens is None:
                            continue
                        pronunciation = " ".join(labels[vocab[token]] for token in tokens)
                        if pronunciation not in alias_pronunciations:
                            alias_pronunciations.append(pronunciation)

            word_lemma = Lemma(orth=orths, phon=pronunciations)
            lexicon.add_lemma(word_lemma)
            word_lemmas.append(word_lemma)
            normalization_aliases.append((word_lemma, alias_pronunciations))

        # Granite can emit punctuation as standalone tokenizer pieces (for
        # example ``.`` between ``▁2`` and ``5``). These pieces are removed by
        # the scoring formatter and must therefore not consume a word-LM token.
        # Add them as empty-syntactic lemmas so the tree can traverse them.
        existing_orths = {orth for lemma in lexicon.lemmata for orth in lemma.orth}
        punctuation_pronunciations: Dict[str, List[str]] = {}
        for token, token_id in vocab.items():
            piece = token[len(WORD_BOUNDARY_MARKER) :] if token.startswith(WORD_BOUNDARY_MARKER) else token
            if not piece:
                continue
            if not all(unicodedata.category(char).startswith("P") for char in piece):
                continue
            if token_id >= vocab_size or token_id == self.blank_index:
                continue
            punctuation_pronunciations.setdefault(piece, []).append(labels[token_id])
        for piece, pronunciations in punctuation_pronunciations.items():
            if piece not in existing_orths:
                lexicon.add_lemma(Lemma(orth=[piece], phon=sorted(set(pronunciations)), synt=[]))
                existing_orths.add(piece)

        omitted_aliases, retained_aliases = _add_uncovered_normalization_aliases(lexicon, normalization_aliases)
        skipped_words = [lemma.orth[0] for lemma in word_lemmas if not lemma.phon]
        num_covered_words = sum(bool(lemma.phon) for lemma in word_lemmas)
        num_pronunciations = sum(len(lemma.phon) for lemma in word_lemmas)
        lexicon.lemmata = [lemma for lemma in lexicon.lemmata if lemma.phon]

        # The model has no sentence boundary labels, so these lemmas exist only to anchor the LM
        # sentence boundary tokens and are never emitted acoustically.
        lexicon.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=[""], synt=["<s>"], special="sentence-begin"))
        lexicon.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=[""], synt=["</s>"], special="sentence-end"))
        if self.add_unknown_lemma:
            # No pronunciation: the model has no label for out-of-lexicon words, so the tree search is
            # closed vocabulary and this lemma only anchors the LM unknown token.
            lexicon.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=[""], synt=["<UNK>"], special="unknown"))
        lexicon.add_lemma(Lemma(orth=["[BLANK]"], phon=[self.blank_label], synt=[], special="blank"))

        write_xml(self.out_lexicon.get_path(), lexicon.to_xml())

        with uopen(self.out_skipped_words, "wt") as file:
            for word in sorted(skipped_words):
                file.write(f"{word}\n")

        coverage = num_covered_words / num_words if num_words else 0.0
        report = {
            "num_words": num_words,
            "num_covered_words": num_covered_words,
            "num_skipped_words": len(skipped_words),
            "coverage": coverage,
            "num_pronunciations": num_pronunciations,
            "pronunciations_per_word": num_pronunciations / num_covered_words if num_covered_words else 0.0,
            "vocab_size": vocab_size,
            "blank_index": self.blank_index,
            "casings": list(self.casings),
            "num_omitted_normalization_aliases": omitted_aliases,
            "num_retained_normalization_aliases": retained_aliases,
        }
        with uopen(self.out_report, "wt") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")

        self.out_num_lemmata.set(num_covered_words)
        self.out_coverage.set(coverage)

        if coverage < self.min_coverage:
            raise ValueError(
                f"Only {coverage:.4f} of the lexicon words could be tokenized, expected at least "
                f"{self.min_coverage}. This usually means the tokenization does not match the "
                f"expected subword conventions, see {self.out_skipped_words.get_path()}."
            )


def get_hf_bpe_bliss_lexicon(
    bliss_lexicon: tk.Path,
    huggingface_repo_dir: tk.Path,
    blank_index: int = 0,
    casings: Sequence[str] = DEFAULT_CASINGS,
) -> tk.Path:
    return HuggingFaceBpeBlissLexiconJob(
        bliss_lexicon=bliss_lexicon,
        huggingface_repo_dir=huggingface_repo_dir,
        blank_index=blank_index,
        casings=casings,
    ).out_lexicon
