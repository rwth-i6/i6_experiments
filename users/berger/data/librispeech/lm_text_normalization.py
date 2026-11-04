"""Controlled original/HF-normalized word-LM estimation for Granite recognition.

Both conditions use the official LibriSpeech LM text plus train-other-960
transcriptions, identical KenLM settings, and the standard lexical inventory.
No evaluation transcripts enter estimation. The updated HF normalizer is used,
not the historical version behind some published leaderboard scores.

The multi-token normalized lexicon is an intermediate for vocabulary extraction,
not a RASR search lexicon. BuildNormalizedWordSearchLexiconJob turns its vocabulary
into single-LM-token lemmas, optionally recovering source pronunciations through
separate word exits. Context-dependent normalization across source lemmas (e.g.
TWENTY ONE -> 21) can still produce training tokens outside this vocabulary;
training-text reports expose those OOVs.
"""

import copy
import json
from collections import Counter
from functools import lru_cache
from typing import Sequence

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.lm.kenlm import KenLMplzJob
from i6_core.util import uopen, write_xml
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from sisyphus import Job, Task, tk

from ...model_pipelines.common.hf_normalizer import EnglishTextNormalizer
from ...tools import ken_lm_binaries
from ..huggingface import DEFAULT_CASINGS, _load_tokenizer, _get_rasr_label_mapping
from .lexicon_oov import LexiconToKenLmVocabularyJob, NormalizeArpaUnknownTokenJob


class NormalizeLexiconLmTokensJob(Job):
    """Normalize syntactic sequences for LM vocabulary extraction/intermediate use.

    Uppercase is only an LM symbol convention, applied after HF normalization
    on both corpus and lexicon. Multi-token expansions and empty sequences are
    retained; special lemmas and existing empty syntactic sequences are preserved.
    Never pass this output directly to TreeTimesyncBeamSearch: it may contain
    multi-token syntactic sequences. Use BuildNormalizedWordSearchLexiconJob.
    """

    def __init__(self, lexicon: tk.Path):
        self.lexicon = lexicon
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("report.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 2})

    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.lexicon.get_path())
        normalize = EnglishTextNormalizer()
        counts = Counter()
        for lemma in lexicon.lemmata:
            if lemma.special is not None or lemma.synt == []:
                continue
            texts = [" ".join(lemma.synt)] if lemma.synt is not None else [orth for orth in lemma.orth if orth]
            alternatives = {tuple(normalize(text).upper().split()) for text in texts}
            if len(alternatives) != 1:
                raise ValueError(f"Ambiguous LM normalization for lemma {lemma.orth!r}: {alternatives!r}")
            tokens = list(alternatives.pop())
            lemma.synt = tokens
            counts["lemmas"] += 1
            counts["empty_sequences"] += not tokens
            counts["multi_token_sequences"] += len(tokens) > 1
        write_xml(self.out_lexicon.get_path(), lexicon.to_xml())
        with uopen(self.out_report, "wt") as dst:
            json.dump(dict(counts), dst, indent=2)


def _split_normalized_pronunciation(labels, words, decode, normalize):
    """Split only at real acoustic-token boundaries, with one normalized word per part.

    No proportional/arbitrary split: e.g. ``don ' t`` must not assign ``don``
    to DO and ``' t`` to NOT. Attached ``zyl ' s`` can split into ZYL and S.
    Return None if no nonempty partition has the required normalized text.
    """

    @lru_cache(None)
    def visit(start, word_index):
        if word_index == len(words):
            return () if start == len(labels) else None
        remaining = len(words) - word_index - 1
        for end in range(start + 1, len(labels) - remaining + 1):
            if tuple(normalize(decode(labels[start:end])).upper().split()) != (words[word_index],):
                continue
            rest = visit(end, word_index + 1)
            if rest is not None:
                return (tuple(labels[start:end]),) + rest
        return None

    return visit(0, 0)


def _has_normalized_word_path(labels, words, pronunciations, neutral_pronunciations):
    """Exact label/LM-token path check, including optional empty-syntactic pieces."""

    @lru_cache(None)
    def visit(start, word_index):
        if start == len(labels):
            return word_index == len(words)
        for end in range(start + 1, len(labels) + 1):
            part = tuple(labels[start:end])
            if part in neutral_pronunciations and visit(end, word_index):
                return True
            if word_index < len(words) and part in pronunciations[words[word_index]] and visit(end, word_index + 1):
                return True
        return False

    return visit(0, 0)


class BuildNormalizedWordSearchLexiconJob(Job):
    """RASR-compatible normalized vocabulary: exactly one LM token per word lemma.

    ``canonical`` uses tokenizer spellings of the normalized words. ``split``
    additionally recovers original pronunciations when they can be partitioned
    into nonempty acoustic chunks that each normalize to the corresponding word.
    Singleton variants (e.g. TWENTY for 20) are also retained in split mode.
    Both modes preserve special and empty-syntactic lemmas, and share LM vocabulary.

    Split chunks are ordinary word pronunciations, not context-constrained suffixes:
    recombination can introduce extra paths. Reports quantify added pronunciations
    and exact coverage, but do not claim identical search spaces to the original.
    """

    def __init__(
        self,
        lexicon: tk.Path,
        huggingface_repo_dir: tk.Path,
        mode: str = "canonical",
        casings: Sequence[str] = DEFAULT_CASINGS,
    ):
        if mode not in {"canonical", "split"}:
            raise ValueError(f"Unsupported normalized lexicon mode: {mode}")
        if any(casing not in {"lower", "capitalize", "upper", "none"} for casing in casings):
            raise ValueError(f"Unsupported casings: {casings}")
        self.lexicon = lexicon
        self.huggingface_repo_dir = huggingface_repo_dir
        self.mode = mode
        self.casings = tuple(casings)
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("report.json")
        self.out_uncovered = self.output_path("uncovered_pronunciations.jsonl.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 8})

    def run(self):
        source = Lexicon()
        source.load(self.lexicon.get_path())
        if any(lemma.synt is None for lemma in source.lemmata):
            raise ValueError("Expected explicit normalized syntactic sequences; run NormalizeLexiconLmTokensJob first")
        tokenizer = _load_tokenizer(self.huggingface_repo_dir)
        label_to_token = _get_rasr_label_mapping(tokenizer.get_vocab().keys())
        token_to_label = {token: label for label, token in label_to_token.items()}
        normalize = EnglishTextNormalizer()
        ordinary = [lemma for lemma in source.lemmata if lemma.special is None and lemma.synt]
        words = sorted({word for lemma in ordinary for word in lemma.synt})
        pronunciations = {word: set() for word in words}

        def decode(labels):
            return tokenizer.convert_tokens_to_string([label_to_token[label] for label in labels])

        for word in words:
            for casing in self.casings:
                text = {"lower": word.lower, "capitalize": word.capitalize, "upper": word.upper, "none": lambda: word}[
                    casing
                ]()
                tokens = tokenizer.tokenize(text)
                if not tokens or getattr(tokenizer, "unk_token", None) in tokens:
                    continue
                labels = tuple(token_to_label[token] for token in tokens)
                if any(label not in source.phonemes for label in labels):
                    continue
                if not tokens[0].startswith("▁") or tokenizer.convert_tokens_to_string(tokens).strip() != text:
                    continue
                pronunciations[word].add(labels)

        counts = Counter()
        canonical_count = sum(map(len, pronunciations.values()))
        if self.mode == "split":
            for lemma in ordinary:
                for pronunciation in lemma.phon:
                    labels = tuple(pronunciation.split())
                    if not labels:
                        raise ValueError(f"Empty acoustic pronunciation for {lemma.orth}")
                    # Singleton remappings intentionally include numeric aliases etc.
                    parts = (
                        (labels,)
                        if len(lemma.synt) == 1
                        else _split_normalized_pronunciation(labels, tuple(lemma.synt), decode, normalize)
                    )
                    if parts is None:
                        counts["unsplittable_source_pronunciations"] += 1
                        continue
                    for word, part in zip(lemma.synt, parts):
                        pronunciations[word].add(part)

        missing = [word for word, variants in pronunciations.items() if not variants]
        if missing:
            raise ValueError(f"Normalized LM words lack pronunciations: {missing[:20]!r} ({len(missing)} total)")
        output = Lexicon()
        output.phonemes = copy.deepcopy(source.phonemes)
        neutral = set()
        for lemma in source.lemmata:
            if lemma.special is not None or lemma.synt == []:
                output.add_lemma(copy.deepcopy(lemma))
                if lemma.special is None:
                    neutral.update(tuple(pron.split()) for pron in lemma.phon if pron.strip())
        for word in words:
            output.add_lemma(Lemma(orth=[word], phon=[" ".join(p) for p in sorted(pronunciations[word])], synt=[word]))

        # Validate the actual serialized-search contract, including special lemmas.
        if any(lemma.synt is None or len(lemma.synt) > 1 for lemma in output.lemmata):
            raise ValueError("RASR search lexicon must have zero or one syntactic token per lemma")
        with uopen(self.out_uncovered, "wt") as dst:
            for lemma in ordinary:
                for pronunciation in lemma.phon:
                    labels = tuple(pronunciation.split())
                    covered = _has_normalized_word_path(labels, tuple(lemma.synt), pronunciations, neutral)
                    counts["source_pronunciations"] += 1
                    counts["covered_source_pronunciations"] += covered
                    if not covered:
                        dst.write(
                            json.dumps({"orth": lemma.orth, "lm_tokens": lemma.synt, "pronunciation": pronunciation})
                            + "\n"
                        )
        write_xml(self.out_lexicon.get_path(), output.to_xml())
        with uopen(self.out_report, "wt") as dst:
            json.dump(
                {
                    **counts,
                    "mode": self.mode,
                    "lm_vocabulary_size": len(words),
                    "canonical_pronunciations": canonical_count,
                    "added_pronunciations": sum(map(len, pronunciations.values())) - canonical_count,
                    "max_syntactic_tokens_per_lemma": max(map(lambda l: len(l.synt), output.lemmata), default=0),
                },
                dst,
                indent=2,
            )


class PrepareLmTrainingTextJob(Job):
    """Stream sentence-delimited text, optionally HF-normalize, and audit LM OOVs.

    Do not replace OOVs here: KenLM's limit_vocab_file performs that mapping.
    Empty normalized sentences are omitted and counted. The output is also
    reusable for future normalized neural LM training.
    """

    def __init__(self, texts: Sequence[tk.Path], vocabulary: tk.Path, normalize_hf: bool):
        self.texts = list(texts)
        self.vocabulary = vocabulary
        self.normalize_hf = normalize_hf
        self.out_text = self.output_path("text.txt.gz")
        self.out_report = self.output_path("report.json")
        self.out_oov_words = self.output_path("oov_words.tsv")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 48})

    def run(self):
        normalize = EnglishTextNormalizer() if self.normalize_hf else lambda text: text
        with uopen(self.vocabulary, "rt") as src:
            vocabulary = {line.strip() for line in src if line.strip()}
        counts, oovs = Counter(), Counter()
        with uopen(self.out_text, "wt") as dst:
            for text in self.texts:
                with uopen(text, "rt") as src:
                    for line in src:
                        counts["input_sentences"] += 1
                        tokens = normalize(line).upper().split()
                        if not tokens:
                            counts["empty_sentences"] += 1
                            continue
                        counts["output_sentences"] += 1
                        counts["tokens"] += len(tokens)
                        oovs.update(token for token in tokens if token not in vocabulary)
                        dst.write(" ".join(tokens) + "\n")
        with uopen(self.out_oov_words, "wt") as dst:
            for token, count in oovs.most_common():
                dst.write(f"{count}\t{token}\n")
        with uopen(self.out_report, "wt") as dst:
            json.dump(
                {
                    **counts,
                    "normalize_hf": self.normalize_hf,
                    "oov_tokens": sum(oovs.values()),
                    "oov_types": len(oovs),
                    "oov_rate": sum(oovs.values()) / counts["tokens"] if counts["tokens"] else 0.0,
                },
                dst,
                indent=2,
            )


@lru_cache
def get_text_normalization_4gram_lms() -> dict[str, tk.Path]:
    """Schedule matched raw/normalized 4-grams; defaults remain unchanged."""
    texts = [
        get_librispeech_normalized_lm_data(),  # OpenSLR normalization, NOT the HF normalizer.
        CorpusToTxtJob(get_bliss_corpus_dict("wav")["train-other-960"], gzip=True).out_txt,
    ]
    original_lexicon = get_bliss_lexicon()
    normalized_lexicon = NormalizeLexiconLmTokensJob(original_lexicon)
    prefix = "datasets/LibriSpeech/lm/hf_text_normalization"
    normalized_lexicon.add_alias(f"{prefix}/lexicon")
    tk.register_output(f"{prefix}/lexicon", normalized_lexicon.out_lexicon)
    tk.register_output(f"{prefix}/lexicon_report", normalized_lexicon.out_report)
    lms = {}
    for name, lexicon in [("original", original_lexicon), ("normalized", normalized_lexicon.out_lexicon)]:
        vocabulary = LexiconToKenLmVocabularyJob(lexicon)
        prepared = PrepareLmTrainingTextJob(texts, vocabulary.out_vocabulary, normalize_hf=name == "normalized")
        prepared.add_alias(f"{prefix}/{name}/training_text")
        estimate = KenLMplzJob(
            text=[prepared.out_text],
            order=4,
            interpolate_unigrams=False,
            pruning=[0, 0, 1, 1],
            vocabulary=vocabulary.out_vocabulary,
            discount_fallback=(),
            kenlm_binary_folder=ken_lm_binaries,
            mem=100,
            time=48,
        )
        estimate.add_alias(f"{prefix}/{name}/estimate_4gram")
        arpa = NormalizeArpaUnknownTokenJob(estimate.out_lm)
        arpa.add_alias(f"{prefix}/{name}/normalize_unknown")
        for label, output in [
            ("text", prepared.out_text),
            ("report", prepared.out_report),
            ("oov_words", prepared.out_oov_words),
            ("vocabulary", vocabulary.out_vocabulary),
            ("arpa", arpa.out_arpa_lm),
        ]:
            tk.register_output(f"{prefix}/{name}/{label}", output)
        lms[name] = arpa.out_arpa_lm
    return lms
