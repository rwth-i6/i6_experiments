__all__ = [
    "CollapseOovScoresInOnnxJob",
    "CountLexiconOovJob",
    "LexiconToKenLmVocabularyJob",
    "NormalizeArpaUnknownTokenJob",
    "BoostArpaUnknownProbabilityJob",
    "ReduceLexiconToTargetOovJob",
]

import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from i6_core.lib.corpus import Corpus
from i6_core.lib.lexicon import Lexicon
from i6_core.util import uopen, write_xml
from sisyphus import Job, Task, tk


def _normalize_word(word: str, casing: str) -> str:
    if casing == "upper":
        return word.upper()
    if casing == "lower":
        return word.lower()
    if casing == "none":
        return word
    raise ValueError(f"Unsupported casing: {casing!r}")


def _count_corpus_words(bliss_corpora: Iterable[tk.Path], casing: str) -> Counter[str]:
    word_counts: Counter[str] = Counter()
    for bliss_corpus in bliss_corpora:
        corpus = Corpus()
        corpus.load(bliss_corpus.get_path())
        for segment in corpus.segments():
            word_counts.update(_normalize_word(word, casing) for word in (segment.orth or "").split())
    return word_counts


def _get_lexicon_words(lexicon: Lexicon, casing: str) -> Set[str]:
    return {
        _normalize_word(orth, casing)
        for lemma in lexicon.lemmata
        for orth in lemma.orth
        if lemma.special is None and orth
    }


def _read_word_frequencies(word_frequencies: tk.Path, casing: str) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    with uopen(word_frequencies, "rt") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                count, word = line.split("\t", maxsplit=1)
                frequencies[_normalize_word(word, casing)] += int(count)
            except ValueError as exc:
                raise ValueError(f"Malformed word-frequency line {line_number}: {line!r}") from exc
    return frequencies


def _read_lm_index_vocabulary(vocab_file: tk.Path) -> Tuple[Dict[str, int], int]:
    token_to_index = {}
    with uopen(vocab_file, "rt") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                token, index_str = line.rsplit(maxsplit=1)
                index = int(index_str)
            except ValueError as exc:
                raise ValueError(f"Malformed LM vocabulary line {line_number}: {line!r}") from exc
            if token in token_to_index and token_to_index[token] != index:
                raise ValueError(f"LM token {token!r} has multiple indices")
            token_to_index[token] = index

    if not token_to_index:
        raise ValueError("LM vocabulary is empty")
    vocab_size = max(token_to_index.values()) + 1
    used_indices = set(token_to_index.values())
    if used_indices != set(range(vocab_size)):
        raise ValueError("LM vocabulary indices must be contiguous")
    return token_to_index, vocab_size


def _read_removed_lemma_tokens(removed_lemmas: tk.Path) -> List[str]:
    tokens = set()
    with uopen(removed_lemmas, "rt") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                _, orthographies = line.split("\t", maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Malformed removed-lemma line {line_number}: {line!r}") from exc
            tokens.update(orthographies.split())
    return sorted(tokens)


def _get_oov_statistics(word_counts: Counter[str], lexicon_words: Set[str]) -> Dict[str, object]:
    oov_counts = Counter({word: count for word, count in word_counts.items() if word not in lexicon_words})
    num_tokens = sum(word_counts.values())
    num_oov_tokens = sum(oov_counts.values())
    num_types = len(word_counts)
    num_oov_types = len(oov_counts)
    return {
        "num_tokens": num_tokens,
        "num_oov_tokens": num_oov_tokens,
        "token_oov_rate": num_oov_tokens / num_tokens if num_tokens else 0.0,
        "num_types": num_types,
        "num_oov_types": num_oov_types,
        "type_oov_rate": num_oov_types / num_types if num_types else 0.0,
        "oov_counts": oov_counts,
    }


def _write_oov_outputs(
    *,
    statistics: Dict[str, object],
    out_oov_words: tk.Path,
    out_report: tk.Path,
    additional_report_values: Optional[Dict[str, object]] = None,
) -> None:
    oov_counts = statistics["oov_counts"]
    assert isinstance(oov_counts, Counter)
    with uopen(out_oov_words, "wt") as file:
        for word, count in sorted(oov_counts.items(), key=lambda item: (-item[1], item[0])):
            file.write(f"{count}\t{word}\n")

    report = {key: value for key, value in statistics.items() if key != "oov_counts"}
    if additional_report_values is not None:
        report.update(additional_report_values)
    with uopen(out_report, "wt") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")


class CountLexiconOovJob(Job):
    """Count token and type OOV rates of one or more Bliss corpora for a Bliss lexicon."""

    def __init__(self, bliss_corpora: Sequence[tk.Path], bliss_lexicon: tk.Path, casing: str = "upper") -> None:
        if not bliss_corpora:
            raise ValueError("At least one Bliss corpus is required")
        if casing not in {"upper", "lower", "none"}:
            raise ValueError(f"Unsupported casing: {casing!r}")

        self.bliss_corpora = list(bliss_corpora)
        self.bliss_lexicon = bliss_lexicon
        self.casing = casing

        self.out_oov_words = self.output_path("oov_words.txt")
        self.out_report = self.output_path("oov_report.json")
        self.out_num_tokens = self.output_var("num_tokens")
        self.out_num_oov_tokens = self.output_var("num_oov_tokens")
        self.out_token_oov_rate = self.output_var("token_oov_rate")
        self.out_num_types = self.output_var("num_types")
        self.out_num_oov_types = self.output_var("num_oov_types")
        self.out_type_oov_rate = self.output_var("type_oov_rate")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())
        statistics = _get_oov_statistics(
            word_counts=_count_corpus_words(self.bliss_corpora, self.casing),
            lexicon_words=_get_lexicon_words(lexicon, self.casing),
        )
        _write_oov_outputs(
            statistics=statistics,
            out_oov_words=self.out_oov_words,
            out_report=self.out_report,
        )

        self.out_num_tokens.set(statistics["num_tokens"])
        self.out_num_oov_tokens.set(statistics["num_oov_tokens"])
        self.out_token_oov_rate.set(statistics["token_oov_rate"])
        self.out_num_types.set(statistics["num_types"])
        self.out_num_oov_types.set(statistics["num_oov_types"])
        self.out_type_oov_rate.set(statistics["type_oov_rate"])


class LexiconToKenLmVocabularyJob(Job):
    """Create a one-token-per-line vocabulary for lexicon-constrained KenLM estimation."""

    def __init__(self, bliss_lexicon: tk.Path) -> None:
        self.bliss_lexicon = bliss_lexicon
        self.out_vocabulary = self.output_path("vocabulary.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())

        # KenLM uses lowercase <unk> internally. It is renamed to RASR's <UNK>
        # after estimation, while sentence-boundary tokens already agree.
        tokens = {"<s>", "</s>", "<unk>"}
        for lemma in lexicon.lemmata:
            if lemma.special is not None:
                continue
            tokens.update(token for token in (lemma.synt if lemma.synt is not None else lemma.orth) if token)

        with uopen(self.out_vocabulary, "wt") as file:
            for token in sorted(tokens):
                file.write(f"{token}\n")


class NormalizeArpaUnknownTokenJob(Job):
    """Rename KenLM's unknown token to the token used by the RASR lexicon."""

    def __init__(self, arpa_lm: tk.Path, source_token: str = "<unk>", target_token: str = "<UNK>") -> None:
        self.arpa_lm = arpa_lm
        self.source_token = source_token
        self.target_token = target_token
        self.out_arpa_lm = self.output_path("lm.arpa.gz")
        self.rqmt = {"cpu": 1, "mem": 2, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self) -> None:
        token_pattern = re.compile(rf"(?<!\S){re.escape(self.source_token)}(?!\S)")
        num_replacements = 0
        with uopen(self.arpa_lm, "rt") as source, uopen(self.out_arpa_lm, "wt") as target:
            for line in source:
                line, replacements = token_pattern.subn(self.target_token, line)
                num_replacements += replacements
                target.write(line)
        if num_replacements == 0:
            raise ValueError(f"ARPA LM does not contain unknown token {self.source_token!r}")


class BoostArpaUnknownProbabilityJob(Job):
    """Multiply the probability of every ARPA n-gram ending in ``unknown_token``."""

    def __init__(self, arpa_lm: tk.Path, probability_factor: float, unknown_token: str = "<UNK>") -> None:
        if probability_factor <= 0.0:
            raise ValueError("probability_factor must be positive")
        self.arpa_lm = arpa_lm
        self.probability_factor = probability_factor
        self.unknown_token = unknown_token
        self.out_arpa_lm = self.output_path("lm.arpa.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        # ARPA probabilities are log10 probabilities.  A multiplicative boost is
        # therefore an additive log10(probability_factor) bonus.
        bonus = math.log10(self.probability_factor)
        changed = 0
        with uopen(self.arpa_lm, "rt") as source, uopen(self.out_arpa_lm, "wt") as target:
            for line in source:
                fields = line.split()
                # ARPA n-gram lines may end in a numeric backoff weight.  Find
                # the final (word) field instead of assuming it is fields[-1].
                words = []
                for field in fields[1:]:
                    try:
                        float(field)
                    except ValueError:
                        words.append(field)
                try:
                    is_ngram = bool(fields)
                    float(fields[0])
                except (IndexError, ValueError):
                    is_ngram = False
                if is_ngram and words and words[-1] == self.unknown_token:
                    fields[0] = f"{float(fields[0]) - bonus:.7f}"
                    line = " ".join(fields) + "\n"
                    changed += 1
                target.write(line)
        if changed == 0:
            raise ValueError(f"ARPA LM contains no n-gram ending in {self.unknown_token!r}")


class CollapseOovScoresInOnnxJob(Job):
    """
    Collapse removed word probabilities into the unknown output of a word-level neural LM.

    The input ONNX model must output negative log-probabilities in ``score_output_name``. Removed output scores are
    masked with a large cost, while the unknown cost becomes ``-log(sum(exp(-score)))`` over the original unknown
    and all removed labels. The decoder vocabulary is also used as the authoritative ONNX output-index mapping.
    """

    def __init__(
        self,
        onnx_model: tk.Path,
        lm_vocab: tk.Path,
        removed_lemmas: tk.Path,
        unknown_token: str = "<UNK>",
        score_output_name: str = "scores",
    ) -> None:
        self.onnx_model = onnx_model
        self.lm_vocab = lm_vocab
        self.removed_lemmas = removed_lemmas
        self.unknown_token = unknown_token
        self.score_output_name = score_output_name

        self.out_onnx_model = self.output_path("model.onnx")
        self.out_report = self.output_path("collapse_oov_report.json")
        self.out_removed_indices = self.output_path("removed_indices.txt")

        self.rqmt = {"cpu": 1, "mem": 16, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self) -> None:
        import numpy as np
        import onnx
        from onnx import external_data_helper, helper, numpy_helper

        token_to_index, vocab_size = _read_lm_index_vocabulary(self.lm_vocab)
        if self.unknown_token not in token_to_index:
            raise ValueError(f"Unknown token {self.unknown_token!r} is missing from the decoder LM vocabulary")
        unknown_index = token_to_index[self.unknown_token]

        removed_tokens = _read_removed_lemma_tokens(self.removed_lemmas)
        missing_tokens = [token for token in removed_tokens if token not in token_to_index]
        if missing_tokens:
            preview = ", ".join(repr(token) for token in missing_tokens[:20])
            raise ValueError(f"Removed lexicon tokens are missing from the decoder LM vocabulary: {preview}")
        removed_indices = sorted({token_to_index[token] for token in removed_tokens})
        if unknown_index in removed_indices:
            raise ValueError("The unknown output itself cannot be part of the removed-label set")
        if not removed_indices:
            raise ValueError("No removed LM output indices were found")

        model_path = self.onnx_model.get_path()
        out_model_path = self.out_onnx_model.get_path()
        model = onnx.load(model_path, load_external_data=False)
        graph = model.graph

        matching_outputs = [output for output in graph.output if output.name == self.score_output_name]
        if len(matching_outputs) != 1:
            raise ValueError(
                f"Expected exactly one ONNX graph output named {self.score_output_name!r}, got {len(matching_outputs)}"
            )
        score_output = matching_outputs[0]
        if score_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("ONNX LM scores must be float32 negative log-probabilities")
        output_dims = score_output.type.tensor_type.shape.dim
        if output_dims and output_dims[-1].HasField("dim_value") and output_dims[-1].dim_value != vocab_size:
            raise ValueError(
                f"ONNX score dimension {output_dims[-1].dim_value} does not match decoder LM vocabulary size "
                f"{vocab_size}"
            )

        producer_outputs = [
            (node, index)
            for node in graph.node
            for index, output_name in enumerate(node.output)
            if output_name == self.score_output_name
        ]
        if len(producer_outputs) != 1:
            raise ValueError(
                f"Expected exactly one producer for ONNX output {self.score_output_name!r}, got {len(producer_outputs)}"
            )

        prefix = "oov_adaptation"
        original_scores = f"{prefix}_original_scores"
        producer_outputs[0][0].output[producer_outputs[0][1]] = original_scores

        selected_indices = np.asarray([unknown_index, *removed_indices], dtype=np.int64)
        score_mask = np.zeros((vocab_size,), dtype=np.float32)
        score_mask[removed_indices] = 1.0e30
        unknown_one_hot = np.zeros((vocab_size,), dtype=np.float32)
        unknown_one_hot[unknown_index] = 1.0

        graph.initializer.extend(
            [
                numpy_helper.from_array(selected_indices, name=f"{prefix}_selected_indices"),
                numpy_helper.from_array(score_mask, name=f"{prefix}_score_mask"),
                numpy_helper.from_array(np.asarray([unknown_index], dtype=np.int64), name=f"{prefix}_unknown_index"),
                numpy_helper.from_array(unknown_one_hot, name=f"{prefix}_unknown_one_hot"),
            ]
        )

        selected_scores = f"{prefix}_selected_scores"
        selected_log_probs = f"{prefix}_selected_log_probs"
        collapsed_log_prob = f"{prefix}_collapsed_log_prob"
        collapsed_unknown_score = f"{prefix}_collapsed_unknown_score"
        masked_scores = f"{prefix}_masked_scores"
        masked_unknown_score = f"{prefix}_masked_unknown_score"
        unknown_delta = f"{prefix}_unknown_delta"
        unknown_delta_vector = f"{prefix}_unknown_delta_vector"

        nodes = [
            helper.make_node(
                "Gather",
                [original_scores, f"{prefix}_selected_indices"],
                [selected_scores],
                axis=-1,
                name=f"{prefix}_gather_collapsed_scores",
            ),
            helper.make_node("Neg", [selected_scores], [selected_log_probs], name=f"{prefix}_scores_to_log_probs"),
        ]

        default_opset = next((opset.version for opset in model.opset_import if not opset.domain), 0)
        if default_opset >= 18:
            graph.initializer.append(
                numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name=f"{prefix}_reduction_axes")
            )
            nodes.append(
                helper.make_node(
                    "ReduceLogSumExp",
                    [selected_log_probs, f"{prefix}_reduction_axes"],
                    [collapsed_log_prob],
                    keepdims=1,
                    name=f"{prefix}_sum_probability_mass",
                )
            )
        else:
            nodes.append(
                helper.make_node(
                    "ReduceLogSumExp",
                    [selected_log_probs],
                    [collapsed_log_prob],
                    axes=[-1],
                    keepdims=1,
                    name=f"{prefix}_sum_probability_mass",
                )
            )

        nodes.extend(
            [
                helper.make_node(
                    "Neg",
                    [collapsed_log_prob],
                    [collapsed_unknown_score],
                    name=f"{prefix}_log_prob_to_score",
                ),
                helper.make_node(
                    "Add",
                    [original_scores, f"{prefix}_score_mask"],
                    [masked_scores],
                    name=f"{prefix}_mask_removed_scores",
                ),
                helper.make_node(
                    "Gather",
                    [masked_scores, f"{prefix}_unknown_index"],
                    [masked_unknown_score],
                    axis=-1,
                    name=f"{prefix}_gather_unknown_score",
                ),
                helper.make_node(
                    "Sub",
                    [collapsed_unknown_score, masked_unknown_score],
                    [unknown_delta],
                    name=f"{prefix}_unknown_score_delta",
                ),
                helper.make_node(
                    "Mul",
                    [unknown_delta, f"{prefix}_unknown_one_hot"],
                    [unknown_delta_vector],
                    name=f"{prefix}_place_unknown_score_delta",
                ),
                helper.make_node(
                    "Add",
                    [masked_scores, unknown_delta_vector],
                    [self.score_output_name],
                    name=f"{prefix}_adapted_scores",
                ),
            ]
        )
        graph.node.extend(nodes)

        model_dir = os.path.dirname(model_path)
        out_model_dir = os.path.dirname(out_model_path)
        external_data_locations = {
            entry.value
            for tensor in external_data_helper._get_all_tensors(model)
            if external_data_helper.uses_external_data(tensor)
            for entry in tensor.external_data
            if entry.key == "location"
        }
        for location in external_data_locations:
            if os.path.isabs(location):
                raise ValueError(f"External ONNX tensor data location must be relative, got: {location!r}")
            source = os.path.normpath(os.path.join(model_dir, location))
            target = os.path.normpath(os.path.join(out_model_dir, location))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)

        onnx.save(model, out_model_path)
        onnx.checker.check_model(out_model_path)

        with uopen(self.out_removed_indices, "wt") as file:
            for token in removed_tokens:
                file.write(f"{token_to_index[token]}\t{token}\n")
        with uopen(self.out_report, "wt") as file:
            json.dump(
                {
                    "score_semantics": "negative_log_probability",
                    "vocab_size": vocab_size,
                    "unknown_token": self.unknown_token,
                    "unknown_index": unknown_index,
                    "num_removed_tokens": len(removed_tokens),
                    "num_removed_indices": len(removed_indices),
                },
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")


class ReduceLexiconToTargetOovJob(Job):
    """
    Remove non-special lemmas in ascending training-frequency order until a target token OOV rate is reached.

    The target is measured over ``evaluation_corpora``. A whole lemma, including all of its orthographies and
    pronunciations, is removed at once. Since lemma frequencies are discrete, the achieved OOV rate can be slightly
    higher than the requested target.
    """

    def __init__(
        self,
        bliss_lexicon: tk.Path,
        evaluation_corpora: Sequence[tk.Path],
        word_frequencies: tk.Path,
        target_oov_rate: float,
        casing: str = "upper",
    ) -> None:
        if not evaluation_corpora:
            raise ValueError("At least one evaluation corpus is required")
        if not 0.0 <= target_oov_rate <= 1.0:
            raise ValueError(f"target_oov_rate must be between 0 and 1, got {target_oov_rate}")
        if casing not in {"upper", "lower", "none"}:
            raise ValueError(f"Unsupported casing: {casing!r}")

        self.bliss_lexicon = bliss_lexicon
        self.evaluation_corpora = list(evaluation_corpora)
        self.word_frequencies = word_frequencies
        self.target_oov_rate = target_oov_rate
        self.casing = casing

        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_removed_lemmas = self.output_path("removed_lemmas.txt")
        self.out_oov_words = self.output_path("oov_words.txt")
        self.out_report = self.output_path("reduction_report.json")
        self.out_num_removed_lemmas = self.output_var("num_removed_lemmas")
        self.out_num_tokens = self.output_var("num_tokens")
        self.out_num_oov_tokens = self.output_var("num_oov_tokens")
        self.out_token_oov_rate = self.output_var("token_oov_rate")
        self.out_num_types = self.output_var("num_types")
        self.out_num_oov_types = self.output_var("num_oov_types")
        self.out_type_oov_rate = self.output_var("type_oov_rate")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())
        evaluation_counts = _count_corpus_words(self.evaluation_corpora, self.casing)
        training_counts = _read_word_frequencies(self.word_frequencies, self.casing)

        active_lemma_counts: Dict[str, int] = defaultdict(int)
        ranked_lemmas: List[Tuple[int, Tuple[str, ...], int]] = []
        for index, lemma in enumerate(lexicon.lemmata):
            if lemma.special is not None:
                continue
            orthographies = tuple(sorted({_normalize_word(orth, self.casing) for orth in lemma.orth if orth}))
            if not orthographies:
                continue
            for orth in orthographies:
                active_lemma_counts[orth] += 1
            frequency = sum(training_counts[orth] for orth in orthographies)
            ranked_lemmas.append((frequency, orthographies, index))

        lexicon_words = set(active_lemma_counts)
        initial_statistics = _get_oov_statistics(evaluation_counts, lexicon_words)
        num_tokens = int(initial_statistics["num_tokens"])
        target_num_oov_tokens = math.ceil(self.target_oov_rate * num_tokens)
        num_oov_tokens = int(initial_statistics["num_oov_tokens"])
        if target_num_oov_tokens < num_oov_tokens:
            raise ValueError(
                f"Target OOV rate {self.target_oov_rate:.6f} is below the baseline rate "
                f"{float(initial_statistics['token_oov_rate']):.6f}"
            )

        removed_indices: Set[int] = set()
        removed_lemmas: List[Tuple[int, Tuple[str, ...]]] = []
        for frequency, orthographies, index in sorted(ranked_lemmas):
            if num_oov_tokens >= target_num_oov_tokens:
                break
            removed_indices.add(index)
            removed_lemmas.append((frequency, orthographies))
            for orth in orthographies:
                active_lemma_counts[orth] -= 1
                if active_lemma_counts[orth] == 0:
                    num_oov_tokens += evaluation_counts[orth]
        if num_oov_tokens < target_num_oov_tokens:
            raise ValueError(
                f"Target OOV rate {self.target_oov_rate:.6f} cannot be reached by removing non-special lemmas"
            )

        lexicon.lemmata = [lemma for index, lemma in enumerate(lexicon.lemmata) if index not in removed_indices]
        write_xml(self.out_lexicon.get_path(), lexicon.to_xml())

        with uopen(self.out_removed_lemmas, "wt") as file:
            for frequency, orthographies in removed_lemmas:
                file.write(f"{frequency}\t{' '.join(orthographies)}\n")

        final_statistics = _get_oov_statistics(evaluation_counts, _get_lexicon_words(lexicon, self.casing))
        _write_oov_outputs(
            statistics=final_statistics,
            out_oov_words=self.out_oov_words,
            out_report=self.out_report,
            additional_report_values={
                "target_token_oov_rate": self.target_oov_rate,
                "baseline_token_oov_rate": initial_statistics["token_oov_rate"],
                "achieved_token_oov_rate": final_statistics["token_oov_rate"],
                "num_removed_lemmas": len(removed_indices),
            },
        )

        self.out_num_removed_lemmas.set(len(removed_indices))
        self.out_num_tokens.set(final_statistics["num_tokens"])
        self.out_num_oov_tokens.set(final_statistics["num_oov_tokens"])
        self.out_token_oov_rate.set(final_statistics["token_oov_rate"])
        self.out_num_types.set(final_statistics["num_types"])
        self.out_num_oov_types.set(final_statistics["num_oov_types"])
        self.out_type_oov_rate.set(final_statistics["type_oov_rate"])
