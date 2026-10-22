"""
BLEU Evaluation Jobs and Scoring Helpers for Unsupervised NMT (UNMT).
Supports evaluating both RETURNN search output dictionaries (search_out.py.gz)
and plain line-by-line hypothesis files against tokenized or raw reference files.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from collections import Counter
import os
import ast
import gzip
import json
import math
import logging

from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.datasets.score_results import ScoreResult
from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob


SPECIAL_TOKENS = {"<pad>", "<unk>", "<s>", "</s>", "<BOS>", "<EOS>", "<MASK>", "<SIL>"}


def clean_mt_tokens(text: str, remove_unk: bool = False) -> str:
    """
    Strips special BOS/EOS/PAD tokens from a decoded or reference string.
    """
    tokens = text.strip().split()
    cleaned = []
    for tok in tokens:
        if tok in ("<s>", "<BOS>"):
            continue
        if tok in ("</s>", "<EOS>", "<pad>"):
            break
        if remove_unk and tok in ("<unk>", "<OOV>"):
            continue
        if tok in SPECIAL_TOKENS and tok != "<unk>":
            continue
        cleaned.append(tok)
    return " ".join(cleaned)


def compute_corpus_bleu_fallback(
    hypotheses: List[str],
    references: List[str],
    max_order: int = 4,
) -> Dict[str, Any]:
    """
    Standard corpus-level BLEU-4 implementation (Papineni et al., 2002 / multi-bleu.perl)
    used both for exact tokenized BLEU verification and as a fallback if sacrebleu is unavailable.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for hyp_line, ref_line in zip(hypotheses, references):
        hyp_tokens = hyp_line.strip().split()
        ref_tokens = ref_line.strip().split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for n in range(1, max_order + 1):
            hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(max(0, len(hyp_tokens) - n + 1)))
            ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
            overlap = hyp_ngrams & ref_ngrams
            matches_by_order[n - 1] += sum(overlap.values())
            possible_matches_by_order[n - 1] += max(0, len(hyp_tokens) - n + 1)

    precisions = [0.0] * max_order
    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
            precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    if hyp_length == 0:
        bp = 0.0
        ratio = 0.0
    else:
        ratio = hyp_length / max(1, ref_length)
        bp = 1.0 if hyp_length > ref_length else math.exp(1.0 - ref_length / hyp_length)

    bleu = geo_mean * bp * 100.0
    prec_100 = [p * 100.0 for p in precisions]
    summary_str = (
        f"BLEU = {bleu:.2f}, {prec_100[0]:.1f}/{prec_100[1]:.1f}/{prec_100[2]:.1f}/{prec_100[3]:.1f} "
        f"(BP={bp:.3f}, ratio={ratio:.3f}, hyp_len={hyp_length}, ref_len={ref_length})"
    )
    return {
        "bleu": float(bleu),
        "precisions": prec_100,
        "bp": float(bp),
        "ratio": float(ratio),
        "hyp_len": int(hyp_length),
        "ref_len": int(ref_length),
        "summary": summary_str,
    }


def _load_text_or_dict(filepath: str) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Loads a file that is either a Python dictionary {seq_tag: text} (optionally gzipped)
    or a plain line-by-line text file.
    Returns (dict_or_none, list_of_lines).
    """
    open_fn = gzip.open if filepath.endswith(".gz") else open
    with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()

    if content.startswith("{") and content.endswith("}"):
        try:
            d = ast.literal_eval(content)
            if isinstance(d, dict):
                # Sort by numeric line index if tags look like "line-123"
                def _tag_sort_key(item):
                    k = str(item[0])
                    parts = k.replace("/", "-").split("-")
                    for p in reversed(parts):
                        if p.isdigit():
                            return (0, int(p), k)
                    return (1, 0, k)

                sorted_items = sorted(d.items(), key=_tag_sort_key)
                lines = []
                norm_dict = {}
                for k, v in sorted_items:
                    if isinstance(v, list):
                        v_str = v[0][1] if (len(v) > 0 and isinstance(v[0], tuple)) else str(v[0])
                    else:
                        v_str = str(v)
                    norm_dict[str(k)] = v_str
                    lines.append(v_str)
                return norm_dict, lines
        except Exception as e:
            logging.warning(f"Failed to parse dict from {filepath}: {e}")

    lines = [line.strip() for line in content.splitlines()]
    return None, lines


class ComputeBleuJob(Job):
    """
    Computes corpus-level BLEU-4 score between system hypotheses and reference translations.
    Supports both RETURNN search output dicts (search_out.py.gz) and plain text files.
    """
    _version = 1

    def __init__(
        self,
        hyp_file: tk.Path,
        ref_file: tk.Path,
        lowercase: bool = False,
        remove_unk: bool = False,
        sacrebleu_tokenize: str = "none",  # "none" for pre-tokenized WMT (multi-bleu equivalent) or "13a"
    ):
        super().__init__()
        self.hyp_file = hyp_file
        self.ref_file = ref_file
        self.lowercase = lowercase
        self.remove_unk = remove_unk
        self.sacrebleu_tokenize = sacrebleu_tokenize

        self.out_bleu = self.output_var("bleu", pickle=False)
        self.out_report = self.output_path("bleu_report.txt")
        self.out_summary_json = self.output_path("bleu_summary.json")
        self.out_hyp_txt = self.output_path("hypotheses.clean.txt")
        self.out_ref_txt = self.output_path("references.clean.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        hyp_dict, hyp_lines = _load_text_or_dict(self.hyp_file.get_path())
        ref_dict, ref_lines = _load_text_or_dict(self.ref_file.get_path())

        if hyp_dict is not None and ref_dict is not None:
            # Align by sequence tag
            common_tags = [k for k in ref_dict.keys() if k in hyp_dict]
            if not common_tags:
                # Try matching by basename / line index suffix
                hyp_by_base = {k.split("/")[-1]: v for k, v in hyp_dict.items()}
                aligned_hyps = []
                aligned_refs = []
                for k, r_val in ref_dict.items():
                    base_k = k.split("/")[-1]
                    if base_k in hyp_by_base:
                        aligned_hyps.append(hyp_by_base[base_k])
                        aligned_refs.append(r_val)
            else:
                aligned_hyps = [hyp_dict[k] for k in common_tags]
                aligned_refs = [ref_dict[k] for k in common_tags]
        else:
            aligned_hyps = hyp_lines
            aligned_refs = ref_lines

        assert len(aligned_hyps) == len(aligned_refs) and len(aligned_refs) > 0, (
            f"Hypothesis/Reference count mismatch: {len(aligned_hyps)} vs {len(aligned_refs)}"
        )

        clean_hyps = []
        clean_refs = []
        for h, r in zip(aligned_hyps, aligned_refs):
            hc = clean_mt_tokens(h, remove_unk=self.remove_unk)
            rc = clean_mt_tokens(r, remove_unk=False)
            if self.lowercase:
                hc = hc.lower()
                rc = rc.lower()
            clean_hyps.append(hc)
            clean_refs.append(rc)

        with open(self.out_hyp_txt.get_path(), "w", encoding="utf-8") as fh:
            for h in clean_hyps:
                fh.write(h + "\n")
        with open(self.out_ref_txt.get_path(), "w", encoding="utf-8") as fr:
            for r in clean_refs:
                fr.write(r + "\n")

        # 1. Exact multi-bleu / tokenized BLEU
        multi_bleu_res = compute_corpus_bleu_fallback(clean_hyps, clean_refs)
        bleu_score = multi_bleu_res["bleu"]
        sacrebleu_str = multi_bleu_res["summary"]

        # 2. SacreBLEU evaluation
        try:
            import sacrebleu
            sb_res = sacrebleu.corpus_bleu(
                clean_hyps,
                [clean_refs],
                tokenize=self.sacrebleu_tokenize,
                lowercase=False,
            )
            bleu_score = float(sb_res.score)
            sacrebleu_str = str(sb_res)
        except Exception as e:
            logging.warning(f"sacrebleu invocation fell back to built-in BLEU-4: {e}")

        self.out_bleu.set(round(bleu_score, 2))

        summary_data = {
            "bleu": round(bleu_score, 2),
            "multi_bleu": round(multi_bleu_res["bleu"], 2),
            "precisions": [round(p, 2) for p in multi_bleu_res["precisions"]],
            "bp": round(multi_bleu_res["bp"], 4),
            "ratio": round(multi_bleu_res["ratio"], 4),
            "hyp_len": multi_bleu_res["hyp_len"],
            "ref_len": multi_bleu_res["ref_len"],
            "num_sentences": len(clean_refs),
            "sacrebleu_signature": sacrebleu_str,
        }

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(summary_data, fj, indent=2)

        with open(self.out_report.get_path(), "w", encoding="utf-8") as frep:
            frep.write("==========================================================\n")
            frep.write("                 UNMT BLEU Evaluation Report              \n")
            frep.write("==========================================================\n")
            frep.write(f"Sentences evaluated : {len(clean_refs)}\n")
            frep.write(f"Corpus BLEU-4       : {bleu_score:.2f}\n")
            frep.write(f"Multi-BLEU summary  : {multi_bleu_res['summary']}\n")
            frep.write(f"SacreBLEU output    : {sacrebleu_str}\n")
            frep.write("----------------------------------------------------------\n")
            frep.write("Sample Translations (first 10 sentences):\n")
            for idx in range(min(10, len(clean_refs))):
                frep.write(f"\n[{idx}] REF: {clean_refs[idx]}\n")
                frep.write(f"[{idx}] HYP: {clean_hyps[idx]}\n")


def generic_bleu_score_recog_out(
    dataset: Dict[str, Any],
    recog_output: tk.Path,
    corpus_name: str,
    target_key: str = "text",
    vocab_opts: Optional[Dict] = None,
    ref_text_file: Optional[tk.Path] = None,
    use_lowercase: bool = False,
    apply_text_norm: bool = False,
    alias: Optional[str] = None,
) -> ScoreResult:
    """
    Scores RETURNN search output using BLEU-4 against either an explicit reference text file
    or the reference extracted from the RETURNN dataset via ReturnnDatasetToTextDictJob.
    """
    if ref_text_file is None:
        ref_text_file = ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset,
            data_key=target_key,
            vocab=vocab_opts,
        ).out_txt

    bleu_job = ComputeBleuJob(
        hyp_file=recog_output,
        ref_file=ref_text_file,
        lowercase=use_lowercase,
        remove_unk=True,
    )
    if alias:
        bleu_job.add_alias(alias)

    return ScoreResult(
        dataset_name=corpus_name,
        main_measure_value=bleu_job.out_bleu,
        report=bleu_job.out_report,
    )
