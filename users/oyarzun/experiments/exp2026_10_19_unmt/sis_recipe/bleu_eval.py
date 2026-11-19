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

    raw_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if any(line.startswith("hypothesis:") for line in raw_lines):
        hyp_lines = [
            line[len("hypothesis:"):].strip()
            for line in raw_lines
            if line.startswith("hypothesis:")
        ]
        return None, hyp_lines
    elif any(line.startswith("reference:") for line in raw_lines):
        ref_lines = [
            line[len("reference:"):].strip()
            for line in raw_lines
            if line.startswith("reference:")
        ]
        return None, ref_lines

    # Detect if lines are formatted with sequence tags: "<tag> <text...>"
    if raw_lines:
        first_tokens = [line.split(None, 1)[0] for line in raw_lines]
        if all("-" in tok or "/" in tok or tok.startswith(("dev", "line", "train", "test")) for tok in first_tokens):
            norm_dict = {}
            lines = []
            for line in raw_lines:
                parts = line.split(None, 1)
                tag = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                norm_dict[tag] = text
                lines.append(text)
            return norm_dict, lines

    lines = raw_lines
    return None, lines


class ComputeBleuJob(Job):
    """
    Computes corpus-level BLEU-4 score between system hypotheses and reference translations.
    Supports both RETURNN search output dicts (search_out.py.gz) and plain text files.
    """
    __sis_version__ = 2

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
            for idx, (h, r) in enumerate(zip(clean_hyps, clean_refs)):
                if idx > 0:
                    fh.write("\n")
                fh.write(f"reference: {r}\n")
                fh.write(f"hypothesis: {h}\n")
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


# =============================================================================
# Score Joining & Evaluation Checkpoint Summary Jobs
# =============================================================================

class JoinScoresJob(Job):
    """
    Joins multiple evaluation scores (e.g. baseline word-by-word BLEU across directions
    and retrieval metrics: de2en_csls, de2en_cosine, en2de_csls, en2de_cosine, or just the
    2 cosine results if not using CSLS) into a single human-readable score table and JSON summary.
    """
    __sis_version__ = 1

    def __init__(self, scores: Dict[str, Any], title: str = "Evaluation Scores Summary"):
        super().__init__()
        self.scores = scores
        self.title = title
        self.out_summary_txt = self.output_path("score_results.txt")
        self.out_summary_json = self.output_path("score_results.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    @staticmethod
    def _extract_val(v: Any) -> Any:
        if hasattr(v, "get"):
            v = v.get()
        if hasattr(v, "get_path"):
            v = v.get_path()
        if isinstance(v, str) and os.path.exists(v):
            if v.endswith(".json"):
                try:
                    with open(v, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("bleu", data.get("score", data.get("per", data)))
                except Exception:
                    pass
            else:
                try:
                    with open(v, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    return float(content)
                except Exception:
                    pass
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        return v

    def run(self):
        res = {}
        for k, v in self.scores.items():
            res[k] = self._extract_val(v)

        max_key_len = max([len(k) for k in res.keys()] + [15])
        with open(self.out_summary_txt.get_path(), "w", encoding="utf-8") as f:
            width = max_key_len + 18
            f.write("=" * width + "\n")
            f.write(f" {self.title}\n")
            f.write("=" * width + "\n")
            f.write(f"  {'Metric / Task'.ljust(max_key_len)} | {'Score':>10}\n")
            f.write("  " + "-" * max_key_len + "-+-----------" + "\n")
            for k, val in res.items():
                val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                f.write(f"  {k.ljust(max_key_len)} | {val_str:>10}\n")
            f.write("=" * width + "\n")

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(res, fj, indent=2)


class SummarizeEpochCheckpointsJob(Job):
    """
    Summarizes the evaluation checkpoints of a single UNdreaMT training ablation into one file,
    similarly to ASR evaluation summaries.
    Takes a dict mapping epoch/checkpoint (int or str) to a dict of {metric_name: score_val}.
    Outputs:
      - out_summary_txt: clean ASCII table displaying all checkpoints and their metrics side-by-side
      - out_summary_json: JSON mapping each epoch to its evaluation metrics
    """
    __sis_version__ = 1

    def __init__(
        self,
        epoch_scores: Dict[Union[int, str], Dict[str, Any]],
        title: str = "UNdreaMT Checkpoint Evaluation Summary",
        main_key: Optional[str] = None,
    ):
        super().__init__()
        self.epoch_scores = epoch_scores
        self.title = title
        self.main_key = main_key
        self.out_summary_txt = self.output_path("results_all_epochs.txt")
        self.out_summary_json = self.output_path("results_all_epochs.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        res = {}
        all_metric_keys = []
        # Sort checkpoints numerically if possible
        sorted_eps = sorted(
            self.epoch_scores.keys(),
            key=lambda x: (int(str(x).replace("epoch_", "").replace("ep", "")) if str(x).replace("epoch_", "").replace("ep", "").isdigit() else str(x))
        )
        for ep in sorted_eps:
            scores_dict = self.epoch_scores[ep]
            ep_str = f"epoch_{ep}" if str(ep).isdigit() else str(ep)
            res[ep_str] = {}
            for k, v in scores_dict.items():
                val = JoinScoresJob._extract_val(v)
                res[ep_str][k] = val
                if k not in all_metric_keys:
                    all_metric_keys.append(k)

        col_widths = {k: max(len(k), 10) for k in all_metric_keys}
        ep_col_len = max([len(ep) for ep in res.keys()] + [18])

        header = f"  {'Checkpoint / Epoch'.ljust(ep_col_len)} | " + " | ".join(k.rjust(col_widths[k]) for k in all_metric_keys)
        divider = "  " + "-" * ep_col_len + "-+-" + "-+-".join("-" * col_widths[k] for k in all_metric_keys)
        total_width = len(header) + 2

        with open(self.out_summary_txt.get_path(), "w", encoding="utf-8") as f:
            f.write("=" * total_width + "\n")
            f.write(f" {self.title}\n")
            f.write("=" * total_width + "\n")
            f.write(header + "\n")
            f.write(divider + "\n")

            best_ep = None
            best_val = -1.0
            eval_key = self.main_key or (all_metric_keys[0] if all_metric_keys else None)

            for ep_str, scores in res.items():
                row_vals = []
                for k in all_metric_keys:
                    v = scores.get(k, "-")
                    v_str = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                    row_vals.append(v_str.rjust(col_widths[k]))
                f.write(f"  {ep_str.ljust(ep_col_len)} | " + " | ".join(row_vals) + "\n")

                if eval_key and isinstance(scores.get(eval_key), (int, float)):
                    if scores[eval_key] > best_val:
                        best_val = scores[eval_key]
                        best_ep = ep_str

            f.write("=" * total_width + "\n")
            if best_ep is not None and eval_key:
                f.write(f" Best Checkpoint: {best_ep} ({eval_key}: {best_val:.2f})\n")
                f.write("=" * total_width + "\n")

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(res, fj, indent=2)
