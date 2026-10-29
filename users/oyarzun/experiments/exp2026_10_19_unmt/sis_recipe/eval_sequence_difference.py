"""
Sequence Length and Difference Evaluation Job for Machine Translation and ASR.
Measures the absolute sequence difference (and statistics: mean, std, median, quartiles, length ratio)
between sequences:
  - Input - Hypothesis (if input is provided)
  - Input - Reference (if input is provided)
  - Hypothesis - Reference
Computes metrics at both token/word level and character level.
Supports both plain line-by-line text files and RETURNN dictionary dumps (*.py, *.py.gz, *.txt.gz).
"""

from typing import Dict, List, Optional, Tuple, Iterator, Union, Any
import os
import ast
import gzip
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk


def load_text_or_dict_file(filepath: str) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Loads text from either a RETURNN Python dictionary dump (*.py, *.py.gz, *.txt.gz)
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
                norm_dict = {}
                for k, v in d.items():
                    if isinstance(v, list):
                        v_str = v[0][1] if (len(v) > 0 and isinstance(v[0], tuple)) else str(v[0])
                    else:
                        v_str = str(v)
                    norm_dict[str(k)] = v_str
                return norm_dict, list(norm_dict.values())
        except Exception as e:
            logging.warning(f"Failed to parse dict from {filepath}: {e}")

    lines = [line.strip() for line in content.splitlines()]
    return None, lines


def compute_pair_difference_stats(
    a_lines: List[str],
    b_lines: List[str],
    level: str = "token",
) -> Dict[str, Any]:
    """
    Computes absolute difference statistics between two lists of text sentences.
    
    :param a_lines: list of strings for sequence A
    :param b_lines: list of strings for sequence B
    :param level: "token" (split on whitespace) or "char" (string length)
    :return: dictionary of summary statistics
    """
    assert len(a_lines) == len(b_lines), f"Line count mismatch: {len(a_lines)} vs {len(b_lines)}"

    if level == "token":
        a_lens = np.array([len(s.strip().split()) for s in a_lines], dtype=np.int32)
        b_lens = np.array([len(s.strip().split()) for s in b_lines], dtype=np.int32)
    elif level == "char":
        a_lens = np.array([len(s.strip()) for s in a_lines], dtype=np.int32)
        b_lens = np.array([len(s.strip()) for s in b_lines], dtype=np.int32)
    else:
        raise ValueError(f"Unknown level: {level!r}, must be 'token' or 'char'")

    abs_diffs = np.abs(a_lens - b_lens)
    signed_diffs = a_lens - b_lens

    q25, q50, q75 = np.percentile(abs_diffs, [25, 50, 75])
    mean_a = float(np.mean(a_lens))
    mean_b = float(np.mean(b_lens))

    return {
        "num_sentences": int(len(abs_diffs)),
        "mean_abs_difference": float(np.mean(abs_diffs)),
        "std_abs_difference": float(np.std(abs_diffs, ddof=1)) if len(abs_diffs) > 1 else 0.0,
        "median_abs_difference": float(q50),
        "q25_quartile": float(q25),
        "q75_quartile": float(q75),
        "iqr": float(q75 - q25),
        "min_abs_difference": int(np.min(abs_diffs)),
        "max_abs_difference": int(np.max(abs_diffs)),
        "mean_signed_difference": float(np.mean(signed_diffs)),
        "mean_len_a": mean_a,
        "mean_len_b": mean_b,
        "length_ratio": float(mean_a / max(1e-6, mean_b)),
        "_abs_diffs": abs_diffs,
        "_signed_diffs": signed_diffs,
        "_a_lens": a_lens,
        "_b_lens": b_lens,
    }


def compute_all_sequence_differences(
    hypothesis_lines: List[str],
    reference_lines: List[str],
    input_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes sequence difference statistics across pairs:
      1. input - hypothesis (if input_lines provided)
      2. input - reference  (if input_lines provided)
      3. hypothesis - reference
    Both token-level and character-level.
    """
    pairs = []
    if input_lines is not None:
        pairs.append(("input_hypothesis", "Input - Hypothesis", input_lines, hypothesis_lines))
        pairs.append(("input_reference", "Input - Reference", input_lines, reference_lines))
    pairs.append(("hypothesis_reference", "Hypothesis - Reference", hypothesis_lines, reference_lines))

    results = {
        "num_sentences": len(hypothesis_lines),
        "token_level": {},
        "character_level": {},
    }

    for key, label, a, b in pairs:
        tok_stats = compute_pair_difference_stats(a, b, level="token")
        char_stats = compute_pair_difference_stats(a, b, level="char")

        results["token_level"][key] = {
            "label": label,
            **{k: v for k, v in tok_stats.items() if not k.startswith("_")}
        }
        results["character_level"][key] = {
            "label": label,
            **{k: v for k, v in char_stats.items() if not k.startswith("_")}
        }

    return results


def format_sequence_difference_report(
    summary_data: Dict[str, Any],
    input_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    title: str = "Sequence Difference Evaluation Report",
) -> str:
    """Formats sequence difference statistics into a readable markdown/text table."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"   {title.upper()}   ")
    lines.append("=" * 78)
    if input_lang:
        lines.append(f"Input Language       : {input_lang}")
    if target_lang:
        lines.append(f"Target Language      : {target_lang}")
    lines.append(f"Sentences Evaluated  : {summary_data.get('num_sentences', 0)}")
    lines.append("-" * 78)

    for level, level_label in [("token_level", "TOKEN / WORD LEVEL (Sequence Length in Tokens)"),
                               ("character_level", "CHARACTER LEVEL (Sequence Length in Characters)")]:
        lines.append(f"\n[ {level_label} ]")
        lines.append(f"{'Pair':<24} | {'Mean |Δ|':<9} | {'Std':<8} | {'Median':<8} | {'Q1 (25%)':<8} | {'Q3 (75%)':<8} | {'IQR':<6} | {'Min-Max':<8}")
        lines.append("-" * 92)

        data = summary_data.get(level, {})
        for pair_key in ["input_hypothesis", "input_reference", "hypothesis_reference"]:
            if pair_key not in data:
                continue
            p = data[pair_key]
            lbl = p.get("label", pair_key)
            mean_val = f"{p['mean_abs_difference']:.2f}"
            std_val = f"{p['std_abs_difference']:.2f}"
            med_val = f"{p['median_abs_difference']:.1f}"
            q25_val = f"{p['q25_quartile']:.1f}"
            q75_val = f"{p['q75_quartile']:.1f}"
            iqr_val = f"{p['iqr']:.1f}"
            min_max = f"{p['min_abs_difference']}-{p['max_abs_difference']}"
            lines.append(f"{lbl:<24} | {mean_val:<9} | {std_val:<8} | {med_val:<8} | {q25_val:<8} | {q75_val:<8} | {iqr_val:<6} | {min_max:<8}")

        lines.append("-" * 92)

    lines.append("\nDetailed Breakdown (Signed & Sequence Length Context):")
    for level, level_label in [("token_level", "Tokens"), ("character_level", "Characters")]:
        data = summary_data.get(level, {})
        lines.append(f"\n  • {level_label}:")
        for pair_key in ["input_hypothesis", "input_reference", "hypothesis_reference"]:
            if pair_key not in data:
                continue
            p = data[pair_key]
            lbl = p.get("label", pair_key)
            ratio_str = f", Ratio (A/B): {p['length_ratio']:.3f}" if "length_ratio" in p else ""
            lines.append(
                f"    - {lbl:<22}: Mean |Δ| = {p['mean_abs_difference']:.2f} ± {p['std_abs_difference']:.2f} "
                f"(Median: {p['median_abs_difference']:.1f}, Q1: {p['q25_quartile']:.1f}, Q3: {p['q75_quartile']:.1f}, IQR: {p['iqr']:.1f}, "
                f"Range: [{p['min_abs_difference']}, {p['max_abs_difference']}], Mean signed (A - B): {p['mean_signed_difference']:+.2f}, "
                f"Avg Len A: {p['mean_len_a']:.1f}, Avg Len B: {p['mean_len_b']:.1f}{ratio_str})"
            )

    lines.append("\n" + "=" * 78 + "\n")
    return "\n".join(lines)


class SequenceDifferenceEvalJob(Job):
    """
    Sisyphus Job to measure absolute sequence differences across:
      1. input - hypothesis (if input is provided)
      2. input - reference  (if input is provided)
      3. hypothesis - reference
    Reports mean, stdev, median, quartiles (Q1, Q2, Q3, IQR), min, max, and length ratio.
    Supports plain line-by-line text files and RETURNN dictionary dumps (*.py, *.py.gz, *.txt.gz).
    """
    __sis_hash_exclude__ = {}

    def __init__(
        self,
        hypothesis_txt: Optional[tk.Path] = None,
        reference_txt: Optional[tk.Path] = None,
        input_txt: Optional[tk.Path] = None,
        name: Optional[str] = "seq_diff",
        input_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ):
        super().__init__()
        self.hypothesis_txt = hypothesis_txt
        self.reference_txt = reference_txt
        self.input_txt = input_txt
        self.name_tag = name
        self.input_lang = input_lang
        self.target_lang = target_lang

        # Outputs
        self.out_summary_json = self.output_path("sequence_difference_summary.json")
        self.out_report_txt = self.output_path("sequence_difference_report.txt")
        self.out_sentence_diffs_tsv = self.output_path("sentence_differences.tsv")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True, rqmt={"cpu": 1, "mem": 2, "time": 1})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting Sequence Difference / Length Analysis Evaluation Job...")

        hyp_dict, hyp_lines = load_text_or_dict_file(self.hypothesis_txt.get_path())
        ref_dict, ref_lines = load_text_or_dict_file(self.reference_txt.get_path())
        inp_dict, inp_lines = load_text_or_dict_file(self.input_txt.get_path()) if self.input_txt else (None, None)

        tags = None
        if hyp_dict is not None and ref_dict is not None:
            common_keys = sorted(set(hyp_dict.keys()) & set(ref_dict.keys()))
            if inp_dict is not None:
                common_keys = sorted(set(common_keys) & set(inp_dict.keys()))
            tags = common_keys
            hyp_lines = [hyp_dict[k] for k in common_keys]
            ref_lines = [ref_dict[k] for k in common_keys]
            inp_lines = [inp_dict[k] for k in common_keys] if inp_dict is not None else None
        else:
            assert len(hyp_lines) == len(ref_lines), (
                f"Line count mismatch between hypothesis ({len(hyp_lines)}) and reference ({len(ref_lines)})"
            )
            if inp_lines is not None:
                assert len(inp_lines) == len(hyp_lines), (
                    f"Line count mismatch between input ({len(inp_lines)}) and hypothesis ({len(hyp_lines)})"
                )

        summary_data = compute_all_sequence_differences(
            hypothesis_lines=hyp_lines,
            reference_lines=ref_lines,
            input_lines=inp_lines,
        )
        if self.input_lang:
            summary_data["input_lang"] = self.input_lang
        if self.target_lang:
            summary_data["target_lang"] = self.target_lang
        summary_data["name"] = self.name_tag

        # Write summary JSON
        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(summary_data, fj, indent=2)

        # Write text report
        report_text = format_sequence_difference_report(
            summary_data=summary_data,
            input_lang=str(self.input_lang) if self.input_lang else None,
            target_lang=str(self.target_lang) if self.target_lang else None,
            title=f"Sequence Difference / Length Analysis Report ({self.name_tag})",
        )
        with open(self.out_report_txt.get_path(), "w", encoding="utf-8") as frep:
            frep.write(report_text)

        # Write per-sentence TSV differences
        with open(self.out_sentence_diffs_tsv.get_path(), "w", encoding="utf-8") as ft:
            if inp_lines is not None:
                headers = [
                    "idx", "tag",
                    "tok_inp", "tok_hyp", "tok_ref",
                    "abs_diff_tok_inp_hyp", "abs_diff_tok_inp_ref", "abs_diff_tok_hyp_ref",
                    "char_inp", "char_hyp", "char_ref",
                    "abs_diff_char_inp_hyp", "abs_diff_char_inp_ref", "abs_diff_char_hyp_ref",
                ]
                ft.write("\t".join(headers) + "\n")
                for i in range(len(hyp_lines)):
                    tag_str = tags[i] if tags else str(i)
                    tok_inp = len(inp_lines[i].split())
                    tok_hyp = len(hyp_lines[i].split())
                    tok_ref = len(ref_lines[i].split())
                    char_inp = len(inp_lines[i])
                    char_hyp = len(hyp_lines[i])
                    char_ref = len(ref_lines[i])
                    row = [
                        str(i), tag_str,
                        str(tok_inp), str(tok_hyp), str(tok_ref),
                        str(abs(tok_inp - tok_hyp)),
                        str(abs(tok_inp - tok_ref)),
                        str(abs(tok_hyp - tok_ref)),
                        str(char_inp), str(char_hyp), str(char_ref),
                        str(abs(char_inp - char_hyp)),
                        str(abs(char_inp - char_ref)),
                        str(abs(char_hyp - char_ref)),
                    ]
                    ft.write("\t".join(row) + "\n")
            else:
                headers = [
                    "idx", "tag",
                    "tok_hyp", "tok_ref",
                    "abs_diff_tok_hyp_ref", "signed_diff_tok_hyp_ref",
                    "char_hyp", "char_ref",
                    "abs_diff_char_hyp_ref", "signed_diff_char_hyp_ref",
                ]
                ft.write("\t".join(headers) + "\n")
                for i in range(len(hyp_lines)):
                    tag_str = tags[i] if tags else str(i)
                    tok_hyp = len(hyp_lines[i].split())
                    tok_ref = len(ref_lines[i].split())
                    char_hyp = len(hyp_lines[i])
                    char_ref = len(ref_lines[i])
                    row = [
                        str(i), tag_str,
                        str(tok_hyp), str(tok_ref),
                        str(abs(tok_hyp - tok_ref)), str(tok_hyp - tok_ref),
                        str(char_hyp), str(char_ref),
                        str(abs(char_hyp - char_ref)), str(char_hyp - char_ref),
                    ]
                    ft.write("\t".join(row) + "\n")

        logging.info("Sequence Difference / Length Analysis Evaluation Job finished successfully.")
