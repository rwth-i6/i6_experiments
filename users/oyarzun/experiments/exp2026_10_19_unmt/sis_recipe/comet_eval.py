"""
COMET Evaluation Job and Scoring Helpers for Unsupervised NMT (UNMT).
Evaluates cross-lingual neural translation quality using Unbabel/COMET (wmt22-comet-da / wmt20-comet-da).
Supports evaluating both RETURNN search output dictionaries (search_out.py.gz / best_search_results.py.gz)
and plain line-by-line hypothesis files against source inputs and reference files.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import os
import ast
import gzip
import json
import logging
import statistics

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


class CometEvalJob(Job):
    """
    Sisyphus Job that evaluates MT hypotheses using Unbabel/COMET neural metric.
    Computes system-level and segment-level COMET scores comparing (source, hypothesis, reference).
    """
    _version = 1

    def __init__(
        self,
        src_file: tk.Path,
        hyp_file: tk.Path,
        ref_file: tk.Path,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 32,
        gpus: int = 1,
        remove_unk: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.src_file = src_file
        self.hyp_file = hyp_file
        self.ref_file = ref_file
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.remove_unk = remove_unk
        self.name_tag = name or "comet"

        self.out_comet = self.output_var("comet", pickle=False)
        self.out_comet_100 = self.output_var("comet_100", pickle=False)
        self.out_report = self.output_path("comet_report.txt")
        self.out_summary_json = self.output_path("comet_summary.json")
        self.out_segment_scores_tsv = self.output_path("comet_segments.tsv")
        self.out_hyp_txt = self.output_path("hypotheses.clean.txt")

    def tasks(self) -> Iterator[Task]:
        if self.gpus > 0:
            yield Task("run", rqmt={"device": "gpu", "gpu": self.gpus, "cpu": 4, "mem": 16, "time": 1})
        else:
            yield Task("run", mini_task=True, rqmt={"cpu": 4, "mem": 16, "time": 1})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting COMET Evaluation Job...")

        src_dict, src_lines = _load_text_or_dict(self.src_file.get_path())
        hyp_dict, hyp_lines = _load_text_or_dict(self.hyp_file.get_path())
        ref_dict, ref_lines = _load_text_or_dict(self.ref_file.get_path())

        tags = None
        if hyp_dict is not None and ref_dict is not None:
            common_tags = [k for k in ref_dict.keys() if k in hyp_dict]
            if src_dict is not None:
                common_tags = [k for k in common_tags if k in src_dict]

            if not common_tags:
                # Basename matching
                hyp_by_base = {k.split("/")[-1]: v for k, v in hyp_dict.items()}
                src_by_base = {k.split("/")[-1]: v for k, v in src_dict.items()} if src_dict else {}
                aligned_srcs = []
                aligned_hyps = []
                aligned_refs = []
                matched_tags = []
                for k, r_val in ref_dict.items():
                    base_k = k.split("/")[-1]
                    if base_k in hyp_by_base and (not src_dict or base_k in src_by_base):
                        aligned_refs.append(r_val)
                        aligned_hyps.append(hyp_by_base[base_k])
                        if src_dict:
                            aligned_srcs.append(src_by_base[base_k])
                        matched_tags.append(k)
                tags = matched_tags
                aligned_src = aligned_srcs if src_dict else src_lines
            else:
                tags = common_tags
                aligned_refs = [ref_dict[k] for k in common_tags]
                aligned_hyps = [hyp_dict[k] for k in common_tags]
                aligned_src = [src_dict[k] for k in common_tags] if src_dict else src_lines
        else:
            aligned_src = src_lines
            aligned_hyps = hyp_lines
            aligned_refs = ref_lines

        assert len(aligned_hyps) == len(aligned_refs), (
            f"Count mismatch: hyps ({len(aligned_hyps)}) vs refs ({len(aligned_refs)})"
        )
        assert len(aligned_src) == len(aligned_refs), (
            f"Count mismatch: srcs ({len(aligned_src)}) vs refs ({len(aligned_refs)})"
        )
        assert len(aligned_refs) > 0, "No sentences found to evaluate"

        clean_srcs = [clean_mt_tokens(s, remove_unk=False) for s in aligned_src]
        clean_hyps = [clean_mt_tokens(h, remove_unk=self.remove_unk) for h in aligned_hyps]
        clean_refs = [clean_mt_tokens(r, remove_unk=False) for r in aligned_refs]

        # Write cleaned hypotheses
        with open(self.out_hyp_txt.get_path(), "w", encoding="utf-8") as fh:
            for h in clean_hyps:
                fh.write(h + "\n")

        # Load COMET model
        from comet import download_model, load_from_checkpoint
        import torch

        # Check GPU availability
        use_gpus = self.gpus if (self.gpus > 0 and torch.cuda.is_available()) else 0
        logging.info(f"Loading COMET model '{self.model_name}' (use_gpus={use_gpus})...")

        model_path = download_model(self.model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(clean_srcs, clean_hyps, clean_refs)
        ]

        logging.info(f"Scoring {len(data)} sentences with batch_size={self.batch_size}...")
        model_output = model.predict(data, batch_size=self.batch_size, gpus=use_gpus)

        system_score = float(model_output.system_score)
        segment_scores = [float(s) for s in model_output.scores]

        # Set output variables
        self.out_comet.set(round(system_score, 4))
        self.out_comet_100.set(round(system_score * 100.0, 2))

        # Statistics
        mean_score = statistics.mean(segment_scores)
        std_score = statistics.stdev(segment_scores) if len(segment_scores) > 1 else 0.0
        median_score = statistics.median(segment_scores)
        min_score = min(segment_scores)
        max_score = max(segment_scores)

        summary_data = {
            "model": self.model_name,
            "system_score": round(system_score, 4),
            "system_score_100": round(system_score * 100.0, 2),
            "num_sentences": len(segment_scores),
            "mean_segment_score": round(mean_score, 4),
            "std_segment_score": round(std_score, 4),
            "median_segment_score": round(median_score, 4),
            "min_segment_score": round(min_score, 4),
            "max_segment_score": round(max_score, 4),
        }

        # Save summary JSON
        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(summary_data, fj, indent=2)

        # Save segment scores TSV
        with open(self.out_segment_scores_tsv.get_path(), "w", encoding="utf-8") as ft:
            ft.write("idx\ttag\tcomet_score\tsrc\thyp\tref\n")
            for i in range(len(data)):
                tag_str = tags[i] if tags else str(i)
                ft.write(
                    f"{i}\t{tag_str}\t{segment_scores[i]:.4f}\t"
                    f"{clean_srcs[i]}\t{clean_hyps[i]}\t{clean_refs[i]}\n"
                )

        # Save detailed report
        with open(self.out_report.get_path(), "w", encoding="utf-8") as frep:
            frep.write("==============================================================================\n")
            frep.write(f"              COMET EVALUATION REPORT ({self.name_tag.upper()})              \n")
            frep.write("==============================================================================\n")
            frep.write(f"Model Name          : {self.model_name}\n")
            frep.write(f"Sentences Evaluated : {len(segment_scores)}\n")
            frep.write(f"COMET System Score  : {system_score:.4f}  ({system_score * 100.0:.2f}%)\n")
            frep.write(f"Segment Mean ± Std  : {mean_score:.4f} ± {std_score:.4f}\n")
            frep.write(f"Segment Median      : {median_score:.4f}\n")
            frep.write(f"Segment Range       : [{min_score:.4f}, {max_score:.4f}]\n")
            frep.write("------------------------------------------------------------------------------\n")
            frep.write("Sample Translations with COMET Scores (first 10 sentences):\n")
            for idx in range(min(10, len(data))):
                frep.write(f"\n[{idx}] COMET: {segment_scores[idx]:.4f}\n")
                frep.write(f"     SRC : {clean_srcs[idx]}\n")
                frep.write(f"     HYP : {clean_hyps[idx]}\n")
                frep.write(f"     REF : {clean_refs[idx]}\n")
            frep.write("\n==============================================================================\n")

        logging.info(f"COMET Evaluation finished successfully: System Score = {system_score:.4f}")


def generic_comet_score_recog_out(
    dataset: Dict[str, Any],
    recog_output: tk.Path,
    corpus_name: str,
    src_text_file: tk.Path,
    ref_text_file: Optional[tk.Path] = None,
    target_key: str = "target",
    vocab_opts: Optional[Dict] = None,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 32,
    gpus: int = 1,
    alias: Optional[str] = None,
) -> ScoreResult:
    """
    Scores RETURNN search output using COMET against source and reference files.
    """
    if ref_text_file is None:
        ref_text_file = ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset,
            data_key=target_key,
            vocab=vocab_opts,
        ).out_txt

    comet_job = CometEvalJob(
        src_file=src_text_file,
        hyp_file=recog_output,
        ref_file=ref_text_file,
        model_name=model_name,
        batch_size=batch_size,
        gpus=gpus,
        name=corpus_name,
    )
    if alias:
        comet_job.add_alias(alias)

    return ScoreResult(
        dataset_name=corpus_name,
        main_measure_value=comet_job.out_comet_100,
        report=comet_job.out_report,
    )
