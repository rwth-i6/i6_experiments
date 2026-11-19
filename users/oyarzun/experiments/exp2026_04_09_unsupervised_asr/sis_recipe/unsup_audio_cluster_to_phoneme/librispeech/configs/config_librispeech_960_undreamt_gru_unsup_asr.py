"""
Unsupervised ASR with UNdreaMT RNN AED (BiGRU Encoder + StackedGRU Decoder with GlobalAttention)
over Discrete Cross-Modal Codebook Representations.

Replaces the Conformer/Transformer architecture with UNdreaMT's 2-layer BiGRU shared encoder and
2-layer StackedGRU decoder with bilinear GlobalAttention, trained on:
  - Source: Monolingual audio cluster codes (from self-learning segment-level k-means clustering, e.g. v5.5 / v5.6)
  - Target: Monolingual phoneme / n-gram text sequences (e.g. 80% coverage diphones/triphones)
  - Frozen aligned cross-modal embeddings (GAN + CSLS Procrustes alignment)
  - Token-swap denoising autoencoding + on-the-fly greedy backtranslation (100 epochs)

Comprehensive Evaluation:
  1. Beam search decoding and PER evaluation against ground-truth reference phonemes.
  2. Sequence difference & length mismatch analysis (SequenceDifferenceEvalJob).
  3. Shared encoder 2D PCA representation alignment visualization (analyze_encoder_states).
  4. Cross-attention alignment heatmaps and .npz array exports (analyze_cross_attention).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator, Union
import ast
import gzip
import json
import logging
import os
import numpy as np

from sisyphus import Job, Task, tk
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchTakeBestJob
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset

# UNMT / UNdreaMT tooling
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.pipeline import (
    training,
    forward_single,
    get_checkpoint,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.config import (
    get_forward_config,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.default_tools import (
    RETURNN_EXE,
    RETURNN_ROOT,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.data.common import (
    TrainingDatasets,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.data.wmt14_de_en import (
    DumpTextToReturnnHdfJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.per_eval import (
    ComputePerJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.bleu_eval import (
    JoinScoresJob,
    SummarizeEpochCheckpointsJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.eval_sequence_difference import (
    SequenceDifferenceEvalJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.analysis import (
    analyze_encoder_states,
    analyze_encoder_states_contrastive,
    analyze_cross_attention,
)

# Unsupervised ASR & Self-Learning Cluster tooling
from i6_experiments.users.oyarzun.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.cluster_adapter import (
    get_cluster,
    set_cluster,
    get_hubert_seq_tags_path,
    get_oracle_idx_to_phoneme_path,
    get_cheating_phoneme_ref_hdfs,
)
from i6_experiments.users.oyarzun.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.eval_cross_modal_cheating_seg import (
    CheatingSegmentationNearestNeighborEvalJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_09_15_self_learning_emb_asr.sis_recipe.train_exp import (
    build_self_learning_asr_alignment_pipeline,
)


# =============================================================================
# Helper Mini-Task Jobs
# =============================================================================

class DumpReturnnVocabJob(Job):
    """
    Creates a RETURNN dictionary file ({token: id}) from a vocabulary text file
    (newline-separated or python dict string), following UNdreaMT convention:
      <pad>: 0, <unk>: 1, </s>: 2, <s>: 3, regular tokens: 4..V+3.
    """
    __sis_version__ = 1

    def __init__(self, vocab_file: tk.Path):
        super().__init__()
        self.vocab_file = vocab_file
        self.out_returnn_vocab = self.output_path("returnn_vocab.py")
        self.out_vocab_size_txt = self.output_path("vocab_size.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.vocab_file.get_path(), "r", encoding="utf-8") as f:
            content = f.read().strip()

        vocab_words: List[str] = []
        if content.startswith("{") and content.endswith("}"):
            try:
                vocab_dict = ast.literal_eval(content)
                vocab_words = [k for k, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]
            except Exception:
                vocab_words = [line.strip().split()[0] for line in content.splitlines() if line.strip()]
        else:
            vocab_words = [line.strip().split()[0] for line in content.splitlines() if line.strip()]

        full_vocab: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "</s>": 2,
            "<s>": 3,
        }
        idx = 4
        for w in vocab_words:
            if w not in full_vocab:
                full_vocab[w] = idx
                idx += 1

        with open(self.out_returnn_vocab.get_path(), "w", encoding="utf-8") as f:
            f.write("{\n")
            for w, i in full_vocab.items():
                f.write(f"  {w!r}: {i},\n")
            f.write("}\n")

        with open(self.out_vocab_size_txt.get_path(), "w", encoding="utf-8") as f:
            f.write(f"{len(vocab_words)} {len(full_vocab)}\n")


class TakeFirstNLinesJob(Job):
    """Takes the first N lines of a text file."""
    __sis_version__ = 1

    def __init__(self, text_file: tk.Path, num_lines: int = 3000):
        super().__init__()
        self.text_file = text_file
        self.num_lines = num_lines
        self.out_text = self.output_path("subset.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.text_file.get_path(), "r", encoding="utf-8") as f_in, \
             open(self.out_text.get_path(), "w", encoding="utf-8") as f_out:
            for i, line in enumerate(f_in):
                if i >= self.num_lines:
                    break
                f_out.write(line)


class MatchCorporaLinesJob(Job):
    """
    Given a primary text corpus (e.g. audio cluster sequences) and a secondary corpus
    (e.g. text phonemes), ensures that both corpora have the exact same number of lines
    by cyclically repeating the secondary corpus if needed (or truncating to max_sentences),
    and outputs matching sequence tags for RETURNN MetaDataset.
    """
    __sis_version__ = 1

    def __init__(
        self,
        src_text_file: tk.Path,
        trg_text_file: tk.Path,
        max_sentences: Optional[int] = None,
        seq_tag_prefix: str = "line",
    ):
        super().__init__()
        self.src_text_file = src_text_file
        self.trg_text_file = trg_text_file
        self.max_sentences = max_sentences
        self.seq_tag_prefix = seq_tag_prefix

        self.out_src_txt = self.output_path("src_corpus.txt")
        self.out_trg_txt = self.output_path("trg_corpus.txt")
        self.out_seq_tags = self.output_path("seq_tags.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        def _read_lines(p: str, max_n: Optional[int] = None) -> List[str]:
            open_fn = gzip.open if p.endswith(".gz") else open
            res = []
            with open_fn(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        res.append(s)
                        if max_n is not None and len(res) >= max_n:
                            break
            return res

        src_lines = _read_lines(self.src_text_file.get_path(), self.max_sentences)
        num_seqs = len(src_lines)
        assert num_seqs > 0, "Source corpus is empty!"

        trg_lines = _read_lines(self.trg_text_file.get_path())
        assert len(trg_lines) > 0, "Target corpus is empty!"

        with open(self.out_src_txt.get_path(), "w", encoding="utf-8") as f_out_src, \
             open(self.out_trg_txt.get_path(), "w", encoding="utf-8") as f_out_trg, \
             open(self.out_seq_tags.get_path(), "w", encoding="utf-8") as f_out_tags:
            for idx in range(num_seqs):
                f_out_src.write(src_lines[idx] + "\n")
                f_out_trg.write(trg_lines[idx % len(trg_lines)] + "\n")
                f_out_tags.write(f"{self.seq_tag_prefix}-{idx}\n")


class UnpackNgramToPhonemeTextJob(Job):
    """
    Unpacks ngram tokens (triphone 'AA_B_C' -> center phone 'B', diphone 'AA_B' -> 'AA')
    into monophones for PER evaluation against reference monophones.
    Supports both gzipped Python dictionary dumps (*.py.gz) and plain line-by-line text files.
    """
    __sis_version__ = 2

    def __init__(self, text_file: tk.Path, code_type: str = "triphone"):
        super().__init__()
        self.text_file = text_file
        self.code_type = code_type
        self.out_unpacked_txt = self.output_path("unpacked_phonemes.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        def _to_phone(token: str) -> str:
            if "_" not in token:
                return token
            parts = token.split("_")
            if self.code_type == "triphone" or len(parts) >= 3:
                return parts[len(parts) // 2]
            return parts[0]

        open_fn = gzip.open if self.text_file.get_path().endswith(".gz") else open
        with open_fn(self.text_file.get_path(), "rt", encoding="utf-8", errors="replace") as f_in:
            content = f_in.read().strip()

        if content.startswith("{") and content.endswith("}"):
            try:
                data_dict = ast.literal_eval(content)
            except Exception:
                data_dict = eval(content, {"nan": float("nan"), "inf": float("inf")})

            with open(self.out_unpacked_txt.get_path(), "w", encoding="utf-8") as f_out:
                f_out.write("{\n")
                for tag, entry in data_dict.items():
                    if isinstance(entry, list):
                        entry_str = entry[0][1] if (len(entry) > 0 and isinstance(entry[0], tuple)) else str(entry[0])
                    else:
                        entry_str = str(entry)
                    tokens = entry_str.strip().split()
                    phones = [_to_phone(t) for t in tokens if t not in ("<pad>", "<unk>", "</s>", "<s>", "<BOS>", "<EOS>", "<SIL>")]
                    unpacked_str = " ".join(phones)
                    f_out.write(f"{repr(str(tag))}: {repr(unpacked_str)},\n")
                f_out.write("}\n")
        else:
            with open(self.out_unpacked_txt.get_path(), "w", encoding="utf-8") as f_out:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) > 1 and ("-" in parts[0] or "/" in parts[0] or parts[0].startswith(("dev", "line", "train", "test"))):
                        tag, tokens = parts[0], parts[1:]
                        phones = [_to_phone(t) for t in tokens if t not in ("<pad>", "<unk>", "</s>", "<s>", "<BOS>", "<EOS>", "<SIL>")]
                        f_out.write(f"{tag} {' '.join(phones)}\n")
                    else:
                        phones = [_to_phone(t) for t in parts if t not in ("<pad>", "<unk>", "</s>", "<s>", "<BOS>", "<EOS>", "<SIL>")]
                        f_out.write(" ".join(phones) + "\n")


class JoinAsrEpochScoresJob(Job):
    """
    Joins PER evaluation and SequenceDifferenceEvalJob length mismatch metrics
    for a single checkpoint evaluation into score_results.txt and score_results.json.
    """
    __sis_version__ = 1

    def __init__(
        self,
        per: Any,
        per_summary_json: Optional[tk.Path] = None,
        seq_diff_summary_json: Optional[tk.Path] = None,
        title: str = "ASR Evaluation Score Results",
    ):
        super().__init__()
        self.per = per
        self.per_summary_json = per_summary_json
        self.seq_diff_summary_json = seq_diff_summary_json
        self.title = title

        self.out_summary_txt = self.output_path("score_results.txt")
        self.out_summary_json = self.output_path("score_results.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        # 1. PER score
        per_val = None
        if hasattr(self.per, "get"):
            per_val = self.per.get()
        elif hasattr(self.per, "get_path"):
            p = self.per.get_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    per_val = float(f.read().strip())
        elif isinstance(self.per, (int, float)):
            per_val = float(self.per)

        if per_val is None and self.per_summary_json:
            p = self.per_summary_json.get_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    per_val = d.get("per", d.get("score", None))

        results = {
            "per": round(float(per_val), 2) if per_val is not None else "-",
        }

        # 2. Sequence Difference / Length Mismatch metrics
        if self.seq_diff_summary_json and os.path.exists(self.seq_diff_summary_json.get_path()):
            with open(self.seq_diff_summary_json.get_path(), "r", encoding="utf-8") as f:
                sd = json.load(f)

            tok_hyp_ref = sd.get("token_level", {}).get("hypothesis_reference", {})
            char_hyp_ref = sd.get("character_level", {}).get("hypothesis_reference", {})
            tok_inp_hyp = sd.get("token_level", {}).get("input_hypothesis", {})

            if tok_hyp_ref:
                results["length_ratio"] = round(float(tok_hyp_ref.get("length_ratio", 0.0)), 4)
                results["mean_abs_diff_tokens"] = round(float(tok_hyp_ref.get("mean_abs_difference", 0.0)), 2)
                results["mean_signed_diff_tokens"] = round(float(tok_hyp_ref.get("mean_signed_difference", 0.0)), 2)
                results["median_abs_diff_tokens"] = round(float(tok_hyp_ref.get("median_abs_difference", 0.0)), 1)
                results["mean_hyp_tokens"] = round(float(tok_hyp_ref.get("mean_len_a", 0.0)), 2)
                results["mean_ref_tokens"] = round(float(tok_hyp_ref.get("mean_len_b", 0.0)), 2)

                hyp_longer = tok_hyp_ref.get("hyp_longer", {})
                ref_longer = tok_hyp_ref.get("ref_longer", {})
                equal_len = tok_hyp_ref.get("equal_length", {})
                results["hyp_longer_pct"] = round(float(hyp_longer.get("percentage", 0.0)), 2)
                results["ref_longer_pct"] = round(float(ref_longer.get("percentage", 0.0)), 2)
                results["equal_len_pct"] = round(float(equal_len.get("percentage", 0.0)), 2)

            if char_hyp_ref:
                results["char_length_ratio"] = round(float(char_hyp_ref.get("length_ratio", 0.0)), 4)
                results["mean_abs_diff_chars"] = round(float(char_hyp_ref.get("mean_abs_difference", 0.0)), 2)

            if tok_inp_hyp:
                results["input_hyp_ratio"] = round(float(tok_inp_hyp.get("length_ratio", 0.0)), 4)

            tok_inp_ref = sd.get("token_level", {}).get("input_reference", {})
            if tok_inp_ref:
                results["input_ref_ratio"] = round(float(tok_inp_ref.get("length_ratio", 0.0)), 4)
                results["mean_input_tokens"] = round(float(tok_inp_ref.get("mean_len_a", 0.0)), 2)
                results["mean_signed_diff_input_ref"] = round(float(tok_inp_ref.get("mean_signed_difference", 0.0)), 2)

        # Write text report
        max_k_len = max([len(k) for k in results.keys()] + [25])
        width = max_k_len + 20
        with open(self.out_summary_txt.get_path(), "w", encoding="utf-8") as f:
            f.write("=" * width + "\n")
            f.write(f" {self.title}\n")
            f.write("=" * width + "\n")
            f.write(f"  {'Metric'.ljust(max_k_len)} | {'Value':>12}\n")
            f.write("  " + "-" * max_k_len + "-+-------------\n")
            for k, val in results.items():
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                f.write(f"  {k.ljust(max_k_len)} | {val_str:>12}\n")
            f.write("=" * width + "\n")

        # Write JSON
        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(results, fj, indent=2)


class SummarizeAsrEpochCheckpointsJob(Job):
    """
    Summarizes evaluation results across all evaluated epochs for Unsupervised ASR,
    including both PER and detailed sequence length mismatch statistics from SequenceDifferenceEvalJob.
    """
    __sis_version__ = 1

    def __init__(
        self,
        epoch_data: Dict[Union[int, str], Dict[str, Any]],
        title: str = "UNdreaMT Unsupervised ASR Checkpoint Summary",
    ):
        super().__init__()
        self.epoch_data = epoch_data
        self.title = title
        self.out_summary_txt = self.output_path("results_all_epochs.txt")
        self.out_summary_json = self.output_path("results_all_epochs.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        res = {}
        sorted_eps = sorted(
            self.epoch_data.keys(),
            key=lambda x: (int(str(x).replace("epoch_", "").replace("ep", "")) if str(x).replace("epoch_", "").replace("ep", "").isdigit() else str(x))
        )

        for ep in sorted_eps:
            data = self.epoch_data[ep]
            ep_str = f"epoch_{ep}" if str(ep).isdigit() else str(ep)
            ep_res = {}

            # 1. PER
            per_val = None
            per_in = data.get("per")
            if hasattr(per_in, "get"):
                per_val = per_in.get()
            elif hasattr(per_in, "get_path"):
                p = per_in.get_path()
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        per_val = float(f.read().strip())
            elif isinstance(per_in, (int, float)):
                per_val = float(per_in)

            if per_val is None and "per_summary" in data:
                p = data["per_summary"].get_path() if hasattr(data["per_summary"], "get_path") else str(data["per_summary"])
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                        per_val = d.get("per", d.get("score", None))

            ep_res["per"] = round(float(per_val), 2) if per_val is not None else None

            # 2. Length difference metrics
            sd_path = data.get("seq_diff_summary")
            if sd_path:
                p = sd_path.get_path() if hasattr(sd_path, "get_path") else str(sd_path)
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        sd = json.load(f)

                    tok_hyp_ref = sd.get("token_level", {}).get("hypothesis_reference", {})
                    tok_inp_hyp = sd.get("token_level", {}).get("input_hypothesis", {})

                    if tok_hyp_ref:
                        ep_res["len_ratio"] = round(float(tok_hyp_ref.get("length_ratio", 0.0)), 4)
                        ep_res["mean_abs_diff"] = round(float(tok_hyp_ref.get("mean_abs_difference", 0.0)), 2)
                        ep_res["mean_signed_diff"] = round(float(tok_hyp_ref.get("mean_signed_difference", 0.0)), 2)
                        ep_res["median_abs_diff"] = round(float(tok_hyp_ref.get("median_abs_difference", 0.0)), 1)
                        ep_res["hyp_longer_pct"] = round(float(tok_hyp_ref.get("hyp_longer", {}).get("percentage", 0.0)), 2)
                        ep_res["ref_longer_pct"] = round(float(tok_hyp_ref.get("ref_longer", {}).get("percentage", 0.0)), 2)
                        ep_res["equal_len_pct"] = round(float(tok_hyp_ref.get("equal_length", {}).get("percentage", 0.0)), 2)
                        ep_res["mean_hyp_len"] = round(float(tok_hyp_ref.get("mean_len_a", 0.0)), 2)
                        ep_res["mean_ref_len"] = round(float(tok_hyp_ref.get("mean_len_b", 0.0)), 2)

                    if tok_inp_hyp:
                        ep_res["input_hyp_ratio"] = round(float(tok_inp_hyp.get("length_ratio", 0.0)), 4)

            res[ep_str] = ep_res

        # Format ASCII table
        cols = [
            ("PER (%)", "per", 10, "{:.2f}"),
            ("Len Ratio", "len_ratio", 11, "{:.4f}"),
            ("Mean |Δ|", "mean_abs_diff", 10, "{:.2f}"),
            ("Signed Δ", "mean_signed_diff", 10, "{:+.2f}"),
            ("Median |Δ|", "median_abs_diff", 10, "{:.1f}"),
            ("Hyp > Ref %", "hyp_longer_pct", 12, "{:.2f}"),
            ("Ref > Hyp %", "ref_longer_pct", 12, "{:.2f}"),
            ("Equal %", "equal_len_pct", 9, "{:.2f}"),
        ]

        ep_col_len = max([len(ep) for ep in res.keys()] + [18])
        header_items = [f"{'Checkpoint / Epoch'.ljust(ep_col_len)}"] + [c[0].rjust(c[2]) for c in cols]
        header = "  " + " | ".join(header_items)
        divider = "  " + "-" * ep_col_len + "-+-" + "-+-".join("-" * c[2] for c in cols)
        total_width = len(header) + 2

        # Find best checkpoint by PER (lower is better)
        best_ep = None
        best_per = float("inf")
        for ep_str, metrics in res.items():
            per_v = metrics.get("per")
            if per_v is not None and isinstance(per_v, (int, float)):
                if per_v < best_per:
                    best_per = per_v
                    best_ep = ep_str

        with open(self.out_summary_txt.get_path(), "w", encoding="utf-8") as f:
            f.write("=" * total_width + "\n")
            f.write(f" {self.title}\n")
            f.write("=" * total_width + "\n")
            f.write(header + "\n")
            f.write(divider + "\n")

            for ep_str, metrics in res.items():
                row_items = [f"{ep_str.ljust(ep_col_len)}"]
                for col_name, key, col_w, fmt in cols:
                    val = metrics.get(key)
                    if val is not None and isinstance(val, (int, float)):
                        val_str = fmt.format(val).rjust(col_w)
                    else:
                        val_str = "-".rjust(col_w)
                    row_items.append(val_str)
                is_best = "  <--" if ep_str == best_ep else ""
                f.write("  " + " | ".join(row_items) + is_best + "\n")

            f.write("=" * total_width + "\n")
            if best_ep is not None and best_per < float("inf"):
                best_len_ratio = res[best_ep].get("len_ratio", "-")
                f.write(f" Best Checkpoint: {best_ep} (PER: {best_per:.2f}%, Length Ratio: {best_len_ratio})\n")
                f.write("=" * total_width + "\n")

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(res, fj, indent=2)


# =============================================================================
# Main Unsupervised ASR UNdreaMT Pipeline
# =============================================================================

def run_undreamt_gru_unsup_asr_pipeline(
    cluster: Optional[str] = None,
    setup_version: str = "v5.5",
    norm_tag: str = "unit_sphere",
    pca_dim: int = 512,
    code_types: Tuple[str, ...] = ("diphone", "triphone"),
    cluster_configs: Optional[List[Tuple[str, int, str]]] = None,
    w2v_dim: int = 128,
    w2v_win: int = 2,
    metric: str = "csls",
    alignment_method: str = "gan",
    hidden_size: int = 600,
    layers: int = 2,
    dropout: float = 0.3,
    num_epochs: int = 100,
    eval_epochs: Optional[Sequence[int]] = None,
    partition_epoch: int = 10,
    batch_size: int = 6000,
    batch_max_seqs: int = 24,
    learning_rate: float = 0.0002,
    beam_size: int = 8,
    lp_alpha: float = 0.6,
    device: str = "gpu",
    dev_max_sentences: int = 3000,
    max_mono_sentences: int = 300000,
    alignment_results: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs the complete Sisyphus graph for Unsupervised ASR with the UNdreaMT RNN AED model.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()

    if eval_epochs is None:
        eval_epochs = [25, 50, 75, num_epochs] if num_epochs >= 50 else [num_epochs]
    eval_epochs = sorted(set(e for e in eval_epochs if 1 <= e <= num_epochs))
    if not eval_epochs:
        eval_epochs = [num_epochs]

    if cluster_configs is None:
        cluster_configs = []
        if "diphone" in code_types:
            cluster_configs.append(("diphone", 1500, "clus_1500_1500"))
        if "triphone" in code_types:
            cluster_configs.append(("triphone", 10000, "clus_10000_10000"))

    # Build or retrieve alignment pipeline results if not passed
    if alignment_results is None:
        norm_clean = norm_tag.strip().lower()
        alignment_results = build_self_learning_asr_alignment_pipeline(
            librispeech_key="train-clean-100",
            audio_layer=6,
            word2vec_dims=[w2v_dim],
            word2vec_windows=[w2v_win],
            word2vec_epochs=20,
            cluster_counts=[k for _, k, _ in cluster_configs],
            code_types=list(code_types),
            version=f"{setup_version}_{norm_clean}",
            text_version=setup_version,
            coverage_threshold=0.8,
            segmentation_type="cheat_seg",
            pca_dim=pca_dim,
            retrieval_metrics=[metric],
            csls_k=10,
            alignment_method=alignment_method,
            code_cluster_map={
                ct: [(k, c_name)] for ct, k, c_name in cluster_configs
            },
            normalize_segments=norm_clean,
            embedding_model="structured_skipgram" if "v5.6" in setup_version else "word2vec",
            word2vec_postprocessing="none",
            all_but_the_top_D=0,
            run_isotropy_analysis=False,
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    dev_tags_job = TakeFirstNLinesJob(text_file=oracle_seq_tags, num_lines=dev_max_sentences)
    dev_tags_file = dev_tags_job.out_text

    pipeline_outputs: Dict[str, Any] = {}

    for config_item in cluster_configs:
        if len(config_item) == 4:
            code_type, k_clusters, clus_str, explicit_trg_vocab_size = config_item
        else:
            code_type, k_clusters, clus_str = config_item
            explicit_trg_vocab_size = None

        gan_metric_key = f"gan_{metric}" if alignment_method == "gan" else metric
        res_key = (code_type, clus_str, w2v_dim, w2v_win, gan_metric_key)
        res_entry = alignment_results.get(res_key)
        if res_entry is None:
            # Fallback to pure metric key
            res_entry = alignment_results.get((code_type, clus_str, w2v_dim, w2v_win, metric))
        if res_entry is None:
            logging.warning(f"No alignment entry found for {res_key}, skipping.")
            continue

        a_clus = res_entry["cluster_job"]
        t_w2v = res_entry["text_w2v_job"]
        align_gan = res_entry.get("align_gan_job", res_entry.get("align_freq_job"))

        src_tokens_file = a_clus.out_collapsed_tokens_txt
        src_vocab_file = a_clus.out_cluster_vocab_txt
        src_pretrained_emb = align_gan.out_mapped_audio_emb_npy

        raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
        trg_tokens_file = getattr(raw_t_w2v, "token_text_file", getattr(t_w2v, "token_text_file", None))
        trg_vocab_file = getattr(t_w2v, "out_vocab_txt", getattr(t_w2v, "vocab_txt", None))
        trg_pretrained_emb = t_w2v.out_embeddings_npy

        # Vocab sizes: text diphone or triphone
        if explicit_trg_vocab_size is not None:
            trg_vocab_size = explicit_trg_vocab_size
        else:
            trg_vocab_size = 483 if code_type == "diphone" else (6161 if code_type == "triphone" else 100)

        prefix_name = (
            f"unsup_audio_cluster_to_phoneme/librispeech/undreamt_gru/{setup_version}_{norm_tag}/"
            f"{code_type}_{clus_str}_d{w2v_dim}_w{w2v_win}_{metric}"
        )

        # 1. RETURNN Vocabularies
        src_returnn_vocab_job = DumpReturnnVocabJob(src_vocab_file)
        src_returnn_vocab_job.add_alias(f"{prefix_name}/vocab/audio_clusters")
        src_returnn_vocab = src_returnn_vocab_job.out_returnn_vocab

        trg_returnn_vocab_job = DumpReturnnVocabJob(trg_vocab_file)
        trg_returnn_vocab_job.add_alias(f"{prefix_name}/vocab/text_phonemes")
        trg_returnn_vocab = trg_returnn_vocab_job.out_returnn_vocab

        tk.register_output(f"{prefix_name}/vocab/audio_returnn_vocab.py", src_returnn_vocab)
        tk.register_output(f"{prefix_name}/vocab/text_returnn_vocab.py", trg_returnn_vocab)

        # 2. Nearest-Neighbor Baseline & Ground-Truth Reference for 3k Dev Utterances
        dw_tag = f"dim{w2v_dim}_win{w2v_win}"
        eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
            audio_embeddings_npy=src_pretrained_emb,
            text_embeddings_npy=trg_pretrained_emb,
            phoneme_vocab_file=trg_vocab_file,
            cheating_segmentation_path=src_tokens_file,
            audio_cluster_hdfs=None,
            reference_phonemes=oracle_ref_hdfs,
            reference_phoneme_vocab_file=oracle_idx_to_phoneme,
            seq_tags_file=oracle_seq_tags,
            code_type=code_type,
            max_seqs=dev_max_sentences,
            ignore_silence=True,
            retrieval_metric=metric,
            csls_k=10,
            center_text_embeddings=True,
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_{setup_version}",
        )
        eval_train_job.add_alias(f"{prefix_name}/baseline_nearest_neighbor")
        tk.register_output(f"{prefix_name}/baseline_nearest_neighbor/per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/baseline_nearest_neighbor/per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/baseline_nearest_neighbor/hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/baseline_nearest_neighbor/reference.txt", eval_train_job.out_reference_txt)

        dev_ref_txt = eval_train_job.out_reference_txt

        dev_src_job = TakeFirstNLinesJob(text_file=src_tokens_file, num_lines=dev_max_sentences)
        dev_src_job.add_alias(f"{prefix_name}/data/dev_audio_tokens")
        dev_src_txt = dev_src_job.out_text

        # 3. Match source and target corpora lengths & RETURNN SimpleHDFWriter Datasets
        match_train_job = MatchCorporaLinesJob(
            src_text_file=src_tokens_file,
            trg_text_file=trg_tokens_file,
            max_sentences=max_mono_sentences,
            seq_tag_prefix="line",
        )
        match_train_job.add_alias(f"{prefix_name}/data/match_train_corpora")

        train_src_hdf_job = DumpTextToReturnnHdfJob(
            text_file=match_train_job.out_src_txt,
            returnn_vocab_file=src_returnn_vocab,
            max_sentences=max_mono_sentences,
            seq_tag_prefix="line",
            seq_tags_file=match_train_job.out_seq_tags,
        )
        train_src_hdf_job.add_alias(f"{prefix_name}/hdf/train_audio_clusters")

        train_trg_hdf_job = DumpTextToReturnnHdfJob(
            text_file=match_train_job.out_trg_txt,
            returnn_vocab_file=trg_returnn_vocab,
            max_sentences=max_mono_sentences,
            seq_tag_prefix="line",
            seq_tags_file=match_train_job.out_seq_tags,
        )
        train_trg_hdf_job.add_alias(f"{prefix_name}/hdf/train_text_phonemes")

        dev_src_hdf_job = DumpTextToReturnnHdfJob(
            text_file=dev_src_txt,
            returnn_vocab_file=src_returnn_vocab,
            max_sentences=dev_max_sentences,
            seq_tag_prefix="dev",
            seq_tags_file=dev_tags_file,
        )
        dev_src_hdf_job.add_alias(f"{prefix_name}/hdf/dev_audio_clusters")

        dev_trg_hdf_job = DumpTextToReturnnHdfJob(
            text_file=dev_ref_txt,
            returnn_vocab_file=trg_returnn_vocab,
            max_sentences=dev_max_sentences,
            seq_tag_prefix="dev",
            seq_tags_file=dev_tags_file,
        )
        dev_trg_hdf_job.add_alias(f"{prefix_name}/hdf/dev_text_reference")

        tk.register_output(f"{prefix_name}/hdf/train_audio.hdf", train_src_hdf_job.out_hdf)
        tk.register_output(f"{prefix_name}/hdf/train_text.hdf", train_trg_hdf_job.out_hdf)
        tk.register_output(f"{prefix_name}/hdf/dev_audio.hdf", dev_src_hdf_job.out_hdf)
        tk.register_output(f"{prefix_name}/hdf/dev_text.hdf", dev_trg_hdf_job.out_hdf)

        # 4. MetaDatasets & Datastreams
        train_dataset = MetaDataset(
            datasets={
                "src_mono": HdfDataset(
                    files=[train_src_hdf_job.out_hdf],
                    partition_epoch=partition_epoch,
                    seq_ordering="random",
                ),
                "trg_mono": HdfDataset(
                    files=[train_trg_hdf_job.out_hdf],
                    partition_epoch=partition_epoch,
                    seq_ordering="random",
                ),
            },
            data_map={
                "data": ("src_mono", "data"),
                "target": ("trg_mono", "data"),
            },
            seq_order_control_dataset="src_mono",
        )

        dev_dataset = MetaDataset(
            datasets={
                "src_dev": HdfDataset(
                    files=[dev_src_hdf_job.out_hdf],
                    partition_epoch=1,
                    seq_ordering="sorted_reverse",
                ),
                "trg_dev": HdfDataset(
                    files=[dev_trg_hdf_job.out_hdf],
                    partition_epoch=1,
                    seq_ordering="sorted_reverse",
                ),
            },
            data_map={
                "data": ("src_dev", "data"),
                "target": ("trg_dev", "data"),
            },
            seq_order_control_dataset="src_dev",
        )

        test_s2t_dataset = MetaDataset(
            datasets={
                "src_test": HdfDataset(files=[dev_src_hdf_job.out_hdf]),
                "trg_test": HdfDataset(files=[dev_trg_hdf_job.out_hdf]),
            },
            data_map={
                "data": ("src_test", "data"),
                "target": ("trg_test", "data"),
            },
            seq_order_control_dataset="src_test",
        )

        datastreams = {
            "data": LabelDatastream(
                available_for_inference=True,
                vocab=src_returnn_vocab,
                vocab_size=k_clusters + 4,
            ),
            "target": LabelDatastream(
                available_for_inference=False,
                vocab=trg_returnn_vocab,
                vocab_size=trg_vocab_size + 4,
            ),
        }

        training_datasets = TrainingDatasets(
            train=train_dataset,
            eval_datasets={"dev": dev_dataset},
            datastreams=datastreams,
        )

        # 5. Network & Training Arguments
        network_module = "definitions.undreamt_rnn_aed_v1.Model"
        train_step_module = "train_steps.undreamt_backtranslation_train_step.train_step"
        forward_step_module = "recognition.undreamt_mt.forward_step.forward_step"
        callback_module = "recognition.undreamt_mt.callback.RecognitionToTextDictCallback"

        net_args = {
            "src_vocab_size": k_clusters,
            "trg_vocab_size": trg_vocab_size,
            "embedding_size": w2v_dim,
            "hidden_size": hidden_size,
            "layers": layers,
            "dropout": dropout,
            "param_init": 0.1,
            "src_pretrained_emb_path": src_pretrained_emb,
            "trg_pretrained_emb_path": trg_pretrained_emb,
            "freeze_encoder_embeddings": True,
            "freeze_decoder_embeddings": False,
            "use_embedding_generator": False,
            "alignment_function": "general",
        }

        train_step_args = {
            "default_data_key": "data",
            "default_target_key": "target",
            "enable_denoising": True,
            "enable_backtranslation": True,
            "denoise_backtranslation": True,
            "bt_max_ratio": 2.0,
            "bt_src2trg_max_ratio": 1.5,
            "bt_trg2src_max_ratio": 1.5,
            "denoise_scale": 1.0,
            "bt_scale": 1.0,
        }

        returnn_train_config_dict = {
            "default_data_key": "data",
            "default_target_key": "target",
            "batch_size": batch_size,
            "max_seqs": batch_max_seqs,
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "learning_rate": learning_rate,
            "gradient_clip": 1.0,
            "gradient_clip_global_norm": 5.0,
            "num_allowed_consec_invalid_gradient_steps": 10,
        }

        train_rqmt = {
            "mem_rqmt": 24,
            "time_rqmt": 24,
            "cpu_rqmt": 4,
            "gpu_mem": 24,
        }
        if device != "gpu":
            train_rqmt["device"] = device

        train_args = {
            "config": returnn_train_config_dict,
            "post_config": {
                "cleanup_old_models": {
                    "keep_last_n": 4,
                    "keep_best_n": 4,
                    "keep": sorted(set(list(eval_epochs) + [num_epochs])),
                }
            },
            "python_prolog": [
                "import os",
                "import sys",
                'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"',
            ],
            "network_module": network_module,
            "train_step_module": train_step_module,
            "net_args": net_args,
            "train_args": train_step_args,
            "rqmt": train_rqmt,
        }

        undreamt_train_name = f"{prefix_name}/training"
        train_job = training(
            training_name=undreamt_train_name,
            datasets=training_datasets,
            train_args=train_args,
            num_epochs=num_epochs,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
        )

        pipeline_outputs[f"{code_type}_train_job"] = train_job

        # 6. Checkpoint Evaluation (Search, PER, Sequence Length Mismatch)
        epoch_scores: Dict[int, Dict[str, Any]] = {}
        for eval_ep in eval_epochs:
            ckpt = get_checkpoint(
                training_name=undreamt_train_name,
                train_job=train_job,
                get_specific_checkpoint=eval_ep,
            )

            fwd_cfg = get_forward_config(
                config={
                    "default_data_key": "data",
                    "default_target_key": "target",
                    "batch_size": min(batch_size, 4000),
                    "max_seqs": 16,
                },
                network_module=network_module,
                extra_config=ReturnnConfig(
                    config={},
                    python_prolog=[
                        "import os, sys",
                        'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                    ],
                ),
                net_args=net_args,
                decoder=forward_step_module,
                callback_module=callback_module,
                callback_opts={"include_beam": True},
                decoder_args={
                    "beam_size": beam_size,
                    "lp_alpha": lp_alpha,
                    "max_ratio": 1.5,
                    "src_lang": "src",
                    "trg_lang": "trg",
                    "input_data_key": "data",
                },
                datastreams=datastreams,
                vocab_key="target",
            )

            recog_prefix = f"{undreamt_train_name}/recog_ep{eval_ep}/dev3k"
            search_out = forward_single(
                prefix_name=recog_prefix,
                returnn_config=fwd_cfg,
                checkpoint=ckpt,
                dataset_dict=test_s2t_dataset.as_returnn_opts(),
                returnn_exe=RETURNN_EXE,
                returnn_root=RETURNN_ROOT,
                rqmt={"device": device, "gpu": 1 if device == "gpu" else 0, "cpu": 4, "mem": 16, "time": 2},
            )
            best_hyp = SearchTakeBestJob(search_out, output_gzip=True).out_best_search_results

            # Unpack ngram tokens (diphone/triphone) into monophones for PER
            unpacked_hyp_job = UnpackNgramToPhonemeTextJob(best_hyp, code_type=code_type)
            unpacked_hyp_job.add_alias(f"{recog_prefix}/unpack_phonemes")
            unpacked_hyp = unpacked_hyp_job.out_unpacked_txt

            per_job = ComputePerJob(
                hyp_file=unpacked_hyp,
                ref_file=dev_ref_txt,
                remove_unk=True,
            )
            per_job.add_alias(f"{recog_prefix}/per_eval")
            tk.register_output(f"{recog_prefix}/per", per_job.out_per)
            tk.register_output(f"{recog_prefix}/per_report.txt", per_job.out_report)
            tk.register_output(f"{recog_prefix}/per_summary.json", per_job.out_summary_json)
            tk.register_output(f"{recog_prefix}/hypotheses.unpacked.txt", unpacked_hyp)
            tk.register_output(f"{recog_prefix}/hypotheses.raw.txt", best_hyp)

            # Sequence difference & length mismatch analysis
            seq_diff_job = SequenceDifferenceEvalJob(
                input_txt=dev_src_txt,
                hypothesis_txt=unpacked_hyp,
                reference_txt=dev_ref_txt,
                name=f"{code_type}_{clus_str}_ep{eval_ep}",
                input_lang=f"audio_{code_type}",
                target_lang="phonemes",
                input_seq_tags=dev_tags_file,
            )
            seq_diff_job.add_alias(f"{recog_prefix}/length_analysis")
            tk.register_output(f"{recog_prefix}/length_analysis/report.txt", seq_diff_job.out_report_txt)
            tk.register_output(f"{recog_prefix}/length_analysis/summary.json", seq_diff_job.out_summary_json)
            tk.register_output(f"{recog_prefix}/length_analysis/sentence_differences.tsv", seq_diff_job.out_sentence_diffs_tsv)

            # Join score results for this epoch (PER + length mismatch metrics)
            score_job = JoinAsrEpochScoresJob(
                per=per_job.out_per,
                per_summary_json=per_job.out_summary_json,
                seq_diff_summary_json=seq_diff_job.out_summary_json,
                title=f"UNdreaMT ASR Evaluation ({code_type} {clus_str} ep{eval_ep})",
            )
            score_job.add_alias(f"{recog_prefix}/score_results")
            tk.register_output(f"{recog_prefix}/score_results.txt", score_job.out_summary_txt)
            tk.register_output(f"{recog_prefix}/score_results.json", score_job.out_summary_json)

            epoch_scores[eval_ep] = {
                "per": per_job.out_per,
                "per_summary": per_job.out_summary_json,
                "seq_diff_summary": seq_diff_job.out_summary_json,
            }

        if epoch_scores:
            summary_epochs_job = SummarizeAsrEpochCheckpointsJob(
                epoch_data=epoch_scores,
                title=f"UNdreaMT Unsupervised ASR Summary ({code_type} {clus_str})",
            )
            summary_epochs_job.add_alias(f"{undreamt_train_name}/results_all_epochs")
            tk.register_output(f"{undreamt_train_name}/results_all_epochs.txt", summary_epochs_job.out_summary_txt)
            tk.register_output(f"{undreamt_train_name}/results_all_epochs.json", summary_epochs_job.out_summary_json)

        # 7. Shared-Encoder 2D PCA Representation Alignment Visualization
        pca_checkpoints = sorted(set(list(eval_epochs) + [num_epochs]))
        pca_jobs = analyze_encoder_states(
            config={
                "default_data_key": "data",
                "default_target_key": "target",
                "batch_size": 3000,
                "max_seqs": 16,
            },
            training_name=undreamt_train_name,
            train_job=train_job,
            train_args=train_args,
            train_data=training_datasets,
            test_data_dict={"dev3k": test_s2t_dataset},
            checkpoints=pca_checkpoints,
            analysis_name=f"vis_encoder_pca_audio_vs_phn",
            cosine_similarity_summary=True,
            src_label=f"audio_{code_type}",
            trg_label="text_phonemes",
            src_vocab_file=src_returnn_vocab,
            trg_vocab_file=trg_returnn_vocab,
            src_text_file=dev_src_txt,
            trg_text_file=dev_ref_txt,
            seq_tags_file=dev_tags_file,
            extra_forward_config=ReturnnConfig(
                config={},
                python_prolog=[
                    "import os, sys",
                    'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                ],
            ),
            rqmt={"device": device, "gpu": 1 if device == "gpu" else 0, "cpu": 4, "mem": 16, "time": 2},
        )
        pipeline_outputs[f"{code_type}_pca_jobs"] = pca_jobs

        # 7b. Comparative / Contrastive BERTScore Shared-Encoder 2D PCA Visualization
        # (Compares utterances that are semantically close vs. semantically far apart via BERTScore)
        contrastive_pca_jobs = analyze_encoder_states_contrastive(
            config={
                "default_data_key": "data",
                "default_target_key": "target",
                "batch_size": 3000,
                "max_seqs": 16,
            },
            training_name=undreamt_train_name,
            train_job=train_job,
            train_args=train_args,
            train_data=training_datasets,
            test_data_dict={"dev3k": test_s2t_dataset},
            checkpoints=pca_checkpoints,
            analysis_name="pca-contrastive",
            cosine_similarity_summary=True,
            valid_seq_tags_files={"dev3k": dev_tags_file},
            extra_forward_config=ReturnnConfig(
                config={},
                python_prolog=[
                    "import os, sys",
                    'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                ],
            ),
            rqmt={"device": device, "gpu": 1 if device == "gpu" else 0, "cpu": 4, "mem": 24, "time": 2},
        )
        pipeline_outputs[f"{code_type}_contrastive_pca_jobs"] = contrastive_pca_jobs

        # 8. Note: Cross-Attention visualization (analyze_cross_attention) is omitted here
        # because UNdreaMT GRU uses single-layer GlobalAttention (Bahdanau/Luong dot/general)
        # rather than multi-head CrossAttentionV1 layers found in Conformer/Transformer AED models.

    return pipeline_outputs


# =============================================================================
# v5.9.2 Compositional Audio N-Grams without Silence ASR-GRU Pipeline
# =============================================================================

def run_v5_9_2_wo_sil_undreamt_gru_unsup_asr_pipeline(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    base_cluster_count: int = 40,
    spin_codebook_size: int = 2048,
    audio_layer: int = 12,
    d_value: int = 2,
    code_types: Tuple[str, ...] = ("diphone", "triphone"),
    num_epochs: int = 100,
    eval_epochs: Optional[Sequence[int]] = None,
    alignment_results: Optional[Dict[Any, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unsupervised ASR with UNdreaMT RNN AED (BiGRU Encoder + StackedGRU Decoder with GlobalAttention)
    initialized with v5.9.2 Spin-HuBERT 2048 + Compositional Audio N-Grams without Silence
    (100% symmetric, D=2, Unit-Sphere Normalization, PCA 512):
      - Audio: 40 base clusters -> 2-grams (432 clusters) and 3-grams (7325 clusters) with 80% coverage
      - Text: monophone text without silence -> 2-grams (321 diphones) and 3-grams (3524 triphones) with 80% coverage
      - Pretrained aligned embeddings: Word2Vec dim=128, win=2 + All-But-The-Top (D=2) + GAN CSLS refinement.
    """
    if alignment_results is None:
        from i6_experiments.users.oyarzun.experiments.exp2026_09_15_self_learning_emb_asr.sis_recipe.train_exp import (
            build_self_learning_asr_alignment_pipeline,
        )
        alignment_results = build_self_learning_asr_alignment_pipeline(
            librispeech_key="train-clean-100",
            audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
            audio_layer=audio_layer,
            word2vec_dims=[128],
            word2vec_windows=[2],
            word2vec_epochs=20,
            cluster_counts=[base_cluster_count],
            coverage_threshold=0.80,
            code_types=list(code_types),
            code_cluster_map={
                "diphone": [(base_cluster_count, f"clus_{base_cluster_count}_2gram")],
                "triphone": [(base_cluster_count, f"clus_{base_cluster_count}_3gram")],
            },
            segmentation_type="cheating",
            pca_dim=pca_dim,
            version=f"v5.9.2_spin_{spin_codebook_size}_base{base_cluster_count}_wo_sil_unit_sphere_isotropic_D{d_value}",
            text_version="v5.9.2_wo_sil",
            text_model_tag="hubert_l6",
            normalize_segments="unit_sphere",
            retrieval_metrics=["csls"],
            csls_k=10,
            alignment_method="gan",
            embedding_model="word2vec",
            word2vec_postprocessing="all_but_the_top",
            all_but_the_top_D=d_value,
            run_isotropy_analysis=False,
            spin_codebook_size=spin_codebook_size,
            use_audio_ngrams=True,
            sil_prob=0.0,
            surround_w_sil=False,
        )

    cluster_configs = []
    if "diphone" in code_types:
        cluster_configs.append(("diphone", 432, f"clus_{base_cluster_count}_2gram", 321))
    if "triphone" in code_types:
        cluster_configs.append(("triphone", 7325, f"clus_{base_cluster_count}_3gram", 3524))

    return run_undreamt_gru_unsup_asr_pipeline(
        cluster=cluster,
        setup_version="v5.9.2_wo_sil_spin_audio_ngrams_base40",
        norm_tag="unit_sphere",
        pca_dim=pca_dim,
        code_types=code_types,
        cluster_configs=cluster_configs,
        w2v_dim=128,
        w2v_win=2,
        metric="csls",
        alignment_method="gan",
        num_epochs=num_epochs,
        eval_epochs=eval_epochs,
        alignment_results=alignment_results,
        **kwargs,
    )


