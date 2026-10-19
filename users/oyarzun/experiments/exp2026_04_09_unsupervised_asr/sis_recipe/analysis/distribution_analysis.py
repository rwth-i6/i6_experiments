"""
Corpus Distribution Analysis Jobs for Text N-grams and Discovered Speech Clusters.

1. TextNgramDistributionAnalysisJob:
   Computes frequency distributions of bigrams and trigrams in text (e.g. phonemized text,
   word text, or precomputed n-grams), and plots histograms sorted descending from most frequent
   to least frequent with dual y-axes (counts and percentage), summary JSON, and ASCII report.

2. SpeechClusterDistributionAnalysisJob:
   Analyzes discovered acoustic speech clusters (frame counts per cluster across the overall corpus),
   computing duration in hours, Shannon entropy, perplexity, Gini coefficient, and plotting:
   - Descending-sorted frame assignment histograms (both top-K zoomed and full corpus distribution)
   - Cumulative frame coverage curve showing cluster concentration across the corpus
   - Exporting cluster_counts.txt, summary JSON, and ASCII report.
"""

import collections
import gzip
import io
import json
import logging
import math
import os
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sisyphus import tk
from sisyphus.task import Task


def _open_text(path: str):
    """Opens a file supporting both plain text and gzip."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _resolve(p: Optional[Union[tk.Path, str]]) -> Optional[str]:
    """Resolves tk.Path or string to an absolute path."""
    if p is None:
        return None
    if isinstance(p, tk.Path):
        path_str = p.get_path()
    else:
        path_str = os.path.expandvars(str(p))
    if path_str.startswith("/hpcwork/"):
        path_str = "/rwthfs/rz/cluster/hpcwork/" + path_str[len("/hpcwork/"):]
    return path_str


def _resolve_list(paths: Optional[Union[List[Union[tk.Path, str]], tk.Path, str]]) -> List[str]:
    """Resolves a single path or list of paths."""
    if paths is None:
        return []
    if isinstance(paths, (tk.Path, str)):
        res = _resolve(paths)
        return [res] if res else []
    resolved = []
    for p in paths:
        r = _resolve(p)
        if r:
            resolved.append(r)
    return resolved


# =============================================================================
# 1. Text N-gram Distribution Analysis Job
# =============================================================================

class TextNgramDistributionAnalysisJob(tk.Job):
    """
    Computes frequency distributions for bigrams and trigrams from a text corpus,
    and plots descending-sorted histograms with detailed summary statistics.
    """

    def __init__(
        self,
        text_file: Optional[Union[tk.Path, str]] = None,
        bigram_counts_file: Optional[Union[tk.Path, str]] = None,
        trigram_counts_file: Optional[Union[tk.Path, str]] = None,
        top_k: int = 50,
        vocab_file: Optional[Union[tk.Path, str]] = None,
        ignore_silence: bool = True,
        title_prefix: str = "LibriSpeech Phoneme N-grams",
        name_prefix: str = "text_ngram_analysis",
    ):
        super().__init__()
        self.text_file = text_file
        self.bigram_counts_file = bigram_counts_file
        self.trigram_counts_file = trigram_counts_file
        self.top_k = top_k
        self.vocab_file = vocab_file
        self.ignore_silence = ignore_silence
        self.title_prefix = title_prefix
        self.name_prefix = name_prefix

        # Outputs
        self.out_bigram_plot = self.output_path("bigram_histogram.png")
        self.out_trigram_plot = self.output_path("trigram_histogram.png")
        self.out_combined_plot = self.output_path("ngram_combined_histogram.png")
        self.out_coverage_plot = self.output_path("ngram_cumulative_coverage.png")
        self.out_counts_txt = self.output_path("ngram_counts.txt")
        self.out_summary_json = self.output_path("ngram_distribution.json")
        self.out_report_txt = self.output_path("ngram_report.txt")

    @classmethod
    def hash(cls, parsed_args):
        parsed_args["version"] = "v5"
        return super().hash(parsed_args)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 1})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

        silence_tokens = {"<sil>", "[sil]", "<SIL>", "SIL", "<eps>", "blank", ""} if self.ignore_silence else set()

        bigram_counts = collections.Counter()
        trigram_counts = collections.Counter()
        unigram_counts = collections.Counter()

        total_lines = 0
        total_tokens = 0
        text_path = ""
        bg_path = None
        tg_path = None

        # Load from precomputed counts files if provided
        if self.bigram_counts_file is not None:
            bg_path = _resolve(self.bigram_counts_file)
            if bg_path and os.path.exists(bg_path):
                logging.info(f"Loading precomputed bigram counts from: {bg_path}")
                with _open_text(bg_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            bg = parts[0]
                            if self.ignore_silence and ("<SIL>" in bg or "<sil>" in bg or "SIL" in bg.split("_")):
                                continue
                            bigram_counts[bg] += int(parts[1])

        if self.trigram_counts_file is not None:
            tg_path = _resolve(self.trigram_counts_file)
            if tg_path and os.path.exists(tg_path):
                logging.info(f"Loading precomputed trigram counts from: {tg_path}")
                with _open_text(tg_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            tg = parts[0]
                            if self.ignore_silence and ("<SIL>" in tg or "<sil>" in tg or "SIL" in tg.split("_")):
                                continue
                            trigram_counts[tg] += int(parts[1])

        # If text_file is provided, compute counts from token stream
        if self.text_file is not None:
            text_path = _resolve(self.text_file)
            if text_path and os.path.exists(text_path):
                logging.info(f"Analyzing N-gram distributions from text stream: {text_path}")
                with _open_text(text_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total_lines += 1
                        parts = line.split()

                        # If first element is a sequence tag, skip tag
                        if len(parts) > 1 and ("/" in parts[0] or (parts[0].count("-") >= 2 and any(c.isdigit() for c in parts[0]))):
                            tokens = parts[1:]
                        else:
                            tokens = parts

                        if self.ignore_silence:
                            tokens = [t for t in tokens if t not in silence_tokens]

                        if not tokens:
                            continue

                        total_tokens += len(tokens)
                        for t in tokens:
                            unigram_counts[t] += 1

                        if self.bigram_counts_file is None:
                            for i in range(len(tokens) - 1):
                                bg = f"{tokens[i]}_{tokens[i + 1]}"
                                bigram_counts[bg] += 1

                        if self.trigram_counts_file is None:
                            for i in range(len(tokens) - 2):
                                tg = f"{tokens[i]}_{tokens[i + 1]}_{tokens[i + 2]}"
                                trigram_counts[tg] += 1

        total_bigrams = sum(bigram_counts.values())
        total_trigrams = sum(trigram_counts.values())
        unique_bigrams = len(bigram_counts)
        unique_trigrams = len(trigram_counts)
        unique_unigrams = len(unigram_counts)

        logging.info(
            f"Processed {total_lines:,} lines, {total_tokens:,} tokens. "
            f"Found {unique_unigrams:,} unique unigrams, {unique_bigrams:,} unique bigrams ({total_bigrams:,} total), "
            f"and {unique_trigrams:,} unique trigrams ({total_trigrams:,} total)."
        )

        # Sorted distributions (descending: most frequent to least frequent)
        sorted_bigrams = bigram_counts.most_common()
        sorted_trigrams = trigram_counts.most_common()

        # Compute entropy
        def _calc_entropy(counts: collections.Counter) -> float:
            total = sum(counts.values())
            if total == 0:
                return 0.0
            h = 0.0
            for c in counts.values():
                p = c / total
                h -= p * math.log2(p)
            return h

        bg_entropy = _calc_entropy(bigram_counts)
        tg_entropy = _calc_entropy(trigram_counts)

        # -------------------------------------------------------------
        # Generate Plots using Matplotlib
        # -------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_k_bi = min(self.top_k, max(len(sorted_bigrams), 1))

        # 1. Bigram Histogram Plot
        self._plot_histogram(
            items=sorted_bigrams[:top_k_bi],
            total_count=total_bigrams,
            title=f"{self.title_prefix} - Top {top_k_bi} Bigrams (Most to Least Frequent)",
            xlabel="Bigram Token",
            color="#2b5c8f",
            out_path=self.out_bigram_plot.get_path(),
        )

        # 2. Trigram Histogram Plot
        top_k_tri = min(self.top_k, max(len(sorted_trigrams), 1))
        self._plot_histogram(
            items=sorted_trigrams[:top_k_tri],
            total_count=total_trigrams,
            title=f"{self.title_prefix} - Top {top_k_tri} Trigrams (Most to Least Frequent)",
            xlabel="Trigram Token",
            color="#d95f02",
            out_path=self.out_trigram_plot.get_path(),
        )

        # 3. Combined Side-by-Side / Two-Panel Plot
        self._plot_combined(
            sorted_bigrams=sorted_bigrams[:min(30, len(sorted_bigrams))],
            total_bigrams=total_bigrams,
            sorted_trigrams=sorted_trigrams[:min(30, len(sorted_trigrams))],
            total_trigrams=total_trigrams,
            title=f"{self.title_prefix} - Top N-gram Distributions",
            out_path=self.out_combined_plot.get_path(),
        )

        # 4. Compute Cumulative Coverage Thresholds (50%, 70%, 80%, 90%)
        def _calc_thresholds(items, total):
            counts = np.array([x[1] for x in items], dtype=np.int64)
            cum = np.cumsum(counts)
            pcts = cum / total * 100.0 if total > 0 else np.zeros_like(cum)
            res = {}
            for target in [50, 70, 80, 90]:
                idx = int(np.searchsorted(pcts, target))
                n_types = idx + 1 if idx < len(counts) else len(counts)
                pct_vocab = (n_types / len(counts) * 100.0) if len(counts) > 0 else 0.0
                res[f"{target}%"] = {
                    "types_count": n_types,
                    "percentage_of_vocabulary": round(pct_vocab, 2),
                    "exact_cumulative_percentage": round(float(pcts[min(idx, len(pcts) - 1)]), 3) if len(pcts) > 0 else 0.0,
                }
            return res, pcts

        bg_thresholds, _ = _calc_thresholds(sorted_bigrams, total_bigrams)
        tg_thresholds, _ = _calc_thresholds(sorted_trigrams, total_trigrams)

        # 5. Cumulative Coverage Plot
        self._plot_cumulative_coverage(
            sorted_bigrams=sorted_bigrams,
            total_bigrams=total_bigrams,
            sorted_trigrams=sorted_trigrams,
            total_trigrams=total_trigrams,
            bg_thresholds=bg_thresholds,
            tg_thresholds=tg_thresholds,
            title=f"{self.title_prefix} - Cumulative N-Gram Coverage Curves (50%, 70%, 80%, 90%)",
            out_path=self.out_coverage_plot.get_path(),
        )

        # -------------------------------------------------------------
        # Dump Counts Text File
        # -------------------------------------------------------------
        with open(self.out_counts_txt.get_path(), "w", encoding="utf-8") as f:
            f.write("# TYPE\tNGRAM\tCOUNT\tPERCENTAGE\n")
            for bg, cnt in sorted_bigrams:
                pct = cnt / total_bigrams * 100.0 if total_bigrams > 0 else 0
                f.write(f"BIGRAM\t{bg}\t{cnt}\t{pct:.4f}\n")
            for tg, cnt in sorted_trigrams:
                pct = cnt / total_trigrams * 100.0 if total_trigrams > 0 else 0
                f.write(f"TRIGRAM\t{tg}\t{cnt}\t{pct:.4f}\n")

        # -------------------------------------------------------------
        # JSON Output
        # -------------------------------------------------------------
        source_file = text_path or bg_path or tg_path or "precomputed_counts"
        summary_data = {
            "title": self.title_prefix,
            "source_file": source_file,
            "total_lines": total_lines,
            "total_tokens": total_tokens,
            "unigrams": {
                "unique_count": unique_unigrams,
                "top_20": [
                    {"unigram": u, "count": c, "percentage": round(c / total_tokens * 100, 3)}
                    for u, c in unigram_counts.most_common(20)
                ],
            },
            "bigrams": {
                "total_count": total_bigrams,
                "unique_count": unique_bigrams,
                "entropy_bits": round(bg_entropy, 3),
                "coverage_thresholds": bg_thresholds,
                "top_100": [
                    {"ngram": bg, "count": c, "percentage": round(c / total_bigrams * 100, 3)}
                    for bg, c in sorted_bigrams[:100]
                ],
            },
            "trigrams": {
                "total_count": total_trigrams,
                "unique_count": unique_trigrams,
                "entropy_bits": round(tg_entropy, 3),
                "coverage_thresholds": tg_thresholds,
                "top_100": [
                    {"ngram": tg, "count": c, "percentage": round(c / total_trigrams * 100, 3)}
                    for tg, c in sorted_trigrams[:100]
                ],
            },
        }

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # -------------------------------------------------------------
        # Text Report Output
        # -------------------------------------------------------------
        with open(self.out_report_txt.get_path(), "w", encoding="utf-8") as rf:
            rf.write("=" * 85 + "\n")
            rf.write("           TEXT N-GRAM FREQUENCY ANALYSIS REPORT\n")
            rf.write("=" * 85 + "\n")
            rf.write(f"Corpus:                {self.title_prefix}\n")
            rf.write(f"Source File:           {text_path}\n")
            rf.write(f"Total Sentences:       {total_lines:,}\n")
            rf.write(f"Total Tokens:          {total_tokens:,}\n")
            rf.write(f"Unique Unigrams:       {unique_unigrams:,}\n")
            rf.write(f"Unique Bigrams:        {unique_bigrams:,}  (Total: {total_bigrams:,}, Entropy: {bg_entropy:.2f} bits)\n")
            rf.write(f"Unique Trigrams:       {unique_trigrams:,}  (Total: {total_trigrams:,}, Entropy: {tg_entropy:.2f} bits)\n")
            rf.write("=" * 85 + "\n\n")

            rf.write("CUMULATIVE COVERAGE THRESHOLDS:\n")
            rf.write("-" * 85 + "\n")
            rf.write(f"{'Target Coverage':<18} {'Bigrams Needed':<22} {'% of Bigram Vocab':<20} {'Trigrams Needed':<22} {'% of Trigram Vocab':<20}\n")
            rf.write("-" * 85 + "\n")
            for target in ["50%", "70%", "80%", "90%"]:
                b_info = bg_thresholds[target]
                t_info = tg_thresholds[target]
                rf.write(
                    f"{target:<18} {b_info['types_count']:<22,d} {b_info['percentage_of_vocabulary']:>6.2f}%               "
                    f"{t_info['types_count']:<22,d} {t_info['percentage_of_vocabulary']:>6.2f}%\n"
                )
            rf.write("=" * 85 + "\n\n")

            rf.write("TOP 50 BIGRAMS (Sorted Descending: Most Frequent to Least Frequent):\n")
            rf.write("-" * 85 + "\n")
            rf.write(f"{'Rank':<6} {'Bigram':<25} {'Count':<15} {'Percentage':<15} {'Cumulative %':<15}\n")
            rf.write("-" * 85 + "\n")
            cum_bg = 0
            for rank, (bg, cnt) in enumerate(sorted_bigrams[:50], 1):
                cum_bg += cnt
                pct = cnt / total_bigrams * 100 if total_bigrams > 0 else 0
                cum_pct = cum_bg / total_bigrams * 100 if total_bigrams > 0 else 0
                rf.write(f"{rank:<6} {bg:<25} {cnt:<15,d} {pct:>8.3f}% {cum_pct:>13.2f}%\n")

            rf.write("\n" + "=" * 85 + "\n")
            rf.write("TOP 50 TRIGRAMS (Sorted Descending: Most Frequent to Least Frequent):\n")
            rf.write("-" * 85 + "\n")
            rf.write(f"{'Rank':<6} {'Trigram':<32} {'Count':<15} {'Percentage':<15} {'Cumulative %':<15}\n")
            rf.write("-" * 85 + "\n")
            cum_tg = 0
            for rank, (tg, cnt) in enumerate(sorted_trigrams[:50], 1):
                cum_tg += cnt
                pct = cnt / total_trigrams * 100 if total_trigrams > 0 else 0
                cum_pct = cum_tg / total_trigrams * 100 if total_trigrams > 0 else 0
                rf.write(f"{rank:<6} {tg:<32} {cnt:<15,d} {pct:>8.3f}% {cum_pct:>13.2f}%\n")
            rf.write("=" * 85 + "\n")

        logging.info("Text N-gram Distribution Analysis completed successfully.")

    def _plot_histogram(
        self,
        items: List[Tuple[str, int]],
        total_count: int,
        title: str,
        xlabel: str,
        color: str,
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        if not items:
            return

        labels = [item[0] for item in items]
        counts = [item[1] for item in items]
        percentages = [(c / total_count * 100.0) if total_count > 0 else 0.0 for c in counts]

        fig, ax1 = plt.subplots(figsize=(max(12, len(items) * 0.28), 6.5), dpi=200)

        x_indices = np.arange(len(labels))
        bars = ax1.bar(x_indices, counts, color=color, alpha=0.88, edgecolor="#222222", linewidth=0.6, width=0.75)

        ax1.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=10)
        ax1.set_ylabel("Occurrence Count", fontsize=11, fontweight="bold", color="#222222")
        ax1.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.35)

        # Dual y-axis for percentage
        ax2 = ax1.twinx()
        max_pct = max(percentages) if percentages else 1.0
        ax2.set_ylim(0, max_pct * 1.15)
        ax2.set_ylabel("% of Total Occurrences", fontsize=11, fontweight="bold", color="#444444")
        ax2.grid(False)

        # Annotate top 15 bars with count
        for i in range(min(15, len(bars))):
            bar = bars[i]
            height = bar.get_height()
            pct_val = percentages[i]
            ax1.annotate(
                f"{pct_val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="semibold",
            )

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_combined(
        self,
        sorted_bigrams: List[Tuple[str, int]],
        total_bigrams: int,
        sorted_trigrams: List[Tuple[str, int]],
        total_trigrams: int,
        title: str,
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=200)

        # Subplot 1: Bigrams
        if sorted_bigrams:
            b_labels = [item[0] for item in sorted_bigrams]
            b_counts = [item[1] for item in sorted_bigrams]
            ax1.bar(range(len(b_labels)), b_counts, color="#2b5c8f", alpha=0.88, edgecolor="#222222", width=0.75)
            ax1.set_title("Top Bigrams (Sorted Descending: Most Frequent to Least Frequent)", fontsize=11, fontweight="bold")
            ax1.set_xticks(range(len(b_labels)))
            ax1.set_xticklabels(b_labels, rotation=45, ha="right", fontsize=9)
            ax1.set_ylabel("Occurrence Count", fontsize=10, fontweight="bold")
            ax1.grid(axis="y", linestyle="--", alpha=0.35)

        # Subplot 2: Trigrams
        if sorted_trigrams:
            t_labels = [item[0] for item in sorted_trigrams]
            t_counts = [item[1] for item in sorted_trigrams]
            ax2.bar(range(len(t_labels)), t_counts, color="#d95f02", alpha=0.88, edgecolor="#222222", width=0.75)
            ax2.set_title("Top Trigrams (Sorted Descending: Most Frequent to Least Frequent)", fontsize=11, fontweight="bold")
            ax2.set_xticks(range(len(t_labels)))
            ax2.set_xticklabels(t_labels, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Occurrence Count", fontsize=10, fontweight="bold")
            ax2.grid(axis="y", linestyle="--", alpha=0.35)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_cumulative_coverage(
        self,
        sorted_bigrams: List[Tuple[str, int]],
        total_bigrams: int,
        sorted_trigrams: List[Tuple[str, int]],
        total_trigrams: int,
        bg_thresholds: Dict,
        tg_thresholds: Dict,
        title: str,
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)

        # 1. Panel 1: Coverage vs. Number of Unique Types (Log Scale)
        bg_counts = np.array([x[1] for x in sorted_bigrams], dtype=np.int64)
        bg_cum = np.cumsum(bg_counts) / total_bigrams * 100.0 if total_bigrams > 0 else np.zeros_like(bg_counts)
        bg_x = np.arange(1, len(bg_counts) + 1)

        tg_counts = np.array([x[1] for x in sorted_trigrams], dtype=np.int64)
        tg_cum = np.cumsum(tg_counts) / total_trigrams * 100.0 if total_trigrams > 0 else np.zeros_like(tg_counts)
        tg_x = np.arange(1, len(tg_counts) + 1)

        ax1.plot(bg_x, bg_cum, color="#2b5c8f", linewidth=2.5, label=f"Bigrams ({len(bg_counts):,} unique)")
        ax1.plot(tg_x, tg_cum, color="#d95f02", linewidth=2.5, label=f"Trigrams ({len(tg_counts):,} unique)")
        ax1.set_xscale("log")

        colors = {"50%": "#1b9e77", "70%": "#7570b3", "80%": "#e7298a", "90%": "#d95f02"}
        for target, col in colors.items():
            pct_val = float(target.replace("%", ""))
            ax1.axhline(pct_val, color=col, linestyle="--", alpha=0.45, linewidth=1.0)

            # Bigram point
            b_cnt = bg_thresholds[target]["types_count"]
            ax1.plot(b_cnt, pct_val, marker="o", color="#2b5c8f", markersize=6)
            ax1.annotate(
                f"{target}: {b_cnt:,}",
                xy=(b_cnt, pct_val),
                xytext=(b_cnt * 1.25, pct_val - 2.5),
                fontsize=8,
                fontweight="semibold",
                color="#2b5c8f",
            )

            # Trigram point
            t_cnt = tg_thresholds[target]["types_count"]
            ax1.plot(t_cnt, pct_val, marker="s", color="#d95f02", markersize=6)
            ax1.annotate(
                f"{target}: {t_cnt:,}",
                xy=(t_cnt, pct_val),
                xytext=(t_cnt * 1.15, pct_val + 1.5),
                fontsize=8,
                fontweight="semibold",
                color="#d95f02",
            )

        ax1.set_xlabel("Number of Unique N-Gram Types (Ranked Descending, Log Scale)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Cumulative Corpus Occurrences (%)", fontsize=11, fontweight="bold")
        ax1.set_title("Cumulative Coverage vs. Unique Types (Log Scale)", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 105)
        ax1.grid(True, which="both", linestyle="--", alpha=0.35)
        ax1.legend(loc="lower right", fontsize=10)

        # 2. Panel 2: Coverage vs. % of Vocabulary (Linear Scale)
        bg_pct_vocab = bg_x / len(bg_counts) * 100.0 if len(bg_counts) > 0 else bg_x
        tg_pct_vocab = tg_x / len(tg_counts) * 100.0 if len(tg_counts) > 0 else tg_x

        ax2.plot(bg_pct_vocab, bg_cum, color="#2b5c8f", linewidth=2.5, label="Bigrams")
        ax2.plot(tg_pct_vocab, tg_cum, color="#d95f02", linewidth=2.5, label="Trigrams")
        ax2.plot([0, 100], [0, 100], color="gray", linestyle=":", linewidth=1.2, label="Uniform Baseline")

        for target, col in colors.items():
            pct_val = float(target.replace("%", ""))
            ax2.axhline(pct_val, color=col, linestyle="--", alpha=0.45, linewidth=1.0)
            b_pct = bg_thresholds[target]["percentage_of_vocabulary"]
            t_pct = tg_thresholds[target]["percentage_of_vocabulary"]
            ax2.plot(b_pct, pct_val, marker="o", color="#2b5c8f", markersize=6)
            ax2.plot(t_pct, pct_val, marker="s", color="#d95f02", markersize=6)

        ax2.set_xlabel("% of Unique Vocabulary (Ranked Most to Least Frequent)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Cumulative Corpus Occurrences (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Cumulative Coverage vs. % of Vocabulary", fontsize=12, fontweight="bold")
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 105)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="lower right", fontsize=10)

        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


# =============================================================================
# 2. Speech Cluster Distribution Analysis Job
# =============================================================================

class SpeechClusterDistributionAnalysisJob(tk.Job):
    """
    Analyzes the frame assignment distribution across discovered acoustic speech clusters
    in the overall corpus. Plots descending-sorted frame assignment histograms (both top-K
    zoomed and full corpus distribution) and cumulative coverage curves, along with entropy,
    perplexity, Gini coefficient, and audio hours statistics.
    """

    def __init__(
        self,
        cluster_counts_file: Optional[Union[tk.Path, str]] = None,
        raw_clusters_file: Optional[Union[tk.Path, str]] = None,
        cluster_hdfs: Optional[Union[List[Union[tk.Path, str]], tk.Path, str]] = None,
        cluster_vocab_file: Optional[Union[tk.Path, str]] = None,
        k_clusters: Optional[int] = None,
        top_k: int = 50,
        frame_shift_sec: float = 0.02,  # 20ms frame shift for HuBERT
        title_prefix: str = "Discovered Speech Clusters",
        name_prefix: str = "speech_cluster_analysis",
    ):
        super().__init__()
        self.cluster_counts_file = cluster_counts_file
        self.raw_clusters_file = raw_clusters_file
        self.cluster_hdfs = cluster_hdfs
        self.cluster_vocab_file = cluster_vocab_file
        self.k_clusters = k_clusters
        self.top_k = top_k
        self.frame_shift_sec = frame_shift_sec
        self.title_prefix = title_prefix
        self.name_prefix = name_prefix

        # Outputs
        self.out_distribution_plot = self.output_path("cluster_distribution.png")
        self.out_cumulative_plot = self.output_path("cluster_cumulative_coverage.png")
        self.out_counts_txt = self.output_path("cluster_counts.txt")
        self.out_summary_json = self.output_path("cluster_distribution.json")
        self.out_report_txt = self.output_path("cluster_report.txt")

    @classmethod
    def hash(cls, parsed_args):
        parsed_args["version"] = "v2"
        return super().hash(parsed_args)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 1})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
        counts_path = _resolve(self.cluster_counts_file)
        raw_path = _resolve(self.raw_clusters_file)
        hdf_paths = _resolve_list(self.cluster_hdfs)

        cluster_counts = collections.Counter()

        if counts_path and os.path.exists(counts_path):
            logging.info(f"Loading precomputed cluster counts from: {counts_path}")
            with open(counts_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            c_id = int(parts[0])
                            cnt = int(parts[1])
                            cluster_counts[c_id] = cnt
                        except ValueError:
                            continue
        elif raw_path and os.path.exists(raw_path):
            logging.info(f"Computing cluster frame counts from raw clusters file: {raw_path}")
            with _open_text(raw_path) as f:
                for line in f:
                    for tok in line.strip().split():
                        try:
                            c_id = int(tok)
                            cluster_counts[c_id] += 1
                        except ValueError:
                            continue
        elif hdf_paths:
            import h5py
            logging.info(f"Computing cluster frame counts from {len(hdf_paths)} HDF files...")
            for hdf_path in hdf_paths:
                if not os.path.exists(hdf_path):
                    logging.warning(f"HDF file not found: {hdf_path}")
                    continue
                with h5py.File(hdf_path, "r") as f:
                    target_key = None
                    for k in ["inputs", "data", "labels", "classes"]:
                        if k in f:
                            target_key = k
                            break
                    if target_key is not None:
                        data = f[target_key][:]
                    elif "targets" in f and "data" in f["targets"]:
                        data = f["targets"]["data"][:]
                    elif "targets" in f and "labels" in f["targets"]:
                        data = f["targets"]["labels"][:]
                    else:
                        continue
                    
                    if data.ndim > 1:
                        data = data.reshape(-1)
                    unique, counts = np.unique(data, return_counts=True)
                    for u, c in zip(unique, counts):
                        try:
                            cluster_counts[int(u)] += int(c)
                        except (ValueError, TypeError):
                            continue
        else:
            raise FileNotFoundError(
                f"No valid input found! Checked cluster_counts_file ({counts_path}), "
                f"raw_clusters_file ({raw_path}), and {len(hdf_paths)} HDF paths."
            )

        num_clusters = len(cluster_counts)
        total_frames = sum(cluster_counts.values())
        total_hours = total_frames * self.frame_shift_sec / 3600.0

        # Sort descending (most frequent to least frequent)
        sorted_clusters = cluster_counts.most_common()

        # Entropy & perplexity
        entropy = 0.0
        for _, cnt in sorted_clusters:
            if total_frames > 0:
                p = cnt / total_frames
                entropy -= p * math.log2(p)
        perplexity = 2.0 ** entropy
        k_val = self.k_clusters if self.k_clusters is not None else num_clusters
        utilization_pct = (perplexity / k_val * 100.0) if k_val > 0 else 0.0

        # Gini coefficient of inequality in cluster assignment
        # G = 2 * sum(i * y_i) / (n * sum(y_i)) - (n + 1) / n
        y = np.array(sorted([c for _, c in sorted_clusters]))
        n = len(y)
        if n > 0 and y.sum() > 0:
            index = np.arange(1, n + 1)
            gini = (2.0 * np.sum(index * y) / (n * np.sum(y))) - ((n + 1.0) / n)
        else:
            gini = 0.0

        logging.info(
            f"Analyzed {num_clusters} clusters across {total_frames:,} total frames ({total_hours:.1f} hours). "
            f"Entropy: {entropy:.2f} bits, Perplexity: {perplexity:.1f} / {k_val} ({utilization_pct:.1f}% utilization), "
            f"Gini coefficient: {gini:.3f}."
        )

        # -------------------------------------------------------------
        # Generate Plots using Matplotlib
        # -------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Bar Chart: Cluster Frame Counts (Descending: Top-K & Full Distribution)
        self._plot_cluster_distribution(
            sorted_clusters=sorted_clusters,
            total_frames=total_frames,
            total_hours=total_hours,
            k_val=k_val,
            top_k=self.top_k,
            out_path=self.out_distribution_plot.get_path(),
        )

        # 2. Cumulative Coverage Curve
        self._plot_cumulative_curve(
            sorted_clusters=sorted_clusters,
            total_frames=total_frames,
            out_path=self.out_cumulative_plot.get_path(),
        )

        # -------------------------------------------------------------
        # Export cluster_counts.txt
        # -------------------------------------------------------------
        with open(self.out_counts_txt.get_path(), "w", encoding="utf-8") as f:
            for c_id, cnt in sorted_clusters:
                f.write(f"{c_id} {cnt}\n")

        # -------------------------------------------------------------
        # JSON Output
        # -------------------------------------------------------------
        dist_json = {
            "title": self.title_prefix,
            "total_clusters": num_clusters,
            "configured_k": k_val,
            "total_frames": total_frames,
            "total_audio_hours": round(total_hours, 2),
            "frame_shift_sec": self.frame_shift_sec,
            "entropy_bits": round(entropy, 3),
            "perplexity": round(perplexity, 2),
            "cluster_utilization_pct": round(utilization_pct, 2),
            "gini_coefficient": round(float(gini), 3),
            "clusters_sorted_by_frequency": [
                {
                    "rank": rank,
                    "cluster_id": c_id,
                    "frames": cnt,
                    "audio_hours": round(cnt * self.frame_shift_sec / 3600.0, 3),
                    "percentage": round(cnt / total_frames * 100.0, 3) if total_frames > 0 else 0,
                }
                for rank, (c_id, cnt) in enumerate(sorted_clusters, 1)
            ],
        }

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as f:
            json.dump(dist_json, f, indent=2)

        # -------------------------------------------------------------
        # Text Report Output
        # -------------------------------------------------------------
        with open(self.out_report_txt.get_path(), "w", encoding="utf-8") as rf:
            rf.write("=" * 85 + "\n")
            rf.write("           DISCOVERED SPEECH CLUSTERS CORPUS DISTRIBUTION REPORT\n")
            rf.write("=" * 85 + "\n")
            rf.write(f"Analysis:              {self.title_prefix}\n")
            rf.write(f"Active Clusters:       {num_clusters} (configured K={k_val})\n")
            rf.write(f"Total Corpus Frames:   {total_frames:,}\n")
            rf.write(f"Total Audio Duration:  {total_hours:,.2f} hours (frame shift = {self.frame_shift_sec * 1000:.0f} ms)\n")
            rf.write(f"Cluster Entropy:       {entropy:.3f} bits (max = {math.log2(k_val):.3f} bits)\n")
            rf.write(f"Perplexity:            {perplexity:.1f} ({utilization_pct:.1f}% of K)\n")
            rf.write(f"Gini Coefficient:      {gini:.3f} (0 = uniform, 1 = concentrated in 1 cluster)\n")
            rf.write("=" * 85 + "\n\n")

            rf.write(f"{'Rank':<6} {'Cluster ID':<12} {'Frames':<15} {'Hours':<12} {'% of Corpus':<15} {'Cumulative %':<15}\n")
            rf.write("-" * 85 + "\n")
            cum = 0
            for rank, (c_id, cnt) in enumerate(sorted_clusters, 1):
                cum += cnt
                pct = cnt / total_frames * 100.0 if total_frames > 0 else 0
                cum_pct = cum / total_frames * 100.0 if total_frames > 0 else 0
                hrs = cnt * self.frame_shift_sec / 3600.0
                rf.write(f"{rank:<6} {c_id:<12} {cnt:<15,d} {hrs:<12.2f} {pct:>10.3f}% {cum_pct:>13.2f}%\n")
            rf.write("=" * 85 + "\n")

        logging.info("Speech Cluster Distribution Analysis completed successfully.")

    def _plot_cluster_distribution(
        self,
        sorted_clusters: List[Tuple[int, int]],
        total_frames: int,
        total_hours: float,
        k_val: int,
        top_k: int,
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        display_k = min(len(sorted_clusters), max(top_k, 30))
        top_items = sorted_clusters[:display_k]
        c_ids = [f"c{item[0]}" for item in top_items]
        counts = [item[1] for item in top_items]
        pcts = [cnt / total_frames * 100.0 if total_frames > 0 else 0.0 for cnt in counts]

        # 2-row figure: Top-K zoomed view on top, Full-corpus sorted distribution on bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=200)

        # Panel 1: Top-K Clusters
        norm_vals = np.linspace(0.85, 0.45, len(counts))
        colors = [plt.cm.viridis(v) for v in norm_vals]
        bars = ax1.bar(range(display_k), counts, color=colors, edgecolor="#222222", linewidth=0.5, width=0.75)
        ax1.set_xlabel(f"Cluster ID (Top {display_k} / {len(sorted_clusters)} Ranked by Frame Count)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Corpus Frames", fontsize=10, fontweight="bold", color="#222222")
        ax1.set_title(
            f"Top {display_k} Most Frequent Clusters (Total Frames: {total_frames:,} | ~{total_hours:.1f} Hours)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xticks(range(display_k))
        ax1.set_xticklabels(c_ids, rotation=70, ha="right", fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.35)

        # Dual y-axis for percentage
        ax1_pct = ax1.twinx()
        max_pct = max(pcts) if pcts else 1.0
        ax1_pct.set_ylim(0, max_pct * 1.15)
        ax1_pct.set_ylabel("% of Corpus Frames", fontsize=10, fontweight="bold", color="#444444")
        ax1_pct.grid(False)

        # Annotate top 15 bars
        for i in range(min(15, len(bars))):
            b = bars[i]
            h = b.get_height()
            p = pcts[i]
            ax1.annotate(
                f"{p:.1f}%",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="semibold",
            )

        # Panel 2: Full Distribution across All Clusters
        all_counts = [item[1] for item in sorted_clusters]
        all_ranks = np.arange(1, len(all_counts) + 1)
        ax2.bar(all_ranks, all_counts, color="#3b528b", width=1.0, edgecolor="none", alpha=0.85)
        ax2.set_xlabel(f"Cluster Rank (1 to {len(all_counts)}, Most to Least Frequent)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Corpus Frames", fontsize=10, fontweight="bold")
        ax2.set_title(f"Full Corpus Cluster Distribution (All {len(all_counts)} Discovered Clusters)", fontsize=11, fontweight="bold")
        ax2.set_xlim(0, len(all_counts) + 1)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)

        fig.suptitle(f"{self.title_prefix} - Frame Assignments Across Corpus", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_cumulative_curve(
        self,
        sorted_clusters: List[Tuple[int, int]],
        total_frames: int,
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        counts = [item[1] for item in sorted_clusters]
        cum_counts = np.cumsum(counts)
        cum_percentages = (cum_counts / total_frames * 100.0) if total_frames > 0 else np.zeros_like(cum_counts)

        ranks = np.arange(1, len(counts) + 1)
        cluster_percentages = ranks / len(counts) * 100.0

        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)

        ax.plot(cluster_percentages, cum_percentages, color="#2b5c8f", linewidth=2.5, label="Cumulative Frame Coverage")
        ax.plot([0, 100], [0, 100], color="gray", linestyle=":", linewidth=1.2, label="Uniform Baseline (Gini = 0)")

        # Reference markers (50%, 80%, 90% coverage)
        for target_pct, color in [(50, "#e377c2"), (80, "#ff7f0e"), (90, "#2ca02c")]:
            idx = int(np.searchsorted(cum_percentages, target_pct))
            if idx < len(cluster_percentages):
                cl_pct = cluster_percentages[idx]
                ax.axhline(target_pct, color=color, linestyle="--", alpha=0.6, linewidth=1.0)
                ax.axvline(cl_pct, color=color, linestyle="--", alpha=0.6, linewidth=1.0)
                ax.plot(cl_pct, target_pct, marker="o", color=color, markersize=6)
                ax.annotate(
                    f"{target_pct}% frames: top {cl_pct:.1f}% clusters ({idx + 1}/{len(counts)})",
                    xy=(cl_pct, target_pct),
                    xytext=(cl_pct + 3, target_pct - 4),
                    fontsize=8,
                    fontweight="semibold",
                    color=color,
                )

        ax.set_xlabel("% of Discovered Clusters (Ranked Most to Least Frequent)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Corpus Frames (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{self.title_prefix} - Cumulative Frame Assignment Coverage Curve", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
