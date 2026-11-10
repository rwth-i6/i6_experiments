__all__ = [
    "ContrastiveEncoderStatePcaCallback",
    "ComputeBertScoreSimilarityMatrixJob",
    "compute_bertscore_similarity_matrix",
]

import json
import os
import sys
import textwrap
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from sisyphus import Job, Task, tk
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict

from .callback import (
    _avg_pairwise_cosine_similarity,
    _avg_pairwise_l2_distance,
    _fit_pca_2d,
    _project,
)


def _ensure_container_libs_on_path() -> None:
    """Ensure container_libs (where bert_score is installed) is on sys.path."""
    candidate_dirs = [
        "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs",
        "/u/benjamin.oyarzun/experiments/2026_05_07_first_experiments/container_libs",
    ]
    for d in candidate_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


def load_seq_tag_to_transcript(
    text_file: Optional[str],
    seq_tags_file: Optional[str],
    valid_seq_tags_file: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, int]]:
    """
    Loads parallel (seq_tags, transcripts) from LibriSpeech text and seq_tags files.
    Optionally filters to only seq_tags present in `valid_seq_tags_file`.

    Returns:
      - ordered_seq_tags: List[str]
      - ordered_transcripts: List[str]
      - seq_tag_to_transcript: Dict[str, str]
      - seq_tag_to_corpus_idx: Dict[str, int]
    """
    if not text_file or not seq_tags_file or not os.path.isfile(text_file) or not os.path.isfile(seq_tags_file):
        return [], [], {}, {}

    valid_set = None
    if valid_seq_tags_file and os.path.isfile(valid_seq_tags_file):
        with open(valid_seq_tags_file, "r", encoding="utf-8") as vf:
            valid_set = {line.strip() for line in vf if line.strip()}

    ordered_seq_tags: List[str] = []
    ordered_transcripts: List[str] = []
    seq_tag_to_transcript: Dict[str, str] = {}
    seq_tag_to_corpus_idx: Dict[str, int] = {}

    with open(text_file, "r", encoding="utf-8") as tf, open(seq_tags_file, "r", encoding="utf-8") as sf:
        for raw_idx, (t_line, s_line) in enumerate(zip(tf, sf)):
            tag = s_line.strip()
            txt = " ".join(t_line.strip().split())
            if not tag or not txt:
                continue
            seq_tag_to_transcript[tag] = txt
            seq_tag_to_corpus_idx[tag] = raw_idx
            if valid_set is not None and tag not in valid_set:
                continue
            ordered_seq_tags.append(tag)
            ordered_transcripts.append(txt)

    return ordered_seq_tags, ordered_transcripts, seq_tag_to_transcript, seq_tag_to_corpus_idx


def compute_bertscore_similarity_matrix(
    sentences: List[str],
    *,
    lang: str = "en",
    model_type: Optional[str] = None,
    idf: bool = True,
    rescale_with_baseline: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes the all-pairs N x N BERTScore (P, R, F1) similarity matrix for `sentences`
    using:
      - IDF importance weighting (`idf=True`, computed over `sentences` with `nthreads=0`)
      - Baseline rescaling (`rescale_with_baseline=True`)
      - Normalization to [0, 1]:
          * `f1_rescaled`: official baseline-rescaled BERTScore (F1 - b) / (1 - b), where
            1.0 = identical and 0.0 = random baseline (can be slightly < 0 for pairs even more
            dissimilar than random; used for strict tie-free ranking of most dissimilar pairs).
          * `f1_norm_0_1`: baseline-rescaled BERTScore clipped to [0.0, 1.0].
          * `f1_minmax_0_1`: baseline-rescaled BERTScore linearly mapped onto [0.0, 1.0] from
            [min(F1_rescaled), 1.0].
          * `f1_raw`: unrescaled raw BERTScore F1 in [0, 1].
    """
    _ensure_container_libs_on_path()
    import torch
    import pandas as pd
    from bert_score.utils import (
        get_bert_embedding,
        get_idf_dict,
        get_model,
        get_tokenizer,
        lang2model,
        model2layers,
    )

    n = len(sentences)
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {
            "f1_norm_0_1": empty,
            "f1_rescaled": empty,
            "f1_minmax_0_1": empty,
            "f1_raw": empty,
            "precision_rescaled": empty,
            "recall_rescaled": empty,
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type is None:
        model_type = lang2model[lang.lower()]
    num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast=False)
    model = get_model(model_type, num_layers, all_layers=False)
    model.to(device)
    model.eval()

    if idf:
        # Use nthreads=0 to avoid multiprocessing Pool deadlocks in Sisyphus/Slurm jobs
        idf_dict = get_idf_dict(sentences, tokenizer, nthreads=0)
    else:
        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[tokenizer.sep_token_id] = 0.0
        idf_dict[tokenizer.cls_token_id] = 0.0

    # Extract per-sentence token embeddings and IDF weights in O(N) forward passes
    embs_list: List[torch.Tensor] = []
    idfs_list: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_sents = sentences[start : start + batch_size]
            embs, masks, padded_idf = get_bert_embedding(
                batch_sents,
                model,
                tokenizer,
                idf_dict,
                batch_size=len(batch_sents),
                device=device,
                all_layers=False,
            )
            # Normalize token embeddings to unit L2 norm
            norms = torch.norm(embs, dim=-1, keepdim=True).clamp(min=1e-12)
            embs = (embs / norms).cpu()
            masks = masks.cpu()
            padded_idf = padded_idf.cpu()

            for i in range(len(batch_sents)):
                seq_len = int(masks[i].sum().item())
                e_i = embs[i, :seq_len]
                w_i = padded_idf[i, :seq_len].float()
                w_sum = w_i.sum().clamp(min=1e-12)
                w_i = w_i / w_sum
                embs_list.append(e_i)
                idfs_list.append(w_i)

    # Pad all N sentences into (N, L_max, D) on device for fast vectorized all-pairs greedy_cos_idf
    lens = [e.size(0) for e in embs_list]
    max_len = max(lens)
    dim = embs_list[0].size(-1)

    E_pad = torch.zeros((n, max_len, dim), dtype=torch.float32, device=device)
    W_pad = torch.zeros((n, max_len), dtype=torch.float32, device=device)
    M_pad = torch.zeros((n, max_len), dtype=torch.float32, device=device)

    for i, (e_i, w_i, l_i) in enumerate(zip(embs_list, idfs_list, lens)):
        E_pad[i, :l_i] = e_i.to(device)
        W_pad[i, :l_i] = w_i.to(device)
        M_pad[i, :l_i] = 1.0

    P_mat = torch.zeros((n, n), dtype=torch.float32, device=device)
    R_mat = torch.zeros((n, n), dtype=torch.float32, device=device)
    F_mat = torch.zeros((n, n), dtype=torch.float32, device=device)

    row_chunk = max(1, min(32, 4096 // max(1, max_len)))
    with torch.no_grad():
        for i_start in range(0, n, row_chunk):
            i_end = min(n, i_start + row_chunk)
            E_cand = E_pad[i_start:i_end]  # [B, L, D]
            W_cand = W_pad[i_start:i_end]  # [B, L]
            M_cand = M_pad[i_start:i_end]  # [B, L]

            # Pairwise token cosine similarity: [B, N, L_cand, L_ref]
            sim = torch.einsum("bld,nmd->bnlm", E_cand, E_pad)
            pair_mask = M_cand[:, None, :, None] * M_pad[None, :, None, :]
            sim = sim * pair_mask

            # Greedy matching (exact match with bert_score.utils.greedy_cos_idf)
            word_precision = sim.max(dim=3)[0]  # [B, N, L_cand]
            word_recall = sim.max(dim=2)[0]     # [B, N, L_ref]

            P_chunk = (word_precision * W_cand[:, None, :]).sum(dim=2)  # [B, N]
            R_chunk = (word_recall * W_pad[None, :, :]).sum(dim=2)      # [B, N]
            F_chunk = 2.0 * P_chunk * R_chunk / (P_chunk + R_chunk).clamp(min=1e-12)

            P_mat[i_start:i_end] = P_chunk
            R_mat[i_start:i_end] = R_chunk
            F_mat[i_start:i_end] = F_chunk

    P_raw = P_mat.cpu().numpy()
    R_raw = R_mat.cpu().numpy()
    F_raw = F_mat.cpu().numpy()
    np.fill_diagonal(P_raw, 1.0)
    np.fill_diagonal(R_raw, 1.0)
    np.fill_diagonal(F_raw, 1.0)

    if rescale_with_baseline:
        import bert_score

        baseline_path = os.path.join(
            os.path.dirname(bert_score.__file__),
            f"rescale_baseline/{lang.lower()}/{model_type}.tsv",
        )
        if os.path.isfile(baseline_path):
            baselines = pd.read_csv(baseline_path).iloc[num_layers].to_numpy()[1:].astype(np.float32)
            b_p, b_r, b_f = float(baselines[0]), float(baselines[1]), float(baselines[2])
        else:
            # Default roberta-large layer 17 English baselines
            b_p, b_r, b_f = 0.8315159, 0.8315159, 0.8315159

        P_rescaled = (P_raw - b_p) / max(1.0 - b_p, 1e-12)
        R_rescaled = (R_raw - b_r) / max(1.0 - b_r, 1e-12)
        F_rescaled = (F_raw - b_f) / max(1.0 - b_f, 1e-12)
    else:
        P_rescaled = P_raw.copy()
        R_rescaled = R_raw.copy()
        F_rescaled = F_raw.copy()

    # Symmetrize F1 matrix (F1 is theoretically symmetric, ensures numerical exactness)
    F_rescaled = 0.5 * (F_rescaled + F_rescaled.T)
    np.fill_diagonal(F_rescaled, 1.0)

    # Normalized [0, 1] matrices:
    # 1) Clipped baseline-rescaled BERTScore in [0, 1]
    F_norm_0_1 = np.clip(F_rescaled, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(F_norm_0_1, 1.0)

    # 2) Min-max [0, 1] rescaled BERTScore (preserves strict ordering below 0)
    f_min = float(np.min(F_rescaled))
    if f_min < 0.0 and (1.0 - f_min) > 1e-12:
        F_minmax_0_1 = ((F_rescaled - f_min) / (1.0 - f_min)).astype(np.float32)
    else:
        F_minmax_0_1 = F_norm_0_1.copy()
    np.fill_diagonal(F_minmax_0_1, 1.0)

    return {
        "f1_norm_0_1": F_norm_0_1,
        "f1_rescaled": F_rescaled.astype(np.float32),
        "f1_minmax_0_1": F_minmax_0_1,
        "f1_raw": F_raw.astype(np.float32),
        "precision_rescaled": P_rescaled.astype(np.float32),
        "recall_rescaled": R_rescaled.astype(np.float32),
    }


def extract_bertscore_extremal_pairs(
    seq_tags: List[str],
    transcripts: List[str],
    corpus_indices: Dict[str, int],
    bert_matrices: Dict[str, np.ndarray],
    top_k: int = 25,
) -> Dict[str, Any]:
    """
    Given the BERTScore similarity matrices over `seq_tags`, extracts:
      - `per_seq_extremes`: for each sequence i, its most dissimilar (least similar) and most similar
        partner sequence j != i, with matrix index, corpus index, seq_tag, transcript, and scores.
      - `global_least_similar_pairs`: top-K least similar pairs (i < j) sorted ascending by score.
      - `global_most_similar_pairs`: top-K most similar pairs (i < j) sorted descending by score.
    """
    n = len(seq_tags)
    f_rescaled = bert_matrices["f1_rescaled"]
    f_norm_01 = bert_matrices["f1_norm_0_1"]
    f_minmax_01 = bert_matrices["f1_minmax_0_1"]
    f_raw = bert_matrices["f1_raw"]

    per_seq_extremes: Dict[str, Dict[str, Any]] = {}
    all_pairs: List[Dict[str, Any]] = []

    for i in range(n):
        tag_i = seq_tags[i]
        txt_i = transcripts[i]
        c_idx_i = corpus_indices.get(tag_i, i)

        if n > 1:
            row_scores = f_rescaled[i].copy()
            row_scores[i] = np.inf
            min_j = int(np.argmin(row_scores))

            row_scores_max = f_rescaled[i].copy()
            row_scores_max[i] = -np.inf
            max_j = int(np.argmax(row_scores_max))

            per_seq_extremes[tag_i] = {
                "seq_tag": tag_i,
                "matrix_index": i,
                "corpus_index": c_idx_i,
                "transcript": txt_i,
                "most_dissimilar": {
                    "seq_tag": seq_tags[min_j],
                    "matrix_index": min_j,
                    "corpus_index": corpus_indices.get(seq_tags[min_j], min_j),
                    "transcript": transcripts[min_j],
                    "bertscore_f1_norm_0_1": float(f_norm_01[i, min_j]),
                    "bertscore_f1_minmax_0_1": float(f_minmax_01[i, min_j]),
                    "bertscore_f1_rescaled": float(f_rescaled[i, min_j]),
                    "bertscore_f1_raw": float(f_raw[i, min_j]),
                },
                "most_similar": {
                    "seq_tag": seq_tags[max_j],
                    "matrix_index": max_j,
                    "corpus_index": corpus_indices.get(seq_tags[max_j], max_j),
                    "transcript": transcripts[max_j],
                    "bertscore_f1_norm_0_1": float(f_norm_01[i, max_j]),
                    "bertscore_f1_minmax_0_1": float(f_minmax_01[i, max_j]),
                    "bertscore_f1_rescaled": float(f_rescaled[i, max_j]),
                    "bertscore_f1_raw": float(f_raw[i, max_j]),
                },
            }

        for j in range(i + 1, n):
            all_pairs.append(
                {
                    "seq_tag_1": tag_i,
                    "seq_tag_2": seq_tags[j],
                    "matrix_index_1": i,
                    "matrix_index_2": j,
                    "corpus_index_1": c_idx_i,
                    "corpus_index_2": corpus_indices.get(seq_tags[j], j),
                    "transcript_1": txt_i,
                    "transcript_2": transcripts[j],
                    "bertscore_f1_norm_0_1": float(f_norm_01[i, j]),
                    "bertscore_f1_minmax_0_1": float(f_minmax_01[i, j]),
                    "bertscore_f1_rescaled": float(f_rescaled[i, j]),
                    "bertscore_f1_raw": float(f_raw[i, j]),
                }
            )

    least_similar_pairs = sorted(all_pairs, key=lambda x: x["bertscore_f1_rescaled"])[:top_k]
    most_similar_pairs = sorted(all_pairs, key=lambda x: x["bertscore_f1_rescaled"], reverse=True)[:top_k]

    return {
        "num_sequences": n,
        "per_seq_extremes": per_seq_extremes,
        "global_least_similar_pairs": least_similar_pairs,
        "global_most_similar_pairs": most_similar_pairs,
    }


class ComputeBertScoreSimilarityMatrixJob(Job):
    """
    Sisyphus job that computes the pairwise BERTScore similarity matrix over LibriSpeech
    transcriptions (with IDF importance weighting and baseline rescaling normalized to [0, 1]),
    and retrieves the indices, sequence tags, and transcripts of the most similar and least
    similar (most dissimilar) audio-text pairs.
    """

    def __init__(
        self,
        text_file: tk.Path,
        seq_tags_file: tk.Path,
        valid_seq_tags_file: Optional[tk.Path] = None,
        max_seqs: Optional[int] = 200,
        priority_seq_tags: Optional[List[str]] = None,
        lang: str = "en",
        top_k_pairs: int = 30,
    ):
        super().__init__()
        self.text_file = text_file
        self.seq_tags_file = seq_tags_file
        self.valid_seq_tags_file = valid_seq_tags_file
        self.max_seqs = max_seqs
        self.priority_seq_tags = priority_seq_tags
        self.lang = lang
        self.top_k_pairs = top_k_pairs

        self.out_similarity_matrix_npy = self.output_path("similarity_matrix.npy")
        self.out_similarity_rescaled_npy = self.output_path("similarity_matrix_rescaled.npy")
        self.out_similarity_report_json = self.output_path("bertscore_similarity_report.json")
        self.out_most_similar_txt = self.output_path("most_similar_pairs.txt")
        self.out_least_similar_txt = self.output_path("least_similar_pairs.txt")
        self.out_heatmap_png = self.output_path("bertscore_similarity_heatmap.png")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ordered_tags, ordered_txts, tag_to_txt, tag_to_cidx = load_seq_tag_to_transcript(
            text_file=self.text_file.get_path(),
            seq_tags_file=self.seq_tags_file.get_path(),
            valid_seq_tags_file=self.valid_seq_tags_file.get_path() if self.valid_seq_tags_file else None,
        )

        # Ensure priority_seq_tags (e.g. the 20 standard PCA sequences) are placed first
        selected_tags: List[str] = []
        selected_set = set()
        if self.priority_seq_tags:
            for t in self.priority_seq_tags:
                if t in tag_to_txt and t not in selected_set:
                    selected_tags.append(t)
                    selected_set.add(t)

        for t in ordered_tags:
            if self.max_seqs is not None and len(selected_tags) >= self.max_seqs:
                break
            if t not in selected_set:
                selected_tags.append(t)
                selected_set.add(t)

        selected_txts = [tag_to_txt[t] for t in selected_tags]

        mats = compute_bertscore_similarity_matrix(
            selected_txts,
            lang=self.lang,
            idf=True,
            rescale_with_baseline=True,
        )

        np.save(self.out_similarity_matrix_npy.get_path(), mats["f1_norm_0_1"])
        np.save(self.out_similarity_rescaled_npy.get_path(), mats["f1_rescaled"])

        extremes = extract_bertscore_extremal_pairs(
            seq_tags=selected_tags,
            transcripts=selected_txts,
            corpus_indices=tag_to_cidx,
            bert_matrices=mats,
            top_k=self.top_k_pairs,
        )

        report_data = {
            "model": "roberta-large (lang=en, layer=17, idf=True, rescale_with_baseline=True)",
            "num_sequences": len(selected_tags),
            "seq_tags": selected_tags,
            "corpus_indices": [tag_to_cidx.get(t, i) for i, t in enumerate(selected_tags)],
            "transcripts": {t: tag_to_txt[t] for t in selected_tags},
            "per_seq_extremes": extremes["per_seq_extremes"],
            "global_most_similar_pairs": extremes["global_most_similar_pairs"],
            "global_least_similar_pairs": extremes["global_least_similar_pairs"],
        }
        with open(self.out_similarity_report_json.get_path(), "w", encoding="utf-8") as jf:
            json.dump(report_data, jf, indent=2)

        # Write human-readable tables for most similar and least similar pairs
        self._write_pairs_table(
            self.out_most_similar_txt.get_path(),
            title="MOST SIMILAR AUDIO-TEXT PAIRS (BY BERTSCORE WITH IDF & BASELINE RESCALING)",
            pairs=extremes["global_most_similar_pairs"],
        )
        self._write_pairs_table(
            self.out_least_similar_txt.get_path(),
            title="LEAST SIMILAR / MOST DISSIMILAR AUDIO-TEXT PAIRS (BY BERTSCORE WITH IDF & BASELINE RESCALING)",
            pairs=extremes["global_least_similar_pairs"],
        )

        # Plot heatmap of the top 30 sequences
        n_plot = min(30, len(selected_tags))
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(mats["f1_minmax_0_1"][:n_plot, :n_plot], cmap="viridis", vmin=0.0, vmax=1.0)
        short_labels = [f"[{tag_to_cidx.get(t, i)}] {t.split('/')[-1]}" for i, t in enumerate(selected_tags[:n_plot])]
        ax.set_xticks(np.arange(n_plot))
        ax.set_yticks(np.arange(n_plot))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(
            f"BERTScore Similarity Matrix (IDF + Baseline Rescaled in [0, 1], Top {n_plot} Sequences)"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized BERTScore F1 [0, 1]")
        fig.tight_layout()
        fig.savefig(self.out_heatmap_png.get_path(), dpi=200)
        plt.close(fig)

    @staticmethod
    def _write_pairs_table(path: str, title: str, pairs: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 110 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 110 + "\n")
            for rank, p in enumerate(pairs, 1):
                f.write(
                    f"Rank #{rank:02d} | BERTScore [0,1] (clipped)={p['bertscore_f1_norm_0_1']:.4f} | "
                    f"BERTScore [0,1] (min-max)={p['bertscore_f1_minmax_0_1']:.4f} | "
                    f"Rescaled={p['bertscore_f1_rescaled']:.4f} | Raw F1={p['bertscore_f1_raw']:.4f}\n"
                )
                f.write(
                    f"  Seq 1 (matrix_idx={p['matrix_index_1']}, corpus_idx={p['corpus_index_1']}): "
                    f"{p['seq_tag_1']}\n"
                    f"    Transcript 1: \"{p['transcript_1']}\"\n"
                )
                f.write(
                    f"  Seq 2 (matrix_idx={p['matrix_index_2']}, corpus_idx={p['corpus_index_2']}): "
                    f"{p['seq_tag_2']}\n"
                    f"    Transcript 2: \"{p['transcript_2']}\"\n"
                )
                f.write("-" * 110 + "\n")


class ContrastiveEncoderStatePcaCallback(ForwardCallbackIface):
    """
    Forward callback for the `pca-contrastive` shared-encoder state PCA analysis.

    For each plotted sequence `seq_tag_1` (Audio 1 + Text 1), retrieves its most dissimilar
    sequence `seq_tag_2` (Audio 2 + Text 2) with respect to BERTScore (IDF-weighted, baseline-rescaled),
    and projects all 4 sequences into the shared 2D PCA space on a single plot:
      - Audio 1 (Anchor):          Dark Blue (`#084594`)
      - Audio 2 (Most Dissimilar): Cyan      (`#00bcd4`)
      - Text 1  (Anchor):          Orange    (`#e6550d`)
      - Text 2  (Most Dissimilar): Yellow    (`#f1c40f`, with dark edge `#7f6000`)

    Also displays the reference textual transcriptions of both utterances directly on the plot,
    and writes per-pair `.npz` files, `summary.txt`, `bertscore_pairs_summary.txt`, and
    `cosine_similarities.txt`.
    """

    # Color palette:
    # Audio colors close to each other: Dark Blue & Cyan
    COLOR_AUDIO_1 = "#084594"
    COLOR_AUDIO_2 = "#00bcd4"
    # Text colors close to each other: Orange & Yellow
    COLOR_TEXT_1 = "#e6550d"
    COLOR_TEXT_2 = "#f1c40f"
    EDGE_TEXT_2 = "#7f6000"

    def __init__(
        self,
        *,
        out_dir: str = "pca_contrastive",
        max_points_per_modality: int = 50_000,
        plot_seq_tags: Optional[List[str]] = None,
        max_plotted_seqs: int = 20,
        cosine_similarity_summary: bool = True,
        transcript_text_file: Optional[Any] = None,
        transcript_seq_tags_file: Optional[Any] = None,
        bertscore_report_json: Optional[Any] = None,
        audio_output: str = "audio_states",
        text_output: str = "text_states",
        vocab=None,
    ):
        self.out_dir = out_dir
        self.max_points_per_modality = max_points_per_modality
        self.plot_seq_tags = set(plot_seq_tags) if plot_seq_tags is not None else None
        self.max_plotted_seqs = max_plotted_seqs
        self.cosine_similarity_summary = cosine_similarity_summary
        self.transcript_text_file = (
            transcript_text_file.get_path() if hasattr(transcript_text_file, "get_path") else transcript_text_file
        )
        self.transcript_seq_tags_file = (
            transcript_seq_tags_file.get_path()
            if hasattr(transcript_seq_tags_file, "get_path")
            else transcript_seq_tags_file
        )
        self.bertscore_report_json = (
            bertscore_report_json.get_path()
            if hasattr(bertscore_report_json, "get_path")
            else bertscore_report_json
        )
        self.audio_output = audio_output
        self.text_output = text_output

        self._anchor_selected: List[str] = []
        self._anchor_selected_set: set = set()

        # Capped pools used for fitting the shared 2D PCA
        self._audio_pool: List[np.ndarray] = []
        self._text_pool: List[np.ndarray] = []
        self._audio_pool_size = 0
        self._text_pool_size = 0

        # Keep states for all candidate sequences so any most-dissimilar partner is available
        self._audio_seqs: Dict[str, np.ndarray] = {}
        self._text_seqs: Dict[str, np.ndarray] = {}

    def init(self, *args, **kwargs):
        os.makedirs(self.out_dir, exist_ok=True)

    def _add_to_pool(self, states: np.ndarray, *, audio: bool) -> None:
        pool = self._audio_pool if audio else self._text_pool
        size = self._audio_pool_size if audio else self._text_pool_size
        remaining = self.max_points_per_modality - size
        if remaining <= 0:
            return
        if states.shape[0] > remaining:
            states = states[:remaining]
        pool.append(states)
        if audio:
            self._audio_pool_size += states.shape[0]
        else:
            self._text_pool_size += states.shape[0]

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        if self.plot_seq_tags is not None:
            if seq_tag in self.plot_seq_tags and seq_tag not in self._anchor_selected_set:
                self._anchor_selected.append(seq_tag)
                self._anchor_selected_set.add(seq_tag)
        else:
            if seq_tag not in self._anchor_selected_set and len(self._anchor_selected) < self.max_plotted_seqs:
                self._anchor_selected.append(seq_tag)
                self._anchor_selected_set.add(seq_tag)

        if self.audio_output in outputs.data:
            audio = np.asarray(outputs[self.audio_output].raw_tensor, dtype=np.float32)
            if audio.shape[0] > 0:
                self._add_to_pool(audio, audio=True)
                self._audio_seqs[seq_tag] = audio
        if self.text_output in outputs.data:
            text = np.asarray(outputs[self.text_output].raw_tensor, dtype=np.float32)
            if text.shape[0] > 0:
                self._add_to_pool(text, audio=False)
                self._text_seqs[seq_tag] = text

    def finish(self, **kwargs):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._audio_pool or not self._text_pool:
            raise ValueError(
                "Need encoder states from both audio and text to fit the shared PCA"
                f" (audio_points={self._audio_pool_size}, text_points={self._text_pool_size})."
            )

        mean, components = _fit_pca_2d(np.concatenate(self._audio_pool + self._text_pool, axis=0))

        all_paired_tags = sorted(set(self._audio_seqs) & set(self._text_seqs))
        anchor_tags = [t for t in sorted(self._anchor_selected) if t in self._audio_seqs and t in self._text_seqs]
        if not anchor_tags:
            anchor_tags = all_paired_tags[: self.max_plotted_seqs]

        # Load reference transcripts
        _, _, tag_to_txt, tag_to_cidx = load_seq_tag_to_transcript(
            self.transcript_text_file,
            self.transcript_seq_tags_file,
        )

        # Load precomputed BERTScore report if available, and also compute exact BERTScore over the
        # active candidate pool (all anchor sequences + precomputed extremes)
        precomputed_report = None
        if self.bertscore_report_json and os.path.isfile(self.bertscore_report_json):
            try:
                with open(self.bertscore_report_json, "r", encoding="utf-8") as f:
                    precomputed_report = json.load(f)
                if "transcripts" in precomputed_report:
                    for k, v in precomputed_report["transcripts"].items():
                        tag_to_txt.setdefault(k, v)
            except Exception:
                precomputed_report = None

        # Build the pool of sequences over which we evaluate BERTScore similarity
        candidate_tags: List[str] = []
        candidate_set = set()
        for t in anchor_tags:
            if t not in candidate_set:
                candidate_tags.append(t)
                candidate_set.add(t)

        if precomputed_report and "seq_tags" in precomputed_report:
            for t in precomputed_report["seq_tags"]:
                if t in self._audio_seqs and t in self._text_seqs and t not in candidate_set:
                    candidate_tags.append(t)
                    candidate_set.add(t)

        # Fill up to at least 100 candidate sequences from all_paired_tags for rich contrastive matching
        for t in all_paired_tags:
            if len(candidate_tags) >= 100:
                break
            if t not in candidate_set:
                candidate_tags.append(t)
                candidate_set.add(t)

        candidate_txts = [tag_to_txt.get(t, t) for t in candidate_tags]
        bert_mats = compute_bertscore_similarity_matrix(
            candidate_txts,
            lang="en",
            idf=True,
            rescale_with_baseline=True,
        )
        extremes = extract_bertscore_extremal_pairs(
            seq_tags=candidate_tags,
            transcripts=candidate_txts,
            corpus_indices=tag_to_cidx,
            bert_matrices=bert_mats,
            top_k=25,
        )

        # Save the BERTScore similarity matrix and extremal pairs report inside out_dir as well
        np.save(os.path.join(self.out_dir, "bertscore_similarity_matrix.npy"), bert_mats["f1_norm_0_1"])
        with open(os.path.join(self.out_dir, "bertscore_similarity_report.json"), "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "candidate_seq_tags": candidate_tags,
                    "per_seq_extremes": extremes["per_seq_extremes"],
                    "global_most_similar_pairs": extremes["global_most_similar_pairs"],
                    "global_least_similar_pairs": extremes["global_least_similar_pairs"],
                },
                jf,
                indent=2,
            )

        ComputeBertScoreSimilarityMatrixJob._write_pairs_table(
            os.path.join(self.out_dir, "most_similar_pairs.txt"),
            title="MOST SIMILAR AUDIO-TEXT PAIRS (BERTSCORE IDF + BASELINE RESCALED)",
            pairs=extremes["global_most_similar_pairs"],
        )
        ComputeBertScoreSimilarityMatrixJob._write_pairs_table(
            os.path.join(self.out_dir, "least_similar_pairs.txt"),
            title="LEAST SIMILAR / MOST DISSIMILAR AUDIO-TEXT PAIRS (BERTSCORE IDF + BASELINE RESCALED)",
            pairs=extremes["global_least_similar_pairs"],
        )

        # Also compute the 20x20 similarity matrix specifically among `anchor_tags` so every anchor
        # sequence has both its global most-dissimilar partner and its within-plotted-set partner
        anchor_txts = [tag_to_txt.get(t, t) for t in anchor_tags]
        anchor_mats = compute_bertscore_similarity_matrix(
            anchor_txts,
            lang="en",
            idf=True,
            rescale_with_baseline=True,
        )
        anchor_extremes = extract_bertscore_extremal_pairs(
            seq_tags=anchor_tags,
            transcripts=anchor_txts,
            corpus_indices=tag_to_cidx,
            bert_matrices=anchor_mats,
            top_k=20,
        )

        cosine_similarities: Dict[str, Dict[str, float]] = {}

        with open(os.path.join(self.out_dir, "summary.txt"), "w", encoding="utf-8") as summary:
            summary.write(f"pooled_audio_points={self._audio_pool_size}\n")
            summary.write(f"pooled_text_points={self._text_pool_size}\n")
            summary.write(f"num_anchor_seqs={len(anchor_tags)}\n")
            summary.write(f"num_bertscore_candidate_seqs={len(candidate_tags)}\n\n")

            for seq_tag_1 in anchor_tags:
                ext_info = extremes["per_seq_extremes"].get(seq_tag_1)
                if ext_info is None:
                    ext_info = anchor_extremes["per_seq_extremes"].get(seq_tag_1)
                if ext_info is None:
                    continue

                dissim_info = ext_info["most_dissimilar"]
                sim_info = ext_info["most_similar"]

                seq_tag_2 = dissim_info["seq_tag"]
                seq_tag_sim = sim_info["seq_tag"]

                audio_1 = self._audio_seqs[seq_tag_1]
                text_1 = self._text_seqs[seq_tag_1]
                audio_2 = self._audio_seqs[seq_tag_2]
                text_2 = self._text_seqs[seq_tag_2]
                audio_sim = self._audio_seqs.get(seq_tag_sim, audio_1)
                text_sim = self._text_seqs.get(seq_tag_sim, text_1)

                audio_1_coords = _project(audio_1, mean, components)
                text_1_coords = _project(text_1, mean, components)
                audio_2_coords = _project(audio_2, mean, components)
                text_2_coords = _project(text_2, mean, components)
                audio_sim_coords = _project(audio_sim, mean, components)
                text_sim_coords = _project(text_sim, mean, components)

                transcript_1 = tag_to_txt.get(seq_tag_1, "(transcript unavailable)")
                transcript_2 = tag_to_txt.get(seq_tag_2, "(transcript unavailable)")
                transcript_sim = tag_to_txt.get(seq_tag_sim, "(transcript unavailable)")
                c_idx_1 = tag_to_cidx.get(seq_tag_1, -1)
                c_idx_2 = tag_to_cidx.get(seq_tag_2, -1)
                c_idx_sim = tag_to_cidx.get(seq_tag_sim, -1)

                b_norm_01 = dissim_info["bertscore_f1_norm_0_1"]
                b_minmax_01 = dissim_info["bertscore_f1_minmax_0_1"]
                b_rescaled = dissim_info["bertscore_f1_rescaled"]
                b_raw = dissim_info["bertscore_f1_raw"]

                sim_b_norm_01 = sim_info["bertscore_f1_norm_0_1"]
                sim_b_minmax_01 = sim_info["bertscore_f1_minmax_0_1"]
                sim_b_rescaled = sim_info["bertscore_f1_rescaled"]
                sim_b_raw = sim_info["bertscore_f1_raw"]

                cos = {
                    "a1_t1": _avg_pairwise_cosine_similarity(audio_1, text_1),
                    "a2_t2": _avg_pairwise_cosine_similarity(audio_2, text_2),
                    "a1_a2": _avg_pairwise_cosine_similarity(audio_1, audio_2),
                    "t1_t2": _avg_pairwise_cosine_similarity(text_1, text_2),
                    "a1_t2": _avg_pairwise_cosine_similarity(audio_1, text_2),
                    "a2_t1": _avg_pairwise_cosine_similarity(audio_2, text_1),
                    "a1_a1": _avg_pairwise_cosine_similarity(audio_1),
                    "t1_t1": _avg_pairwise_cosine_similarity(text_1),
                }
                dist = {
                    "a1_t1": _avg_pairwise_l2_distance(audio_1, text_1),
                    "a2_t2": _avg_pairwise_l2_distance(audio_2, text_2),
                    "a1_a2": _avg_pairwise_l2_distance(audio_1, audio_2),
                    "t1_t2": _avg_pairwise_l2_distance(text_1, text_2),
                    "a1_t2": _avg_pairwise_l2_distance(audio_1, text_2),
                    "a2_t1": _avg_pairwise_l2_distance(audio_2, text_1),
                }

                cos_sim = {
                    "a1_t1": cos["a1_t1"],
                    "asim_tsim": _avg_pairwise_cosine_similarity(audio_sim, text_sim),
                    "a1_asim": _avg_pairwise_cosine_similarity(audio_1, audio_sim),
                    "t1_tsim": _avg_pairwise_cosine_similarity(text_1, text_sim),
                    "a1_tsim": _avg_pairwise_cosine_similarity(audio_1, text_sim),
                    "asim_t1": _avg_pairwise_cosine_similarity(audio_sim, text_1),
                }
                dist_sim = {
                    "a1_t1": dist["a1_t1"],
                    "asim_tsim": _avg_pairwise_l2_distance(audio_sim, text_sim),
                    "a1_asim": _avg_pairwise_l2_distance(audio_1, audio_sim),
                    "t1_tsim": _avg_pairwise_l2_distance(text_1, text_sim),
                    "a1_tsim": _avg_pairwise_l2_distance(audio_1, text_sim),
                    "asim_t1": _avg_pairwise_l2_distance(audio_sim, text_1),
                }
                cosine_similarities[seq_tag_1] = cos

                safe_tag_1 = seq_tag_1.replace("/", "_")
                safe_tag_2 = seq_tag_2.replace("/", "_")
                safe_tag_sim = seq_tag_sim.replace("/", "_")

                npz_data = dict(
                    seq_tag_1=seq_tag_1,
                    seq_tag_2=seq_tag_2,
                    seq_tag_sim=seq_tag_sim,
                    corpus_index_1=c_idx_1,
                    corpus_index_2=c_idx_2,
                    corpus_index_sim=c_idx_sim,
                    transcript_1=transcript_1,
                    transcript_2=transcript_2,
                    transcript_sim=transcript_sim,
                    bertscore_f1_norm_0_1=b_norm_01,
                    bertscore_f1_minmax_0_1=b_minmax_01,
                    bertscore_f1_rescaled=b_rescaled,
                    bertscore_f1_raw=b_raw,
                    sim_bertscore_f1_norm_0_1=sim_b_norm_01,
                    sim_bertscore_f1_minmax_0_1=sim_b_minmax_01,
                    sim_bertscore_f1_rescaled=sim_b_rescaled,
                    sim_bertscore_f1_raw=sim_b_raw,
                    audio_1_coords=audio_1_coords,
                    text_1_coords=text_1_coords,
                    audio_2_coords=audio_2_coords,
                    text_2_coords=text_2_coords,
                    audio_sim_coords=audio_sim_coords,
                    text_sim_coords=text_sim_coords,
                    **{f"cos_{k}": v for k, v in cos.items()},
                    **{f"l2_{k}": v for k, v in dist.items()},
                )
                np.savez(os.path.join(self.out_dir, f"{safe_tag_1}.npz"), **npz_data)

                self._plot_contrastive_pca_figure(
                    out_png_path=os.path.join(self.out_dir, f"{safe_tag_1}.png"),
                    seq_tag_1=seq_tag_1,
                    seq_tag_2=seq_tag_2,
                    seq_tag_sim=seq_tag_sim,
                    c_idx_1=c_idx_1,
                    c_idx_2=c_idx_2,
                    c_idx_sim=c_idx_sim,
                    transcript_1=transcript_1,
                    transcript_2=transcript_2,
                    transcript_sim=transcript_sim,
                    audio_1_coords=audio_1_coords,
                    text_1_coords=text_1_coords,
                    audio_2_coords=audio_2_coords,
                    text_2_coords=text_2_coords,
                    audio_sim_coords=audio_sim_coords,
                    text_sim_coords=text_sim_coords,
                    b_norm_01=b_norm_01,
                    b_minmax_01=b_minmax_01,
                    b_rescaled=b_rescaled,
                    sim_b_norm_01=sim_b_norm_01,
                    sim_b_minmax_01=sim_b_minmax_01,
                    sim_b_rescaled=sim_b_rescaled,
                    cos=cos,
                    dist=dist,
                    cos_sim=cos_sim,
                    dist_sim=dist_sim,
                )

                summary.write(
                    f"seq_1={seq_tag_1} (idx={c_idx_1})\n"
                    f"  most_dissimilar={seq_tag_2} (idx={c_idx_2}) | "
                    f"bertscore_01={b_norm_01:.4f} (minmax={b_minmax_01:.4f}, rescaled={b_rescaled:.4f})\n"
                    f"  transcript_1: \"{transcript_1}\"\n"
                    f"  transcript_dissim: \"{transcript_2}\"\n"
                    f"  most_similar={seq_tag_sim} (idx={c_idx_sim}) | "
                    f"bertscore_01={sim_b_norm_01:.4f} (minmax={sim_b_minmax_01:.4f}, rescaled={sim_b_rescaled:.4f})\n"
                    f"  transcript_sim: \"{transcript_sim}\"\n\n"
                )

        if self.cosine_similarity_summary and cosine_similarities:
            self._write_contrastive_cosine_summary(cosine_similarities)

    def _plot_contrastive_pca_figure(
        self,
        *,
        out_png_path: str,
        seq_tag_1: str,
        seq_tag_2: str,
        seq_tag_sim: str,
        c_idx_1: int,
        c_idx_2: int,
        c_idx_sim: int,
        transcript_1: str,
        transcript_2: str,
        transcript_sim: str,
        audio_1_coords: np.ndarray,
        text_1_coords: np.ndarray,
        audio_2_coords: np.ndarray,
        text_2_coords: np.ndarray,
        audio_sim_coords: np.ndarray,
        text_sim_coords: np.ndarray,
        b_norm_01: float,
        b_minmax_01: float,
        b_rescaled: float,
        sim_b_norm_01: float,
        sim_b_minmax_01: float,
        sim_b_rescaled: float,
        cos: Dict[str, float],
        dist: Dict[str, float],
        cos_sim: Dict[str, float],
        dist_sim: Dict[str, float],
    ) -> None:
        import matplotlib.pyplot as plt

        fig, (ax_dissim, ax_sim) = plt.subplots(1, 2, figsize=(20, 9.5))

        short_1 = seq_tag_1.split("/")[-1]
        short_2 = seq_tag_2.split("/")[-1]
        short_sim = seq_tag_sim.split("/")[-1]

        # =========================================================================
        # SUBPLOT 1: Anchor vs. Most Dissimilar Pair (BERTScore Minimum)
        # =========================================================================
        # Audio 1 & Audio 2
        ax_dissim.scatter(
            audio_1_coords[:, 0],
            audio_1_coords[:, 1],
            s=26,
            alpha=0.80,
            c=self.COLOR_AUDIO_1,
            label=f"Audio 1 (Anchor: {short_1})",
            zorder=3,
        )
        if len(audio_1_coords) > 1:
            ax_dissim.plot(audio_1_coords[:, 0], audio_1_coords[:, 1], alpha=0.40, c=self.COLOR_AUDIO_1, linewidth=1.2)

        ax_dissim.scatter(
            audio_2_coords[:, 0],
            audio_2_coords[:, 1],
            s=26,
            alpha=0.80,
            c=self.COLOR_AUDIO_2,
            label=f"Audio 2 (Dissimilar: {short_2})",
            zorder=3,
        )
        if len(audio_2_coords) > 1:
            ax_dissim.plot(audio_2_coords[:, 0], audio_2_coords[:, 1], alpha=0.40, c=self.COLOR_AUDIO_2, linewidth=1.2)

        # Text 1 & Text 2
        ax_dissim.scatter(
            text_1_coords[:, 0],
            text_1_coords[:, 1],
            s=34,
            alpha=0.88,
            c=self.COLOR_TEXT_1,
            label=f"Text 1 (Anchor: {short_1})",
            zorder=4,
        )
        if len(text_1_coords) > 1:
            ax_dissim.plot(text_1_coords[:, 0], text_1_coords[:, 1], alpha=0.45, c=self.COLOR_TEXT_1, linewidth=1.3)

        ax_dissim.scatter(
            text_2_coords[:, 0],
            text_2_coords[:, 1],
            s=38,
            alpha=0.92,
            c=self.COLOR_TEXT_2,
            edgecolors=self.EDGE_TEXT_2,
            linewidths=0.6,
            label=f"Text 2 (Dissimilar: {short_2})",
            zorder=4,
        )
        if len(text_2_coords) > 1:
            ax_dissim.plot(text_2_coords[:, 0], text_2_coords[:, 1], alpha=0.55, c="#d4ac0d", linewidth=1.3)

        self._mark_start_end(ax_dissim, audio_1_coords, color=self.COLOR_AUDIO_1, prefix="Audio 1", show_legend=True)
        self._mark_start_end(ax_dissim, audio_2_coords, color=self.COLOR_AUDIO_2, prefix="Audio 2", show_legend=False)
        self._mark_start_end(ax_dissim, text_1_coords, color=self.COLOR_TEXT_1, prefix="Text 1", show_legend=False)
        self._mark_start_end(ax_dissim, text_2_coords, color=self.COLOR_TEXT_2, prefix="Text 2", show_legend=False)

        title_dissim = [
            f"(A) MOST DISSIMILAR: {short_1} (idx {c_idx_1}) vs. {short_2} (idx {c_idx_2})",
            f"BERTScore (IDF, Baseline-Rescaled): [0,1]-Clipped={b_norm_01:.3f} | MinMax={b_minmax_01:.3f} | Rescaled={b_rescaled:.3f}",
            f"Cos-Sim: a1-t1={cos['a1_t1']:.3f} a2-t2={cos['a2_t2']:.3f} | a1-a2={cos['a1_a2']:.3f} t1-t2={cos['t1_t2']:.3f} a1-t2={cos['a1_t2']:.3f}",
            f"L2-Dist: a1-t1={dist['a1_t1']:.2f} a2-t2={dist['a2_t2']:.2f} | a1-a2={dist['a1_a2']:.2f} t1-t2={dist['t1_t2']:.2f} a1-t2={dist['a1_t2']:.2f}",
        ]
        ax_dissim.set_title("\n".join(title_dissim), fontsize=9.0, pad=10)
        ax_dissim.set_xlabel("PC1")
        ax_dissim.set_ylabel("PC2")
        ax_dissim.grid(True, linestyle="--", alpha=0.25)
        ax_dissim.legend(loc="best", fontsize=8.0, framealpha=0.9)

        # =========================================================================
        # SUBPLOT 2: Anchor vs. Most Similar Pair (BERTScore Maximum)
        # =========================================================================
        # Audio 1 & Audio Sim
        ax_sim.scatter(
            audio_1_coords[:, 0],
            audio_1_coords[:, 1],
            s=26,
            alpha=0.80,
            c=self.COLOR_AUDIO_1,
            label=f"Audio 1 (Anchor: {short_1})",
            zorder=3,
        )
        if len(audio_1_coords) > 1:
            ax_sim.plot(audio_1_coords[:, 0], audio_1_coords[:, 1], alpha=0.40, c=self.COLOR_AUDIO_1, linewidth=1.2)

        ax_sim.scatter(
            audio_sim_coords[:, 0],
            audio_sim_coords[:, 1],
            s=26,
            alpha=0.80,
            c=self.COLOR_AUDIO_2,
            label=f"Audio Sim (Similar: {short_sim})",
            zorder=3,
        )
        if len(audio_sim_coords) > 1:
            ax_sim.plot(audio_sim_coords[:, 0], audio_sim_coords[:, 1], alpha=0.40, c=self.COLOR_AUDIO_2, linewidth=1.2)

        # Text 1 & Text Sim
        ax_sim.scatter(
            text_1_coords[:, 0],
            text_1_coords[:, 1],
            s=34,
            alpha=0.88,
            c=self.COLOR_TEXT_1,
            label=f"Text 1 (Anchor: {short_1})",
            zorder=4,
        )
        if len(text_1_coords) > 1:
            ax_sim.plot(text_1_coords[:, 0], text_1_coords[:, 1], alpha=0.45, c=self.COLOR_TEXT_1, linewidth=1.3)

        ax_sim.scatter(
            text_sim_coords[:, 0],
            text_sim_coords[:, 1],
            s=38,
            alpha=0.92,
            c=self.COLOR_TEXT_2,
            edgecolors=self.EDGE_TEXT_2,
            linewidths=0.6,
            label=f"Text Sim (Similar: {short_sim})",
            zorder=4,
        )
        if len(text_sim_coords) > 1:
            ax_sim.plot(text_sim_coords[:, 0], text_sim_coords[:, 1], alpha=0.55, c="#d4ac0d", linewidth=1.3)

        self._mark_start_end(ax_sim, audio_1_coords, color=self.COLOR_AUDIO_1, prefix="Audio 1", show_legend=True)
        self._mark_start_end(ax_sim, audio_sim_coords, color=self.COLOR_AUDIO_2, prefix="Audio Sim", show_legend=False)
        self._mark_start_end(ax_sim, text_1_coords, color=self.COLOR_TEXT_1, prefix="Text 1", show_legend=False)
        self._mark_start_end(ax_sim, text_sim_coords, color=self.COLOR_TEXT_2, prefix="Text Sim", show_legend=False)

        title_sim = [
            f"(B) MOST SIMILAR: {short_1} (idx {c_idx_1}) vs. {short_sim} (idx {c_idx_sim})",
            f"BERTScore (IDF, Baseline-Rescaled): [0,1]-Clipped={sim_b_norm_01:.3f} | MinMax={sim_b_minmax_01:.3f} | Rescaled={sim_b_rescaled:.3f}",
            f"Cos-Sim: a1-t1={cos_sim['a1_t1']:.3f} as-ts={cos_sim['asim_tsim']:.3f} | a1-as={cos_sim['a1_asim']:.3f} t1-ts={cos_sim['t1_tsim']:.3f} a1-ts={cos_sim['a1_tsim']:.3f}",
            f"L2-Dist: a1-t1={dist_sim['a1_t1']:.2f} as-ts={dist_sim['asim_tsim']:.2f} | a1-as={dist_sim['a1_asim']:.2f} t1-ts={dist_sim['t1_tsim']:.2f} a1-ts={dist_sim['a1_tsim']:.2f}",
        ]
        ax_sim.set_title("\n".join(title_sim), fontsize=9.0, pad=10)
        ax_sim.set_xlabel("PC1")
        ax_sim.set_ylabel("PC2")
        ax_sim.grid(True, linestyle="--", alpha=0.25)
        ax_sim.legend(loc="best", fontsize=8.0, framealpha=0.9)

        # =========================================================================
        # REFERENCE TRANSCRIPTIONS BOX AT THE BOTTOM
        # =========================================================================
        wrapped_t1 = textwrap.fill(f'Anchor Seq 1 [{c_idx_1}] ({seq_tag_1}): "{transcript_1}"', width=220)
        wrapped_t2 = textwrap.fill(
            f'Most Dissimilar Seq 2 [{c_idx_2}] ({seq_tag_2}, BERTScore [0,1]={b_norm_01:.3f}, rescaled={b_rescaled:.3f}): "{transcript_2}"',
            width=220,
        )
        wrapped_tsim = textwrap.fill(
            f'Most Similar Seq [{c_idx_sim}] ({seq_tag_sim}, BERTScore [0,1]={sim_b_norm_01:.3f}, rescaled={sim_b_rescaled:.3f}): "{transcript_sim}"',
            width=220,
        )
        transcript_box_text = f"Reference Transcriptions:\n{wrapped_t1}\n{wrapped_t2}\n{wrapped_tsim}"

        fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.18)
        fig.text(
            0.02,
            0.015,
            transcript_box_text,
            fontsize=8.5,
            family="monospace",
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#adb5bd", alpha=0.95),
        )

        fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _mark_start_end(
        ax,
        coords: np.ndarray,
        *,
        color: str,
        prefix: str,
        show_legend: bool = False,
    ) -> None:
        if len(coords) == 0:
            return
        ax.scatter(
            coords[0, 0],
            coords[0, 1],
            s=130,
            marker="*",
            c=color,
            edgecolors="black",
            linewidths=0.9,
            zorder=6,
            label="Seq Start (*)" if show_legend else None,
        )
        if len(coords) > 1:
            ax.scatter(
                coords[-1, 0],
                coords[-1, 1],
                s=110,
                marker="X",
                c=color,
                edgecolors="black",
                linewidths=0.9,
                zorder=6,
                label="Seq End (X)" if show_legend else None,
            )

    def _write_contrastive_cosine_summary(self, cosine_similarities: Dict[str, Dict[str, float]]) -> None:
        keys = ("a1_t1", "a2_t2", "a1_a2", "t1_t2", "a1_t2", "a2_t1")
        with open(os.path.join(self.out_dir, "cosine_similarities.txt"), "w", encoding="utf-8") as f:
            f.write("# Average pairwise cosine similarity in the original encoder feature space\n")
            f.write("# a1_t1 = Anchor Audio<->Text, a2_t2 = Dissimilar Audio<->Text\n")
            f.write("# a1_a2 = Anchor Audio<->Dissimilar Audio, t1_t2 = Anchor Text<->Dissimilar Text\n")
            header = f"{'anchor_seq_tag':<55}" + "".join(f"{k:>9}" for k in keys)
            f.write(header + "\n")
            for tag in sorted(cosine_similarities):
                cos = cosine_similarities[tag]
                row = f"{tag:<55}" + "".join(f"{cos[k]:>9.4f}" for k in keys)
                f.write(row + "\n")
            f.write("-" * len(header) + "\n")
            means = {
                k: float(np.nanmean([c[k] for c in cosine_similarities.values()]))
                for k in keys
            }
            mean_row = f"{'MEAN over seqs':<55}" + "".join(f"{means[k]:>9.4f}" for k in keys)
            f.write(mean_row + "\n")

