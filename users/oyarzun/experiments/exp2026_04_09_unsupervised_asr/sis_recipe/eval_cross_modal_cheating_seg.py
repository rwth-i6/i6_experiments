"""
Cheating Segmentation Nearest-Neighbor Evaluation Job (No Learnable Parameters)
Evaluates the baseline translation quality of cross-modal codes trained via HuBERT and clustering
using an oracle (cheating) segmentation approach.

For each segment defined by the cheating segmentation:
  1. Average representations within the segment (producing 1 vector per segment)
  2. Find the nearest neighbor text phoneme via cosine similarity against text phoneme embeddings
  3. Evaluate against ground-truth phoneme sequences using Levenshtein distance (PER).
"""

from typing import List, Dict, Optional, Tuple, Iterator, Union
import os
import glob
import json
import logging
import numpy as np

from sisyphus import tk, Task


def compute_edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance between reference and hypothesis tokens.
    Returns: (substitutions, deletions, insertions, total_reference_tokens)
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,        # Deletion
                dp[i, j - 1] + 1,        # Insertion
                dp[i - 1, j - 1] + cost   # Match / Substitution
            )

    # Backtrack to count operations
    i, j = m, n
    subs = dels = inss = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                if cost == 1:
                    subs += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            inss += 1
            j -= 1
        else:
            break

    return subs, dels, inss, m


def normalize_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Length-normalizes 2D matrix along axis 1."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = eps
    return v / norms


class CheatingSegmentationNearestNeighborEvalJob(tk.Job):
    """
    Zero-parameter evaluation job that applies oracle segmentation, averages representations
    per segment, and predicts phonemes via nearest-neighbor search against text cross-modal embeddings.
    """

    def __init__(
        self,
        audio_embeddings_npy: Union[tk.Path, str],
        text_embeddings_npy: Union[tk.Path, str],
        phoneme_vocab_file: Union[tk.Path, str],
        cheating_segmentation_path: Union[tk.Path, str],
        audio_cluster_hdfs: Optional[Union[List[Union[tk.Path, str]], tk.Path, str]] = None,
        reference_phonemes: Optional[Union[List[Union[tk.Path, str]], tk.Path, str]] = None,
        seq_tags_file: Optional[Union[tk.Path, str]] = None,
        centroids_npy: Optional[Union[tk.Path, str]] = None,
        cluster_to_phoneme_dict: Optional[Union[tk.Path, str]] = None,
        max_seqs: Optional[int] = None,
        ignore_silence: bool = True,
        name_prefix: str = "cheating_seg_eval",
    ):
        super().__init__()
        self.audio_embeddings_npy = audio_embeddings_npy
        self.text_embeddings_npy = text_embeddings_npy
        self.phoneme_vocab_file = phoneme_vocab_file
        self.cheating_segmentation_path = cheating_segmentation_path
        self.audio_cluster_hdfs = audio_cluster_hdfs
        self.reference_phonemes = reference_phonemes
        self.seq_tags_file = seq_tags_file
        self.centroids_npy = centroids_npy
        self.cluster_to_phoneme_dict = cluster_to_phoneme_dict
        self.max_seqs = max_seqs
        self.ignore_silence = ignore_silence
        self.name_prefix = name_prefix

        # Outputs
        self.out_hypotheses_txt = self.output_path("hypotheses.txt")
        self.out_reference_txt = self.output_path("reference.txt")
        self.out_per_summary_json = self.output_path("per_summary.json")
        self.out_per_report_txt = self.output_path("per_report.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4, "gpu": 1})

    def _resolve_path(self, p: Optional[Union[tk.Path, str]]) -> Optional[str]:
        if p is None:
            return None
        if isinstance(p, tk.Path):
            path_str = p.get_path()
        else:
            path_str = os.path.expandvars(str(p))
        if path_str.startswith("/hpcwork/"):
            path_str = "/rwthfs/rz/cluster/hpcwork/" + path_str[len("/hpcwork/"):]
        return path_str

    def _load_hdf_sequences(self, hdf_paths: List[str]) -> Dict[str, np.ndarray]:
        """Loads sequences keyed by seqTag from one or more RETURNN-format HDF files."""
        import h5py

        data_dict = {}
        for hdf_path in hdf_paths:
            if not os.path.exists(hdf_path):
                logging.warning(f"HDF file not found: {hdf_path}")
                continue
            with h5py.File(hdf_path, "r") as f:
                if "seqTags" not in f or "seqLengths" not in f:
                    continue
                seq_tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][:]]
                seq_lens = f["seqLengths"][:]
                
                # Check target dataset
                target_key = "inputs" if "inputs" in f else "data"
                if target_key not in f:
                    # check groups
                    if "targets" in f and "data" in f["targets"]:
                        target_data = f["targets"]["data"][:]
                    else:
                        continue
                else:
                    target_data = f[target_key][:]

                offset = 0
                for idx, tag in enumerate(seq_tags):
                    length = int(seq_lens[idx, 0]) if seq_lens.ndim == 2 else int(seq_lens[idx])
                    seq = target_data[offset : offset + length]
                    offset += length
                    data_dict[tag] = seq
        return data_dict

    def _load_segment_boundaries(self, seg_path: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Loads segment boundaries for each sequence.
        Supports:
          1. Text format: seq_tag start_frame end_frame (or start_sec end_sec)
          2. JSON format: {seq_tag: [[start, end], ...]}
          3. HDF format: reads segment lengths/labels
        """
        import h5py

        seg_dict = {}
        if os.path.isdir(seg_path):
            files = sorted(glob.glob(os.path.join(seg_path, "*.hdf")) + glob.glob(os.path.join(seg_path, "*.txt")))
        else:
            files = [seg_path]

        for file_path in files:
            if not os.path.exists(file_path):
                logging.warning(f"Segmentation file not found: {file_path}")
                continue

            if file_path.endswith(".hdf"):
                with h5py.File(file_path, "r") as f:
                    if "seqTags" in f and "seqLengths" in f:
                        seq_tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][:]]
                        seq_lens = f["seqLengths"][:]
                        target_key = "inputs" if "inputs" in f else ("data" if "data" in f else None)
                        target_data = f[target_key][:] if target_key else None
                        offset = 0
                        for idx, tag in enumerate(seq_tags):
                            length = int(seq_lens[idx, 0]) if seq_lens.ndim == 2 else int(seq_lens[idx])
                            # If file contains pre-segmented cluster labels (1 label per segment)
                            if target_data is not None:
                                cluster_ids = target_data[offset : offset + length]
                                seg_dict[tag] = [(int(c), int(c)) for c in cluster_ids]
                            else:
                                seg_dict[tag] = [(s_i, s_i) for s_i in range(length)]
                            offset += length
            elif file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    seg_dict.update(json.load(f))
            else:
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            tag = parts[0]
                            try:
                                s = int(float(parts[1]))
                                e = int(float(parts[2]))
                                seg_dict.setdefault(tag, []).append((s, e))
                            except ValueError:
                                continue
        return seg_dict

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.info("Starting Cheating Segmentation Nearest Neighbor Evaluation...")

        # 1. Load Vocabularies & Embeddings
        audio_emb_path = self._resolve_path(self.audio_embeddings_npy)
        text_emb_path = self._resolve_path(self.text_embeddings_npy)
        vocab_path = self._resolve_path(self.phoneme_vocab_file)

        logging.info(f"Loading audio embeddings from: {audio_emb_path}")
        audio_emb = np.load(audio_emb_path)  # (K, d)
        logging.info(f"Loading text embeddings from: {text_emb_path}")
        text_emb = np.load(text_emb_path)    # (P, d)

        # Normalize text phoneme embeddings for cosine similarity
        text_emb_norm = normalize_vectors(text_emb)
        P, d_text = text_emb_norm.shape

        # Load phoneme vocabulary
        phoneme_vocab = []
        with open(vocab_path, "r") as f:
            for line in f:
                token = line.strip().split()[0]
                phoneme_vocab.append(token)
        logging.info(f"Loaded {len(phoneme_vocab)} phonemes into vocabulary.")

        # Load cluster-to-phoneme dictionary if precomputed
        cluster_to_phon_map = None
        if self.cluster_to_phoneme_dict is not None:
            c2p_path = self._resolve_path(self.cluster_to_phoneme_dict)
            if os.path.exists(c2p_path):
                with open(c2p_path, "r") as f:
                    cluster_to_phon_map = json.load(f)

        # 2. Load Cheating Segmentation
        seg_path = self._resolve_path(self.cheating_segmentation_path)
        logging.info(f"Loading cheating segmentation from: {seg_path}")
        segmentations = self._load_segment_boundaries(seg_path)
        logging.info(f"Found segmentation for {len(segmentations)} sequences.")

        # 3. Load Frame-Level Audio Clusters / Features
        audio_seqs = {}
        if self.audio_cluster_hdfs is not None:
            if isinstance(self.audio_cluster_hdfs, list):
                hdf_list = [self._resolve_path(p) for p in self.audio_cluster_hdfs]
            else:
                hdf_list = [self._resolve_path(self.audio_cluster_hdfs)]
            logging.info(f"Loading audio cluster HDFs ({len(hdf_list)} files)...")
            audio_seqs = self._load_hdf_sequences(hdf_list)
            logging.info(f"Loaded audio frame sequences for {len(audio_seqs)} utterances.")

        # 4. Load Reference Phonemes
        ref_seqs = {}
        if self.reference_phonemes is not None:
            if isinstance(self.reference_phonemes, list):
                ref_paths = [self._resolve_path(p) for p in self.reference_phonemes]
            else:
                ref_paths = [self._resolve_path(self.reference_phonemes)]
            logging.info(f"Loading reference phonemes ({len(ref_paths)} files)...")
            if any(p.endswith(".hdf") for p in ref_paths):
                ref_idx_dict = self._load_hdf_sequences(ref_paths)
                for tag, indices in ref_idx_dict.items():
                    ref_seqs[tag] = [phoneme_vocab[int(idx)] for idx in indices if int(idx) < len(phoneme_vocab)]
            else:
                for rp in ref_paths:
                    if os.path.exists(rp):
                        with open(rp, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    tag = parts[0]
                                    ref_seqs[tag] = parts[1:]

        # Filter sequence tags if a whitelist was given
        active_tags = sorted(list(segmentations.keys()))
        if self.seq_tags_file is not None:
            tags_path = self._resolve_path(self.seq_tags_file)
            if os.path.exists(tags_path):
                with open(tags_path, "r") as f:
                    whitelist = set(line.strip().split()[0] for line in f if line.strip())
                active_tags = [t for t in active_tags if t in whitelist]

        if self.max_seqs is not None and self.max_seqs > 0:
            active_tags = active_tags[: self.max_seqs]

        logging.info(f"Evaluating {len(active_tags)} utterances...")
        if len(active_tags) == 0:
            logging.warning(
                f"Warning: No matching sequences found between segmentation ({len(segmentations)} sequences) "
                f"and whitelist/tags file ({self.seq_tags_file})! "
                f"Check whether the dataset split (e.g. dev-other vs train-other-960) matches the segmentation data."
            )

        # 5. Main Inference & Nearest-Neighbor Search Loop
        total_subs = 0
        total_dels = 0
        total_inss = 0
        total_ref_tokens = 0
        total_correct = 0

        hypotheses_records = []
        references_records = []

        for idx, tag in enumerate(active_tags):
            segments = segmentations[tag]
            pred_phonemes = []

            if tag in audio_seqs:
                # We have frame-level tokens/representations
                frame_data = audio_seqs[tag]
                for (start_f, end_f) in segments:
                    start_f = max(0, start_f)
                    end_f = min(len(frame_data) - 1, end_f)
                    if start_f > end_f:
                        continue

                    seg_frames = frame_data[start_f : end_f + 1]
                    valid_frames = seg_frames[seg_frames < len(audio_emb)]
                    if len(valid_frames) == 0:
                        valid_frames = seg_frames % len(audio_emb)

                    # Segment-level representations: average cross-modal embedding of frames
                    seg_emb_vectors = audio_emb[valid_frames.astype(np.int64)]  # (|seg|, d)
                    avg_vector = np.mean(seg_emb_vectors, axis=0, keepdims=True)  # (1, d)
                    avg_vector_norm = normalize_vectors(avg_vector)

                    # Cosine nearest neighbor to text phoneme embeddings
                    sims = np.dot(avg_vector_norm, text_emb_norm.T)[0]  # (P,)
                    best_phon_idx = int(np.argmax(sims))
                    best_phoneme = phoneme_vocab[best_phon_idx]

                    if self.ignore_silence and best_phoneme in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", ""):
                        continue
                    pred_phonemes.append(best_phoneme)

            else:
                # Direct segment labels provided in segmentation (e.g. cluster_labels_k512 HDF)
                # Each segment already has a cluster ID
                for (s_idx, _) in segments:
                    if s_idx < len(audio_emb):
                        seg_vec = audio_emb[s_idx : s_idx + 1]
                        seg_vec_norm = normalize_vectors(seg_vec)
                        sims = np.dot(seg_vec_norm, text_emb_norm.T)[0]
                        best_phon_idx = int(np.argmax(sims))
                        best_phoneme = phoneme_vocab[best_phon_idx]
                        if self.ignore_silence and best_phoneme in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", ""):
                            continue
                        pred_phonemes.append(best_phoneme)

            hyp_str = " ".join(pred_phonemes)
            hypotheses_records.append(f"{tag} {hyp_str}")

            # Evaluate against reference if available
            if tag in ref_seqs:
                ref_phonemes = ref_seqs[tag]
                if self.ignore_silence:
                    ref_phonemes = [p for p in ref_phonemes if p not in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")]

                ref_str = " ".join(ref_phonemes)
                references_records.append(f"{tag} {ref_str}")

                subs, dels, inss, ref_len = compute_edit_distance(ref_phonemes, pred_phonemes)
                total_subs += subs
                total_dels += dels
                total_inss += inss
                total_ref_tokens += ref_len
                total_correct += (ref_len - subs - dels)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(active_tags):
                current_errs = total_subs + total_dels + total_inss
                current_per = (current_errs / max(1, total_ref_tokens)) * 100.0
                logging.info(
                    f"Processed {idx + 1}/{len(active_tags)} utts | "
                    f"Current PER: {current_per:.2f}% (Ref tokens: {total_ref_tokens}, S: {total_subs}, D: {total_dels}, I: {total_inss})"
                )

        # 6. Write Output Files
        with open(self.out_hypotheses_txt.get_path(), "w") as f:
            for rec in hypotheses_records:
                f.write(rec + "\n")

        with open(self.out_reference_txt.get_path(), "w") as f:
            for rec in references_records:
                f.write(rec + "\n")

        total_errs = total_subs + total_dels + total_inss
        final_per = (total_errs / max(1, total_ref_tokens)) * 100.0
        accuracy = (total_correct / max(1, total_ref_tokens)) * 100.0

        summary = {
            "total_sequences": len(active_tags),
            "total_ref_tokens": total_ref_tokens,
            "total_errors": total_errs,
            "substitutions": total_subs,
            "deletions": total_dels,
            "insertions": total_inss,
            "correct_tokens": total_correct,
            "phoneme_error_rate_pct": round(final_per, 2),
            "token_accuracy_pct": round(accuracy, 2),
            "audio_embeddings_path": audio_emb_path,
            "text_embeddings_path": text_emb_path,
            "cheating_segmentation_path": seg_path,
        }

        with open(self.out_per_summary_json.get_path(), "w") as f:
            json.dump(summary, f, indent=2)

        report = (
            "======================================================================\n"
            "   CHEATING SEGMENTATION NEAREST-NEIGHBOR EVALUATION REPORT\n"
            "======================================================================\n"
            f"Evaluated Sequences:    {len(active_tags)}\n"
            f"Total Reference Tokens: {total_ref_tokens}\n"
            f"Substitutions (S):      {total_subs}\n"
            f"Deletions (D):          {total_dels}\n"
            f"Insertions (I):         {total_inss}\n"
            f"Total Errors (S+D+I):   {total_errs}\n"
            f"----------------------------------------------------------------------\n"
            f"PHONEME ERROR RATE (PER): {final_per:.2f}%\n"
            f"TOKEN ACCURACY:           {accuracy:.2f}%\n"
            "======================================================================\n"
        )
        with open(self.out_per_report_txt.get_path(), "w") as f:
            f.write(report)

        logging.info(f"Evaluation complete. Final PER: {final_per:.2f}%. Report written to {self.out_per_report_txt.get_path()}")
