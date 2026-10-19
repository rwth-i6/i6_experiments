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


def compute_edit_distance(
    ref_tokens: List[str], hyp_tokens: List[str], return_alignment: bool = False
) -> Union[Tuple[int, int, int, int], Tuple[int, int, int, int, List[Tuple[str, str, str]]]]:
    """
    Computes Levenshtein edit distance between reference and hypothesis tokens.
    Returns: (substitutions, deletions, insertions, total_reference_tokens)
    If return_alignment is True, also returns alignment: List of (ref_tok, hyp_tok, op)
    where op is ' ' (correct), 'S' (substitution), 'D' (deletion), or 'I' (insertion).
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

    # Backtrack to count operations and reconstruct alignment
    i, j = m, n
    subs = dels = inss = 0
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                op = " " if cost == 0 else "S"
                if cost == 1:
                    subs += 1
                if return_alignment:
                    alignment.append((ref_tokens[i - 1], hyp_tokens[j - 1], op))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            if return_alignment:
                alignment.append((ref_tokens[i - 1], "*" * max(1, len(ref_tokens[i - 1])), "D"))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            inss += 1
            if return_alignment:
                alignment.append(("*" * max(1, len(hyp_tokens[j - 1])), hyp_tokens[j - 1], "I"))
            j -= 1
        else:
            break

    if return_alignment:
        alignment.reverse()
        return subs, dels, inss, m, alignment
    return subs, dels, inss, m


def format_sclite_pra_sentence(
    utt_idx: int,
    utt_tag: str,
    correct: int,
    subs: int,
    dels: int,
    inss: int,
    alignment: List[Tuple[str, str, str]],
    max_line_width: int = 120,
) -> str:
    """
    Formats utterance alignment matching NIST SCLITE .pra format:
    Speaker sentences{idx}:  {tag}   #utts: 1
    id: ({tag})
    Scores: (#C #S #D #I) {C} {S} {D} {I}
    REF:  ...
    HYP:  ...
    Eval: ...
    """
    clean_tag = utt_tag.split("/")[-1] if "/" in utt_tag else utt_tag
    header = (
        f"Speaker sentences{utt_idx}:  {utt_tag}   #utts: 1\n"
        f"id: ({clean_tag})\n"
        f"Scores: (#C #S #D #I) {correct} {subs} {dels} {inss}\n"
    )

    blocks = []
    ref_row = "REF:  "
    hyp_row = "HYP:  "
    eval_row = "Eval: "

    for r_tok, h_tok, op in alignment:
        col_w = max(len(r_tok), len(h_tok), len(op))
        rc = r_tok.ljust(col_w)
        hc = h_tok.ljust(col_w)
        ec = (op if op != " " else "").ljust(col_w)

        if len(ref_row) + col_w + 1 > max_line_width and ref_row != "REF:  ":
            blocks.append(f"{ref_row}\n{hyp_row}\n{eval_row}\n")
            ref_row = "REF:  "
            hyp_row = "HYP:  "
            eval_row = "Eval: "

        ref_row += rc + " "
        hyp_row += hc + " "
        eval_row += ec + " "

    if ref_row != "REF:  ":
        blocks.append(f"{ref_row}\n{hyp_row}\n{eval_row}\n")

    return header + "".join(blocks) + "\n"


def normalize_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Length-normalizes 2D matrix along axis 1."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = eps
    return v / norms


def normalize_and_center(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Length-normalizes (unit L2 norm) and mean-centers an embedding matrix.
    Matches Artetxe et al. (2017) preprocessing in SelfLearningEmbeddingAlignmentJob.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = eps
    norm_emb = embeddings / norms
    mean = np.mean(norm_emb, axis=0, keepdims=True)
    centered_emb = norm_emb - mean
    final_norms = np.linalg.norm(centered_emb, axis=1, keepdims=True)
    final_norms[final_norms == 0] = eps
    return centered_emb / final_norms


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
        reference_phoneme_vocab_file: Optional[Union[tk.Path, str]] = None,
        seq_tags_file: Optional[Union[tk.Path, str]] = None,
        centroids_npy: Optional[Union[tk.Path, str]] = None,
        cluster_to_phoneme_dict: Optional[Union[tk.Path, str]] = None,
        mapping_matrix_npy: Optional[Union[tk.Path, str]] = None,
        code_type: str = "auto",
        max_seqs: Optional[int] = None,
        ignore_silence: bool = True,
        name_prefix: str = "cheating_seg_eval",
        retrieval_metric: str = "cosine",
        csls_k: int = 10,
        center_text_embeddings: bool = True,
    ):
        super().__init__()
        self.audio_embeddings_npy = audio_embeddings_npy
        self.text_embeddings_npy = text_embeddings_npy
        self.phoneme_vocab_file = phoneme_vocab_file
        self.cheating_segmentation_path = cheating_segmentation_path
        self.audio_cluster_hdfs = audio_cluster_hdfs
        self.reference_phonemes = reference_phonemes
        self.reference_phoneme_vocab_file = reference_phoneme_vocab_file
        self.seq_tags_file = seq_tags_file
        self.centroids_npy = centroids_npy
        self.cluster_to_phoneme_dict = cluster_to_phoneme_dict
        self.mapping_matrix_npy = mapping_matrix_npy
        self.code_type = code_type
        self.max_seqs = max_seqs
        self.ignore_silence = ignore_silence
        self.name_prefix = name_prefix
        self.retrieval_metric = retrieval_metric
        self.csls_k = csls_k
        self.center_text_embeddings = center_text_embeddings

        # Outputs
        self.out_hypotheses_txt = self.output_path("hypotheses.txt")
        self.out_reference_txt = self.output_path("reference.txt")
        self.out_per_summary_json = self.output_path("per_summary.json")
        self.out_per_report_txt = self.output_path("per_report.txt")
        self.out_pra_report_txt = self.output_path("sclite.pra")

    @classmethod
    def hash(cls, parsed_args):
        m = parsed_args.get("retrieval_metric", "cosine")
        if "v4" in str(parsed_args.get("name_prefix", "")):
            parsed_args["eval_version"] = f"v4_segment_kmeans_{m}"
        elif parsed_args.get("audio_cluster_hdfs") is None:
            parsed_args["eval_version"] = f"v6_direct_seg_{m}"
        else:
            if m == "csls":
                parsed_args["eval_version"] = "v5_real_pooling_csls"
            else:
                parsed_args["eval_version"] = "v5_real_pooling"
        return super().hash(parsed_args)


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
                target_key = None
                for k in ["inputs", "data", "labels", "classes"]:
                    if k in f:
                        target_key = k
                        break
                if target_key is not None:
                    target_data = f[target_key][:]
                elif "targets" in f and "data" in f["targets"]:
                    target_data = f["targets"]["data"][:]
                elif "targets" in f and "labels" in f["targets"]:
                    target_data = f["targets"]["labels"][:]
                else:
                    continue

                offset = 0
                for idx, tag in enumerate(seq_tags):
                    length = int(seq_lens[idx, 0]) if seq_lens.ndim == 2 else int(seq_lens[idx])
                    seq = target_data[offset : offset + length]
                    offset += length
                    data_dict[tag] = seq
                    data_dict[tag.split("/")[-1]] = seq
        return data_dict

    def _load_audio_sequences(self, paths: List[str]) -> Dict[str, np.ndarray]:
        """Loads sequences keyed by seqTag from RETURNN HDF files or raw_clusters text files."""
        data_dict = {}
        hdf_paths = [p for p in paths if p and p.endswith(".hdf")]
        txt_paths = [p for p in paths if p and p.endswith(".txt")]

        if hdf_paths:
            data_dict.update(self._load_hdf_sequences(hdf_paths))

        if txt_paths and self.seq_tags_file is not None:
            tags_path = self._resolve_path(self.seq_tags_file)
            if tags_path and os.path.exists(tags_path):
                with open(tags_path, "r") as tf:
                    tags = [line.strip().split()[0] for line in tf if line.strip()]
                for tp in txt_paths:
                    if not os.path.exists(tp):
                        continue
                    logging.info(f"Loading audio frame cluster sequences from text file: {tp}")
                    with open(tp, "r") as f:
                        for idx, line in enumerate(f):
                            if idx >= len(tags):
                                break
                            tag = tags[idx]
                            seq = np.array([int(c) for c in line.strip().split()], dtype=np.int32)
                            data_dict[tag] = seq
                            data_dict[tag.split("/")[-1]] = seq
        return data_dict

    def _load_segment_boundaries(
        self,
        seg_path: Union[str, List[str]],
        ref_dict_out: Optional[Dict[str, List[str]]] = None,
        idx_to_phone: Optional[Dict[int, str]] = None,
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Loads segment boundaries for each sequence.
        Supports:
          1. GMM Oracle frame-level phoneme reference HDF files (extracts start_frame, end_frame runs and references)
          2. Segment cluster label HDF files (pre-segmented)
          3. Text format: seq_tag start_frame end_frame
          4. JSON format: {seq_tag: [[start, end], ...]}
        """
        import h5py

        seg_dict = {}
        if isinstance(seg_path, list):
            files = [self._resolve_path(p) for p in seg_path]
        elif os.path.isdir(seg_path):
            files = sorted(glob.glob(os.path.join(seg_path, "*.hdf")) + glob.glob(os.path.join(seg_path, "*.txt")))
        else:
            files = [seg_path]

        for file_path in files:
            if not file_path or not os.path.exists(file_path):
                logging.warning(f"Segmentation file not found: {file_path}")
                continue

            if file_path.endswith(".hdf"):
                with h5py.File(file_path, "r") as f:
                    if "seqTags" not in f or "seqLengths" not in f:
                        continue
                    seq_tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][:]]
                    seq_lens = f["seqLengths"][:]
                    target_key = "inputs" if "inputs" in f else ("data" if "data" in f else None)
                    target_data = f[target_key][:] if target_key else None
                    if target_data is None:
                        continue

                    offset = 0
                    is_frame_phoneme_ref = "phoneme_ref" in file_path

                    for idx, tag in enumerate(seq_tags):
                        length = int(seq_lens[idx, 0]) if seq_lens.ndim == 2 else int(seq_lens[idx])
                        clean_tag = tag.split("/")[-1]

                        if is_frame_phoneme_ref:
                            # Frame-level phoneme alignments: extract consecutive identical label runs
                            labels = target_data[offset : offset + length]
                            cur_label = None
                            cur_start = None
                            segs = []
                            ref_phons = []
                            for t, lab in enumerate(labels):
                                lab_val = int(lab)
                                is_sil = (lab_val == 0) or (
                                    idx_to_phone and idx_to_phone.get(lab_val, "") in ("[SILENCE]", "<SIL>", "SIL", "<sil>", "<eps>")
                                )
                                if is_sil:
                                    if cur_label is not None:
                                        segs.append((cur_start, t - 1))
                                        if idx_to_phone:
                                            ref_phons.append(idx_to_phone.get(cur_label, str(cur_label)))
                                        cur_label = None
                                else:
                                    if cur_label is None or lab_val != cur_label:
                                        if cur_label is not None:
                                            segs.append((cur_start, t - 1))
                                            if idx_to_phone:
                                                ref_phons.append(idx_to_phone.get(cur_label, str(cur_label)))
                                        cur_start = t
                                        cur_label = lab_val
                            if cur_label is not None:
                                segs.append((cur_start, len(labels) - 1))
                                if idx_to_phone:
                                    ref_phons.append(idx_to_phone.get(cur_label, str(cur_label)))

                            seg_dict[tag] = segs
                            seg_dict[clean_tag] = segs
                            if ref_dict_out is not None and len(ref_phons) > 0:
                                ref_dict_out[tag] = ref_phons
                                ref_dict_out[clean_tag] = ref_phons
                        else:
                            # Pre-segmented cluster labels (1 label per segment)
                            cluster_ids = target_data[offset : offset + length]
                            seg_dict[tag] = [(int(c), int(c)) for c in cluster_ids]
                            seg_dict[clean_tag] = seg_dict[tag]

                        offset += length
            elif file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        seg_dict[k] = v
                        seg_dict[k.split("/")[-1]] = v
            else:
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    f.seek(0)
                    parts_0 = first_line.split()
                    is_token_list = len(parts_0) > 0 and all(p.isdigit() for p in parts_0[:min(10, len(parts_0))])
                    if is_token_list and self.seq_tags_file is not None:
                        tags_path = self._resolve_path(self.seq_tags_file)
                        with open(tags_path, "r") as tf:
                            file_tags = [line.strip().split()[0] for line in tf if line.strip()]
                        logging.info(f"Loading direct collapsed token sequences from: {file_path} ({len(file_tags)} tags)")
                        for idx, line in enumerate(f):
                            if idx >= len(file_tags):
                                break
                            tag = file_tags[idx]
                            toks = [int(c) for c in line.strip().split() if c.isdigit()]
                            segs = [(c, c) for c in toks]
                            seg_dict[tag] = segs
                            seg_dict[tag.split("/")[-1]] = segs
                    else:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                tag = parts[0]
                                try:
                                    s = int(float(parts[1]))
                                    e = int(float(parts[2]))
                                    seg_dict.setdefault(tag, []).append((s, e))
                                    seg_dict.setdefault(tag.split("/")[-1], []).append((s, e))
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

        # Check if mapped audio embeddings file exists alongside W_star.npy
        if audio_emb_path and os.path.exists(audio_emb_path):
            if os.path.basename(audio_emb_path) == "W_star.npy":
                sib_mapped = os.path.join(os.path.dirname(audio_emb_path), "mapped_audio_embeddings.npy")
                real_mapped = os.path.join(os.path.dirname(os.path.realpath(audio_emb_path)), "mapped_audio_embeddings.npy")
                if os.path.exists(sib_mapped):
                    audio_emb_path = sib_mapped
                elif os.path.exists(real_mapped):
                    audio_emb_path = real_mapped

        logging.info(f"Loading audio embeddings from: {audio_emb_path}")
        audio_emb = np.load(audio_emb_path)  # (K, d)
        if self.mapping_matrix_npy is not None:
            w_path = self._resolve_path(self.mapping_matrix_npy)
            if os.path.exists(w_path):
                logging.info(f"Applying mapping matrix from: {w_path}")
                W = np.load(w_path)
                audio_emb = normalize_vectors(audio_emb) @ W

        logging.info(f"Loading text embeddings from: {text_emb_path}")
        text_emb = np.load(text_emb_path)    # (P, d)

        # Normalize text phoneme embeddings for cosine similarity
        if self.center_text_embeddings:
            text_emb_norm = normalize_and_center(text_emb)
        else:
            text_emb_norm = normalize_vectors(text_emb)
        P, d_text = text_emb_norm.shape

        # Load phoneme vocabulary
        phoneme_vocab = []
        with open(vocab_path, "r") as f:
            for line in f:
                token = line.strip().split()[0]
                phoneme_vocab.append(token)
        logging.info(f"Loaded {len(phoneme_vocab)} phonemes into vocabulary.")

        # Determine code_type (monophone, diphone, triphone)
        detected_code = self.code_type.lower()
        if detected_code == "auto":
            has_triphone = any(t.count("_") >= 2 for t in phoneme_vocab)
            has_diphone = any(t.count("_") == 1 for t in phoneme_vocab)
            if has_triphone:
                detected_code = "triphone"
            elif has_diphone:
                detected_code = "diphone"
            else:
                detected_code = "monophone"
        logging.info(f"Evaluation code_type: {detected_code}")

        def _to_phone(token: str) -> str:
            if "_" not in token:
                return token
            parts = token.split("_")
            if detected_code == "triphone" or len(parts) >= 3:
                return parts[len(parts) // 2]
            return parts[0]

        # Load cluster-to-phoneme dictionary if precomputed
        cluster_to_phon_map = None
        if self.cluster_to_phoneme_dict is not None:
            c2p_path = self._resolve_path(self.cluster_to_phoneme_dict)
            if os.path.exists(c2p_path):
                with open(c2p_path, "r") as f:
                    cluster_to_phon_map = json.load(f)

        # Precompute target phoneme CSLS statistics if requested
        r_S = None
        if self.retrieval_metric == "csls":
            audio_emb_for_csls = normalize_vectors(audio_emb)
            S_base = np.dot(audio_emb_for_csls, text_emb_norm.T)  # (K, P)
            k_S = min(self.csls_k, len(audio_emb_for_csls))
            if k_S > 0:
                topk_S = np.partition(S_base, -k_S, axis=0)[-k_S:, :]
                r_S = np.mean(topk_S, axis=0)  # (P,)
            else:
                r_S = np.zeros(P, dtype=np.float32)
            logging.info(f"Initialized CSLS target neighborhood density r_S for {P} phonemes (k={k_S})")

        # Precompute direct cluster-to-phoneme mapping when no frame-level pooling is needed
        precomputed_cluster_tokens = None
        precomputed_cluster_phones = None
        if self.audio_cluster_hdfs is None and len(audio_emb) > 0:
            audio_emb_all = normalize_vectors(audio_emb)
            cos_all = np.dot(audio_emb_all, text_emb_norm.T)  # (K, P)
            if self.retrieval_metric == "csls" and r_S is not None:
                k_T = min(self.csls_k, P)
                if k_T > 0:
                    topk = np.partition(cos_all, -k_T, axis=1)[:, -k_T:]
                    r_T = np.mean(topk, axis=1, keepdims=True)
                else:
                    r_T = 0.0
                scores_all = 2.0 * cos_all - r_T - r_S[None, :]
            else:
                scores_all = cos_all
            best_all_indices = np.argmax(scores_all, axis=1)
            precomputed_cluster_tokens = [phoneme_vocab[int(b)] for b in best_all_indices]
            precomputed_cluster_phones = [_to_phone(t) for t in precomputed_cluster_tokens]
            logging.info(f"Precomputed direct cluster-to-phoneme mapping for all {len(audio_emb)} clusters.")

        # Load reference phoneme vocabulary if provided
        idx_to_phone = {}
        ref_vocab = phoneme_vocab
        if self.reference_phoneme_vocab_file is not None:
            rv_path = self._resolve_path(self.reference_phoneme_vocab_file)
            if rv_path and os.path.exists(rv_path):
                with open(rv_path, "r") as rf:
                    ref_vocab_lines = [line.strip() for line in rf if line.strip()]
                for line in ref_vocab_lines:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        idx = int(parts[0])
                        sym = parts[1].replace("#", "").replace('"', '').replace(':', '')
                        idx_to_phone[idx] = sym
                    elif len(parts) >= 1:
                        idx = len(idx_to_phone)
                        sym = parts[0].replace("#", "").replace('"', '').replace(':', '')
                        idx_to_phone[idx] = sym

        # 2. Load Cheating Segmentation & Ground-Truth References
        ref_seqs = {}
        if isinstance(self.cheating_segmentation_path, list):
            seg_path = [self._resolve_path(p) for p in self.cheating_segmentation_path]
        else:
            seg_path = self._resolve_path(self.cheating_segmentation_path)
        logging.info(f"Loading cheating segmentation from: {seg_path}")
        segmentations = self._load_segment_boundaries(seg_path, ref_dict_out=ref_seqs, idx_to_phone=idx_to_phone)
        logging.info(f"Found segmentation for {len(segmentations)} sequences (extracted {len(ref_seqs)} references).")

        # 3. Load Frame-Level Audio Clusters / Features
        audio_seqs = {}
        if self.audio_cluster_hdfs is not None:
            if isinstance(self.audio_cluster_hdfs, list):
                paths_list = [self._resolve_path(p) for p in self.audio_cluster_hdfs]
            else:
                paths_list = [self._resolve_path(self.audio_cluster_hdfs)]
            logging.info(f"Loading audio cluster sequences ({len(paths_list)} paths)...")
            audio_seqs = self._load_audio_sequences(paths_list)
            logging.info(f"Loaded audio frame sequences for {len(audio_seqs)} utterances.")

        # 4. Load Additional Reference Phonemes if explicitly provided
        if self.reference_phonemes is not None:
            if isinstance(self.reference_phonemes, list):
                ref_paths = [self._resolve_path(p) for p in self.reference_phonemes]
            else:
                ref_paths = [self._resolve_path(self.reference_phonemes)]
            logging.info(f"Loading additional reference phonemes ({len(ref_paths)} files)...")
            if any(p.endswith(".hdf") for p in ref_paths):
                if any("phoneme_ref" in p for p in ref_paths):
                    self._load_segment_boundaries(ref_paths, ref_dict_out=ref_seqs, idx_to_phone=idx_to_phone)
                else:
                    ref_idx_dict = self._load_hdf_sequences(ref_paths)
                    for tag, indices in ref_idx_dict.items():
                        decoded = [idx_to_phone.get(int(idx), ref_vocab[int(idx)] if int(idx) < len(ref_vocab) else str(idx)) for idx in indices]
                        ref_seqs[tag] = decoded
                        ref_seqs[tag.split("/")[-1]] = decoded
            else:
                for rp in ref_paths:
                    if os.path.exists(rp):
                        with open(rp, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    tag = parts[0]
                                    ref_seqs[tag] = parts[1:]
                                    ref_seqs[tag.split("/")[-1]] = parts[1:]

        # Filter sequence tags if a whitelist was given
        active_tags = sorted(list(set(segmentations.keys())))
        if self.seq_tags_file is not None:
            tags_path = self._resolve_path(self.seq_tags_file)
            if os.path.exists(tags_path) and os.path.getsize(tags_path) > 0:
                with open(tags_path, "r") as f:
                    whitelist = set(line.strip().split()[0] for line in f if line.strip())
                if len(whitelist) > 0:
                    matched_tags = [t for t in active_tags if (t in whitelist or t.split("/")[-1] in whitelist)]
                    if len(matched_tags) > 0:
                        active_tags = matched_tags
                    else:
                        logging.error(f"Whitelist provided ({len(whitelist)} tags) but none matched segmentations! Setting active_tags = []")
                        active_tags = []

        # Intersect with available audio sequences if audio sequences loaded
        if len(audio_seqs) > 0 and len(active_tags) > 0:
            audio_keys = set(audio_seqs.keys())
            matched_audio = [t for t in active_tags if (t in audio_keys or t.split("/")[-1] in audio_keys)]
            if len(matched_audio) > 0:
                active_tags = matched_audio
            else:
                logging.error(f"Audio sequences loaded ({len(audio_seqs)}) but none matched active tags! Setting active_tags = []")
                active_tags = []

        # If reference phonemes are loaded, filter to sequences with available references
        if len(ref_seqs) > 0 and len(active_tags) > 0:
            ref_keys = set(ref_seqs.keys())
            ref_suffixes = set(k.split("/")[-1] for k in ref_keys)
            overlap_tags = [t for t in active_tags if (t in ref_keys or t.split("/")[-1] in ref_suffixes)]
            if len(overlap_tags) > 0:
                # Deduplicate tags to avoid evaluating both full tag and basename
                seen_basenames = set()
                unique_tags = []
                for t in overlap_tags:
                    b = t.split("/")[-1]
                    if b not in seen_basenames:
                        seen_basenames.add(b)
                        unique_tags.append(t)
                active_tags = unique_tags
            else:
                logging.error(
                    f"Error: No matching sequences between segmentation ({len(active_tags)}) "
                    f"and reference phonemes ({len(ref_seqs)})! Setting active_tags = []"
                )
                active_tags = []

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

        # Run-collapsed phoneme metrics (consecutive identical emitted phonemes collapsed)
        total_collapsed_subs = 0
        total_collapsed_dels = 0
        total_collapsed_inss = 0

        # Token metrics (for diphones/triphones)
        total_token_subs = 0
        total_token_dels = 0
        total_token_inss = 0
        total_ref_ngram_tokens = 0

        hypotheses_records = []
        references_records = []
        pra_records = []

        for idx, tag in enumerate(active_tags):
            segments = segmentations[tag]
            pred_phonemes = []
            pred_raw_tokens = []

            if tag in audio_seqs:
                # We have frame-level tokens/representations
                frame_data = audio_seqs[tag]
                seg_vecs_list = []
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
                    avg_vector = np.mean(seg_emb_vectors, axis=0)  # (d,)
                    seg_vecs_list.append(avg_vector)

                if len(seg_vecs_list) > 0:
                    avg_matrix = np.stack(seg_vecs_list, axis=0)  # (num_segs, d)
                    avg_matrix_norm = normalize_vectors(avg_matrix)
                    cos_sims_batch = np.dot(avg_matrix_norm, text_emb_norm.T)  # (num_segs, P)

                    if self.retrieval_metric == "csls" and r_S is not None:
                        k_T = min(self.csls_k, P)
                        if k_T > 0:
                            topk = np.partition(cos_sims_batch, -k_T, axis=1)[:, -k_T:]
                            r_T = np.mean(topk, axis=1, keepdims=True)
                        else:
                            r_T = 0.0
                        scores_batch = 2.0 * cos_sims_batch - r_T - r_S[None, :]
                    else:
                        scores_batch = cos_sims_batch

                    best_phon_indices = np.argmax(scores_batch, axis=1)
                    for b_idx in best_phon_indices:
                        best_token = phoneme_vocab[int(b_idx)]
                        best_phone = _to_phone(best_token)
                        if self.ignore_silence and (
                            best_phone in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")
                            or best_token in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")
                        ):
                            continue
                        pred_phonemes.append(best_phone)
                        pred_raw_tokens.append(best_token)

            else:
                # Direct segment labels provided in segmentation (e.g. collapsed_tokens.txt or cluster_labels_k512 HDF)
                if precomputed_cluster_phones is not None:
                    for (s_idx, _) in segments:
                        idx_val = s_idx if s_idx < len(precomputed_cluster_phones) else (s_idx % len(precomputed_cluster_phones))
                        best_phone = precomputed_cluster_phones[idx_val]
                        best_token = precomputed_cluster_tokens[idx_val]
                        if self.ignore_silence and (
                            best_phone in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")
                            or best_token in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")
                        ):
                            continue
                        pred_phonemes.append(best_phone)
                        pred_raw_tokens.append(best_token)
                else:
                    seg_indices = []
                    for (s_idx, _) in segments:
                        idx_val = s_idx if s_idx < len(audio_emb) else (s_idx % len(audio_emb))
                        seg_indices.append(idx_val)

                    if len(seg_indices) > 0:
                        seg_matrix = audio_emb[np.array(seg_indices, dtype=np.int64)]  # (num_segs, d)
                        seg_matrix_norm = normalize_vectors(seg_matrix)
                        cos_sims_batch = np.dot(seg_matrix_norm, text_emb_norm.T)

                        if self.retrieval_metric == "csls" and r_S is not None:
                            k_T = min(self.csls_k, P)
                            if k_T > 0:
                                topk = np.partition(cos_sims_batch, -k_T, axis=1)[:, -k_T:]
                                r_T = np.mean(topk, axis=1, keepdims=True)
                            else:
                                r_T = 0.0
                            scores_batch = 2.0 * cos_sims_batch - r_T - r_S[None, :]
                        else:
                            scores_batch = cos_sims_batch

                        best_phon_indices = np.argmax(scores_batch, axis=1)
                        for b_idx in best_phon_indices:
                            best_token = phoneme_vocab[int(b_idx)]
                            best_phone = _to_phone(best_token)
                            if self.ignore_silence and (
                                best_phone in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")
                                or best_token in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")
                            ):
                                continue
                            pred_phonemes.append(best_phone)
                            pred_raw_tokens.append(best_token)

            hyp_str = " ".join(pred_phonemes)
            hypotheses_records.append(f"{tag} {hyp_str}")

            # Evaluate against reference if available
            ref_phonemes = ref_seqs.get(tag)
            if ref_phonemes is None:
                utt_id = tag.split("/")[-1] if "/" in tag else tag
                ref_phonemes = ref_seqs.get(utt_id)
            if ref_phonemes is None:
                utt_id = tag.split("/")[-1] if "/" in tag else tag
                for r_k in ref_seqs:
                    if r_k == utt_id or r_k.endswith("/" + utt_id) or r_k.endswith(utt_id) or utt_id.endswith(r_k):
                        ref_phonemes = ref_seqs[r_k]
                        break

            if ref_phonemes is not None:
                if self.ignore_silence:
                    ref_phonemes = [p for p in ref_phonemes if p not in ("[SIL]", "SIL", "<sil>", "<eps>", "blank", "")]

                ref_str = " ".join(ref_phonemes)
                references_records.append(f"{tag} {ref_str}")

                subs, dels, inss, ref_len, alignment = compute_edit_distance(
                    ref_phonemes, pred_phonemes, return_alignment=True
                )
                total_subs += subs
                total_dels += dels
                total_inss += inss
                total_ref_tokens += ref_len
                correct = ref_len - subs - dels
                total_correct += correct

                # Evaluate run-collapsed phonemes (consecutive identical emitted phonemes collapsed)
                collapsed_pred = []
                for p in pred_phonemes:
                    if len(collapsed_pred) == 0 or collapsed_pred[-1] != p:
                        collapsed_pred.append(p)
                c_s, c_d, c_i, _ = compute_edit_distance(ref_phonemes, collapsed_pred)
                total_collapsed_subs += c_s
                total_collapsed_dels += c_d
                total_collapsed_inss += c_i

                pra_block = format_sclite_pra_sentence(
                    len(pra_records) + 1, tag, correct, subs, dels, inss, alignment
                )
                pra_records.append(pra_block)

                # N-gram token error rate
                if detected_code in ("diphone", "triphone"):
                    n_order = 3 if detected_code == "triphone" else 2
                    if len(ref_phonemes) < n_order:
                        ref_ngrams = ["_".join(ref_phonemes)] if ref_phonemes else []
                    else:
                        ref_ngrams = ["_".join(ref_phonemes[i : i + n_order]) for i in range(len(ref_phonemes) - n_order + 1)]

                    t_subs, t_dels, t_inss, t_ref_len = compute_edit_distance(ref_ngrams, pred_raw_tokens)
                    total_token_subs += t_subs
                    total_token_dels += t_dels
                    total_token_inss += t_inss
                    total_ref_ngram_tokens += t_ref_len

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

        with open(self.out_pra_report_txt.get_path(), "w") as f:
            for block in pra_records:
                f.write(block)

        total_errs = total_subs + total_dels + total_inss
        if total_ref_tokens > 0:
            final_per = round((total_errs / total_ref_tokens) * 100.0, 2)
            accuracy = round((total_correct / total_ref_tokens) * 100.0, 2)
        else:
            final_per = None
            accuracy = None

        total_collapsed_errs = total_collapsed_subs + total_collapsed_dels + total_collapsed_inss
        if total_ref_tokens > 0:
            final_collapsed_per = round((total_collapsed_errs / total_ref_tokens) * 100.0, 2)
        else:
            final_collapsed_per = None

        summary = {
            "code_type": detected_code,
            "retrieval_metric": self.retrieval_metric,
            "csls_k": self.csls_k if self.retrieval_metric == "csls" else None,
            "total_sequences": len(active_tags),
            "total_ref_tokens": total_ref_tokens,
            "total_errors": total_errs,
            "substitutions": total_subs,
            "deletions": total_dels,
            "insertions": total_inss,
            "correct_tokens": total_correct,
            "phoneme_error_rate_pct": final_per,
            "phoneme_error_rate_collapsed_runs_pct": final_collapsed_per,
            "collapsed_substitutions": total_collapsed_subs,
            "collapsed_deletions": total_collapsed_dels,
            "collapsed_insertions": total_collapsed_inss,
            "token_accuracy_pct": accuracy,
            "audio_embeddings_path": audio_emb_path,
            "text_embeddings_path": text_emb_path,
            "cheating_segmentation_path": seg_path,
        }
        if total_ref_tokens == 0:
            summary["status"] = "ERROR: No reference tokens matched. Check dataset split compatibility with oracle segmentation."

        token_report_section = ""
        if detected_code in ("diphone", "triphone") and total_ref_ngram_tokens > 0:
            tot_t_errs = total_token_subs + total_token_dels + total_token_inss
            token_ter = (tot_t_errs / total_ref_ngram_tokens) * 100.0
            summary["ngram_token_error_rate_pct"] = round(token_ter, 2)
            summary["ngram_total_errors"] = tot_t_errs
            summary["ngram_substitutions"] = total_token_subs
            summary["ngram_deletions"] = total_token_dels
            summary["ngram_insertions"] = total_token_inss
            summary["ngram_total_ref_tokens"] = total_ref_ngram_tokens
            token_report_section = (
                f"----------------------------------------------------------------------\n"
                f"NGRAM TOKEN ERROR RATE ({detected_code.upper()}): {token_ter:.2f}%\n"
                f"  N-gram Ref Tokens: {total_ref_ngram_tokens}, S: {total_token_subs}, D: {total_token_dels}, I: {total_token_inss}\n"
            )

        with open(self.out_per_summary_json.get_path(), "w") as f:
            json.dump(summary, f, indent=2)

        per_str = f"{final_per:.2f}%" if final_per is not None else "N/A"
        collapsed_per_str = f"{final_collapsed_per:.2f}%" if final_collapsed_per is not None else "N/A"
        acc_str = f"{accuracy:.2f}%" if accuracy is not None else "N/A"

        sample_section = ""
        if pra_records:
            sample_count = min(10, len(pra_records))
            sample_section = (
                f"----------------------------------------------------------------------\n"
                f"SAMPLE TRANSCRIPTION ALIGNMENTS (FIRST {sample_count} UTTERANCES)\n"
                f"Full pairwise NIST SCLITE alignment written to: {self.out_pra_report_txt.get_path()}\n"
                f"----------------------------------------------------------------------\n"
                + "".join(pra_records[:sample_count])
            )

        metric_desc = f"{self.retrieval_metric} (k={self.csls_k})" if self.retrieval_metric == "csls" else self.retrieval_metric
        report = (
            "======================================================================\n"
            "   CHEATING / DIRECT SEGMENTATION NEAREST-NEIGHBOR EVALUATION REPORT\n"
            "======================================================================\n"
            f"Code Type:              {detected_code}\n"
            f"Retrieval Metric:       {metric_desc}\n"
            f"Evaluated Sequences:    {len(active_tags)}\n"
            f"Total Reference Tokens: {total_ref_tokens}\n"
            f"Substitutions (S):      {total_subs}\n"
            f"Deletions (D):          {total_dels}\n"
            f"Insertions (I):         {total_inss}\n"
            f"Total Errors (S+D+I):   {total_errs}\n"
            f"----------------------------------------------------------------------\n"
            f"PHONEME ERROR RATE (PER):       {per_str}\n"
            f"COLLAPSED-RUNS PER (p,p->p):    {collapsed_per_str}\n"
            f"TOKEN ACCURACY:                 {acc_str}\n"
            f"{token_report_section}"
            f"{sample_section}"
            "======================================================================\n"
        )
        with open(self.out_per_report_txt.get_path(), "w") as f:
            f.write(report)

        logging.info(f"Evaluation complete. Final PER: {per_str}. Report written to {self.out_per_report_txt.get_path()}")
