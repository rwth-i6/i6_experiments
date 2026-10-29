"""
Word-by-Word Cross-Lingual Embedding Translation Baseline Job (No Neural Sequence Model Parameters).
Translates sentences word-by-word using nearest-neighbor lookup in the aligned cross-lingual
word embedding space (Artetxe et al., ACL 2017 / UNdreaMT baseline) and evaluates with BLEU-4.
"""

from typing import Dict, List, Optional, Tuple, Iterator, Union
import os
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.datasets.score_results import ScoreResult
from .bleu_eval import compute_corpus_bleu_fallback, clean_mt_tokens
from .eval_sequence_difference import compute_all_sequence_differences, format_sequence_difference_report


def normalize_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Length-normalizes a 2D matrix along axis 1."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = eps
    return (v / norms).astype(np.float32)


def load_vocab_list(vocab_path: str) -> List[str]:
    """
    Loads an ordered list of words (index 0..V-1) from vocab.txt or a RETURNN vocab dict file.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{") and content.endswith("}"):
        import ast
        vocab_dict = ast.literal_eval(content)
        # Filter out special symbols if offset by 4
        items = sorted(vocab_dict.items(), key=lambda x: int(x[1]))
        if len(items) >= 4 and items[0][0] == "<pad>":
            items = items[4:]
        return [str(k) for k, _ in items]

    return [line.strip().split()[0] for line in content.splitlines() if line.strip()]


class WordByWordCrossLingualEvalJob(Job):
    """
    Baseline translation system that uses aligned cross-lingual embeddings to translate
    source sentences into target sentences word-by-word via nearest neighbor retrieval
    (supporting cosine similarity or CSLS) and evaluates the output with BLEU-4.
    """
    __sis_hash_exclude__ = {"num_sample_sentences": 500}

    def __init__(
        self,
        src_embeddings_npy: tk.Path,
        trg_embeddings_npy: tk.Path,
        src_vocab_file: tk.Path,
        trg_vocab_file: tk.Path,
        src_test_txt: tk.Path,
        trg_ref_txt: tk.Path,
        mapping_matrix_npy: Optional[tk.Path] = None,
        direction: str = "src2trg",  # "src2trg" (X_mapped -> Z) or "trg2src" (Z -> X_mapped)
        retrieval_metric: str = "csls",  # "cosine" or "csls"
        csls_k: int = 10,
        copy_oov: bool = True,
        lowercase_eval: bool = False,
        batch_size: int = 2048,
        num_sample_sentences: int = 500,
    ):
        super().__init__()
        self.src_embeddings_npy = src_embeddings_npy
        self.trg_embeddings_npy = trg_embeddings_npy
        self.src_vocab_file = src_vocab_file
        self.trg_vocab_file = trg_vocab_file
        self.src_test_txt = src_test_txt
        self.trg_ref_txt = trg_ref_txt
        self.mapping_matrix_npy = mapping_matrix_npy
        self.direction = direction
        self.retrieval_metric = retrieval_metric
        self.csls_k = csls_k
        self.copy_oov = copy_oov
        self.lowercase_eval = lowercase_eval
        self.batch_size = batch_size
        self.num_sample_sentences = num_sample_sentences

        # Outputs
        self.out_hypotheses_txt = self.output_path("hypotheses.txt")
        self.out_search_dict = self.output_path("search_out.py")
        self.out_word_translation_table = self.output_path("word_translation_table.json")
        self.out_bleu = self.output_var("bleu", pickle=False)
        self.out_bleu_summary_json = self.output_path("bleu_summary.json")
        self.out_bleu_report_txt = self.output_path("bleu_report.txt")
        self.out_sequence_difference_summary_json = self.output_path("sequence_difference_summary.json")
        self.out_sequence_difference_report_txt = self.output_path("sequence_difference_report.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO)

        if getattr(self, "out_sequence_difference_summary_json", None) is None:
            self.out_sequence_difference_summary_json = self.output_path("sequence_difference_summary.json")
        if getattr(self, "out_sequence_difference_report_txt", None) is None:
            self.out_sequence_difference_report_txt = self.output_path("sequence_difference_report.txt")

        X = np.load(self.src_embeddings_npy.get_path()).astype(np.float32)
        Z = np.load(self.trg_embeddings_npy.get_path()).astype(np.float32)

        if self.mapping_matrix_npy is not None and os.path.exists(self.mapping_matrix_npy.get_path()):
            W = np.load(self.mapping_matrix_npy.get_path()).astype(np.float32)
            X = normalize_vectors(X) @ W

        X_norm = normalize_vectors(X)
        Z_norm = normalize_vectors(Z)

        src_vocab = load_vocab_list(self.src_vocab_file.get_path())
        trg_vocab = load_vocab_list(self.trg_vocab_file.get_path())

        # Trim to matching lengths if needed
        X_norm = X_norm[: len(src_vocab)]
        Z_norm = Z_norm[: len(trg_vocab)]

        if self.direction == "src2trg":
            query_emb = X_norm
            key_emb = Z_norm
            query_vocab = src_vocab
            key_vocab = trg_vocab
        elif self.direction == "trg2src":
            query_emb = Z_norm
            key_emb = X_norm
            query_vocab = trg_vocab
            key_vocab = src_vocab
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        Q = len(query_vocab)
        K_len = len(key_vocab)

        # Precompute nearest neighbor in key_vocab for every word in query_vocab
        if self.retrieval_metric == "csls":
            k_T = min(self.csls_k, K_len)
            k_S = min(self.csls_k, Q)
            r_T = np.zeros(Q, dtype=np.float32)
            for i in range(0, Q, self.batch_size):
                sim_b = query_emb[i : i + self.batch_size] @ key_emb.T
                topk_T = np.partition(sim_b, -k_T, axis=1)[:, -k_T:]
                r_T[i : i + self.batch_size] = np.mean(topk_T, axis=1)

            r_S = np.zeros(K_len, dtype=np.float32)
            for j in range(0, K_len, self.batch_size):
                sim_b = key_emb[j : j + self.batch_size] @ query_emb.T
                topk_S = np.partition(sim_b, -k_S, axis=1)[:, -k_S:]
                r_S[j : j + self.batch_size] = np.mean(topk_S, axis=1)

            best_key_indices = np.zeros(Q, dtype=np.int64)
            for i in range(0, Q, self.batch_size):
                sim_b = (
                    2.0 * (query_emb[i : i + self.batch_size] @ key_emb.T)
                    - r_T[i : i + self.batch_size, None]
                    - r_S[None, :]
                )
                best_key_indices[i : i + self.batch_size] = np.argmax(sim_b, axis=1)
        else:
            best_key_indices = np.zeros(Q, dtype=np.int64)
            for i in range(0, Q, self.batch_size):
                sim_b = query_emb[i : i + self.batch_size] @ key_emb.T
                best_key_indices[i : i + self.batch_size] = np.argmax(sim_b, axis=1)

        # Build word -> translated_word lookup dictionary (also supporting case-insensitive fallback)
        word2trans: Dict[str, str] = {}
        lower2trans: Dict[str, str] = {}
        for q_idx, k_idx in enumerate(best_key_indices):
            q_word = query_vocab[q_idx]
            t_word = key_vocab[int(k_idx)]
            word2trans[q_word] = t_word
            if q_word.lower() not in lower2trans:
                lower2trans[q_word.lower()] = t_word

        with open(self.out_word_translation_table.get_path(), "w", encoding="utf-8") as f_tbl:
            json.dump(word2trans, f_tbl, indent=2, ensure_ascii=False)

        # Translate test sentences word-by-word
        with open(self.src_test_txt.get_path(), "r", encoding="utf-8") as f_src:
            src_lines = [line.strip() for line in f_src]
        with open(self.trg_ref_txt.get_path(), "r", encoding="utf-8") as f_ref:
            ref_lines = [line.strip() for line in f_ref]

        assert len(src_lines) == len(ref_lines), (
            f"Source/Reference sentence count mismatch: {len(src_lines)} vs {len(ref_lines)}"
        )

        hyp_lines = []
        total_tokens = 0
        in_vocab_tokens = 0
        oov_tokens = 0

        with (
            open(self.out_hypotheses_txt.get_path(), "w", encoding="utf-8") as f_hyp,
            open(self.out_search_dict.get_path(), "w", encoding="utf-8") as f_dict,
        ):
            f_dict.write("{\n")
            for idx, s_line in enumerate(src_lines):
                tokens = s_line.split()
                trans_tokens = []
                for tok in tokens:
                    total_tokens += 1
                    if tok in word2trans:
                        trans_tokens.append(word2trans[tok])
                        in_vocab_tokens += 1
                    elif tok.lower() in lower2trans:
                        trans_tokens.append(lower2trans[tok.lower()])
                        in_vocab_tokens += 1
                    else:
                        oov_tokens += 1
                        if self.copy_oov:
                            trans_tokens.append(tok)
                        else:
                            trans_tokens.append("<unk>")
                hyp_str = " ".join(trans_tokens)
                hyp_lines.append(hyp_str)
                f_hyp.write(hyp_str + "\n")
                f_dict.write(f"  'line-{idx}': {hyp_str!r},\n")
            f_dict.write("}\n")

        eval_hyps = [h.lower() if self.lowercase_eval else h for h in hyp_lines]
        eval_refs = [r.lower() if self.lowercase_eval else r for r in ref_lines]

        multi_bleu_res = compute_corpus_bleu_fallback(eval_hyps, eval_refs)
        bleu_score = multi_bleu_res["bleu"]
        sacrebleu_str = multi_bleu_res["summary"]

        try:
            import sacrebleu
            sb_res = sacrebleu.corpus_bleu(eval_hyps, [eval_refs], tokenize="none", lowercase=False)
            bleu_score = float(sb_res.score)
            sacrebleu_str = str(sb_res)
        except Exception as e:
            logging.warning(f"SacreBLEU fallback to built-in BLEU-4: {e}")

        self.out_bleu.set(round(bleu_score, 2))
        oov_rate = (oov_tokens / max(1, total_tokens)) * 100.0

        summary = {
            "direction": self.direction,
            "retrieval_metric": self.retrieval_metric,
            "csls_k": self.csls_k,
            "bleu": round(bleu_score, 2),
            "precisions": [round(p, 2) for p in multi_bleu_res["precisions"]],
            "bp": round(multi_bleu_res["bp"], 4),
            "ratio": round(multi_bleu_res["ratio"], 4),
            "total_tokens": total_tokens,
            "in_vocab_tokens": in_vocab_tokens,
            "oov_tokens": oov_tokens,
            "oov_rate_percent": round(oov_rate, 2),
            "num_sentences": len(src_lines),
            "sacrebleu_signature": sacrebleu_str,
        }

        # Sequence Length and Absolute Difference Evaluation
        input_lang = "de" if self.direction == "src2trg" else "en"
        target_lang = "en" if self.direction == "src2trg" else "de"
        seq_diff_summary = compute_all_sequence_differences(
            input_lines=src_lines,
            hypothesis_lines=hyp_lines,
            reference_lines=ref_lines,
        )
        seq_diff_summary["direction"] = self.direction
        seq_diff_summary["input_lang"] = input_lang
        seq_diff_summary["target_lang"] = target_lang
        seq_diff_summary["retrieval_metric"] = self.retrieval_metric
        summary["sequence_differences"] = seq_diff_summary

        with open(self.out_sequence_difference_summary_json.get_path(), "w", encoding="utf-8") as fs:
            json.dump(seq_diff_summary, fs, indent=2)

        seq_diff_report = format_sequence_difference_report(
            summary_data=seq_diff_summary,
            input_lang=input_lang,
            target_lang=target_lang,
            title=f"Sequence Difference Evaluation ({self.direction}, {self.retrieval_metric})",
        )
        with open(self.out_sequence_difference_report_txt.get_path(), "w", encoding="utf-8") as fr:
            fr.write(seq_diff_report)

        with open(self.out_bleu_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(summary, fj, indent=2)

        with open(self.out_bleu_report_txt.get_path(), "w", encoding="utf-8") as frep:
            frep.write("====================================================================\n")
            frep.write("   Word-by-Word Cross-Lingual Embedding Translation Baseline Report \n")
            frep.write("====================================================================\n")
            frep.write(f"Direction           : {self.direction}\n")
            frep.write(f"Retrieval Metric    : {self.retrieval_metric} (k={self.csls_k})\n")
            frep.write(f"Sentences Evaluated : {len(src_lines)}\n")
            frep.write(f"Total Source Tokens : {total_tokens} (In-Vocab: {in_vocab_tokens}, OOV: {oov_tokens} [{oov_rate:.2f}%])\n")
            frep.write(f"Corpus BLEU-4       : {bleu_score:.2f}\n")
            frep.write(f"Multi-BLEU Summary  : {multi_bleu_res['summary']}\n")
            frep.write(f"SacreBLEU Output    : {sacrebleu_str}\n")
            frep.write("\n" + seq_diff_report + "\n")
            frep.write("--------------------------------------------------------------------\n")
            num_samples = min(getattr(self, "num_sample_sentences", 500), len(src_lines))
            frep.write(f"Sample Word-by-Word Translations (first {num_samples} sentences):\n")
            for idx in range(num_samples):
                frep.write(f"\n[{idx}] SRC: {src_lines[idx]}\n")
                frep.write(f"[{idx}] REF: {ref_lines[idx]}\n")
                frep.write(f"[{idx}] HYP: {hyp_lines[idx]}\n")
