
import shutil 
from sisyphus import Job, Task, tk

"""
Compute wav2vec-U unsupervised metric from already generated Viterbi transcripts.

Inputs:
  - transcripts_dir: folder with transcript files (one per checkpoint)
  - kenlm: phoneme LM in KenLM binary format
  - vocab: phoneme vocabulary
  - sil token: silence token to drop before computing LM scores

Transcript file format:
  Each file must contain:
     token1 token2 token3 ...
  per line (one utterance per line)

Naming:
  File name (without extension) is treated as checkpoint ID.
"""

import os
import math
import kenlm
from pathlib import Path

def load_vocab(path):
    vocab = set()
    for line in open(path):
        tok = line.split()[0].strip()
        if tok:
            vocab.add(tok)
    return vocab

def load_transcripts(path):
    seqs = []
    for line in open(path):
        toks = line.strip().split()
        if toks:
            seqs.append(toks)
    return seqs

def score_with_lm(seqs, lm, vocab, sil=None):
    """
    Implements the LM scoring + vocabulary usage exactly as in the wav2vec-U paper.
    Returns:
       avg_NLL   : average per-utterance negative log-likelihood per token
       total_logprob : sum of all log probabilities across tokens (natural log)
       vocab_usage: fraction of vocab types used
    """
    total_logprob = 0.0
    total_nll_sum = 0.0
    total_utts = 0
    vocab_used = set()

    ln10 = math.log(10.0)

    for seq in seqs:
        # remove SIL
        toks = [t for t in seq if t != sil] if sil else seq
        if not toks:
            continue

        sent = " ".join(toks)

        # kenlm gives log10 probability
        log10_p = lm.score(sent, bos=False, eos=False)
        ln_p = log10_p * ln10

        M = len(toks)

        avg_nll_utt = -(ln_p) / M

        total_nll_sum += avg_nll_utt
        total_logprob += ln_p
        total_utts += 1

        # vocabulary usage
        for t in toks:
            if t in vocab:
                vocab_used.add(t)

    if total_utts == 0:
        avg_nll = float("inf")
    else:
        avg_nll = total_nll_sum / total_utts

    vocab_usage = len(vocab_used) / max(1, len(vocab))

    return avg_nll, total_logprob, vocab_usage, len(vocab_used)

def select_best(results, allow_margin=1.2):
    """
    Apply Section 4.3 selection:

      1. Anchor = argmin (NLL - log(U))
      2. Filter using:
             NLL(P) < NLL(anchor) + log(U(P)/U(anchor)) + log(1.2)
      3. Among survivors, pick argmax total_logprob

    results is a dict: ckpt -> {avg_nll, total_logprob, vocab_usage}
    """

    # --- Step 1: anchor ---
    anchor = None
    anchor_score = None
    for ck, v in results.items():
        U = max(v["vocab_usage"], 1e-12)
        score = v["avg_nll"] - math.log(U)
        if anchor_score is None or score < anchor_score:
            anchor_score = score
            anchor = ck

    NLL_anchor = results[anchor]["avg_nll"]
    U_anchor = max(results[anchor]["vocab_usage"], 1e-12)

    # --- Step 2: filter ---
    survivors = {}
    margin = math.log(allow_margin)

    for ck, v in results.items():
        U = max(v["vocab_usage"], 1e-12)
        rhs = NLL_anchor + math.log(U / U_anchor) + margin
        if v["avg_nll"] < rhs:
            survivors[ck] = v

    if not survivors:
        return anchor, results[anchor]

    # --- Step 3: pick best among survivors ---
    sorted_survivors = sorted(
        survivors.items(),
        key=lambda kv: kv[1]["total_logprob"],
        reverse=True  # Highest total_logprob = best
    )
    
    return sorted_survivors

def unsupervised_metric(transcripts_and_models: list[tk.Path], kenlm_path: tk.Path, vocab_path: tk.Path, sil: str = "<SIL>", allow_margin: float = 1.2):
    vocab = load_vocab(vocab_path.get_path())
    lm = kenlm.Model(kenlm_path.get_path())

    results = {}
    for p in transcripts_and_models:
        trans_path = p[0]
        model_path = p[1]
        seqs = load_transcripts(trans_path.get_path())
        avg_nll, total_logprob, vocab_usage, vocab_used_count = score_with_lm(seqs, lm, vocab, sil=sil)
        results[(trans_path, model_path)] = {
            "avg_nll": avg_nll,
            "total_logprob": total_logprob,
            "vocab_usage": vocab_usage,
            "vocab_used_count": vocab_used_count,
        }
        print(f"{(trans_path, model_path)}: NLL={avg_nll:.5f}, total_logprob={total_logprob:.2f}, vocab_usage={vocab_usage:.4f}, vocab_used_count={vocab_used_count}")

    print("\nSelecting best checkpoint...")
    sorted_survivors = select_best(results, allow_margin=allow_margin)
    return sorted_survivors


class UnsupervisedVocabUsageAndLMScoreMetric(Job):
    def __init__(
        self,
        vocab_path: tk.Path,
        kenlm_path: tk.Path,
        transcripts_and_models: list[tk.Path],
        sil: str = "<SIL>",
        allow_margin: float = 1.2,
    ):
        #transcripts_and_models each entry must be a tuple (transcript_path: tk.Path, model_checkpoint: tk.Path)
        self.vocab_path = vocab_path
        self.kenlm_path = kenlm_path
        self.transcripts_and_models = transcripts_and_models
        self.sil = sil
        self.allow_margin = allow_margin

        self.best_to_worst_models =[]
        for n in range(len(transcripts_and_models)):
            self.best_to_worst_models.append(self.output_var(f"model_rank_{n}"))

        self.rqmt = {"time": 1000, "cpu": 1, "mem": 16} 

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    def run(self):
        sorted_survivors = unsupervised_metric(
            vocab_path=self.vocab_path,
            kenlm_path=self.kenlm_path,
            transcripts_and_models=self.transcripts_and_models,
            sil=self.sil,
            allow_margin=self.allow_margin,
        )
        print("\nBest to worst models:", sorted_survivors)
        print("Best model:", sorted_survivors[0])

        for rank in range(len(sorted_survivors)):
            model_info = sorted_survivors[rank]
            self.best_to_worst_models[rank].set(model_info[0][1])  # model_info[0] is (transcript_path, model_checkpoint)

        
class Dummypath():
    def __init__(self, path:str):
        self.path=path
    def get_path(self):
        return self.path

def py():
    vocab_path=tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/PrepareWav2VecTextDataJob.wOMeudriybr1/output/text/phones/dict.phn.txt")
    kenlm_path=tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/PrepareWav2VecTextDataJob.wOMeudriybr1/output/text/phones/lm.phones.filtered.04.bin")
    transcripts=[tk.Path(path) for path in [f"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/w2vu_generate_job/ViterbiGenerateWav2VecUJob.{p}/output/transcriptions/preprocessed_audio.txt" for p in ['oKlXpcuI0zIO', 'zVrEWkBmMeBA', '3Y3xYTpuS4X6', 's7LH1esWDsrp', '3rNyz7XHaTI8', 'mEBWqjn8gndU', 'CjAhjWhhtJyN', 'xJbuoJ6ZLvbt', 'bYgEhUF7C4CI', 'pbePpO2MFiWz', 'ZLDUzPpOugK2', 'fMtoOW4yi8ZJ', 'DXFtjWtJiFB4', 'rDYEQ6JyX4bD', 'mEdl3G6ezZGv', 'ZICoob1IvdF5', 'h6aFxcDVGaKG', '3fsuo45msmMB', '22WHuwuptFdF', '8iYxVwfJvNaW', 'uD94WI9bxpgQ', 'rnVvnvUbXMvU', 'T2bMBbj3oleL', '8OdJpXJpkVYm', 'ouqMGq2xOY3y', 'RJfaOyKv0ZeC', 'mscffnDTpdoU', 'hxGgRmCvJLhc', 'Fhd4wE2VbSuM', 'ssvURqm3SdwQ', 'o5nD3J9KHeT6', 'A1mD9bo4Hkq6', 'AARs7DNooWb5', 'UkmJxjoruHNb', 'bAzMchOCqsIj', 'bSGmW34OXYAg', 'MmsDA8XKrA7R', 'yHWuiEGWem2q', 'K9DOGXwYBAmv', '1b5v7ENZ9dzX', 'XUYF0CxWCniB']]]
    transcripts_and_models = [(t, Dummypath(f"model_{i+1}.pt")) for i, t in enumerate(transcripts)]
    unsupervised_metric(vocab_path=vocab_path, kenlm_path=kenlm_path, transcripts_and_models=transcripts_and_models)
    unsupervised_metric_job =  UnsupervisedVocabUsageAndLMScoreMetric(vocab_path=vocab_path, kenlm_path=kenlm_path, transcripts_and_models=transcripts_and_models)
    tk.register_output("best_unsup_transcription", unsupervised_metric_job.best_to_worst_models[0])
