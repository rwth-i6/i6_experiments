"""
Count-based **BPE** n-gram LM (KenLM), trained on exactly the same BPE-tokenized text as the BPE
neural LMs (``experiments/lm_bpe/trafo.py``) so the two are a fair head-to-head, and reused by the
lexicon-free BPE CTC search.

That text is the official LibriSpeech normalized LM corpus concatenated with the LS-960 transcripts,
BPE-tokenized with the SAME subword-nmt codes/vocab as the AM lexicon (we reuse
``data/bpe_lm.build_lm_training_datasets`` -> ``ApplyBPEToTextJob``, so the segmentation jobs are shared
1:1 with the neural BPE LM -- no duplicate compute, byte-identical data). KenLM's ``lmplz`` then estimates
a modified Kneser-Ney n-gram and ``build_binary`` produces the mmap-able ``.bin`` consumed at decoding
time by the Python n-gram label scorer (``pytorch_networks/phmm/decoder/ngram_label_scorer.py``, via the
BPE lexfree decoder ``rasr_phmm_lexfree_ngram_bpe_v1``).

KenLM adds ``<s>``/``</s>`` itself; the BPE text is pure whitespace-separated subword tokens (with ``@@``
continuation markers), one sentence per line -- identical to what ``LmDataset`` feeds the neural BPE LM.
The tokens are the RAW subword strings (no ``"." -> "_"`` rewrite); ``CreateCorpusBpeFsaLexiconJob`` asserts
no BPE token contains ``"."`` so the lexicon symbol (= the lexfree scorer's ``token_strings``) equals the
KenLM token. ``build_bpe_count_ngram_lm`` and the AM lexicon MUST be built with the same ``(librispeech_key,
bpe_size)`` so the shared, lru-cached ``get_subword_nmt_bpe_v2`` yields identical codes/vocab.
"""

import os
from typing import Any, Dict, List, Optional

from sisyphus import tk

from i6_core.lm.kenlm import KenLMplzJob, CreateBinaryLMJob

from ...data.bpe_lm import build_lm_training_datasets, LMDatasetSettings
from ...default_tools import KENLM_BINARY_PATH, kenlm_repo
from ..lm_phon.count_ngram import CompileKenLMMaxOrderJob


def build_bpe_count_ngram_lm(
    *,
    prefix: str,
    bpe_size: int,
    librispeech_key: str = "train-other-960",
    order: int = 4,
    kenlm_max_order: int = 10,
    pruning: Optional[List[int]] = None,
    interpolate_unigrams: bool = True,
    discount_fallback: Optional[List[float]] = (0.5, 1.0, 1.5),
    mem: float = 96.0,
    time: float = 48.0,
) -> Dict[str, Any]:
    """
    Train a KenLM n-gram on the BPE neural LM's BPE-tokenized training text.

    :param prefix: alias/output prefix
    :param bpe_size: subword-nmt merge count; MUST match the AM lexicon's ``bpe_size`` so the shared
        ``get_subword_nmt_bpe_v2`` codes/vocab make the KenLM tokens == the lexicon BPE symbols.
    :param librispeech_key: G2P/transcript key; MUST match the neural BPE LM (train-other-960) so the
        reused BPE-segmentation jobs line up.
    :param order: n-gram order. BPE subwords carry more context per token than phonemes (closer to whole
        words), so a 4-gram is a sensible default (vs. the phoneme 8-gram); sweep via the caller.
    :param kenlm_max_order: ``KENLM_MAX_ORDER`` the loading/query binaries are compiled with. Must be
        >= order (KenLM defaults to 6). Set once with headroom (10) so a single rebuild covers any order
        sweep; ``lmplz`` is order-agnostic and reuses the default ``KENLM_BINARY_PATH``.
    :param pruning: per-order absolute count pruning; None = no pruning.
    :param interpolate_unigrams: KenLM default True; False is SRILM-compatible.
    :param discount_fallback: fall back to these KN discounts if closed-form estimation fails (the BPE
        corpus has a small, closed vocab with huge counts, which can break the closed-form estimate).
    :param mem: ``lmplz`` memory budget in GB (NOT hashed; bump freely).
    :param time: ``lmplz`` time budget in hours (NOT hashed).
    :return: dict with "arpa" (lm.gz), "binary" (lm.bin), "text" (BPE corpus), "vocab" (subword vocab),
        "order", "kenlm_max_order".
    """
    # Reuse the EXACT BPE-tokenized training text of the neural BPE LM (data/bpe_lm). The LMDatasetSettings
    # only affect the (unused here) LmDataset objects, not the text jobs, so the segmentation is shared
    # 1:1 with the neural LM (sisyphus dedups the identical jobs).
    train_data = build_lm_training_datasets(
        prefix=prefix,
        librispeech_key=librispeech_key,
        bpe_size=bpe_size,
        settings=LMDatasetSettings(train_partition_epoch=100, train_seq_ordering="laplace:.100"),
    )
    bpe_text = train_data.train.corpus_file  # BPE-tokenized train text (tk.Path)
    label_datastream = train_data.datastreams["data"]

    # KenLM training: BPE subwords are the "words"; vocabulary left unrestricted (small closed set;
    # anything unseen maps to <unk> at query time).
    lmplz_job = KenLMplzJob(
        text=[bpe_text],
        order=order,
        interpolate_unigrams=interpolate_unigrams,
        pruning=pruning,
        vocabulary=None,
        discount_fallback=list(discount_fallback) if discount_fallback is not None else None,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=mem,
        time=time,
    )
    lmplz_job.add_alias(os.path.join(prefix, f"kenlm_lmplz_o{order}"))
    arpa_lm = lmplz_job.out_lm
    tk.register_output(prefix + f"/bpe_count_{order}gram.arpa.gz", arpa_lm)

    # build_binary / queries must be compiled with KENLM_MAX_ORDER >= order (default 6). The
    # CompileKenLMMaxOrderJob is shared (by content hash) with the phoneme count LM.
    assert kenlm_max_order >= order, f"kenlm_max_order {kenlm_max_order} < order {order}"
    maxorder_kenlm = CompileKenLMMaxOrderJob(repository=kenlm_repo, kenlm_max_order=kenlm_max_order).out_binaries

    binary_job = CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=maxorder_kenlm)
    binary_job.add_alias(os.path.join(prefix, f"kenlm_binary_o{order}"))
    binary_lm = binary_job.out_lm
    tk.register_output(prefix + f"/bpe_count_{order}gram.bin", binary_lm)

    return {
        "arpa": arpa_lm,
        "binary": binary_lm,
        "text": bpe_text,
        "vocab": label_datastream.vocab,
        "order": order,
        "kenlm_max_order": kenlm_max_order,
    }
