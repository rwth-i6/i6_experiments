"""
Count-based phoneme n-gram LM (KenLM), trained on **exactly the same phonemized text
as the neural Transformer phoneme LM** so the two are a fair head-to-head.

That text is the official LibriSpeech normalized LM corpus concatenated with the LS-960
transcripts, tokenized into EOW-phoneme sequences with the pHMM lexicon + Sequitur-G2P
fallback. We reuse `data/phon_lm.build_phon_lm_training_datasets` and take its phonemized
train text directly, so the G2P/phonemization jobs are *shared* with the neural LM (no
duplicate compute, byte-identical data). KenLM's `lmplz` then estimates a modified
Kneser-Ney n-gram and `build_binary` produces the mmap-able `.bin` consumed at decoding
time by the Python n-gram label scorer (`pytorch_networks/phmm/decoder/ngram_label_scorer.py`).

KenLM adds `<s>`/`</s>` itself; the phonemized text is pure whitespace-separated phoneme
tokens, one sentence per line (no boundary tokens) -- identical to what `LmDataset` feeds
the neural LM.
"""

import os
import shutil
import subprocess as sp
import tempfile
from typing import Any, Dict, List, Optional

from sisyphus import tk, Task, gs

from i6_core.lm.kenlm import KenLMplzJob, CreateBinaryLMJob, CompileKenLMJob

from ...data.phon_lm import LMDatasetSettings, build_phon_lm_training_datasets
from ...default_tools import KENLM_BINARY_PATH, kenlm_repo


class CompileKenLMMaxOrderJob(CompileKenLMJob):
    """
    Like i6_core's CompileKenLMJob but passes ``-DKENLM_MAX_ORDER`` to cmake.

    KenLM defaults to a max n-gram order of 6 (`set(KENLM_MAX_ORDER 6)` in CMakeLists).
    `build_binary` and the query/Model code (incl. the python `kenlm` module) then refuse
    to LOAD any model with order > 6 ("This model has order N but KenLM was compiled to
    support up to 6"). `lmplz` itself is order-agnostic (it estimates any `-o N`), so only
    the loading/query side needs rebuilding for order > 6.
    """

    def __init__(self, *, repository: tk.Path, kenlm_max_order: int = 10):
        super().__init__(repository=repository)
        self.kenlm_max_order = kenlm_max_order

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as td:
            repo = os.path.join(td, "repo")
            shutil.copytree(self.repository.get_path(), repo)
            build_dir = os.path.join(repo, "build")
            os.mkdir(build_dir)
            sp.check_call(["cmake", f"-DKENLM_MAX_ORDER={self.kenlm_max_order}", ".."], cwd=build_dir)
            sp.check_call(["make", "-j", str(self.rqmt["cpu"])], cwd=build_dir)
            shutil.copytree(os.path.join(build_dir, "bin"), self.out_binaries.get_path())


def build_phon_count_ngram_lm(
    *,
    prefix: str,
    librispeech_key: str = "train-other-960",
    order: int = 8,
    kenlm_max_order: int = 10,
    pruning: Optional[List[int]] = None,
    interpolate_unigrams: bool = True,
    discount_fallback: Optional[List[float]] = (0.5, 1.0, 1.5),
    mem: float = 96.0,
    time: float = 48.0,
) -> Dict[str, Any]:
    """
    Train a KenLM n-gram on the neural phoneme LM's phonemized training text.

    :param prefix: alias/output prefix
    :param librispeech_key: G2P/transcript key; MUST match the neural LM (train-other-960)
        so the reused phonemization jobs line up.
    :param order: n-gram order. Phonemes carry less context per token than words, so this
        should be noticeably higher than a word 4-gram; we start at 8 (sweep via the caller).
    :param kenlm_max_order: KENLM_MAX_ORDER the loading/query binaries are compiled with.
        Must be >= order (KenLM defaults to 6). Set once with headroom (10) so a single
        rebuild covers the whole order sweep. Only build_binary/queries need it; lmplz is
        order-agnostic and reuses the default KENLM_BINARY_PATH (no lm.gz re-run).
    :param pruning: per-order absolute count pruning (e.g. [0,0,0,0,1,1,1,1] for order 8);
        None = no pruning. Useful to bound model size / memory at high order.
    :param interpolate_unigrams: KenLM default True; False is SRILM-compatible.
    :param discount_fallback: fall back to these KN discounts if closed-form estimation
        fails. Phoneme corpora (tiny vocab, huge counts) routinely break the closed-form
        estimate, so we enable the standard fallback by default.
    :param mem: lmplz memory budget in GB (NOT hashed; bump freely). High orders need more.
    :param time: lmplz time budget in hours (NOT hashed).
    :return: dict with "arpa" (lm.gz), "binary" (lm.bin), "text" (phonemized corpus),
        "vocab" (phoneme vocab pkl incl. <s>/</s>), "order".
    """
    # Reuse the EXACT phonemized training text of the neural phoneme LM: LM corpus +
    # LS-960 transcripts, same lexicon + G2P. settings only affect the (unused here)
    # LmDataset objects, not the text job, so the phonemization is shared 1:1 with the
    # neural LM (sisyphus dedups the identical jobs).
    train_data = build_phon_lm_training_datasets(
        prefix=prefix,
        librispeech_key=librispeech_key,
        settings=LMDatasetSettings(train_partition_epoch=100, train_seq_ordering="laplace:.100"),
    )
    phon_text = train_data.train.corpus_file  # phonemized train text (tk.Path)
    label_datastream = train_data.datastreams["data"]

    # KenLM training: phonemes are the "words"; vocabulary is left unrestricted (the
    # phoneme set is small and closed, anything unseen maps to <unk> at query time).
    lmplz_job = KenLMplzJob(
        text=[phon_text],
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
    tk.register_output(prefix + f"/phon_count_{order}gram.arpa.gz", arpa_lm)

    # build_binary / queries must be compiled with KENLM_MAX_ORDER >= order (default 6).
    assert kenlm_max_order >= order, f"kenlm_max_order {kenlm_max_order} < order {order}"
    maxorder_kenlm = CompileKenLMMaxOrderJob(repository=kenlm_repo, kenlm_max_order=kenlm_max_order).out_binaries

    binary_job = CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=maxorder_kenlm)
    binary_job.add_alias(os.path.join(prefix, f"kenlm_binary_o{order}"))
    binary_lm = binary_job.out_lm
    tk.register_output(prefix + f"/phon_count_{order}gram.bin", binary_lm)

    return {
        "arpa": arpa_lm,
        "binary": binary_lm,
        "text": phon_text,
        "vocab": label_datastream.vocab,
        "order": order,
        "kenlm_max_order": kenlm_max_order,
    }
