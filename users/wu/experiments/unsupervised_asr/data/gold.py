"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/eval.py (``GoldPhonesJob``)
and eval_per.py (``compute_gold``), and from speech-llm c49559ce src/speech_llm/sae/emc/init_jobs.py
(``SeedGoldPhonesJob``, ``PhoneTargetHdfJob``).

MFA gold phones from ``gilkeyio/librispeech-alignments``, pinned at :data:`MFA_REVISION` with the
sha256 of every parquet read (:data:`MFA_DEV_SHA256`, :data:`MFA_TRAIN_CLEAN_100_SHA256`), fetched by
:class:`~.hf_hub.DownloadHuggingFaceSnapshotJob`.  The source globbed ``$HF_HOME`` (default a
hardcoded cluster path) for any snapshot; the revision pinned here is the only one that cache holds.

* :class:`GoldPhonesJob` -- the dev PER reference ``{split: {utt id: [phones]}}``: each MFA phone
  through :func:`..phones.canonical_phone` (stress-free fold), silence dropped, NOT rasterised to
  frames and NOT run-length collapsed.  Evaluation only.
* :class:`SeedGoldPhonesJob` -- QUARANTINED: the same convention on the 2,849 seed utterances, for the
  supervised analysis inits only.
* :class:`PhoneTargetHdfJob` -- a phone-label json -> sparse RETURNN HDF of recognizer emission
  indices (the supervised init's targets).
"""

from __future__ import annotations

import json
import os
from functools import cache
from typing import Dict, List, Optional, Sequence

from sisyphus import Job, Task, tk

from .. import default_tools
from .hf_hub import DownloadHuggingFaceSnapshotJob

__all__ = [
    "MFA_REPO_ID",
    "MFA_REVISION",
    "MFA_DEV_SHA256",
    "MFA_TRAIN_CLEAN_100_SHA256",
    "get_mfa_alignments",
    "compute_gold",
    "GoldPhonesJob",
    "SeedGoldPhonesJob",
    "PhoneTargetHdfJob",
    "get_seed_10h_ids",
]

# -------------------------------------------------------------------------------------------------
# Pinned source: gilkeyio/librispeech-alignments (sha256 = the Hub LFS blob ids)
# -------------------------------------------------------------------------------------------------
MFA_REPO_ID = "gilkeyio/librispeech-alignments"
MFA_REVISION = "0daa1eb43dda38ee6ce752e785555380e5628f5c"
MFA_DEV_SHA256: Dict[str, str] = {
    "data/dev_clean-00000-of-00001.parquet": "927736928481fe2f05a7fe2582991cacf38b369c63754b0fc179baa6c171deee",
    "data/dev_other-00000-of-00001.parquet": "81f6bf2e3d871c41f878207820ab18a7ab72040eb4b67112a0de928a58886fb3",
}
MFA_TRAIN_CLEAN_100_SHA256: Dict[str, str] = {
    "data/train_clean_100-00000-of-00014.parquet": "c85b0d96853d139666b811f66ff36743fbdd8e8657ec1f40c66d3bb9231d476f",
    "data/train_clean_100-00001-of-00014.parquet": "95f2fa8a331bb8808c98f32ffeb4de8c611934e3e67aaa842e980cc6dbed0d98",
    "data/train_clean_100-00002-of-00014.parquet": "5fd31fbe786941915aafe9eede50bab94260c3da60fc889a3ecae80d14520158",
    "data/train_clean_100-00003-of-00014.parquet": "20797423b18426a13c072a19e5a7484fdb630d0af054da3a41a1f99363c39911",
    "data/train_clean_100-00004-of-00014.parquet": "ef2ec06a43e82268071f8ba848a6002110c84b40c4ee2275cea8c34d38c6cb60",
    "data/train_clean_100-00005-of-00014.parquet": "50c1c5e4a39883f4397d00fb6c15ee60e9d6b520643d0d956833c364840943be",
    "data/train_clean_100-00006-of-00014.parquet": "db3263790072015ff24c9f643546ea1b5770f90074c40cd58831b9afa64d36d0",
    "data/train_clean_100-00007-of-00014.parquet": "1ef9478c536c95ce211c73ca64ad24c314d14be10409170f66acd9a37a3b195d",
    "data/train_clean_100-00008-of-00014.parquet": "a9d821aae60d700b997668d32b8d3eaee806ddcf8196f80c3877df029237be47",
    "data/train_clean_100-00009-of-00014.parquet": "c833b5bb0f0601fbf62cc84e4c4c8750d451f63dc31583a88c09c000095d1d0c",
    "data/train_clean_100-00010-of-00014.parquet": "e2ad02bbc0feb953981c0fe01eb59862df25caec7cfe9383caf3b935d691f01c",
    "data/train_clean_100-00011-of-00014.parquet": "418ea545694d2eac3fab9543478e741257154630b04b1f4bafc64326c4e53136",
    "data/train_clean_100-00012-of-00014.parquet": "ca64c066e86577675d711eae4a9ad64ffa19f80d4e94320b76fcfd8c5403e22c",
    "data/train_clean_100-00013-of-00014.parquet": "2a1e5e5ebb2e4bddc1146886ce9067b6afa0289709315854eb695f925cb9bd11",
}

#: the 2,849-utterance 10 h seed (``sae.data._take_hours(train-clean-100, 10.0, random_seed=42)``), in
#: the seed dataset's ROW order -- copied from ``TransformAndMapHuggingFaceDatasetJob.DlfDYnmTEkGZ``
#: ``train`` ``id`` column.  The order matters: the k-means fit concatenates seed frames in this order.
SEED_10H_IDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_10h_ids.txt")
SEED_10H_IDS_SHA256 = "fa5bf1922f7d4062eb4eae5e8ed852fcc406d6615581b385acbdcc3a274b5798"
SEED_10H_NUM_UTTS = 2849


def get_seed_10h_ids() -> tk.Path:
    """The shipped seed id list (one id per line, seed row order)."""
    return tk.Path(SEED_10H_IDS_FILE, hash_overwrite=f"unsupervised_asr_seed_10h_ids_{SEED_10H_IDS_SHA256[:16]}")


@cache
def get_mfa_alignments(subset: str = "dev") -> tk.Path:
    """The pinned MFA parquets: ``subset="dev"`` (dev-clean + dev-other) or ``"train-clean-100"``."""
    files = {"dev": MFA_DEV_SHA256, "train-clean-100": MFA_TRAIN_CLEAN_100_SHA256}
    if subset not in files:
        raise ValueError(f"subset must be one of {sorted(files)}, got {subset!r}")
    job = DownloadHuggingFaceSnapshotJob(
        repo_id=MFA_REPO_ID, repo_type="dataset", revision=MFA_REVISION, files=files[subset],
        hf_home=default_tools.HF_HOME,
    )
    job.add_alias(f"datasets/LibriSpeech/mfa_alignments_{subset}_{MFA_REVISION[:8]}")
    return job.out_dir


# -------------------------------------------------------------------------------------------------
# dev gold (w2vu2/eval.py + eval_per.py)
# -------------------------------------------------------------------------------------------------
def compute_gold(split: str, mfa_dir: str) -> Dict[str, List[str]]:
    """utt id -> sil-free gold phone-symbol list, from the gilkeyio MFA parquet under ``mfa_dir``."""
    import glob

    import pandas as pd

    from ..phones import SIL, canonical_phone

    split_map = {"dev-clean": "dev_clean", "dev-other": "dev_other"}
    files = sorted(glob.glob(os.path.join(mfa_dir, "data", f"{split_map[split]}-*.parquet")))
    assert files, f"no gilkeyio parquet for {split}"
    gold = {}
    for fp in files:
        df = pd.read_parquet(fp, columns=["id", "phonemes"])
        for r in df.itertuples():
            # The PER reference is the phone sequence itself -- map each MFA phone to its stress-free
            # class and drop silence. NOT rasterized to frames (loses sub-frame phones) and NOT
            # run-length collapsed (the reference keeps real repeats; only the hypothesis collapses,
            # exactly as fairseq's get_tokens does on the decoded side).
            seq = [canonical_phone(p["phoneme"]) for p in r.phonemes]
            gold[r.id] = [s for s in seq if s != SIL]
    return gold


class GoldPhonesJob(Job):
    """dev gold phone sequences (sil-free, stress-free) as one json {split: {utt_id: [phones]}}.

    Checkpoint-independent, so it is computed once and shared by every PER eval.

    :param mfa_dir: the pinned MFA parquets (:func:`get_mfa_alignments` ``("dev")``); the source read
        ``$HF_HOME`` (default a hardcoded cluster path) instead.
    """

    def __init__(self, *, mfa_dir: tk.Path, splits: Sequence[str] = ("dev-clean", "dev-other")):
        super().__init__()
        self.mfa_dir = mfa_dir
        self.splits = list(splits)
        self.out_gold = self.output_path("gold.json")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        gold = {s: compute_gold(s, self.mfa_dir.get_path()) for s in self.splits}
        for s, d in gold.items():
            assert d, f"no gold for split {s}"
        with open(self.out_gold.get_path(), "w") as f:
            json.dump(gold, f)


# -------------------------------------------------------------------------------------------------
# seed gold + targets (sae/emc/init_jobs.py)
# -------------------------------------------------------------------------------------------------
class SeedGoldPhonesJob(Job):
    """QUARANTINED.  MFA gold phones for the 2849-utterance §3c seed dir -- S1b diagnostic ONLY.

    Its outputs, and every checkpoint trained on them, NEVER feed the ladder: not as a teacher, not
    as a reward, not as a selection signal, not as an initialization of any arm other than S1b
    (``SAE.md:22-31``, ``SAE_2S.md:337-340``, SAE_4A.md:148-150).  The quarantine here is
    structural: the id set is the shipped seed list, so nothing outside the registered 2849
    utterances can enter.

    Convention copied verbatim from the campaign's dev-side gold (:func:`compute_gold`): the gilkeyio
    MFA parquet, ``canonical_phone`` per phone (stress-free fold), silence dropped, NOT rasterised to
    frames and NOT run-length collapsed.

    :param seed_ids: the seed id list, one id per line (:func:`get_seed_10h_ids`).  The source took
        ``seed_dir`` (the seed-only-text dataset ``mQmb6aW1IDH5``) + ``seed_split`` and used its
        text-bearing rows: exactly these 2,849 ids.
    :param mfa_dir: the pinned train-clean-100 MFA parquets (:func:`get_mfa_alignments`
        ``("train-clean-100")``); the source took ``hf_home`` + ``mfa_glob``.
    """

    def __init__(self, *, seed_ids: tk.Path, mfa_dir: tk.Path,
                 mfa_glob: str = "data/train_clean_100-*.parquet"):
        super().__init__()
        self.seed_ids = seed_ids
        self.mfa_dir = mfa_dir
        self.mfa_glob = mfa_glob
        self.out_gold = self.output_path("seed_gold_phones.json")
        self.out_ids = self.output_path("seed_ids.json")
        self.out_stats = self.output_path("seed_gold.stats.txt")
        self.rqmt = {"cpu": 4, "mem": 32, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import glob

        import pandas as pd

        from ..phones import SIL, canonical_phone

        with open(self.seed_ids.get_path()) as fh:
            seed_ids = sorted(line.strip() for line in fh if line.strip())
        assert seed_ids, "empty seed id list"
        assert len(set(seed_ids)) == len(seed_ids), "duplicate seed ids"

        files = sorted(glob.glob(os.path.join(self.mfa_dir.get_path(), self.mfa_glob)))
        assert files, f"no MFA parquet under {self.mfa_dir.get_path()}/{self.mfa_glob}"
        wanted, gold = set(seed_ids), {}
        for fp in files:
            df = pd.read_parquet(fp, columns=["id", "phonemes"])
            for r in df.itertuples():
                uid = str(r.id)
                if uid not in wanted:
                    continue
                seq = [canonical_phone(p["phoneme"]) for p in r.phonemes]
                gold[uid] = [s for s in seq if s != SIL]
        missing = wanted - set(gold)
        assert not missing, f"{len(missing)} seed utts without an MFA alignment, e.g. {sorted(missing)[:3]}"

        with open(self.out_gold.get_path(), "w") as fh:
            json.dump(gold, fh)
        with open(self.out_ids.get_path(), "w") as fh:
            json.dump(seed_ids, fh)
        lines = [
            "QUARANTINED seed gold phones (S1b diagnostic only; never feeds the ladder)",
            f"seed ids = {self.seed_ids.get_path()}",
            f"utts = {len(gold)}   mean_phones = {sum(len(v) for v in gold.values()) / len(gold):.2f}",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


class PhoneTargetHdfJob(Job):
    """{utt id: phone string} json -> a sparse RETURNN HDF of recognizer emission indices.

    The label json is ``{"labels": {uid: "AA B ..."}}`` or a plain ``{uid: [phones]}`` map
    (``SeedGoldPhonesJob``).  Every symbol must resolve in ``phones.PHONES``; the job asserts that
    and prints the symbol census, so a label set that grew a symbol fails loudly instead of silently
    mapping to blank.
    """

    def __init__(self, *, labels_json: tk.Path, ids_json: Optional[tk.Path] = None,
                 label_key: Optional[str] = "labels"):
        super().__init__()
        self.labels_json = labels_json
        self.ids_json = ids_json
        self.label_key = label_key
        self.out_hdf = self.output_path("targets.hdf")
        self.out_stats = self.output_path("targets.stats.txt")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from collections import Counter

        import numpy as np

        from returnn.datasets.hdf import SimpleHDFWriter

        from ..model.recognizer import N_OUT, emission_index
        from ..phones import PHONES

        phone2id = {p: i for i, p in enumerate(PHONES)}
        raw = json.load(open(self.labels_json.get_path()))
        if self.label_key is not None and self.label_key in raw:
            raw = raw[self.label_key]
        if self.ids_json is not None:
            wanted = set(json.load(open(self.ids_json.get_path())))
            raw = {k: v for k, v in raw.items() if k in wanted}
            assert len(raw) == len(wanted), f"{len(wanted) - len(raw)} requested ids have no labels"

        census = Counter()
        writer = SimpleHDFWriter(filename=self.out_hdf.get_path(), dim=N_OUT, ndim=1)
        n_empty = 0
        lens = []
        for tag in sorted(raw):
            toks = raw[tag].split() if isinstance(raw[tag], str) else list(raw[tag])
            census.update(toks)
            unknown = [t for t in toks if t not in phone2id]
            assert not unknown, f"{tag}: symbols outside prior.PHONES: {sorted(set(unknown))[:5]}"
            ids = np.array([emission_index(phone2id[t]) for t in toks], dtype=np.int32)
            n_empty += ids.size == 0
            lens.append(int(ids.size))
            writer.insert_batch(ids[None, :], [int(ids.size)], [tag])
        writer.close()

        lines = [
            f"phone targets <- {self.labels_json.get_path()}",
            f"utts = {len(raw)}   empty = {n_empty}   mean_len = {sum(lens) / max(len(lens), 1):.2f}",
            f"symbol types = {len(census)} of {len(PHONES)} in prior.PHONES",
            f"symbols seen = {sorted(census)}",
            f"symbols NOT seen = {sorted(set(PHONES) - set(census))}",
            "emission index = prior.PHONES id + 1 (0 = CTC blank, 40 = SIL)",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
