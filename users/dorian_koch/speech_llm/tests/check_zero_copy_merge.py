"""Guard: shard merges link instead of copying, and the merged dataset is exactly what the old copy was.

``HfMergeShards`` and ``MergeMoshiOutputsViaSymlinks`` (clip datasets) used to ``concatenate_datasets(...)
.save_to_disk()``, a full second copy of every sharded output (1.8 TiB on 2026-10-16). They now write the same
``state.json`` / ``dataset_info.json`` with ``data-*.arrow`` as relative symlinks to the shards' files
(``jobs.hf.save_linked``); user: "make sure old jobs keep working and new behaviour exactly matches old behaviour".

Drives the real ``HfMergeShards.run`` and ``MergeMoshiOutputsViaSymlinks.merge`` on tiny shards and compares with the
OLD behaviour (``concatenate_datasets(parts).save_to_disk``, i.e. the code before the change) row by row, feature by
feature, fingerprint, format and both JSON files (``_data_files`` aside); checks the output holds no data copy; that
the id-rewrite merge and shards ``concatenate_datasets`` must cast still copy (and still equal the old output); that a
duplicate clip index still raises and a zero-row clip shard is still skipped.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_zero_copy_merge.py
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))

import datasets  # noqa: E402
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk  # noqa: E402

from i6_experiments.users.dorian_koch.jobs.hf import HfMergeShards  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.clip_store import merge_clip_datasets, write_clips  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import MergeMoshiOutputsViaSymlinks  # noqa: E402

datasets.disable_progress_bars()


class P:
    def __init__(self, p):
        self.p = p

    def get(self):
        return self.p


def old_merge(out, parts):
    """The code before 2026-10-16, verbatim: concatenate + save_to_disk."""
    concatenate_datasets(parts).save_to_disk(out)


def same_dataset(a_dir, b_dir, what):
    a, b = load_from_disk(a_dir), load_from_disk(b_dir)
    assert a.column_names == b.column_names, (what, a.column_names, b.column_names)
    assert a.features == b.features, (what, a.features, b.features)
    assert len(a) == len(b), (what, len(a), len(b))
    assert a.to_dict() == b.to_dict(), f"{what}: rows differ"
    assert a._fingerprint == b._fingerprint, (what, a._fingerprint, b._fingerprint)
    assert (a.format, a.split) == (b.format, b.split), (what, a.format, b.format)
    sa, sb = (json.load(open(os.path.join(d, "state.json"))) for d in (a_dir, b_dir))
    sa.pop("_data_files"), sb.pop("_data_files")
    assert sa == sb, (what, sa, sb)
    ia, ib = (open(os.path.join(d, "dataset_info.json")).read() for d in (a_dir, b_dir))
    assert ia == ib, f"{what}: dataset_info.json differs"


def linked_only(d, shard_dirs, what):
    """The output holds only relative links into the shards plus the two small JSON files."""
    real_shards = {os.path.realpath(s) for s in shard_dirs}
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if n in ("state.json", "dataset_info.json"):
            assert os.path.isfile(p) and not os.path.islink(p) and os.path.getsize(p) < 65536, (what, n)
            continue
        assert os.path.islink(p), f"{what}: {n} is a copied file, not a link (merge duplicated the data)"
        t = os.readlink(p)
        assert not os.path.isabs(t), (what, n, t)
        assert os.path.dirname(os.path.realpath(p)) in real_shards, (what, n, os.path.realpath(p))


def copied(d, what):
    arrows = [n for n in os.listdir(d) if n.endswith(".arrow")]
    assert arrows and not any(os.path.islink(os.path.join(d, n)) for n in arrows), f"{what}: expected a copy"


def shard(tmp, name, rows, features=None):
    d = os.path.join(tmp, name)
    Dataset.from_dict(rows, features=features).save_to_disk(d, max_shard_size="2KB")  # several arrow files per shard
    return d


def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
        feats = Features(
            {"id": Value("string"), "audio": Value("binary"), "n": Value("int64"), "v": Sequence(Value("float32"))}
        )
        shards = [
            shard(
                tmp,
                f"s{i}",
                {
                    "id": [f"d{k}" for k in range(40)],
                    "audio": [bytes([i, k]) * 50 for k in range(40)],
                    "n": list(range(40)),
                    "v": [[float(k), 0.5] for k in range(40)],
                },
                feats,
            )
            for i in range(3)
        ]
        assert all(len([n for n in os.listdir(s) if n.endswith(".arrow")]) > 1 for s in shards), (
            "want multi-file shards"
        )

        # 1. HfMergeShards without the id rewrite: links, and exactly the old dataset
        ref = os.path.join(tmp, "ref_hf")
        old_merge(ref, [load_from_disk(s) for s in shards])
        out = os.path.join(tmp, "new_hf")
        HfMergeShards.run(
            types.SimpleNamespace(shard_paths=[P(s) for s in shards], add_shard_to_id=False, out_hf=P(out))
        )
        same_dataset(out, ref, "HfMergeShards")
        linked_only(out, shards, "HfMergeShards")

        # 2. with the id rewrite: still a copy, and still exactly the old output (ids shard_{i}_{id})
        ref2, out2 = os.path.join(tmp, "ref_id"), os.path.join(tmp, "new_id")
        parts = []
        for i, s in enumerate(shards):
            ds = load_from_disk(s)
            parts.append(
                ds.map(lambda x, i=i: {"id": f"shard_{i}_{x['id']}"}, keep_in_memory=True)
            )  # the original rewrite
        old_merge(ref2, parts)
        HfMergeShards.run(
            types.SimpleNamespace(shard_paths=[P(s) for s in shards], add_shard_to_id=True, out_hf=P(out2))
        )
        a, b = load_from_disk(out2), load_from_disk(ref2)
        assert a.to_dict() == b.to_dict() and a.features == b.features, "id-rewrite merge changed"
        copied(out2, "HfMergeShards(add_shard_to_id)")

        # 3. shards concatenate_datasets must cast (a null column vs a string column): falls back to the exact copy
        c0 = shard(tmp, "c0", {"id": ["a", "b"], "x": [None, None]})
        c1 = shard(tmp, "c1", {"id": ["c", "d"], "x": ["u", "w"]})
        ref3, out3 = os.path.join(tmp, "ref_cast"), os.path.join(tmp, "new_cast")
        old_merge(ref3, [load_from_disk(c0), load_from_disk(c1)])
        HfMergeShards.run(types.SimpleNamespace(shard_paths=[P(c0), P(c1)], add_shard_to_id=False, out_hf=P(out3)))
        same_dataset(out3, ref3, "cast fallback")
        copied(out3, "cast fallback")

        # 4. MergeMoshiOutputsViaSymlinks on clip datasets: links, exactly the old merge_clip_datasets output
        clips = []
        for i, idx in enumerate(([0, 2, 4], [1, 3], [])):
            d = os.path.join(tmp, f"clip{i}")
            write_clips(d, [(k, [0.1 * k, -0.2, 0.3], 24000) for k in idx])
            clips.append(d)
        ref4, out4 = os.path.join(tmp, "ref_clip"), os.path.join(tmp, "new_clip")
        merge_clip_datasets(ref4, clips)  # the old behaviour: link=False
        MergeMoshiOutputsViaSymlinks.merge(types.SimpleNamespace(in_dirs=[P(c) for c in clips], out_merged=P(out4)))
        same_dataset(out4, ref4, "MergeMoshiOutputsViaSymlinks")
        linked_only(out4, clips[:2], "MergeMoshiOutputsViaSymlinks")
        try:
            merge_clip_datasets(os.path.join(tmp, "dup"), [clips[0], clips[0]], link=True)
        except ValueError as e:
            assert "overlap" in str(e)
        else:
            raise AssertionError("duplicate clip index no longer raises")

        # 5. a rerun into the same dir replaces its own links (sisyphus reruns a mini_task in place)
        HfMergeShards.run(
            types.SimpleNamespace(shard_paths=[P(s) for s in shards], add_shard_to_id=False, out_hf=P(out))
        )
        same_dataset(out, ref, "HfMergeShards rerun")
    print(
        "check_zero_copy_merge: OK (links only, identical rows/features/fingerprint/JSON; id-rewrite and cast cases copy)"
    )


if __name__ == "__main__":
    main()
