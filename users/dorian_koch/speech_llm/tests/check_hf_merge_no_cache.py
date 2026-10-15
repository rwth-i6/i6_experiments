"""Guard: ``HfMergeShards(add_shard_to_id=True)`` writes nothing into its INPUT shards' dirs.

It used to prefix the ids with ``ds.map(..., num_proc=4)``, which rewrote every row (audio included) and left the
result as ``cache-*.arrow`` next to the shard's data, i.e. inside another job's output: by 2026-10-15 that was 562 GiB
in ``ChatterboxInference`` outputs (two cache sets per shard). The ids are now rewritten as one column. This drives
the real ``HfMergeShards.run`` on two tiny shards and checks the shard dirs are untouched and the merge is exact.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_hf_merge_no_cache.py
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))

from datasets import Dataset, load_from_disk  # noqa: E402

from i6_experiments.users.dorian_koch.jobs.hf import HfMergeShards  # noqa: E402


class P:
    def __init__(self, p):
        self.p = p

    def get(self):
        return self.p


def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
        shards = []
        for i in range(2):
            d = os.path.join(tmp, f"shard{i}")
            rows = {
                "id": [f"r{k}" for k in range(5)],
                "audio": [bytes([i, k]) * 1000 for k in range(5)],
                "n": [k for k in range(5)],
            }
            Dataset.from_dict(rows).save_to_disk(d)
            shards.append(d)
        before = {d: sorted(os.listdir(d)) for d in shards}
        out = os.path.join(tmp, "merged")
        job = types.SimpleNamespace(shard_paths=[P(d) for d in shards], add_shard_to_id=True, out_hf=P(out))
        HfMergeShards.run(job)
        after = {d: sorted(os.listdir(d)) for d in shards}
        bad = {d: sorted(set(after[d]) - set(before[d])) for d in shards if after[d] != before[d]}
        assert not bad, f"HfMergeShards wrote into its input shard dirs: {bad}"
        m = load_from_disk(out)
        assert m.column_names == ["id", "audio", "n"], m.column_names
        assert m["id"] == [f"shard_{i}_r{k}" for i in range(2) for k in range(5)], m["id"]
        assert m["audio"] == [bytes([i, k]) * 1000 for i in range(2) for k in range(5)]
        assert m["n"] == [k for _ in range(2) for k in range(5)]
    print("check_hf_merge_no_cache: OK (input shards untouched, ids prefixed, columns and rows exact)")


if __name__ == "__main__":
    main()
