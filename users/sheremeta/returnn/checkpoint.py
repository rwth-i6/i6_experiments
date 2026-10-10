"""RETURNN training helper jobs: best-checkpoint-by-WER selection."""

import os
from typing import Any, Dict, Optional, Tuple

from sisyphus import Job, Task, tk

__all__ = ["GetBestCheckpointByWerJob", "ResolveHFCheckpointJob", "hf_snapshot_dir", "resolve_names", "resolve_label"]


def hf_snapshot_dir(hub_cache_dir: str) -> str:
    """
    Resolves a HuggingFace hub cache directory to the snapshot directory ``from_pretrained`` requires.
    The cache must hold exactly one repo with exactly one ref.

    :param hub_cache_dir: HF_HUB_CACHE style directory with one ``models--`` or ``datasets--`` entry
    :return: path of the snapshot the single ref points at
    """
    entries = [
        fn
        for fn in os.listdir(hub_cache_dir)
        if fn.startswith(("models--", "datasets--"))
    ]
    assert len(entries) == 1, f"cache dir {hub_cache_dir} has repo entries {entries}"
    model_dir = os.path.join(hub_cache_dir, entries[0])
    refs = os.listdir(os.path.join(model_dir, "refs"))
    assert len(refs) == 1, f"refs dir {model_dir}/refs has entries {refs}"
    ref = open(os.path.join(model_dir, "refs", refs[0])).read().strip()
    snapshot_dir = os.path.join(model_dir, "snapshots", ref)
    assert os.path.isdir(snapshot_dir), snapshot_dir
    return snapshot_dir


class ResolveHFCheckpointJob(Job):
    """Resolve a hub-cache dir to its single-file ``model.safetensors``."""

    def __init__(self, *, hub_cache_dir: tk.Path, filename: str = "model.safetensors"):
        self.hub_cache_dir = hub_cache_dir
        self.filename = filename
        self.out_checkpoint = self.output_path("model.safetensors")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        snap = hf_snapshot_dir(self.hub_cache_dir.get_path())
        assert not any(fn.endswith(".index.json") for fn in os.listdir(snap)), (
            f"sharded checkpoint in {snap}; ResolveHFCheckpointJob expects a single-file safetensors"
        )
        src = os.path.join(snap, self.filename)
        assert os.path.isfile(src), src
        out = self.out_checkpoint.get_path()
        if os.path.lexists(out):
            os.remove(out)
        os.symlink(src, out)


def _format_name(spec: Any) -> str:
    if isinstance(spec, (list, tuple)):
        parts = [_format_name(s) for s in spec]
        return f"avg{len(parts)}_" + "_".join(parts)
    value = spec.get() if hasattr(spec, "get") else spec
    if isinstance(value, bool):
        raise TypeError(f"bool is not a checkpoint name: {value!r}")
    if isinstance(value, int):
        return f"ep{value:03d}"
    return str(value)


def resolve_names(label_names: Optional[Dict[str, Any]]) -> Dict[str, str]:
    resolved = {label: _format_name(spec) for label, spec in (label_names or {}).items()}
    return {label: resolved.get(name, name) if name != label else name for label, name in resolved.items()}


def resolve_label(label: str, names: Dict[str, str]) -> str:
    if label in names:
        return names[label]
    for prefix, name in names.items():
        if label.startswith(prefix + "_"):
            return name + label[len(prefix) :]
    return label


class GetBestCheckpointByWerJob(Job):
    """Pick the lowest-WER checkpoint among a set of already-scored recogs."""

    __sis_hash_exclude__ = {"label_names": None}

    def __init__(
        self,
        wers: Dict[str, tk.Variable],
        checkpoints: Dict[str, Any],
        label_names: Optional[Dict[str, Any]] = None,
    ):
        assert set(wers) == set(checkpoints)
        self.wers = wers
        self.checkpoints = checkpoints
        self.label_names = label_names
        self.out_summary = self.output_path("wer_summary.json")
        self.out_best_checkpoint = self.output_path("best_checkpoint.pt")
        self.out_best_label = self.output_var("best_label")
        self.out_best_wer = self.output_var("best_wer")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json

        names = resolve_names(self.label_names)
        table: Dict[str, Tuple[float, str]] = {}
        for label, var in self.wers.items():
            name = resolve_label(label, names)
            wer = float(var.get())
            if name not in table or wer < table[name][0]:
                table[name] = (wer, label)
        best_name = min(table, key=lambda n: table[n][0])
        best_wer, best_label = table[best_name]
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(
                {
                    "best_label": best_name,
                    "best_wer": best_wer,
                    "wers": {name: wer for name, (wer, _label) in sorted(table.items(), key=lambda kv: kv[1][0])},
                },
                f,
                indent=2,
            )
        ckpt = self.checkpoints[best_label]
        src = ckpt.get_path() if hasattr(ckpt, "get_path") else ckpt.path.get_path()
        dst = self.out_best_checkpoint.get_path()
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        self.out_best_label.set(best_name)
        self.out_best_wer.set(best_wer)
