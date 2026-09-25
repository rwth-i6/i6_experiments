"""New in the port (replaces i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/eval.py
``W2vu2PerEvalJob``, w2vu2/selftrain.py ``GanPseudoLabelJob`` and their fairseq worker w2vu2/eval_per.py).

Every forward of the §1c wav2vec-U 2.0 GAN generator, in RETURNN.  The GAN itself trains in fairseq;
nothing here imports fairseq.  The chain per split:

    W2vu2GeneratorCheckpointJob       fairseq checkpoint + text dict.txt -> generator.pt, vocab.json
      -> ReturnnForwardJobV2          (w2vu2_forward_job) feature HDFs -> hyps.json
      -> W2vu2GanPerJob               dev split: greedy PER against GoldPhonesJob -> per.json
      -> W2vu2GanPseudoLabelJob       train: {"labels": {id: "p1 p2 ..."}} -> labels.json

* :class:`W2vu2GeneratorCheckpointJob` -- the generator weights alone, renamed to
  ``ConvRecognizer``'s parameter names, as a RETURNN checkpoint ``{"model", "epoch", "step"}``, plus
  the generator's output symbols (``vocab.json``: fairseq's ``Dictionary.load(dict.txt)`` order,
  the four specials first).  The fairseq checkpoint is read with ``torch.load(weights_only=True)``,
  torch's restricted unpickler: it rebuilds only tensors, storages, ``OrderedDict`` and plain
  containers and raises on any other global, so no fairseq / omegaconf class is imported or
  constructed.  fairseq 0.12 stores its config as a plain container
  (``OmegaConf.to_container(..., enum_to_str=True)``), so a fairseq 0.12 checkpoint loads this way.
  The job checks the checkpoint's own ``cfg["model"]`` against the forward's net args.
* :func:`build_w2vu2_forward_config` / :func:`w2vu2_forward_job` -- the ``ReturnnForwardJobV2`` over
  one split's feature HDFs.  The RETURNN side is :mod:`..model.w2vu2_generator` (``get_model`` is
  :func:`..model.recognizer_only.get_model`).  Batching as the posterior dump (:mod:`.posterior`).
* :class:`W2vu2GanPerJob` -- the source's PER: summed Levenshtein edits (:func:`.per.edit_counts`)
  of the collapsed SIL-free hypotheses against the dev gold, divided by the summed gold lengths.
  The gold is the port's :class:`..data.gold.GoldPhonesJob`, the same construction as the source's
  (``compute_gold``: MFA phone, stress-free fold, SIL dropped, no collapse).  On the pinned MFA
  revision it equals the banked ``GoldPhonesJob.ZGSp0hxyd2YP`` json on every utterance of both dev
  splits (checked 2026-09-25).
* :class:`W2vu2GanPseudoLabelJob` -- the §1d pseudo-labels in the source's ``labels.json`` format
  ``{"labels": {id: "p1 p2 ..."}, "utts": n, "empty": k}``, which
  :class:`..data.gold.PhoneTargetHdfJob` reads (``label_key="labels"``).

The features are the port's VAD-trimmed L15 HDFs (``data.vad.BlankfreeVadHdfJob`` ``feats``, float16,
tags = utterance ids): the frames the GAN trained on (the VAD counts equal the GAN data's).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

__all__ = [
    "read_fairseq_dict",
    "check_generator_cfg",
    "fairseq_generator_state",
    "W2vu2GeneratorCheckpointJob",
    "w2vu2_generator_checkpoint",
    "build_w2vu2_forward_config",
    "w2vu2_forward_job",
    "W2VU2_FORWARD_OUTPUT_FILES",
    "W2vu2GanPerJob",
    "W2vu2GanPseudoLabelJob",
]

#: the package whose relative code-object paths are hashed (``unhashed_package_root``)
_PKG = __package__.rsplit(".", 1)[0]

_alias_prefix = "sae/1c/gan"

#: the output files of every generator forward (``ReturnnForwardJobV2.out_files`` keys)
W2VU2_FORWARD_OUTPUT_FILES = ("hyps.json", "hyps.stats.txt")

# fairseq ``Generator`` parameter name -> ``ConvRecognizer`` parameter name
_FAIRSEQ_GENERATOR_PREFIX = "generator."
_FAIRSEQ_CONV_PREFIX = "proj.1."  # nn.Sequential(TransposeLast, Conv1d, TransposeLast)


# =============================================================================================
# checkpoint conversion
# =============================================================================================
def read_fairseq_dict(path: str) -> List[str]:
    """The output symbols of fairseq's ``Dictionary.load(path)``, in index order.

    ``Dictionary()`` adds ``<s>``, ``<pad>``, ``</s>``, ``<unk>`` (indices 0..3); ``add_from_file``
    then appends one symbol per ``"<symbol> <count>"`` line, in file order.  The
    ``#fairseq:overwrite`` flag and duplicate symbols are refused here (fairseq would reorder or
    raise); the source's ``dict.txt`` has neither.
    """
    from ..model.w2vu2_generator import FAIRSEQ_SPECIAL_SYMBOLS

    symbols = list(FAIRSEQ_SPECIAL_SYMBOLS)
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip()
            parts = line.rsplit(" ", 1)
            if len(parts) == 2 and parts[1] == "#fairseq:overwrite":
                raise ValueError(f"{path}:{n}: #fairseq:overwrite is not supported")
            if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
                raise ValueError(f"{path}:{n}: expected '<token> <cnt>', got {line!r}")
            if parts[0] in symbols:
                raise ValueError(f"{path}:{n}: duplicate symbol {parts[0]!r}")
            symbols.append(parts[0])
    return symbols


def check_generator_cfg(model_cfg: Dict[str, Any], net_args: Dict[str, Any]) -> None:
    """Raise unless the fairseq ``cfg["model"]`` builds the generator ``net_args`` describes (eval mode)."""
    k = int(model_cfg["generator_kernel"])
    pad = int(model_cfg.get("generator_pad", -1))
    checks = {
        "input_dim": (int(model_cfg["input_dim"]), int(net_args["in_dim"])),
        "generator_kernel": (k, int(net_args["kernel"])),
        "generator_stride": (int(model_cfg["generator_stride"]), int(net_args["stride"])),
        "generator_dilation": (int(model_cfg.get("generator_dilation", 1)), 1),
        "generator_pad (effective)": (k // 2 if pad < 0 else pad, int(net_args["kernel"]) // 2),
        "generator_bias": (bool(model_cfg["generator_bias"]), bool(net_args["bias"])),
        "generator_batch_norm (on)": (model_cfg["generator_batch_norm"] != 0, bool(net_args["batch_norm"])),
        "generator_residual": (bool(model_cfg["generator_residual"]), bool(net_args["residual"])),
        "generator_dropout": (float(model_cfg["generator_dropout"]), float(net_args["dropout"])),
        "n_layers": (1, int(net_args.get("n_layers", 1))),
    }
    bad = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
    if bad:
        raise ValueError(f"fairseq generator config != net args (checkpoint, net args): {bad}")


def fairseq_generator_state(model_state: Dict[str, Any]) -> Dict[str, Any]:
    """``generator.*`` entries of a fairseq ``Wav2vec_U`` state dict, renamed to ``ConvRecognizer``'s
    names: prefix stripped, ``proj.1.`` (the conv inside fairseq's Sequential) -> ``conv.``."""
    out = {}
    for key, value in model_state.items():
        if not key.startswith(_FAIRSEQ_GENERATOR_PREFIX):
            continue
        name = key[len(_FAIRSEQ_GENERATOR_PREFIX):]
        if name.startswith(_FAIRSEQ_CONV_PREFIX):
            name = "conv." + name[len(_FAIRSEQ_CONV_PREFIX):]
        out[name] = value
    return out


class W2vu2GeneratorCheckpointJob(Job):
    """fairseq wav2vec-U 2.0 checkpoint -> RETURNN checkpoint of the generator alone + its vocabulary.

    :param fairseq_checkpoint: a fairseq 0.12 ``Wav2vec_U`` checkpoint (e.g. the GAN's
        ``checkpoint_best.pt``).
    :param text_dict: the text side's ``dict.txt`` (fairseq ``task.text_data``), which fixes the
        generator's output symbols; the checkpoint does not store them.
    :param net_args: the forward's ``ConvRecognizer`` args; default
        :data:`..model.w2vu2_generator.W2VU2_NET_ARGS`.  Checked against the checkpoint's config and
        weights (a strict ``load_state_dict``).
    """

    def __init__(self, *, fairseq_checkpoint: tk.Path, text_dict: tk.Path, net_args: Optional[Dict[str, Any]] = None):
        super().__init__()
        if net_args is None:
            from ..model.w2vu2_generator import W2VU2_NET_ARGS

            net_args = W2VU2_NET_ARGS
        self.fairseq_checkpoint = fairseq_checkpoint
        self.text_dict = text_dict
        self.net_args = dict(net_args)
        self.out_checkpoint = self.output_path("generator.pt")
        self.out_vocab = self.output_path("vocab.json")
        self.out_stats = self.output_path("convert.stats.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import torch

        from ..model.recognizer_only import get_model
        from ..model.w2vu2_generator import SIL_SYMBOL

        state = torch.load(self.fairseq_checkpoint.get_path(), map_location="cpu", weights_only=True)
        model_cfg = state["cfg"]["model"]
        assert model_cfg.get("_name") == "wav2vec_u", model_cfg.get("_name")
        check_generator_cfg(model_cfg, self.net_args)

        vocab = read_fairseq_dict(self.text_dict.get_path())
        assert SIL_SYMBOL in vocab, f"no {SIL_SYMBOL} in {self.text_dict.get_path()}"
        assert len(vocab) == int(self.net_args["n_out"]), (len(vocab), self.net_args["n_out"])

        picked = fairseq_generator_state(state["model"])
        model = get_model(epoch=0, step=0, **self.net_args)
        model.load_state_dict(picked, strict=True)  # exact key set and shapes

        extra = state.get("extra_state") or {}
        history = state.get("optimizer_history") or [{}]
        epoch = int((extra.get("train_iterator") or {}).get("epoch", 0))
        step = int(history[-1].get("num_updates", 0))
        torch.save({"model": picked, "epoch": epoch, "step": step}, self.out_checkpoint.get_path())
        with open(self.out_vocab.get_path(), "w") as fh:
            json.dump(vocab, fh)

        lines = [
            f"source      = {self.fairseq_checkpoint.get_path()}",
            f"text dict   = {self.text_dict.get_path()}   symbols = {len(vocab)} (incl. 4 fairseq specials)",
            "unpickled with torch.load(weights_only=True) (no fairseq import)",
            f"kept keys   = {sorted(picked)} of {len(state['model'])}",
            f"fairseq epoch = {epoch}   num_updates = {step}   "
            f"best = {extra.get('best')}   val_loss = {extra.get('val_loss')}",
            f"net args    = {self.net_args}",
            f"parameters  = {sum(int(v.numel()) for v in picked.values())}",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def w2vu2_generator_checkpoint(
    *, fairseq_checkpoint: tk.Path, text_dict: tk.Path, alias: Optional[str] = None,
    net_args: Optional[Dict[str, Any]] = None,
) -> Tuple[W2vu2GeneratorCheckpointJob, Any]:
    """``(job, PtCheckpoint)`` of the converted generator; ``job.out_vocab`` is the forward's vocab."""
    from i6_core.returnn.training import PtCheckpoint

    job = W2vu2GeneratorCheckpointJob(fairseq_checkpoint=fairseq_checkpoint, text_dict=text_dict, net_args=net_args)
    if alias:
        job.add_alias(alias)
    return job, PtCheckpoint(job.out_checkpoint)


# =============================================================================================
# the forward
# =============================================================================================
def _forward_serializer(net_args: Dict[str, Any], extern_data: Dict[str, Any], callback_args: Dict[str, Any]):
    """``Collection`` for the generator forward (no local package copy: the recipe dir is on sys.path)."""
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import, NonhashedCode, PartialImport
    from returnn.util.pprint import pformat

    return Collection(
        serializer_objects=[
            NonhashedCode(f"extern_data = {pformat(extern_data)}\n"),
            PartialImport(
                code_object_path=f"{_PKG}.model.recognizer_only.get_model",
                unhashed_package_root=_PKG,
                hashed_arguments=net_args,
                unhashed_arguments={},
                import_as="get_model",
            ),
            Import(
                code_object_path=f"{_PKG}.model.w2vu2_generator.w2vu2_forward_step",
                unhashed_package_root=_PKG,
                import_as="forward_step",
            ),
            PartialImport(
                code_object_path=f"{_PKG}.model.w2vu2_generator.W2vu2GreedyDecodeCallback",
                unhashed_package_root=_PKG,
                hashed_arguments=callback_args,
                unhashed_arguments={},
                import_as="forward_callback",
            ),
        ],
        make_local_package_copy=False,
        packages=None,
    )


def build_w2vu2_forward_config(
    *,
    feature_hdfs: Sequence[tk.Path],
    vocab: tk.Path,
    net_args: Optional[Dict[str, Any]] = None,
    expected_num_seqs: int = 0,
    batch_size_frames: Optional[int] = None,
    max_seqs: Optional[int] = None,
):
    """ReturnnConfig of the generator forward over one split's feature HDFs (``HDFDataset``, the
    stream ``data``, float16 1024-d).

    :param vocab: ``W2vu2GeneratorCheckpointJob.out_vocab``.
    :param expected_num_seqs: if > 0, the callback asserts exactly this many utterances.
    :param batch_size_frames, max_seqs: default the posterior dump's
        (``analysis.posterior.POSTERIOR_BATCH_SIZE_FRAMES`` / ``POSTERIOR_MAX_SEQS``).
    """
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

    from .posterior import POSTERIOR_BATCH_SIZE_FRAMES, POSTERIOR_MAX_SEQS
    from .posterior_steps import FORWARD_DATA_KEY

    if net_args is None:
        from ..model.w2vu2_generator import W2VU2_NET_ARGS

        net_args = W2VU2_NET_ARGS
    net_args = dict(net_args)
    data = HDFDataset(files=list(feature_hdfs), seq_ordering="default")
    extern_data = {
        FORWARD_DATA_KEY: {"dim": net_args["in_dim"], "shape": (None, net_args["in_dim"]), "dtype": "float16"},
    }
    config = {
        "forward_data": data.as_returnn_opts(),
        "batch_size": int(batch_size_frames or POSTERIOR_BATCH_SIZE_FRAMES),
        "max_seqs": int(max_seqs or POSTERIOR_MAX_SEQS),
    }
    post_config = {"backend": "torch", "watch_memory": True}
    return ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[
            _forward_serializer(net_args, extern_data,
                                {"vocab_file": vocab, "expected_num_seqs": int(expected_num_seqs)})
        ],
    )


def w2vu2_forward_job(
    *,
    name: str,
    checkpoint,
    vocab: tk.Path,
    feature_hdfs: Sequence[tk.Path],
    expected_num_seqs: int = 0,
    returnn_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    net_args: Optional[Dict[str, Any]] = None,
    device: str = "gpu",
    time_rqmt: float = 2.0,
    mem_rqmt: int = 24,
    cpu_rqmt: int = 4,
    gpu_mem: int = 24,
):
    """The ``ReturnnForwardJobV2`` decoding one split (outputs :data:`W2VU2_FORWARD_OUTPUT_FILES`;
    ``out_files["hyps.json"]`` feeds :class:`W2vu2GanPerJob` / :class:`W2vu2GanPseudoLabelJob`).

    :param checkpoint: the ``PtCheckpoint`` of :func:`w2vu2_generator_checkpoint`.
    :param feature_hdfs: the split's ``BlankfreeVadHdfJob`` feature HDFs (all shards of the split).
    :param returnn_exe, returnn_root: default ``default_tools.RETURNN_EXE`` / ``RETURNN_ROOT``.
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2

    if returnn_exe is None or returnn_root is None:
        from ..default_tools import RETURNN_EXE, RETURNN_ROOT

        returnn_exe = RETURNN_EXE if returnn_exe is None else returnn_exe
        returnn_root = RETURNN_ROOT if returnn_root is None else returnn_root
    job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=build_w2vu2_forward_config(
            feature_hdfs=feature_hdfs, vocab=vocab, net_args=net_args, expected_num_seqs=expected_num_seqs),
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=list(W2VU2_FORWARD_OUTPUT_FILES),
        device=device,
        time_rqmt=time_rqmt,
        mem_rqmt=mem_rqmt,
        cpu_rqmt=cpu_rqmt,
    )
    if device == "gpu":
        job.rqmt["gpu_mem"] = gpu_mem
    job.add_alias(f"{_alias_prefix}/{name}/forward")
    return job


# =============================================================================================
# PER (W2vu2PerEvalJob / eval_per._score_model)
# =============================================================================================
class W2vu2GanPerJob(Job):
    """Greedy PER of one split's generator decode against the dev gold.

    PER = sum over utterances of the Levenshtein distance (substitutions + deletions + insertions,
    :func:`.per.edit_counts`) between the collapsed SIL-free hypothesis and the SIL-free gold, over
    the summed gold lengths -- the source's ``editdistance.eval`` sum.  Every gold utterance of the
    split must be decoded and vice versa.

    :param hyps: ``hyps.json`` of :func:`w2vu2_forward_job` over the split.
    :param gold: :class:`..data.gold.GoldPhonesJob` ``out_gold``.
    :param split: ``"dev-clean"`` or ``"dev-other"``.
    """

    def __init__(self, *, hyps: tk.Path, gold: tk.Path, split: str):
        super().__init__()
        self.hyps = hyps
        self.gold = gold
        self.split = split
        self.out_per = self.output_path("per.json")
        self.out_report = self.output_path("per.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from .per import edit_counts

        with open(self.gold.get_path()) as fh:
            gold = json.load(fh)[self.split]
        with open(self.hyps.get_path()) as fh:
            hyps = json.load(fh)
        assert gold, f"no gold for {self.split}"
        missing, extra = set(gold) - set(hyps), set(hyps) - set(gold)
        assert not missing and not extra, (
            f"{self.split}: {len(missing)} gold utterances not decoded (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} decoded utterances without gold (e.g. {sorted(extra)[:3]})")
        s = d = i = n = 0
        for tag in sorted(gold):
            sub, dele, ins = edit_counts(list(hyps[tag]), list(gold[tag]))
            s, d, i, n = s + sub, d + dele, i + ins, n + len(gold[tag])
        record = {"split": self.split, "per": (s + d + i) / n, "errors": s + d + i, "sub": s, "del": d,
                  "ins": i, "reference_phones": n, "utterances": len(gold),
                  "phones_emitted": sum(len(h) for h in hyps.values()),
                  "empty_hypotheses": sum(1 for h in hyps.values() if not h)}
        with open(self.out_per.get_path(), "w") as fh:
            json.dump(record, fh, indent=2)
        with open(self.out_report.get_path(), "w") as fh:
            fh.write(f"{self.split} PER={record['per']:.6f} errors={s + d + i} S={s} D={d} I={i} N={n}\n")
        print(json.dumps(record, indent=2), flush=True)


# =============================================================================================
# §1d pseudo-labels (GanPseudoLabelJob / eval_per.dump_labels)
# =============================================================================================
class W2vu2GanPseudoLabelJob(Job):
    """The GAN teacher's greedy phone strings as §1d pseudo-labels, in the source's format.

    ``labels.json`` = ``{"labels": {utterance id: "p1 p2 ..."}, "utts": n, "empty": k}``: the
    collapsed SIL-free decode joined by single spaces (an empty decode is ``""``).  The port's
    :class:`..data.gold.PhoneTargetHdfJob` reads it with ``label_key="labels"`` (and refuses any
    symbol outside ``phones.PHONES``, e.g. a fairseq special).

    :param hyps: the ``hyps.json`` of every forward over the labelled set (e.g. one per train
        shard, or one over all shards); an utterance may appear in only one.
    :param expected_num_seqs: if given, the number of labelled utterances must equal it
        (train-clean-100: 28,539).
    """

    def __init__(self, *, hyps: Sequence[tk.Path], expected_num_seqs: Optional[int] = None):
        super().__init__()
        self.hyps = list(hyps)
        assert self.hyps, "no hyps files"
        self.expected_num_seqs = expected_num_seqs
        self.out_labels = self.output_path("labels.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        labels: Dict[str, str] = {}
        for path in self.hyps:
            with open(path.get_path()) as fh:
                part = json.load(fh)
            dup = set(part) & set(labels)
            assert not dup, f"utterances decoded twice, e.g. {sorted(dup)[:3]}"
            for tag, hyp in part.items():
                labels[tag] = " ".join(hyp)
        if self.expected_num_seqs is not None:
            assert len(labels) == int(self.expected_num_seqs), (len(labels), self.expected_num_seqs)
        empty = sum(1 for v in labels.values() if not v)
        with open(self.out_labels.get_path(), "w") as fh:
            json.dump({"labels": {t: labels[t] for t in sorted(labels)}, "utts": len(labels), "empty": empty}, fh)
        print(f"pseudo-labels: {len(labels)} utts, {empty} empty (collapsed, SIL-free)", flush=True)
