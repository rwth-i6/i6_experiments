"""
Grad-based alignment for ASR

Continuation/cleanup of :mod:`exp2025_05_05_align` using the generic,
model-agnostic job classes from :mod:`exp2025_07_07_in_grads`.

Conventions:

- Grad/score extraction:
  :class:`exp2025_07_07_in_grads.jobs.extract_in_grad_scores.ExtractInGradsFromModelJob`
  Parameterized by `(mult_grad_by_inputs: bool, attr_reduction: str)`.
  Output HDF has a single score key "data".

  Mapping to the old `grad_type` strings (see :mod:`exp2025_05_05_align`):
      L01_e_grad  ↔  mult_grad_by_inputs=True,  attr_reduction="L0.1"
      L1_e_grad   ↔  mult_grad_by_inputs=True,  attr_reduction="L1"
      L2_e_grad   ↔  mult_grad_by_inputs=True,  attr_reduction="L2"
      L01_grad    ↔  mult_grad_by_inputs=False, attr_reduction="L0.1"
      L1_grad     ↔  mult_grad_by_inputs=False, attr_reduction="L1"
      L2_grad     ↔  mult_grad_by_inputs=False, attr_reduction="L2"
      dot_e_grad  ↔  mult_grad_by_inputs=True,  attr_reduction="sum"

- Chunked long-form segmentation:
  :class:`exp2025_07_07_in_grads.jobs.chunk_segmentation.ChunkSegmentationFromModelJob`

- Alignment metric (WBE) computation: reuse
  :class:`exp2025_05_05_align.CalcAlignmentMetricsJob` and
  :class:`exp2025_05_05_align.CalcChunkedAlignmentMetricsJob` (already
  model-agnostic — just take a grad-score HDF + dataset).

Next steps, in rough priority:

1. Port the prompt-sensitivity sweep (lines ~172-203 of old recipe).
2. Port the long-form Buckeye block (lines ~86-170 of old recipe), using
   `ChunkSegmentationFromModelJob` + `ExtractInGradsFromModelJob` chained.
3. Add an external-aligner baseline (MFA or WhisperX phoneme-align) computing
   WBE on the same TIMIT/Buckeye references.
4. Optional stretch: Qwen2-Audio model_config — extend
   `exp2025_07_07_in_grads.jobs.models` with a `Qwen2Audio` interface and
   register via the `type` dispatcher.
"""

from __future__ import annotations

from typing import Any, Dict
from sisyphus import tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJobV2,
)
from i6_experiments.users.zeyer.external_models.phi4multimodal import (
    download_phi4multimodal_model,
)

from sisyphus import Job
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_in_grad_scores import (
    ExtractInGradsFromModelJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.phi4mm import Phi4MM

# Alignment-metric jobs and the dict-naming helper live in the old recipe and
# are already model-agnostic. Reusing them avoids a copy-paste.
from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import (
    CalcAlignmentMetricsJob,
    _name_for_dict,
)


# (mult_grad_by_inputs, attr_reduction, old_grad_type_name_for_output_alias)
_GRAD_VARIANTS = [
    (True, "sum", "dot_e_grad"),
    (True, "L0.1", "L01_e_grad"),
    (True, "L1", "L1_e_grad"),
    (True, "L2", "L2_e_grad"),
    (False, "L0.1", "L01_grad"),
    (False, "L1", "L1_grad"),
    (False, "L2", "L2_grad"),
]


# Align-opts grid (taken from the old recipe, lines ~57-72). Trim later;
# for now keep both so we can re-confirm numbers.
_ALIGN_OPTS_GRID = [
    {"apply_softmax_over_time": True, "blank_score": -6},
    {
        "apply_softmax_over_time": True,
        "blank_score": "calc",
        "blank_score_est": "flipped_after_softmax_over_time",
        "non_blank_score_reduce": "log_mean_exp",
        "blank_score_flipped_percentile": 60,
        "apply_softmax_over_labels": True,
    },
]


_DATASET_OFFSET_FACTORS = {"timit": 1, "buckeye": 1000}


# Dotted-path identifier of Phi4MMTorch27 below, for `make_model(type=...)` dispatch.
# The generic registry imports the module via importlib when a dotted path is given.
_PHI4MM_TORCH27_TYPE = "i6_experiments.users.zeyer.experiments.exp2026_05_23_grad_align.Phi4MMTorch27"


def _phi4mm_model_config(
    model_dir: tk.Path,
    *,
    speech_prompt: str = "Transcribe the audio clip into text.",
    fake_loss_grad: bool = False,
    margin_grad: bool = False,
    first_subword_only: bool = False,
    char_level: bool = False,
) -> Dict[str, Any]:
    """
    model_config dict consumed by the generic `make_model(type=..., **opts)`
    registry in :mod:`exp2025_07_07_in_grads.jobs.models`.

    Uses Phi4MMTorch27 (this module) to apply torch 2.7 compatibility patches.

    Each variant flag, when True, adds a key to the config dict and thus
    changes the model_config hash (separate jobs). When False (default), the
    key is omitted so the hash matches the original default-variant runs.
    """
    cfg = {
        "type": _PHI4MM_TORCH27_TYPE,
        "model_dir": model_dir,
        "speech_prompt": speech_prompt,
    }
    if fake_loss_grad:
        cfg["fake_loss_grad"] = True
    if margin_grad:
        cfg["margin_grad"] = True
    if first_subword_only:
        cfg["first_subword_only"] = True
    if char_level:
        cfg["char_level"] = True
    return cfg


def _apply_torch27_patches() -> None:
    """Apply runtime monkey-patches needed to run Phi4-MM on torch 2.7. Idempotent.

    These are scoped to this recipe; they intentionally do NOT modify shared
    code in `recipe/i6_experiments/users/zeyer/...`.
    """
    # i6_experiments...batch_gather.batches_gather: the original incorrectly
    # uses indices.shape[:-1] *after* flatten which (a) does not recover the
    # original batch dims and (b) raises on empty shape on torch>=2.7.
    # Replace with a corrected variant that saves/restores batch_dims_shape.
    from i6_experiments.users.zeyer.torch import batch_gather as _bg_mod

    if not getattr(_bg_mod.batches_gather, "_torch27_patched", False):
        _inner = _bg_mod.batch_gather

        def _batches_gather_fixed(values, *, indices, num_batch_dims):
            assert num_batch_dims >= 1
            batch_dims_shape = indices.shape[:num_batch_dims]
            values = values.flatten(0, num_batch_dims - 1)
            indices = indices.flatten(0, num_batch_dims - 1)
            res = _inner(values=values, indices=indices, batch_dim=0, index_dim=1)
            if num_batch_dims > 1:
                res = res.unflatten(0, batch_dims_shape)
            return res

        _batches_gather_fixed._torch27_patched = True
        _bg_mod.batches_gather = _batches_gather_fixed


def _unwrap_checkpoint_wrappers(model) -> int:
    """Walk `model` and replace any `CheckpointWrapper` (torch.distributed) with
    the wrapped module directly. Returns the number unwrapped.

    Phi4-MM's audio encoder config enables `activation_checkpointing` which
    wraps layers with REENTRANT-impl checkpointing -- incompatible with
    `torch.autograd.grad()` on torch>=2.7. Inference-only use here, so we just
    skip checkpointing entirely (small memory bump, no semantic change).
    """
    import torch.nn as nn
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    count = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, CheckpointWrapper):
                setattr(module, name, child._checkpoint_wrapped_module)
                count += 1
    return count


class Phi4MMTorch27(Phi4MM):
    """Phi4-MM with torch 2.7 compatibility patches + several gradient-signal variants.

    All variant flags default to ``False`` so the default behavior (and Sis hashes)
    is unchanged.

    - ``fake_loss_grad=True``: reproduces the OLD ``exp2025_05_05_align``
      grad-injection trick: gradient injected at logits is
      ``softmax(0) - one_hot(target) = 1/V - one_hot(target)``, independent of
      model confidence.
    - ``margin_grad=True``: contrastive signal. Per (B, t, V) position, replace
      ``log p[v]`` with ``log p[v] - log p[top_non_v_competitor]``. After
      ``batches_gather(., target)`` this gives ``log p[target] - log p[top
      non-target]``. Amplifies the target direction vs its strongest competitor.
      Mutually exclusive with ``fake_loss_grad``.
    - ``first_subword_only=True``: per word, only use the first subword for the
      per-word loss/grad (ignore continuation subwords). Continuations are
      mostly LM-predictable so their gradients are small and add noise; this
      also normalizes signal strength across short / long words.
    - ``char_level=True``: tokenize the reference at character level (one BPE
      token per character via space-separated chars). Per-word grad becomes a
      sum over per-character grads. Hypothesis: finer-granularity log-prob
      signal may sharpen alignment. Implementation: explode words to chars
      before calling ``super().forward()``, then post-process
      ``target_start_end`` to re-group chars back into words.
    """

    def __init__(
        self,
        *,
        fake_loss_grad: bool = False,
        margin_grad: bool = False,
        first_subword_only: bool = False,
        char_level: bool = False,
        **kwargs,
    ):
        _apply_torch27_patches()
        if fake_loss_grad and margin_grad:
            raise ValueError("fake_loss_grad and margin_grad are mutually exclusive")
        self._fake_loss_grad = fake_loss_grad
        self._margin_grad = margin_grad
        self._first_subword_only = first_subword_only
        self._char_level = char_level
        super().__init__(**kwargs)
        n = _unwrap_checkpoint_wrappers(self.model)
        print(
            f"Phi4MMTorch27: unwrapped {n} CheckpointWrapper(s). "
            f"fake_loss_grad={fake_loss_grad} margin_grad={margin_grad} "
            f"first_subword_only={first_subword_only} char_level={char_level}"
        )

    def forward(self, *, raw_targets=None, raw_target_seq_lens=None, **kwargs):
        import torch

        if self._char_level and raw_targets is not None:
            # Explode each word into its characters. Each char becomes a "word"
            # from Phi4MM.forward's perspective, so its BPE tokenization treats
            # space-separated chars as one token each. After super().forward(),
            # we re-group chars back into words in target_start_end.
            assert len(raw_targets) == 1, "char_level supports batch size 1 only"
            orig_words = raw_targets[0]
            chars = [c for word in orig_words for c in word]
            char_targets = [chars]
            char_target_seq_lens = torch.tensor([len(chars)])
            # Track word boundaries in char indices.
            word_char_ranges = []
            idx = 0
            for word in orig_words:
                word_char_ranges.append((idx, idx + len(word)))
                idx += len(word)

            out = super().forward(
                raw_targets=char_targets, raw_target_seq_lens=char_target_seq_lens, **kwargs
            )
            # out.target_start_end has shape [B, num_chars+1, 2] (chars + EOS).
            tse = out.target_start_end  # on CPU
            assert tse.shape[1] == len(chars) + 1, (
                f"char_level: expected {len(chars) + 1} entries (chars + EOS), got {tse.shape[1]}"
            )
            # Build new target_start_end with one entry per WORD covering the
            # subword range of that word's characters, plus the EOS entry.
            new_entries = []
            for cstart, cend in word_char_ranges:
                t0 = int(tse[0, cstart, 0])
                t1 = int(tse[0, cend - 1, 1])
                new_entries.append([t0, t1])
            # Append the EOS entry (unchanged).
            new_entries.append([int(tse[0, -1, 0]), int(tse[0, -1, 1])])
            out.target_start_end = torch.tensor([new_entries], dtype=tse.dtype, device=tse.device)
            # Also fix up target_seq_lens since the parent recomputed from char count.
            # Downstream code uses target_start_end directly, not target_seq_lens, so this
            # is mostly cosmetic but keep it consistent.
            out.target_seq_lens = torch.tensor([len(orig_words)])
        else:
            out = super().forward(raw_targets=raw_targets, raw_target_seq_lens=raw_target_seq_lens, **kwargs)

        if out.target_start_end.device != self.device:
            out.target_start_end = out.target_start_end.to(self.device)
        if self._first_subword_only:
            # Each entry's [t0, t1] gets restricted to [t0, t0 + 1] -- just the
            # first subword. (The trailing EOS entry, added in Phi4MM.forward as
            # [n_targets, n_targets + 1], is already a single token so this is
            # a no-op for it.)
            tse = out.target_start_end  # [B, num_words+1, 2]
            out.target_start_end = torch.stack([tse[..., 0], tse[..., 0] + 1], dim=-1)
        return out

    def log_probs(self, *, forward_output, start, end):
        import torch
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice

        if self._fake_loss_grad:
            log_p = self._log_probs_fake_loss(forward_output=forward_output, start=start, end=end)
        else:
            log_p = super().log_probs(forward_output=forward_output, start=start, end=end)

        if self._margin_grad:
            # For each (B, t, V) position, replace log_p[v] with
            #   log_p[v] - log_p[top_non_v_competitor].
            # We don't know `target` here -- it's used downstream in
            # batches_gather. So we build a tensor where every entry already
            # has its "best non-self" competitor subtracted. When batches_gather
            # picks position v = target, we get
            #   log_p[target] - log_p[top non-target competitor]
            # which is the margin we want.
            top2_vals, top2_idx = log_p.topk(2, dim=-1)  # values, indices: [B, T', 2]
            top1_val = top2_vals[..., 0:1]
            top2_val = top2_vals[..., 1:2]
            top1_idx = top2_idx[..., 0:1]
            V = log_p.shape[-1]
            arange_v = torch.arange(V, device=log_p.device)
            is_top1 = arange_v.view(1, 1, V) == top1_idx  # [B, T', V]
            top_non_i = torch.where(is_top1, top2_val.expand_as(log_p), top1_val.expand_as(log_p))
            log_p = log_p - top_non_i
        return log_p

    def _log_probs_fake_loss(self, *, forward_output, start, end):
        import torch
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice

        # Fake-loss-grad path: replicate the OLD exp2025_05_05_align trick.
        # Forward value of fake_logits is 0 (so log_softmax = log(1/V) for every class),
        # but gradient flows through real `logits`.
        last_out = forward_output.outputs["last_out"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        last_out = batch_slice(last_out, (dst_text_start + start - 1, dst_text_start + end - 1))
        logits = self.model.lm_head(last_out)
        logits = logits.float()
        for f in self.logits_transform:
            logits = f(logits)
        fake_logits = logits + (-logits).detach()  # value: 0; grads -> real logits
        return fake_logits.log_softmax(-1)  # [B, T', V]


class ExtractInGradsFromModelJobTorch27(ExtractInGradsFromModelJob):
    """Wrapper that applies torch 2.7 patches *before* the base run() does its
    local imports (which bind the original function names if patched too late).
    """

    def run(self):
        _apply_torch27_patches()
        super().run()


class AddSizesToGradScoreHdfJob(Job):
    """Augment an ExtractInGradsFromModelJob HDF with a ``sizes`` key, so that
    :class:`CalcAlignmentMetricsJob` (which expects the original single-chunk
    schema) can read it.

    ``sizes[seq_idx] = [num_words_total, num_input_frames_total]`` summed across
    chunks. For short-form (one chunk per seq) this matches the original schema.
    """

    def __init__(self, *, grad_score_hdf: tk.Path):
        super().__init__()
        self.grad_score_hdf = grad_score_hdf
        self.out_hdf = self.output_path("out.hdf")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        from sisyphus import Task

        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import shutil
        import numpy as np
        import h5py

        in_path = self.grad_score_hdf.get_path()
        out_path = self.out_hdf.get_path()
        shutil.copyfile(in_path, out_path)

        with h5py.File(out_path, "r+") as f:
            num_in = f["targets/data/num_input_frames"][:]  # [n_seqs, 1] for short-form (1 chunk)
            num_words = f["targets/data/num_words"][:]  # [n_seqs, 1]
            n_seqs = f["seqTags"].shape[0]
            assert num_in.shape == num_words.shape == (n_seqs, 1), (
                f"AddSizesToGradScoreHdfJob: assumes one chunk per seq; got "
                f"num_in.shape={num_in.shape}, n_seqs={n_seqs}"
            )
            sizes = np.concatenate([num_words, num_in], axis=-1).astype(np.int32)  # [n_seqs, 2]
            f.create_dataset("targets/data/sizes", data=sizes)
            f.create_dataset("targets/labels/sizes", data=np.array([b"classes"]))
            # Register the new stream's (dim, ndim) in the targets/size attrs group.
            # Mirrors what SimpleHDFWriter would have written for extra_type={"sizes": (2, 2, "int32")}.
            f["targets/size"].attrs["sizes"] = np.array([2, 2], dtype=np.int32)
            # Extend seqLengths by one column (one sizes row per seq).
            old_sl = f["seqLengths"][:]
            new_col = np.ones((n_seqs, 1), dtype=old_sl.dtype)
            new_sl = np.concatenate([old_sl, new_col], axis=1)
            del f["seqLengths"]
            f.create_dataset("seqLengths", data=new_sl)


def py():
    """Sisyphus entry point."""
    dl_phi4mi_dir = download_phi4multimodal_model()

    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    tk.register_output("timit-dataset", dl_ds_timit.out_hub_cache_dir)

    dl_ds_buckeye = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/buckeye", repo_type="dataset")
    tk.register_output("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    # --- Short-form: TIMIT val/test, Phi4-multimodal -----------------------
    # Mirrors exp2025_05_05_align lines ~44-84 but via the generic Job.
    # Several model_config variants of the per-word gradient signal.

    for variant_suffix, variant_kwargs in [
        ("", {}),  # default: actual d log p / d inputs
        ("-fakegrad", {"fake_loss_grad": True}),  # OLD-style fake-loss-grad
        ("-margin", {"margin_grad": True}),  # log p(t) - log p(top non-t)
        ("-firstsub", {"first_subword_only": True}),  # only first subword per word
        ("-charlev", {"char_level": True}),  # tokenize ref char-by-char
    ]:
        phi4mm_cfg = _phi4mm_model_config(dl_phi4mi_dir, **variant_kwargs)
        _build_timit_phi4mm(
            phi4mm_cfg=phi4mm_cfg,
            variant_suffix=variant_suffix,
            dl_ds_timit=dl_ds_timit,
        )


def _build_timit_phi4mm(*, phi4mm_cfg: Dict[str, Any], variant_suffix: str, dl_ds_timit) -> None:
    for ds_name, ds_dl, keys in [
        ("timit", dl_ds_timit, ["val", "test"]),
    ]:
        for key in keys:
            for mult_grad_by_inputs, attr_reduction, grad_type_alias in _GRAD_VARIANTS:
                gen = ExtractInGradsFromModelJobTorch27(
                    dataset_dir=ds_dl.out_hub_cache_dir,
                    dataset_key=key,
                    model_config=phi4mm_cfg,
                    mult_grad_by_inputs=mult_grad_by_inputs,
                    attr_reduction=attr_reduction,
                )
                gen.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                name = f"phi4mm-{ds_name}-{key}-{grad_type_alias}{variant_suffix}"
                gen.add_alias(name)
                tk.register_output(f"{name}.hdf", gen.out_hdf)

                # Add `sizes` key to the HDF (CalcAlignmentMetricsJob expects it).
                hdf_with_sizes = AddSizesToGradScoreHdfJob(grad_score_hdf=gen.out_hdf)
                hdf_with_sizes.add_alias(f"{name}-sizes")

                for align_opts in _ALIGN_OPTS_GRID:
                    align_name = f"align/{name}-{_name_for_dict(align_opts)}"
                    align = CalcAlignmentMetricsJob(
                        grad_score_hdf=hdf_with_sizes.out_hdf,
                        grad_score_key="data",
                        dataset_dir=ds_dl.out_hub_cache_dir,
                        dataset_key=key,
                        dataset_offset_factors=_DATASET_OFFSET_FACTORS[ds_name],
                        align_opts=align_opts,
                    )
                    align.add_alias(align_name)
                    tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- TODO: prompt sensitivity sweep ------------------------------------
    # See exp2025_05_05_align lines ~172-203.

    # --- TODO: long-form Buckeye (chunked segmentation + grad-align) -------
    # See exp2025_05_05_align lines ~86-170. Use:
    #   from exp2025_07_07_in_grads.jobs.chunk_segmentation import ChunkSegmentationFromModelJob
    #   from exp2025_05_05_align import CalcChunkedAlignmentMetricsJob

    # --- TODO: external-aligner baselines (MFA / WhisperX phoneme align) ---
    # For comparison against the grad-align word-boundary numbers.
