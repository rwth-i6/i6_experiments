# debugger: SoftLamProbeJob CUDA OOM (224.61 GiB), 2026-09-21

Job `work/speech_llm/sae/emc/soft_scorer_jobs/SoftLamProbeJob.wAJQ26T7iZzX` (Slurm 1925281),
code `recipe/2025-10-speech-llm` @ f99f9f6, torch 2.7.1, GH200 95 GB.
Verdict: **DONE** -- model code (`soft_scorer.py`), one line, value-identical fix available.

## 1. The first real failure

`log.run.1` shows ONE traceback, no earlier warning. Frames:

* `soft_scorer_jobs.py:367` -- `torch.autograd.grad(soft, params, retain_graph=True, allow_unused=True)`
* `torch/autograd/__init__.py:502` -> `graph.py:824` -> C++ engine
* `OutOfMemoryError: Tried to allocate 224.61 GiB. GPU 0 ... 89.06 GiB is free.
  this process has 5.90 GiB in use. Of the allocated memory 4.97 GiB is allocated by PyTorch.`

Two facts from the same traceback that pin the layer:

* the FORWARD completed: `soft = tensor(0.0494, dtype=float64, grad_fn=<DivBackward0>)`, all the
  in-step asserts (path re-add, argmax) passed, and only 4.97 GiB was resident.  The 224.61 GiB is
  allocated INSIDE the backward, by a node whose forward output was a view.
* `soft_scorer_jobs.py:365-366`, `torch.autograd.grad(l_tau, params, ...)`, RAN FIRST AND SUCCEEDED.
  So the lattice DP's backward (the manual log-semiring one) is not implicated at all.

## 2. The offending tensor and the op that created it

`soft_scorer.py:491-494` (inside `segment_conditionals`, the differentiable half of the term):

```python
flat = seg_table.reshape(b, -1)                                   # [B, K*D*(S+1)]
gidx = (arange(k_n).view(1,1,-1) * d_cap + d1.unsqueeze(-1)) * s1 + s0.unsqueeze(-1)   # [B,U,K]
seg = scale * torch.gather(flat.unsqueeze(1).expand(b, u_max, flat.shape[1]), 2, gidx)
```

`expand` is free in the forward (stride-0 view) and `gather`'s OUTPUT is only `[B, U, K]`.  But
`GatherBackward` allocates `zeros(self.sizes())` -- the sizes of its INPUT, i.e. the EXPANDED
shape -- and scatter-adds into it before `ExpandBackward` reduces it back to `[B, K*D*(S+1)]`.

Offending tensor: **`[B, U_max, K*d_cap*(S+1)]`, float64, created by `torch.gather` at
`soft_scorer.py:494`, materialised by `GatherBackward0` (the node directly above
`ExpandBackward0`).**  With the lattice constants `K = cfg.n_phones = 40`,
`d_cap = max(d_max, d_max_sil) = 50` (`lattice.py:224`) and the batch's
`B * (S+1) ~ batch_size["features"] = 88000` (`ctrl_20/returnn.config:80`, `max_seqs = 128`), this
is exactly **`seg_table` replicated `U_max` times**: seg_table itself is ~1.31 GiB in fp64, and
224.61 / 1.31 = **U_max ~ 171 tokens**, the longest Viterbi string in the batch.  The arithmetic
closes on the logged number to three digits, with no free parameter.

Ruled out, each by a positive observation, not by elimination:

* the max-plus/Viterbi pass -- `viterbi_blankfree` is decorated `@torch.no_grad()`
  (`soft_scorer.py:190`) and returns detached tensors; it holds no graph.
* the lattice DP autograd (memory `lattice-dp-autograd-memory.md`, the `band x M x D x |V|`
  transition tensor kept T times) -- that path is `l_tau`, whose grad at line 365 completed.
  Same family of bug (autograd materialising a broadcast operand), different tensor and different
  function; the memory note does not cover this one.
* a dense `[B, S, T, O]` conditional -- `log_f`/`p` are `[B, U, K]` (~7 MB) and the emission path
  `rec` is a `scatter_add` over `[B, T, K]` (`soft_scorer.py:476-481`), already O(B*T*O).
* the scorer's attention -- `model.pt` config is `n_layers 4, n_heads 4, d_model 256,
  max_positions 512, vocab 42`; at L <= 512 its fp32 attention is < 0.6 GiB/layer, and it is a
  FORWARD allocation, which did not fail.
* the three `p_flat[idx]` prior gathers (`soft_scorer.py:512-515`) -- `p_flat` is `[H*K] = 67240`;
  their backward buffers are that size.

Reality-anchored check (CPU, same interpreter `env/conda/envs/speech_llm`, fp64): `x[2, 1e6]`,
`gather(x.unsqueeze(1).expand(2, 64, 1e6), 2, idx[2,64,8])` -- peak RSS unchanged by the forward,
+1.02 GB in the backward = exactly `2*64*1e6*8`.  The mechanism is reproduced, not inferred.

## 3. Upstream

Not an upstream bug and no issue to wait on.  `gather_backward = zeros(self.sizes()).scatter_add_`
is documented PyTorch semantics, unchanged across 2.x; searching the PyTorch tracker turns up only
the adjacent `scatter_add` shape issue (#27614) and the generic OOM-in-backward leak (#82218),
neither of which applies.  Nothing in RETURNN, NCCL or Sisyphus is involved: the failure is inside
one recipe-owned autograd node.  No tooling regression -- `soft_scorer.py` is new in f99f9f6 and
this is its first run at the run shape.

## 4. The minimal, value-identical fix

Gather from the un-expanded `[B, N]` table along dim 1 and reshape:

```python
seg = scale * torch.gather(flat, 1, gidx.reshape(b, u_max * k_n)).view(b, u_max, k_n)
```

Same elements, same order, same dtype; the backward buffer becomes `zeros_like(flat)` =
`[B, K*d_cap*(S+1)]` ~ 1.3 GiB fp64, which the run ALREADY pays -- `l_tau`'s manual backward forms
`seg_post` of exactly that shape and contracts it against `seg_table` (`lattice.py:84`).  Peak goes
from 224.61 GiB to ~6.3 GiB.

Verified in the job's own interpreter at fp64: values `torch.equal` and grads `torch.equal`
between the two spellings, on random `x` and random indices.  It is a rewrite of one gather, not a
change of the term: the Viterbi pass, the conditional, the straight-through estimator, the SIL
rule, the 512-position mask and the per-retained-frame normalisation are all untouched, so
`l_tau_norm`, `soft_norm`, `ratio` and every monitor the probe reports are the numbers the current
code WOULD have produced.

What would change the measured quantity, and is therefore NOT the fix: lowering `max_seqs` /
`batch_size` (moves the probe off the run shape the pre-registration names -- the ratio is read at
the pack's own batch), capping `u_max`, chunking over U with a different reduction order in fp64,
or re-running the term in fp32.

## 5. Rerun

Fix `soft_scorer.py:494` (implementer), then clear `error.run.1` AND `log.run.1` in the job dir;
the hash does not move (the file is `src/`, not a Job class; `SoftLamProbeJob.__sis_version__` is
hand-bumped and the term's code is reached through `code_object_path`, so confirm with a dry run
that `SoftLamProbeJob.wAJQ26T7iZzX` is still the path before deleting the markers).

Artifacts: log `work/speech_llm/sae/emc/soft_scorer_jobs/SoftLamProbeJob.wAJQ26T7iZzX/log.run.1`;
executor report `reports/exec_soft_probe_launch_2026-09-21.md`; implementer report
`reports/impl_soft_arm_r1_2026-09-21.md`.
