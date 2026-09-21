# E0 lexlat census CUDA OOM — diagnosis (2026-09-21)

Job: `work/speech_llm/sae/emc/lexlat_jobs/LexlatCensusJob.RDMTvt4ngAPB` (alias `sae/4a/lexlat/e0_census`),
slurm 1922782, node jpbo-014-09 (GH200, 95.0 GiB), torch 2.7.1, `error.run.1` present, `output/` EMPTY.
Read-only diagnosis: no code edited, no job launched, no GPU used.

## Verdict

First real failure = the last traceback, for once. The OOM at `lexlat.py:987` is the cause, not a
downstream symptom: nothing upstream in the log warns, the 9 pruned cells of the same utterance had
just completed, and the allocation arithmetic closes to the byte. **Owning layer: model/DP code
(`speech_llm/sae/emc/lexlat.py`, `_step`/`_slot_u`), with a contributory wrong-quantity memory guard
in `lexlat_jobs.py:810-825`.** Not the launcher, not the environment, not a tooling regression.

## (1) The 37.44 GiB tensor and the 80.37 GiB already held

`lexlat.py:987`

```python
mass = torch.where(dg.unsqueeze(1), torch.gather(slots["f0"], 2, src_o),
                   torch.gather(slots["both"], 2, src_o))
```

`slots["f0"] / slots["both"]` are `[B, O, C_prev]` float64 slices of `fwd`. `src_o` is
`src.unsqueeze(1).expand(b, o_n, N)` (a view, 0 bytes) with `N = n_new * m_max`, because `_step`
(`lexlat.py:1096-1102`) flattens the destination x multiplicity rectangle into `sidx` before calling
`_slot_u` and only reshapes afterwards (`.view(b, o_n, n_new, m_max)`).

Shape formula of each gather output:

    bytes = B * o_n * n_new * m_max * itemsize
          = B * cfg.n_offsets * C * m_max * 8          (float64: model.lattice_float64 asserted,
                                                        lexlat_jobs.py:801; dq/ds .double())
    o_n = cfg.n_offsets = 2*band + 1 = 2*25 + 1 = 51   (lexlat.py:1153, LatticeConfig band=25)

At the failing frame (traceback locals): B=1, o_n=51, n_new=16384, m_max=6014.

    1 * 51 * 16384 * 6014 = 5,025,202,176 elements
    * 8 B                 = 40,201,617,408 B = 37.4403 GiB     <-- "Tried to allocate 37.44 GiB"

Exact to the reported digits. The tensor is therefore the **gathered source-mass block
`[B, O=51, n_new*m_max]` in float64** — one per `torch.where` operand plus one output.

The 80.37 GiB already held decomposes (N = 16384*6014 = 98,533,376):

| live tensor | where | bytes | GiB |
|---|---|---|---|
| `torch.gather(slots["f0"], 2, src_o)` | :987, arg 2 already evaluated | 40.20e9 | 37.44 |
| `torch.gather(slots["both"], 2, src_o)` | :987, arg 3 already evaluated | 40.20e9 | 37.44 |
| `sidx` int64 [1,N] | `_step`:1101 | 0.788e9 | 0.73 |
| `pos` int64 [1,n_new,m_max] | `_step`:1099 | 0.788e9 | 0.73 |
| `src`, `k`, `kk` int64 [1,N] | `_slot_u`:983-985 | 3x0.788e9 | 2.20 |
| `okm`, `dg` bool [1,N] | :1100, :986 | 2x0.0985e9 | 0.18 |
| `slots` dict, `fwd`, priors, model | — | ~0.9e9 | ~0.8 |
| **sum** | | | **~79.5** |

which matches the message's "79.53 GiB is allocated by PyTorch" (the remaining 0.84 GiB to 80.37 is
the CUDA context, "non-PyTorch memory"). **The decisive point: eager PyTorch materialises BOTH
branches of `torch.where` before selecting, so the peak is 3 x 37.44 = 112.3 GiB on a 95 GiB card.
The allocation that actually fails is the third one, `torch.where`'s output.** This is documented
eager behaviour, not a framework bug (see "upstream" below).

## (2) Peak-memory formula per cell

Per frame, at `lexlat.py:987`:

    peak(C) ~ 3 * o_n * C * m_max(C) * 8 B  +  ~7 * C * m_max(C) * 8 B   (index/mask tensors)
            ~ 1224 * C * m_max(C)  bytes                                 (o_n = 51, float64)

`m_max = sel["cnt"].max()` (`_step`:1093) is the largest number of arcs landing on ONE destination
key in the frame — data dependent, and it grows with C (more source contexts can reach one
destination) and with t. Observed in this run at C=16384: `max_mult = 3100` at t=3 (info dict in the
`_forward` frame), `m_max = 6014` at t=4.

Two readings, and the conclusion is the same under both:

| C | m_max (linear in C, 0.367*C) | peak | m_max frozen at 6014 (worst case) | peak |
|---|---|---|---|---|
| 256   | ~94   | 0.03 GiB  | 6014 | 1.76 GiB |
| 1024  | ~376  | 0.44 GiB  | 6014 | 7.0 GiB  |
| 4096  | ~1504 | 7.0 GiB   | 6014 | 28.1 GiB |
| 16384 | 6014 (observed) | **112.3 GiB** | 6014 | **112.3 GiB** |

**Only the unpruned ceiling cell overflows.** Even under the assumption-free worst case (m_max does
not shrink at all with C), C=4096 peaks at 28.1 GiB and fits in 95 GiB. And this is not just
arithmetic: the traceback shows `log_z = tensor([-242.7951])` and `stats` already BOUND in the
`run()` frame, i.e. a previous cell of the same utterance returned — all 9 pruned cells
(C in {256,1024,4096} x lam in {1/3,2/3,1}) completed on this utterance before the ceiling cell died.
That is the reality anchor.

Note also that m_max was still rising at t=4 of 32, so a ceiling cell that merely fitted at t=4 would
still be at risk later in the utterance.

### The arc guard does not guard this

`lexlat_jobs.py:822` derives `max_candidates = 1.25 * C * (2K+1 + (1+wmax)(2K+1))`
= 1.25*16384*(81 + 14*81) = 24,883,200, and `lexlat.py:964` asserts the per-frame ARC COUNT against it.
The allocation, however, is `o_n * (n_new * m_max)` = 51 * 98,533,376. **`n_new*m_max` = 98.5M is
3.96x the guard's own limit of 24.9M**, because `m_max` pads every one of the 16384 destinations to
the width of the single busiest one. The guard passed and the job still OOMed: it is a guard on the
wrong quantity. Its docstring premise (`lexlat_jobs.py:810-821`) is right about arcs and silent about
the `o_n` factor and the padding factor.

## (3) Order of execution — what ran, what was written

`cells` is built at `lexlat_jobs.py:838-839` as `[(C, lam) for C in (256,1024,4096) for lam in
(1/3,2/3,1)] + [(None, lam) for lam in ...]`, inside a tag-major loop (`for tag in tags:`, :843).
`select_tags` (`blankfree_probe_jobs.py:92`) returns `sorted(Random(0).sample(sorted(tags), 100))`, i.e.
**alphabetical** order, not length order.

- **First cell in the job: `C=256, lam=1/3` on tag `116-288045-0004`** (tags[0], 161 units) — a tag
  NOT in the 20-shortest set, so it skipped the three `None` cells (`lexlat_jobs.py:869`).
- Reconstructed from the gold tag list + the feature HDF `seqLengths` (read-only): the **first
  processed tag that is in `shortest` is index 2, `1255-74899-0009`, 96 units**. That matches the
  traceback exactly (`lens = tensor([96])`, `out_lens = tensor([32])` = ceil(96/3)). So the crash
  utterance is `1255-74899-0009`, the 3rd of 100.
- **Two tags (indices 0 and 1) completed all their cells, plus the third tag's 9 pruned cells and its
  plain lattice.** All of it is lost: `census.json`, `per_utt.json` and `summary.txt` are written only
  after the whole tag loop (`lexlat_jobs.py:~988-996`), and `output/` is empty. **Nothing was banked.**
- Wall clock: model ready ~05:09:54, crash 05:10:29 — 35 s of GPU work, consistent with ~2.5 tags.

## (4) Remedies

### A. Chunk the candidate expansion (recommended, semantics-preserving, hash-neutral)

Loop over a bounded block of the **destination axis `n_new`**, not over `m_max`, in `_step`
(`lexlat.py:1097-1102`) and identically in `lexlat_forward_backward` (`lexlat.py:1485-1490`, same
`_slot_u(...).view(b, o_n, n_new, m_max)` pattern):

    for n0 in range(0, n_new, n_blk):
        sl     = slice(n0, min(n0 + n_blk, n_new))
        cnt_b  = sel["cnt"][:, sl]
        m_b    = int(cnt_b.max())            # per-block width; drops global padding as a bonus
        pos_b  = (sel["start"][:, sl].unsqueeze(-1) + arange(m_b)).clamp(...)
        sidx_b = torch.gather(sel["order"], 1, pos_b.reshape(b, -1))
        um_b   = _slot_u(slots, sidx_b, cx).view(b, o_n, sl_len, m_b)
        ...mask with okm_b & not_rep_b...
        u[:, :, sl] = torch.logsumexp(um_b, dim=3)

The iteration is over **destination contexts**, which is exactly the axis the reduction does NOT sum
over (`logsumexp(dim=3)` reduces `m_max`). Therefore **no value changes — bit-identically, not just
mathematically**: each destination's logsumexp is still taken over its complete arc list in a single
call, so the reduction set and the internal max-shift are unchanged; the per-block `m_b` only removes
padded columns that `_step`:1101 has already set to NEG_INF = -1e30, whose `exp(v - max)` is an exact
0.0 and whose removal cannot move the sum. `log_z`, `pruned`, `contexts_per_frame`, and every census
read are preserved. `max_multiplicity` must keep reporting the GLOBAL `m_max`, not a per-block one.

Peak becomes `1224 * n_blk * m_max` bytes. `n_blk = 1024` gives 7.4 GiB at m_max=6014 and still 24 GiB
at m_max=20000; an adaptive `n_blk = max(1, budget // (1224 * m_max))` is safer than a constant.
Cost: 16 sequential blocks per frame at C=16384; each block is still multi-GiB, so launch overhead is
negligible next to the HBM traffic. The `max_candidates` assert should stay (it is a cheap sanity
bound) but its docstring should be corrected to say it does not bound memory.

### B. Lower the ceiling cell — not recommended

`unpruned_ceiling=16384 -> 8192` gives ~30 GiB (linear-m_max reading) and fits. But: **the ceiling
cell is ALREADY not unpruned.** `info['pruned'] = 9.8055e-13 > 0` at t=3 and `n_keep = 16384 = c_max`
at t=3 and t=4 — `_select` (`lexlat.py:1045-1060`) only sets `n_keep = int(c_max)` when
`n_groups > c_max`, so contexts were being discarded from frame 3 of 32 onward. The cell's `exact`
flag (`lexlat_jobs.py:~908`, `pruned.max() <= 0 AND max_contexts_reached < ceiling`) would come back
**False** whatever we do. Lowering the ceiling therefore costs nothing in exactness but loses the
extrapolation point, and — unlike A — it **moves the sisyphus hash** (see 5). Chunking is strictly
better: it is what would let the ceiling go UP far enough to actually reach pruned == 0.

### C. "The 20 shortest are longer than assumed" — REFUTED

Read off the feature HDF `seqLengths` for the reconstructed subset. The 20 shortest, in units
(50 Hz), and their recognizer output frames T (stride 3):

    units: 64, 78, 80, 92, 96, 96, 99, 103, 104, 106, 109, 111, 111, 113, 117, 119, 120, 124, 124, 128
    T:     22, 26, 27, 31, 32, 32, 33,  35,  35,  36,  37,  37,  37,  38,  39,  40,  40,  42,  42,  43

(all-100 subset: min 64, median 206, max 1154 units.) The crash utterance is 96 units / T=32 — the 5th
shortest, entirely as designed — and it died at **t = 4 of 32**. The cost at :987 is per-frame and
independent of T; T only multiplies the number of times the peak is paid. Length is not the cause,
and no reduction of the subset would help.

## (5) Sisyphus hash

- `LexlatCensusJob.__sis_version__ = 1` (`lexlat_jobs.py:663`) is a **hand-bumped integer, not a file
  sha** — stated explicitly in the module docstring (`lexlat_jobs.py:28-30`), which exists precisely
  to avoid the `sis-version-file-hash-moves-on-any-edit` trap.
- The hash-relevant kwargs are the `PARAMETER:` lines in the job's `info` file: name, arm, split,
  returnn_config, checkpoint, epoch, tau, features, units, gold, greedy_reference, resources,
  n_utterances=100, subset_seed=0, context_budgets=(256,1024,4096), lam_values, escape_budget=64,
  primary_budget=1024, primary_lam=1.0, reference_budget=4096, n_unpruned=20,
  **unpruned_ceiling=16384**, max_candidates=None, and the four bars.
- **`lexlat.py` is not hashed.** It carries no `__sis_version__` and is imported at RUN time from
  inside the job bodies (`lexlat_jobs.py:225`, `:446`, and in `LexlatCensusJob.run`).

Therefore **remedy A is hash-neutral**: the fixed code reruns in the SAME job dir
`LexlatCensusJob.RDMTvt4ngAPB`, cleared with `rm error.run.1 log.run.1` (nothing in `output/` to
protect — it is empty). Two caveats: (i) remedy B moves the hash, because `unpruned_ceiling` is a
constructor kwarg; (ii) if the block size is added as a new `LexlatCensusJob.__init__` kwarg it also
moves the hash unless hash-excluded — keep it a module constant or derive it from
`torch.cuda.mem_get_info()` inside `lexlat.py`.

## Upstream

Checked: this is not a known PyTorch/NCCL/Sisyphus defect and no issue applies. torch 2.7.1 eager
evaluates both `torch.where` branch arguments before dispatch — documented semantics, not a
regression; `pytorch/pytorch#115819` confirms the `unsqueeze().expand()` index idiom used at
`_slot_u:986` is the sanctioned workaround for `gather`'s shape rule, i.e. the code uses the API as
intended. There is no upstream fix to wait for; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(suggested by the error text) would not help — only 129 MiB was reserved-but-unallocated, so this is
a genuine 112 GiB working set, not fragmentation.

## What a fix must change

`recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat.py`, `_step` lines 1097-1102 and
`lexlat_forward_backward` lines 1485-1490: block the `sidx` / `_slot_u` / mask / `logsumexp` sequence
over the `n_new` axis, with the block width derived from `m_max` and a byte budget, keeping the global
`m_max` in the returned `max_mult`. Optionally correct the memory-guard docstring at
`lexlat_jobs.py:810-821`. No change to `lexlat_jobs.py` constructor kwargs.

Report path: /e/project1/spell/wu24/2026-07-13_unsupervised/recipe/i6_experiments/users/wu/exp_logs/SAE/reports/debug_lexlat_e0_oom_2026-09-21.md
