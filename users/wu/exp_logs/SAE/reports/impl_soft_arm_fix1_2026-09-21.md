# Implementer: the soft-arm OOM fix (one gather), 2026-09-21

Applies the fix diagnosed in `reports/debug_soft_probe_oom_2026-09-21.md` (SoftLamProbeJob OOM,
224.61 GiB in the backward of `segment_conditionals`). Commit **0474d7f** on
`haotian_modality_matching_jupiter` of `recipe/2025-10-speech-llm` (explicit paths, NOT pushed).
Nothing launched.

## 1. The change

`src/speech_llm/sae/emc/soft_scorer.py:494` (+1/-1), inside `segment_conditionals`:

```python
-    seg = scale * torch.gather(flat.unsqueeze(1).expand(b, u_max, flat.shape[1]), 2, gidx)
+    seg = scale * torch.gather(flat, 1, gidx.reshape(b, u_max * k_n)).view(b, u_max, k_n)
```

Same elements, same order, same dtype, same `scale`; nothing after it touched (`d_ax`, `seg_ok`,
the three prior terms, the masks, the straight-through estimator). `gidx` is built by broadcast
arithmetic, so `reshape` is a view; the gather is on the un-expanded `[B, K*d_cap*(S+1)]` table and
its backward buffer is `zeros_like(flat)` instead of `U_max` copies of it.

## 2. The test (`test_soft_scorer.py` +67 l)

`test_the_segment_gather_is_the_expanded_gather_in_value_and_gradient`, with the two spellings as
module-level helpers `_seg_gather_expanded` / `_seg_gather_flat` and a `gidx` built by the
production expression (one `(d1, s0)` pair per token, distinct starts within a row, `B=2`,
`U_max=64`, `K=5`, `d_cap=7`, `S+1=64`, fp64):

* value: `torch.equal`, and gradient (via `torch.autograd.grad` against a random upstream):
  `torch.equal` -- bit-identical, not merely close.
* memory (CUDA only; the test returns early with no GPU): peak `max_memory_allocated` growth across
  the backward, measured for both spellings. Asserted `flat <= 2 x table bytes` and, as the control
  that the probe would SEE the regression, `expanded >= U_max x table bytes`.
  Measured on the GH200 this session: table 0.0342 MiB, flat form **1.14 x table**, expanded form
  **65.1 x table** (= `U_max + 1`), i.e. the mechanism the debugger named, reproduced in the test.

Caveat recorded, not asserted: when two tokens share the same `(d1, s0)` (impossible for a real
segmentation, since starts are distinct) the two spellings differ by scatter-add REASSOCIATION at
~4e-16 fp64, not by value. The test's index construction excludes that case deliberately.

## 3. Test counts

`PYTHONPATH=src:tools/sisyphus pytest src/speech_llm/sae/emc/test_soft_scorer.py -q`
(`env/conda/envs/speech_llm`, torch 2.7.1): **23 passed** (22 before + the new one), 5.9 s.
Without `tools/sisyphus` on the path 5 error out on `import sisyphus` (pre-existing, environmental).

## 4. The probe hash does NOT move

Graph load of `config/sae_4a_soft_probe.py` AFTER the fix (`scripts`-style census, one process,
no launch) prints exactly one job:

    speech_llm/sae/emc/soft_scorer_jobs/SoftLamProbeJob.wAJQ26T7iZzX

unchanged from the OOM'd run's dir. As expected: `soft_scorer.py` is model code reached through
`code_object_path` at run time, not a Job class, and `SoftLamProbeJob.__sis_version__` is
hand-bumped and was not touched. The rerun is therefore: clear `error.run.1` AND `log.run.1` in
that job dir (executor), same hash.

## 5. The other expand-then-index sites (surveyed, none is the same bug)

`soft_scorer_jobs.py` has NO `expand`/`broadcast_to`/`repeat` at all. In `soft_scorer.py`:

| site | expression | verdict |
|---|---|---|
| `:357` | `torch.gather(arc, 1, di.view(b,1,1,1).expand(b,1,arc.shape[2],2))` | not the pattern: the expand is on the INDEX; `GatherBackward` sizes come from the SOURCE `arc` `[B,D,P,2]`, which is materialised. Also inside `viterbi_blankfree`, `@torch.no_grad()`. Left as is. |
| `:479-480` | `new_zeros(b,u_max,k_n).scatter_add(1, ft...expand(b,t_max,k_n), contrib)` | not the pattern: expand on the INDEX; `self` is a real `[B,U,K]` buffer and the backward w.r.t. `contrib` is a gather producing `[B,T,K]`. Left as is. |
| `:827-831` | `idx = ...expand(b,u_max,k_n)`; `new_zeros(b,l_max+1,k_n).scatter(1, idx, x)` | not the pattern: expand on the INDEX of a scatter whose `self` is materialised. Left as is. |
| `:125` | `sel.index_select(2, hist.members.reshape(-1))` | source is a real `torch.cat` result, inside the no-grad Viterbi. Left as is. |
| `:406` | `buf.scatter_(1, idx, x)` in `_pack` | `buf` is a real `new_full`; integer path data, no autograd. Left as is. |
| `:525,529,835-845` | `gather`/`index_select` on materialised tensors | not applicable. |

Only `:494` gathered from an EXPANDED VIEW of a per-segment table, which is what makes the
backward buffer scale with `U_max`.

## 6. Not done here (out of the dispatch)

The rerun itself (markers, launch) -- executor. This report is NOT committed: the dispatch names
only `recipe/2025-10-speech-llm`; the `exp_logs` checkout is the planner's to commit.
