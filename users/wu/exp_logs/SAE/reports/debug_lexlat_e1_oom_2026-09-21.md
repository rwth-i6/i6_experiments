# E1 lexlat efficiency probe CUDA OOM -- diagnosis (2026-09-21)

Job: `work/speech_llm/sae/emc/lexlat_train_jobs/LexlatEfficiencyProbeJob.r4Iaa72mU27T`
(alias `sae/4a/lexlat_pack/e1_efficiency`), node jpbo-001-13 (GH200, 95.0 GiB), torch 2.7.1 / CUDA 12.6.
`error.run.1` is 0 bytes; the traceback is in `log.run.1`. `output/` holds only the regenerated
`returnn.config` -- **no measurement was banked.** Read-only: no code edited, no job launched, no GPU used.

## Verdict

First real failure = the OOM at `lexlat.py:1460` -> `lattice.py:663`. It is the cause, not a downstream
symptom: the complete lexicalised forward + manual backward for this batch had **already succeeded**
(the traceback enters through `sae_blankfree.py:125`, i.e. `fd_passes`, which runs only after
`model.lexlat.lattice_loss(...)` and `assert_no_neg_inf` returned), and the allocation arithmetic
closes on `m_max = 7416` to four digits.

**Owning layer: model/DP code -- `speech_llm/sae/emc/lexlat.py`, the MANUAL BACKWARD of
`lexlat_forward_backward` (lines 1449-1460).** Not the launcher, not the environment, not the
config, not the probe, and not a tooling regression.

The obvious wrong answer is "the batch was too long / it hit the longest batch". It did not: it died on
**batch index 0**, while the longest batch of the sub-epoch is index 48 (T = 1137, log line "sub-epoch 10:
57 batches, longest T at index 48 (T = 1137); timing 57 of them"). The failing per-frame working set is
**T-independent** -- it is `B x O x (arcs of that frame) x 8` -- so no reduction of T or of the sub-epoch
would have helped. The second obvious wrong answer is "E0's expansion bug again": E0's fix
(`a05ab16` / `299922f`, `_dest_block`, `DEST_BLOCK_BYTES = 8 GiB`) blocked the **forward** `_step`
(`lexlat.py:1180`) only. The forward completed here. The backward at 1449-1460 has **no blocking of any
kind** and is the part that died.

## (1) The live-tensor budget at the failing point

Constants of this call: `b = 114` (traceback locals at `lexlat.py:1460`), `k_n = K = 40`, `o_n = O = 51`
(`= 2*band+1`, band 25), `d_cap = 50`, `d_pad = 53`, `n_ctx = 41`, trigram `|h| = 1681`, dtype float64
(`lattice_float64: True`, returnn.config:63), `checkpoint = 32` (returnn.config:62), `C = 1024`,
`C_esc = 64`, `max_candidates = 177,292,800` (pinned = `1.25*C*(2K+1)*(2+max_homophones) * B`).

**The failing allocation identifies the frame's slot geometry.** At `lattice.py:663` the tensors created
are one float64 copy of each operand: `a = up.transpose(1,2)`, `b = bp`, both `numel = B*K*m_max*O`.

    bytes(up) = bytes(bp) = B * K * m_max * O * 8 = 1,860,480 * m_max
    12.85 GiB = 13,797,582,438 B  ->  m_max = 7416      (1,860,480 * 7416 = 13,797,319,680 B = 12.8499 GiB)

`m_max` is `_band_group`'s widest `(utterance, emitted phone)` group at that frame (`lexlat.py:1455`),
i.e. the largest number of surviving arcs of one utterance that emit the same phone.
`M = m_n` (the compacted surviving-arc count, `lexlat.py:1445`) obeys `M <= K * m_max = 296,640`.

Per-frame table, at `b = 114`, `O = 51`, `K = 40`, float64:

| tensor | line | shape | bytes | value here |
|---|---|---|---|---|
| `u` (slot forward mass) | 1449-1450 | `[B, O, M]` | `46,512 * M` | <= 12.85 GiB |
| `bn` (slot backward mass) | 1452-1453 | `[B, O, M]` | `46,512 * M` | <= 12.85 GiB |
| `suffix` | 1458 | `[B, O, M]` | `46,512 * M` | <= 12.85 GiB |
| `up` = `_pad_by_k(u)` | 1456 | `[B*K, m_max, O]` | `1,860,480 * m_max` | **12.85 GiB** |
| `bp` = `_pad_by_k(bn)` | 1457 | `[B*K, m_max, O]` | `1,860,480 * m_max` | **12.85 GiB** |
| `_logmm` operand copies | lattice.py:663 | 2 x `[B*K, m_max, O]` | `2 * 1,860,480 * m_max` | 2 x 12.85 GiB |
| `_logmm` product at :1458 | lattice.py:663-665 | `[B*K, m_max, O]` + 2 log-chain temporaries | up to 3 x 12.85 | transient |
| `yk`, `lg`, `mass_k` | 1460-1463 | `[B,K,O,O]`, `[B,K,O,D_pad]` | 94.9 / 98.6 MB | negligible |

**Five of these are simultaneously live when line 1460 asks for the sixth** (`u`, `bn`, `suffix`, `up`,
`bp` are all named locals and none is used-and-dropped): `5 x 12.85 = 64.25 GiB` for ONE frame of the
backward, plus the pair `_logmm` is about to create. That is the 83.20 GiB.

Resident tables, computed from the shapes (T = padded feature frames ~ 766-771, t_max = ceil(T/3) ~ 256,
S+1 ~ 772; see (2)) -- these are the remainder and are NOT where the failure lives:

| tensor | shape | bytes here |
|---|---|---|
| `seg_pad` (per DP call) | `[B, S+1+2W, D_pad, K]` fp64 | 1.48 GiB |
| `seg_post_pad` | `[B, S+1+2W, D, K]` fp64 | 1.40 GiB |
| tilted `seg_table` of this FD pass | `[B, K, D, S+1]` fp64 | 1.31 GiB |
| checkpoint `store`, stride 32 | `ceil(t_max/32)+1` x `[B,O,C,2]` fp64 | 10 x 88.7 MiB = 0.87 GiB |
| `_replay` cache (the live chunk) | `(32+1)` x `[B,O,C,2]` fp64 (+ctx) | 3.0 GiB |
| `bwd` | `[B,O,C,2]` fp64 | 88.7 MiB |
| slot columns of the frame | 9 cols x `[B, n_cand]` = 58 B per slot | ~0.06 GiB per 10k arcs/utt |
| carried from the MAIN pass (alive during FD) | `dp_seg` 1.31 + `out.seg_post` 1.31 + `seg` fp32 0.66 + recognizer activations | ~4-5 GiB |

Summing the resident block (~12-13 GiB) with `3M`-sized + `2 m_max`-sized tensors reproduces the message's
"81.75 GiB allocated by PyTorch" for `M` close to its ceiling `K*m_max`, i.e. an almost uniform phone
distribution over the ~2.9e5 surviving arcs of that frame (~290 arcs per surviving destination context,
against the E0 census maximum of 2562 at C = 1024 -- consistent). The escape/word-LM tables
(`res.child/word_id/unk_*`, `prior [1681,40]`, `w_pair [B,t_max,41,2]` = 19 MB) are all sub-GiB and
play no part. The banked `lattice.py` path's own tensors are NOT alive (see (3)).

Reality anchor, not inference: the message reports 752 MiB reserved-but-unallocated, so this is a genuine
working set and not fragmentation; `expandable_segments:True` would not have saved it.

## (2) Which batch, and the timing evidence

* **Batch index 0**, the first batch of sub-epoch 10, `lexicon=True` leg (`lexlat_train_jobs.py:378`,
  traceback local `index = 0`). B = 114 utterances (traceback local `b = 114`).
* **Padded T is not in the log** -- only the longest batch's is (index 48, T = 1137). Derived bound:
  RETURNN's `batch_size = {"features": 88000}` counts padded frames and `max_seqs = 128`; B = 114 < 128,
  so the frame budget bound: `114*T <= 88,000 -> T <= 771`, and adding a 115th sequence had to exceed it,
  so `T >= 766` under laplace ordering (near-equal lengths inside a batch). **T ~ 766-771,
  t_max ~ 256 recognizer frames.** This affects only the resident block, not the failing tensors.
* **NO per-batch timing was recorded, lexicon or banked.** The only `print` of a step's seconds is at
  `lexlat_train_jobs.py:427`, after BOTH legs of a batch complete; the record dict (with
  `peak_allocated_gib` / `peak_reserved_gib`) is assembled after `optimizer_step`. The job died inside
  the first leg of the first batch. **There is no lexicon full-step time, no banked full-step time, no
  ratio, and no banked peak memory anywhere in this run.** This is a probe-structure defect worth fixing
  separately: an OOM in the priced path yields zero information.
* The only timing datum that exists is wall clock. Walk 1 (the shape census) iterated all 57 batches
  between 06:49:18 and the 06:50:54 log line. The crash is at 07:00:10. So **>= 556 s elapsed inside
  [walk-2 loader re-init + h2d + the COMPLETE lexicalised forward and manual backward + the first of two
  FD tilted passes] for ONE batch**, against a budget of 1202/57 = 21 s per batch. Treat it as a lower
  bound on a single lexicalised step, not a measurement: it is the gap between a sisyphus usage poll and
  the exception. Even so it is 20x+ over the time bar, so the memory fix alone will not produce a PASS.
* The `_dest_block` spy list (`block_calls`) was populated but is only summarised into the unwritten
  record, so the measured destination-block widths are lost too.

## (3) Is the banked path's state alive? No -- this is not a probe-structure leak

`lexlat_train_jobs.py:378-379` runs `_step_once(index, raw, lexicon=True)` **first** and
`_step_once(index, raw, lexicon=False)` second. At the crash the banked `lattice.py` leg had never run
for any batch, so none of its tensors existed. The probe's only GPU-resident extras are
`cold_model` / `cold_optimizer` (`deepcopy` of the state dicts, 1,789,892 params: ~7 MB weights + ~14 MB
Adam state) -- under 25 MB, 0.03 % of the peak. `model.lexlat` is restored in a `finally`, `raw` and
`extern_data` are one batch. **The 83.20 GiB is the method's own cost at this run shape, not probe
structure.** What the probe does add is the FD pass's position: `fd_passes` runs while the main pass's
fp64 outputs and the recognizer's autograd graph are still alive (~4-5 GiB), which is exactly the
headroom the main pass had left. The main pass fits; the first tilted pass does not.

## (4) Remedies that keep the loss VALUE and the run shape (88,000 frames / max_seqs 128)

The DP is `@torch.no_grad()` with a hand-written backward and returns per-utterance
`post_q` / `seg_post` / `log_z`; the surrogate, the optimizer and the batch-level terms are assembled
outside it. So the utterance axis inside the DP can be cut without touching anything else.

**A. Chunk the DP over the utterance axis B (recommended).** Loop `lexlat_forward_backward` (and the
`fd_passes` tilts) over slices of B, concatenating the per-utterance outputs. No gradient accumulation is
needed and none of the caller's arithmetic changes.
Memory after the change, with chunk `b_c`:

    peak(b_c) ~ 6 GiB (fixed: main-pass graph, params, CUDA ctx)
              + (b_c/114) * [ 5 * 12.85 GiB (frame set) + ~10 GiB (resident DP tables) ]
              ~ 6 + 0.65 * b_c  GiB
    b_c = 16 -> ~16 GiB;  b_c = 32 -> ~27 GiB;  b_c = 57 -> ~43 GiB

Value: the batch axis is never contracted, so every per-utterance number is unchanged. It is
bit-identical if the k-padding width `m_max` is kept at the batch-global value (the GEMM's contracted
length is then unchanged); with a per-chunk `m_max` the dropped columns are exactly `NEG_INF = -1e30`,
whose `exp(v - max)` is exactly 0.0, so the value is identical up to fp64 GEMM summation order.
Time: FLOPs are unchanged; the per-frame Python work and kernel launches multiply by `114/b_c`
(~2300 frame-steps per batch today: t_max x 3 passes x (forward + replay + backward)). Expect roughly
+10-15 % at `b_c = 57`, +25-40 % at `b_c = 16`.
**Caveat on the brief's wording:** micro-batching the *train step* with gradient accumulation is NOT
value-identical -- `BatchNorm1d(1024)` in `ConvRecognizer`, `BlankfreeAggLoss` (a batch-aggregate KL) and
the `/ n_keep` normalisation are all batch-coupled. Only the DP call may be chunked.

**B. Release checkpoint tables per segment.** `store` (`lexlat.py:1310`) keeps all `ceil(t_max/32)+1`
entries for the whole backward, but the backward consumes chunks in strictly decreasing order
(`lexlat.py:1424-1426`): dropping `store[cid]` after its replay frees `(n_ck-1) * 92.7 MB` ~ 0.83 GiB
here. Also `chunk_id, cache = cid, _replay(...)` holds the OLD cache while building the new one -- setting
`cache = None` first saves another ~3.0 GiB at the switch. Value-identical, time cost zero.
**Total ~3.8 GiB against a 64 GiB per-frame set: necessary hygiene, not a fix.**

**C. float32 storage with float64 accumulation -- NOT value-identical.** Halves every table
(per-frame 64.25 -> 32.1 GiB, resident ~12 -> ~6 GiB). `_logmm` already accumulates in float64, but its
own docstring records the measured consequence of fp32 *operands*: the max-shift floor sits 87.3 nats
down instead of 708, ~1 % of entries inflate by up to +60 nats, and `post_q` row sums came out in
[3e-9, 3.9e3] (`debug_post_q_rowsum_2026-09-15`) instead of 1 +- 1e-12. This changes `log_z`, the
posteriors and therefore the gradient. It is a different objective; say so if it is ever proposed.

**D. A larger checkpoint stride makes this WORSE.** `ck = 32 -> 64` halves `store` (0.87 -> 0.44 GiB) but
doubles the live `_replay` cache (3.0 -> 6.0 GiB) and doubles the recomputed forward. The memory optimum
is `ck ~ sqrt(t_max) ~ 16` (store 1.7 + cache 1.5 = 3.2 GiB vs 3.9 today) -- a ~0.7 GiB effect either way,
and it touches nothing in the per-frame set. Value-identical (deterministic recomputation), but it is a
`returnn.config` key, so it is NOT hash-neutral (see (6)).

**E. (Beyond the brief, the finer-grained variant.)** Block lines 1449-1460 over the PHONE axis K, which
is the axis `_pad_by_k`/`_band_group` already partitions by: `yk[b,k,...]` contracts only within its own
`(b,k)` group, so a K-block needs no cross-block reduction and cuts all five tensors by `k_blk/40`
(`k_blk = 8` -> 12.85 GiB per frame). `emit_all`/`emit_off` (`_scatter_lse`, :1467-1471) would then need a
log-domain accumulator across blocks -- exact in value, last-bit different. Use it only if B-chunking
costs too much time.

## (5) C = 1024 -> C = 4096 (E0 re-declared C = 4096)

E0's census (`LexlatCensusJob.RDMTvt4ngAPB/output/census.json`, B = 1, dev-other) measures the
destination multiplicity directly: `max_multiplicity` = 2562 at C = 1024 and 7853 at C = 4096 at
lam_lex = 1 (**x3.07**), with `contexts_per_frame_mean` 1008.8 -> 4031.5.

| term | shape | scaling 1024 -> 4096 | value at B = 114 |
|---|---|---|---|
| `u`,`bn`,`suffix` `[B,O,M]` | `46,512 * M` | `M = C x mean multiplicity`: x4 (flat mean) to x12.3 (mean tracks max) | 12.85 -> 51-158 GiB each |
| `up`,`bp` `[B*K,m_max,O]` | `1,860,480 * m_max` | same, m_max ~ M/K | 12.85 -> 51-158 GiB each |
| whole per-frame set | 5 of the above | x4 to x12.3 | **64.25 -> 257-790 GiB** |
| `fwd`,`bwd`,`store`,`_replay` | `[B,O,C,2]` | linear in C: x4 | 3.9 -> 15.6 GiB |
| slot columns | `58 * B * n_cand` | linear in C: x4 | x4 |
| `seg_pad`,`seg_post_pad`,`seg_table`,`post_q`,`w_pair` | - | **C-independent** | ~5.5 GiB |

So at C = 4096 the DP chunk must be `b_c ~ 4-14` utterances merely to reach an 80 GiB footprint, i.e.
8-29 sequential chunks per batch on top of 3 DP passes per step. C = 4096 at this run shape is a
launch-overhead-bound configuration; the E1 time bar (1202 s per sub-epoch) is out of reach there unless
the per-frame cost drops by an order of magnitude.

## (6) Hash implications

* `lexlat.py`, `lattice.py` and `lexlat_train.py` carry no `__sis_version__` and are imported at RUN time
  from inside the job body / the serialized model construction. **Remedies A, B and E are hash-neutral for
  both `LexlatEfficiencyProbeJob` and `PackedBlankfreeTrainJob`**, provided the chunk width is a module
  constant in `lexlat.py` or is derived at run time from `torch.cuda.mem_get_info()`.
* The probe's hash-relevant kwargs are the `PARAMETER:` lines of `info`: name, arm (`lexlat_20`), epoch 10,
  checkpoint, features/units/originals, `returnn_config`, `max_contexts 1024`, `lam_lex 1.0`,
  `batch_frames 88000`, `max_seqs 128`, `n_steps 100`, `n_points 4`, `time_rqmt 2.0`.
  `LexlatEfficiencyProbeJob.__sis_version__ = 1` (`lexlat_train_jobs.py:142`) is a HAND-BUMPED integer, not
  a file sha (module docstring, and memory `sis-version-file-hash-moves-on-any-edit`): do not bump it, and
  editing a neighbouring class in that module does not move this hash.
* **Anything that becomes a `returnn.config` key moves BOTH hashes.** A new knob such as
  `lexlat_dp_batch`, or changing `lattice_checkpoint` (32, line 62) or `lexlat_contexts` (1024, line 68),
  enters `returnn_config` here and `arms`/`EmcArmSpec` in `PackedBlankfreeTrainJob` -- which would re-hash
  and re-run **every arm of the pack**, including the banked `ctrl_50` this probe's checkpoint comes from.
  Remedy D is therefore the expensive one in hash terms as well as the useless one in memory terms.
* Rerun: the job dir has nothing to protect (`output/` holds only the regenerated `returnn.config`), so
  `rm error.run.1 log.run.1` in `LexlatEfficiencyProbeJob.r4Iaa72mU27T` re-picks it up at the same hash.

## Upstream

Checked, and nothing applies. torch 2.7.1 / CUDA 12.6. No PyTorch, NCCL or Sisyphus issue covers this:
the message itself reports only 752.23 MiB reserved-but-unallocated, so the 83.20 GiB is live tensors and
not allocator fragmentation, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` would not change it.
`torch.matmul` materialising one float64 copy of each shifted operand is the documented semantics of the
expression at `lattice.py:663`, exactly as `torch.where` materialising both branches was in the E0 report;
searched the PyTorch tracker for a float64 batched-matmul allocation regression in 2.7 and found none.

## What a fix must change

`recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat.py`, `lexlat_forward_backward` lines 1449-1460
(and the matching `store`/`cache` lifetime at 1424-1426): cut the frame's slot work into chunks whose
per-chunk bytes are derived from a byte budget -- over the utterance axis B for a value-safe fix, over the
phone axis K if a finer cut is needed -- keeping the reported `max_multiplicity` global. Nothing in
`lexlat_train_jobs.py`, `lexlat_train.py` or any `returnn.config` key needs to change.
Separately, `lexlat_train_jobs.py` `_step_once` should emit each stage's seconds as it completes, so that
a failure in the priced path still leaves the timings it already paid for.

Report path: /e/project1/spell/wu24/2026-07-13_unsupervised/recipe/i6_experiments/users/wu/exp_logs/SAE/reports/debug_lexlat_e1_oom_2026-09-21.md
