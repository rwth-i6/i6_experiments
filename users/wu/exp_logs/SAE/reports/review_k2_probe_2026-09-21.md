# Code review: the k2 settling probe (speech-llm ffa5501), 2026-09-21

Read-only review before compute. Verdict: **DONE_WITH_CONCERNS** -- the science of the graph is
sound (H, L, G, tau, the bar arithmetic, the hash story all check out), but `run_probe` **cannot
run**: it dies on a TypeError two lines after loading the graph, so the probe as committed would
burn the GPU allocation and produce nothing.

Files read: `src/speech_llm/sae/emc/lexlat_k2.py`, `lexlat_k2_jobs.py`, `test_lexlat_k2.py`,
`configs/config_sae_4a_lexlat_k2_v1.py`, `config/sae_4a_lexlat_k2.py`, plus `lexlat.py`,
`lattice.py`, `blankfree_seed.py`, `lexlat_train_jobs.py`, `config_sae_4a_lexlat_pack_v1.py`,
`config_sae_4a_attrib_v1.py`, and the installed k2 (`/e/scratch/spell/wu24/envs/sae_k2`).

## Findings

### 1. `lexlat_k2.py:659` (and `:714`): `int(fsa.shape[1])` on an FsaVec is a TypeError -- FATAL
k2's `Fsa.shape` is `(num_states, None)` for an Fsa and `(num_fsas, None, None)` for an FsaVec
(`k2/fsa.py:1218-1229`, the installed build). `run_probe` does

    if not hasattr(hlg, "shape") or len(hlg.shape) == 2:
        hlg = k2.create_fsa_vec([hlg])
    graph = {"states": int(hlg.shape[1]) if len(hlg.shape) == 3 else int(hlg.shape[0]), ...}

`HLG.as_dict()` saves a 2-axis Fsa, so the branch always fires, `len(shape)` is then 3 and
`int(None)` raises. Confirmed by running it in the scratch env:

    single shape (3, None)  vec shape (1, None, None)  len 3
    CRASH: TypeError int() argument must be ... not 'NoneType'

The same expression sits at `:714` (`lattice.shape[1]`; `intersect_dense_pruned` always returns an
FsaVec), so even a fixed `:659` dies inside the first timed cell. The child exits non-zero,
`lexlat_k2_jobs.py:512` raises, and the whole GPU job is lost AFTER the expensive half (two full
walks of the sub-epoch's loader plus nine recognizer forwards). Neither line was ever executed:
`run_probe` demands `cuda:0` and the 22 passing tests only cover CPU graph construction.
Fix: use `fsa.arcs.dim0()`/`fsa.arcs.tot_size(1)` (or `fsa.shape[0]` before `create_fsa_vec`).

### 2. `lexlat_k2.py:692-697`: the max-plus pass is charged to the timed log-semiring window
`t_intersect` ends with a `synchronize()`, then `tot_max = lattice.get_tot_scores(log_semiring=
False, ...)` launches kernels **asynchronously**, and only then does `t0 = perf_counter()` start the
`t_scores` window. Those kernels drain inside that window, and `tot_max`'s buffers enter
`max_memory_reserved`. The reported `seconds` -- the number `lexlat_k2_jobs.py:555` compares with
1202 s -- therefore includes a whole extra total-score pass that the objective does not contain,
and the peak GiB likewise. This one errs AGAINST the arm (a real PASS can be printed FAIL), which
is still a wrong route decision. Fix: `torch.cuda.synchronize()` after `tot_max`, or compute the
max-plus check outside the timed/peak region.

### 3. `lexlat_k2_jobs.py:501-513`: no partial write and no step abort
`run_probe` writes `probe.json` once, at the very end (`lexlat_k2.py:737`); the job writes
`summary.txt` only after the child returns. The job's clock is 2 h
(`config_sae_4a_lexlat_k2_v1.py:94`) for 9 batches x 3 rungs = 27 cells at an unknown per-cell
cost. E1 hit exactly this and the coordinator's 2026-09-21 ruling added `partial.json` after every
timed batch plus `STEP_ABORT_SEC = 1200` -> a MEASURED FAIL (`lexlat_train_jobs.py:118-126,
171-173`). Neither is carried over here, so a slow rung ends as a Slurm timeout with no registered
output and no verdict -- the failure mode that ruling exists to prevent.

### 4. `lexlat_k2.py:153, 525-529`: the priced graph has no ESCAPE, and `summary.txt` does not say so
`NON_EMITTABLE_WORDS` drops the `<unk>` arcs, so HLG is the STRICT lexicon. The phase's Design
fixes ESCAPE inside the marginal verbatim (`SAE_4A_lexlat.md:57`: one `<unk>` word-LM transition
per contiguous non-SIL span plus the Witten-Bell order-1 phone term), precisely because a `-inf` on
non-segmentable prefixes is the zero-probability trap. The disclosure block the job prints
(`lexlat_k2_jobs.py:620-625`) names the reverse segment score, the phone trigram and the optimizer
step as unpriced, but NOT the escape arcs. A PASS would therefore read as "route A is affordable"
for a graph the Design does not permit, with the escape arcs' state-space cost unmeasured. This is
a disclosure fix, not a code fix: add escape to the "what is not measured" paragraph (and to the
amendment's own scope if a PASS arrives).

### 5. `lexlat_k2.py:718`: `n_neg_inf` double-counts a genuine `-inf`
`int((~finite).sum()) + int((tot_cpu <= NEG_INF / 2).sum())` counts a `-inf` total score in BOTH
terms (it is non-finite and it is below -5e29). `summary.txt` prints "non-finite tot scores N" at
twice the real value; the count is also summed over rungs at `lexlat_k2_jobs.py:550`. Only the
reported number is affected -- the stability read itself filters correctly through `_finite`.

## Checked and sound (no finding)

* **H equivalence.** `lattice.py:347-350` makes `same_nonsil` unconditional outside `topology ==
  "ctc"` (the SIL exemption is CTC-only) and `_arc_weights:511-516` fills `w_blank` with NEG_INF, so
  `f` never resets and the frame string's reading IS the adjacent-run collapse. `h_topology` is that
  machine arc for arc. The test is a real call into the bed's primitive with real tensors
  (`test_lexlat_k2.py:202-223`: `H . linear(y)` under `k2.intersect_dense` against
  `blankfree_seed.transcript_logprob`, 5 token strings incl. non-adjacent repeats x 2 frame counts,
  k2 leg at the production float32 dense + double scores, tol 1e-4), plus an exhaustive
  aux-label check over every 4-frame string (`:227-236`) and a `min_frames = 2` negative
  (`:240-255`). SIL is not a special case in H and needs none. `emission_min_frames(2, 3) = 1` is the
  correct reading of a 50 Hz `d_min` on a stride-3 clock; the real `d_min >= 2` lives in the segment
  leg, which this probe does not price, and that is stated.
* **Scale / log-domain.** `lattice.py:46` and `:826-829`: EVERY arc weight is
  `(1/tau)[log q + alpha log q_init + beta log P_psi + G]`. `run_probe` divides both the dense
  emissions and the HLG scores by `tau = model.temperature(epoch)` (`lexlat_k2_jobs.py:424`,
  `lexlat_k2.py:655, 674`), so log Z comes out in the bed's tempered nats and the stability read is
  on E0's scale. Correct. (The anchor term `alpha log q_init` is absent from the dumped emissions;
  at this bed's `anchor_weight` that is a non-issue, and it would only perturb the search's shape,
  not the convention.)
* **G conversion.** `lexlat.parse_arpa_word_lm`'s CSR carries every n-gram of the ARPA
  (151,734 + 3,393,577 + 10,419,405), so it is the full unpruned trigram with its back-off arcs, and
  it is the very object the phase's own scorer reads. `g_fsa` emits one arc per explicit n-gram,
  one `#0` back-off arc per state (correctly suppressed for the null context, `:553-556`), swaps
  `begin_state` with 0, makes every state final at 0 -- the bed's registered "no end-of-sequence
  term" (`prior.PhoneNgramPrior`, `lexlat.string_best_segmentation`). The kaldilm delta
  `log p(</s>|ctx)` IS string-dependent, but it is kaldilm's convention, not ours: our graph never
  scores `</s>`, and the decisive check is `test_lexlat_k2.py:391-434`, HLG's max-plus against
  `lexlat.string_best_segmentation(escape=False)` + `sil_model_log_prob(2)` to 1e-4. SIL is handled
  as in `prior_gap.py` (word-boundary anchor, never inside a word) and its constant is priced
  explicitly, which is what makes that equivalence exact. `prune_lm_tables`' order accounting
  (`:490`) matches the parser's state layout (0 = null, 1..V = unigram contexts, rest = bigram
  contexts), so the reported n-gram counts are right.
* **The bar arithmetic.** `bar_sec = TIME_FACTOR_BAR * SEC_PER_SUBEPOCH = 2.00 * 601 = 1202`,
  `bar_gib = 80` -- read off E1's own class constants, not re-typed. `per_subepoch` is the sum over
  batches when the plan covers the sub-epoch and `mean x n_total` otherwise, with the basis printed
  beside the number; `n_total` is COUNTED from the loader, not assumed to be 57. Comparison is a
  bare `<=`, no rounding either way. Memory is read on RESERVED, the stricter statistic, max over
  batches (the largest-T batch is in the plan). The one favourable slack is sub-GiB: the parent's
  CUDA context survives `del model, engine` + `empty_cache()` and is invisible to the child's
  counter.
* **Single delta / census.** Both modules are NEW files with hand-bumped `__sis_version__`, so no
  banked `__sis_version__` moves. The config calls `attrib._control()`, `attrib._priorshuf_prior`
  and `pack._arm_train_config` with exactly the arguments `pack.efficiency_probe` passes at
  `E1_CONTEXTS_LOWER = 1024`; `prefix`/`alias` reach only `add_alias` and `register_output`
  (`config_sae_4a_attrib_v1.py:143-170`), which move no hash. It calls neither `build()` nor
  `efficiency_probe()` nor `census()`. `LexiconTrieBuildJob.rlMsnTBSZXsB` is consumed as a frozen
  `tk.Path` whose `hash_overwrite` carries a sha of the realpath. The batch plan is
  `LexlatEfficiencyProbeJob._plan` IMPORTED and called with the same `n_steps=100 / n_points=4 /
  max_timed_batches=E1_MAX_TIMED_BATCHES=9`, so the timed indices are E1's. A full census was NOT
  re-run by the implementer and is not re-run here; nothing in the diff suggests one is needed.
* **Label dependence.** None. The probe reads features/units/originals and a checkpoint; no
  transcript, alignment or gold string enters it.
* **Subprocess hand-over.** `PYTHONPATH` is the recipe `src` root alone; `TMPDIR` is passed at the
  call site; `CUDA_VISIBLE_DEVICES` is forwarded when set; the k2 build resolves its libraries from
  an absolute RPATH so no `LD_LIBRARY_PATH` is needed. `ENV_PYTHON` and `gpu_check` are in both job
  hashes, so a rebuilt env at another path is a different measurement.
* **No silent no-ops.** The prune ladder is really walked, rung 0 is asserted to be theta = 0 and
  the chosen rung reaches `build.json` and `summary.txt`; `max_active` reaches
  `intersect_dense_pruned` (`lexlat_k2.py:689`) and the ladder's largest entry is asserted to be the
  stability reference; the GPU check runs FIRST and a non-zero exit is fatal
  (`lexlat_k2_jobs.py:385-396`) -- `/e/scratch/spell/wu24/envs/k2_gpu_check.py` exists and ends in an
  assert, so it does exit non-zero on a dead build. The max-plus/log-semiring split matches the
  spec: max-plus only as the `log-sum >= max` sanity check and in test (d); the objective and the
  backward are the log semiring.
* **Loader determinism** is asserted by a second walk against the recorded shapes
  (`lexlat_k2_jobs.py:467-469`), and the graph is asserted against the LIVE `model.lattice_cfg`
  (`n_phones`, `sil_id`, `d_min`, `recognizer_stride`) before a frame is dumped.

## Recommendation

Do not launch the probe as committed (finding 1 guarantees a wasted GPU job). Findings 1, 2 and 3
are small, local edits in `lexlat_k2.py` / `lexlat_k2_jobs.py`; finding 5 is one line; finding 4 is
a paragraph of disclosure. The HLG build job (`LexlatHLGBuildJob`) is independent of all five and
can be launched now -- it is CPU-only and its output is what the probe consumes.
