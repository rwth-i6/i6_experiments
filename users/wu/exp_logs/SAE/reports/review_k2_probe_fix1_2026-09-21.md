# Re-check of the five k2-probe fixes (speech-llm 05f425f vs ffa5501), 2026-09-21

Read-only, narrow: only the five fixes of `review_k2_probe_2026-09-21.md` plus the hash/census
line. Commit 05f425f touches exactly four files (`lexlat_k2.py`, `lexlat_k2_jobs.py`,
`test_lexlat_k2.py`, `config_sae_4a_lexlat_k2_v1.py`); nothing else is in it (the other files in
`ffa5501..05f425f` belong to the three intermediate commits ff4f005 / ecf846c / 10ccb37).

Verdict: **DONE_WITH_CONCERNS** -- all five fixes are real and effective; one disclosure sentence
that will be printed into `summary.txt` is factually wrong, and one reported memory number is
mislabelled. Neither blocks the launch.

## 1. `_sizes` off the ragged arcs -- PASS
`lexlat_k2.py:676-690`: `axes = fsa.arcs.num_axes()`, `states = fsa.arcs.tot_size(axes - 2)`,
`arcs = fsa.num_arcs`, `n_fsas = 1 | fsa.arcs.dim0()`. Correct for both ranks. `grep '\.shape\['`
over both modules returns only torch-tensor uses (`lexlat_k2_jobs.py:489,505,525-528,532,537`,
`lexlat_k2.py:308,520,523,841`); the only `Fsa.shape` left is the rank test
`len(hlg.shape) == 2` at `:806`, which is safe. Both call sites now go through `_sizes`
(`:808` graph, `:908` lattice). `test_sizes_read_a_saved_fsa_vec` saves `HLG.as_dict()`, reloads it
exactly as `run_probe` does and asserts `hlg.shape[1] is None` -- it exercises the line that raised.
(Nit: the `_sizes` docstring cites a test name that does not exist,
`test_sizes_read_an_fsa_and_an_fsa_vec`; the real one is `test_sizes_read_a_saved_fsa_vec`.)

## 2. partial.json after every cell + per-rung abort -- PASS
`lexlat_k2.py:819-834` `_write_partial` is called at `:935` after EVERY cell (with the in-flight
row), at `:937` after every batch and at `:951` at the end; the job passes
`--partial-json <out_partial>` (`lexlat_k2_jobs.py:548`) and the config registers it
(`config_sae_4a_lexlat_k2_v1.py:203`), so the file is written directly into the job's output dir
and survives a Slurm kill. `step_abort_sec` is `LexlatEfficiencyProbeJob.STEP_ABORT_SEC` imported
(`lexlat_k2_jobs.py:545`), not re-typed; 1200.0 confirmed at `lexlat_train_jobs.py:173`.
Control flow: the cell is recorded (`:921`) BEFORE the abort test (`:928`), the rung key enters
`aborted`, later batches skip it (`:850-853`), the other rungs continue, and `:938` breaks when
all three are aborted -- so `probe.json` is written, the child exits 0, and the job's read block
runs. An aborted rung gets `time_read = FAIL` unconditionally (`lexlat_k2_jobs.py:626`), its
seconds are labelled LOWER BOUND (`:578-581`) and the abort is printed in `summary.txt`
(`:696-699`). The defensive no-cell branch (`:566-571`) and the `gap is None` branch (`:709`) keep
the render from raising, so a verdict is always written.
Residual, inherent to E1's own design and not a regression: the abort can only fire BETWEEN cells,
and there is no total-wall guard (`sp.run` at `:556` has no timeout). Cells at, say, 900 s each
(below the 1200 s trip) x 27 cells overrun the 2 h clock; partial.json then holds the measured
cells but no `summary.txt` is produced. Only a single cell longer than the whole remaining wall
leaves nothing on disk.

## 3. The timed window -- PASS
`lexlat_k2.py:859-877`: `synchronize()` before `t0`; then `DenseFsaVec` + `intersect_dense_pruned`
(sync), `get_tot_scores(log_semiring=True)` (sync), `sum().backward()` (sync); `seconds =
t_intersect + t_scores + t_backward` (`:890`); peaks read at `:876-877`, after the last sync and
before anything else. The max-plus pass is at `:879-885`, after its own
`reset_peak_memory_stats()`, with its own timer and its own sync. `reset_peak_memory_stats()` at
`:855` precedes the `dense_in` allocation, so the peak covers the whole region.
The number compared with `bar_sec = 2.00 x 601 = 1202` is `per_subepoch`, built only from
`r["seconds"]` (`lexlat_k2_jobs.py:574,582,626`) -- no max-plus term enters it, and
`seconds_max_plus_*` is reported separately.
Concern (cosmetic, gates nothing): `peak_reserved_gib_max_plus` is read after
`reset_peak_memory_stats()`, which sets the peak to the CURRENT reserved bytes, so the value is
the total reserved during the max-plus pass, not an increment; `summary.txt:705` prints it as
"GiB on top" and the code comment at `:903-905` says the same. Mislabelled, not used in any read.

## 4. ESCAPE and the determinization question -- PASS, with one wrong sentence
Constants are imported, never re-typed: `from speech_llm.sae.emc.lexlat import
ESCAPE_LENGTH_LOG_PROB, ESCAPE_WORD` (`lexlat_k2.py:158`); `escape_prices` (`:362-377`) is
`LexResources.build`'s three lines (raw order-1 Witten-Bell log prob + `log 0.5`, SIL -> NEG_INF)
and `test_escape_prices_are_lexlats_own` asserts equality with `tiny["res"].escape_price` at
rtol = atol = 0. The loop (`:459-474`): one opening arc per non-SIL phone carrying word `<unk>`
(so G pays `log p(<unk>|state)` once per span), one self-loop per non-SIL phone, an exit pair
through `#(max_disambig+1)` to the word boundary / silence state; SIL is skipped (`:469-470`), so
a SIL can only follow the exit -- SIL closes the span. `#0` self-loops are added BEFORE the escape
arcs (`:454-458`), so the escape state has no back-off loop and the LM state cannot move while a
span is open. `NON_EMITTABLE_WORDS` now keeps `<unk>` and only the strict build drops it
(`:204-207`, `:1016`). The banked npz carries `escape_phone_log_prob` and `words`, and
`words[unk_word=0] == '<unk>'`, so the new build-time reads and the assert at `:1008` pass on the
real input (checked on
`work/.../LexiconTrieBuildJob.rlMsnTBSZXsB/output/lexlat_resources.npz`).
`test_hlg_with_escape_equals_the_lexlat_string_scorer_on_an_oov` calls
`L.string_best_segmentation(phones, res, escape=True)` with the fixture's REAL `LexResources` on
three OOV-leading strings and matches the HLG max-plus to 1e-4, and asserts the strict graph has
no path. I re-ran the file in the k2 env: **25 passed, 1 skipped**.

(a) `k2.determinize` is **tropical-only**: the installed build's own docstring,
`sae_k2/.../k2/fsa_algo.py:811-812`, "it's equivalent to the input `fsa` under tropical semiring".
The probe's objective is the LOG-semiring total. Determinization preserves the max-plus best path
but NOT the log-semiring sum (it keeps, per phone string, the best reading), so a determinized
graph would have been WRONG for this objective: skipping it is the correct choice, not a
compromise, and the undeterminized graph is the one whose log Z is the marginal.
FINDING: `lexlat_k2.py:705-707` and the build `summary.txt` text at `lexlat_k2_jobs.py:250-253`
state "the max-plus and the log-semiring total scores of the HLG are unchanged by dropping it" /
"The scores are unchanged (max-plus and log-semiring alike)". That is false for the log semiring
and it contradicts this file's own strict-branch sentence (`:255-256`: a determinized graph's
log-semiring total "is a sum over phone strings of the BEST reading of each") and the module
docstring's own closing parenthesis (`lexlat_k2.py:107-110`), which get it right. The priced graph
is correct; only the disclosure sentence that will appear in `summary.txt` is wrong, and it invites
a later arm to re-enable determinization believing log Z is preserved.

(b) `remove_epsilon`, `connect`, `arc_sort`, `compose(H, LG)`, `connect`, `arc_sort` all still run
(`lexlat_k2.py:733-741`); only `determinize` + its `connect` are behind the flag (`:728-732`), and
the disambiguation labels -- including the escape's own `#(max_disambig+1)` -- are zeroed at `:733`
before the epsilon removal, so no epsilon arc reaches the dense intersection.
`intersect_dense_pruned`'s only documented requirements (installed `k2/autograd.py:678-686`) are
"a_fsas MUST be arc sorted" and `a_fsas.shape[0] == 1` for a shared graph; both hold
(`lexlat_k2.py:741`, `:806-807`). Determinism is not required. The escape graph is built and
scored end-to-end in the passing test, but through `k2.intersect_dense` (the test helper
`test_lexlat_k2.py:199-204`), not through `intersect_dense_pruned`; my attempt to run the pruned
operator on the fixture from a scratch directory died on the env's library path before it could be
answered, so that one call remains checked by argument rather than by execution.

(c) `build.json` carries `"escape"` and `"determinized"` (`lexlat_k2.py:1026-1027`, plus
`escape_disambig`, `escape_price_min/max`, `dropped_words`); the build `summary.txt` prints a
"GRAPH PRICED: ESCAPE ..." block and a "determinized: NO ..." block
(`lexlat_k2_jobs.py:239-256`); the probe `summary.txt` prints "graph priced: ESCAPE ... NOT
determinized" off `graph_build["escape"]` (`:671-678`) and repeats it in the closing disclosure
(`:720-723`). Present in both files.

## 5. `n_neg_inf` counted once -- PASS
`lexlat_k2.py:913`: `int(((~finite) | (tot_cpu <= NEG_INF / 2)).sum())` -- one test, one count per
utterance. The job sums it over batches within a rung only (`lexlat_k2_jobs.py:621`) and prints it
once (`:710`). No other site computes it.

## Hashes and census -- unchanged
Graph loaded from `config/sae_4a_lexlat_k2.py`'s module (`config_sae_4a_lexlat_k2_v1.py`), 10 jobs:
`speech_llm/sae/emc/lexlat_k2_jobs/LexlatHLGBuildJob.rtX44PBJFNy1` and
`.../LexlatK2ProbeJob.Tkl94t85j4pV` -- both as reported by the implementer.
Census re-run per graph, one process each: `sae_4a_prepro_pack` **133**, `sae_4a_budget_pack`
**775**, `sae_4a_lexlat_probes` **3**. Unchanged.

## Not checked (out of the narrow scope)
The bar arithmetic, the H/G equivalences, the scale convention, the batch plan and the subprocess
hand-over -- all were checked in `review_k2_probe_2026-09-21.md` and are untouched by this commit.
`LexlatK2ProbeJob.run()` still has no end-to-end execution; the full-lexicon escape build (whether
it compiles inside 4 h / 200 GB without determinization) remains an open empirical question that
the build job itself answers.
