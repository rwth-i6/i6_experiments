# SAE 4A k2 probe -- fix round 2: no graph the probe prices is determinized

Dispatch 2026-09-21 (after `review_k2_probe_fix1_2026-09-21.md`, item 4). Ruling implemented: k2's
`determinize` is tropical-only (`k2/fsa_algo.py`, the scratch env `sae_k2`), so NO graph the probe
prices is determinized -- the escape graph and the strict rung alike. Code: `recipe/2025-10-speech-llm`
at `05f425f`, branch `haotian_modality_matching_jupiter`.

## 1. What changed

* `compile_hlg(..., determinize=False)` is now the default and the only path the job uses;
  `_cmd_build_hlg` passes `determinize=False` explicitly for BOTH `escape=True` and
  `--strict-lexicon`, and `build.json` records `"determinized": False` unconditionally (it was
  `not escape`). Epsilon removal, `connect` and `arc_sort` are untouched. The parameter survives
  only as a test hook, defaults to `False`, and the build job never passes `True`.
* Docstrings/comments corrected in `lexlat_k2.py` (module docstring, `compile_hlg`, the
  `build-hlg` console lines) and `lexlat_k2_jobs.py` (class docstring, build `summary.txt`, probe
  `summary.txt`): determinization is tropical-only, it merges readings of a label string by max
  and so does not preserve the log-semiring total, a determinized graph would under-count that
  marginal, no graph priced is determinized, scores are compared on the undeterminized graph. The
  old claim "scores are unchanged (max-plus and log-semiring alike)" is REMOVED -- it is false
  (see 2). Both `summary.txt` files now print a `determinized: False` line of their own, for
  every rung, and the probe's `summary.txt` gained that line (it only had it inside the escape
  branch before).
* `test_lexlat_k2.py`: `_build_hlg` no longer determinizes anything, so the tests compile what the
  job compiles. Two new checks, brute-forcing the STRICT graph's readings (every cut of the phone
  string into lexicon words, scored with `_lookup` -- the numpy twin of `lexlat.lm_score` -- plus
  `sil_model_log_prob`):
  * `test_hlg_log_semiring_exceeds_the_sum_over_parses` (passes): the graph's MAX-PLUS total
    equals the best parse to 1e-4 -- which validates the Python reference against the graph
    itself -- and its LOG-semiring total is strictly larger than the sum over parses.
  * `test_hlg_log_semiring_equals_the_sum_over_parses`: the equality a marginal would need,
    declared `xfail(strict=True)` with the measured gap, so a construction that makes the two
    agree is noticed here.

## 2. THE FINDING: the log-semiring total is not the sum over parses (either way)

Measured on the tiny fixture (`test_lexlat`'s lexicon + real `lmplz` trigram), strict graph, one
phone per recognizer frame, nats:

| phone string | parses | sum over parses | HLG undeterminized | HLG determinized | max-plus |
|---|---|---|---|---|---|
| AY S K R IY M | 2 (`I SCREAM`, `ICE CREAM`) | -2.802934 | **-2.739123** | -2.736247 | -2.803281 |
| F AO R | 1 | -2.102869 | **-2.038123** | -2.038123 | -2.102869 |
| AY S | 1 | -2.106963 | **-2.071956** | -2.071956 | -2.106963 |
| F AO R AY S | 1 | -2.801069 | **-2.766935** | -2.766935 | -2.801069 |

So `get_tot_scores(log_semiring=True)` -- the probe's own number -- is an UPPER bound on the
marginal over parses, +0.03 to +0.07 nats on 3-6 phone strings, and it over-counts even where the
string has a single parse. Two causes, both verified:

1. `G`'s `#0` back-off arcs give one word sequence several routes (explicit arc, or back off and
   take the lower-order arc). Summing over those routes alone gives -2.786577 for AY S K R IY M
   (+0.016 above the parse sum).
2. icefall's `add_self_loops` puts a `#0:#0` self-loop on EVERY `L` state with a non-epsilon
   outgoing token, so the same back-off route can also be taken at several positions (mid-word as
   well as at the word boundary). Those copies are distinct paths of `L . G`; `k2.remove_epsilon`
   is itself documented tropical-only ("equivalent to the input `fsa` under tropical semiring"),
   so it merges epsilon routes only by MAX and the surviving copies are summed. This accounts for
   the rest (-2.786577 -> -2.739123).

Determinization does not repair it here: the determinized strict graph's log total is -2.736247,
i.e. slightly HIGHER, not lower, than the undeterminized one on this fixture (it also came out
larger: 100 states / 519 arcs against 97 / 454). The dispatch's rationale ("a determinized graph
would under-count") is the standard tropical argument and is what the docstrings now say; the
measured direction on this fixture is the opposite, and it is recorded here rather than in the
code. Either way the ruling stands -- neither graph's log total is the marginal.

Consequence for the probe: the settling read compares pruned against full totals ON THE SAME
graph, so the gap the probe reports is unaffected. Any later use of that total as `log Z` of the
lexicalised marginal is not licensed by this graph. NOT acted on -- it is a construction change
(e.g. dropping the mid-word `#0` self-loops, or a log-semiring epsilon removal) and a decision
about what the probe is meant to measure.

## 3. Checks run

| check | result |
|---|---|
| `test_lexlat_k2.py`, `sae_k2` env | **26 passed, 1 skipped, 1 xfailed** (skip = pre-existing `kaldilm` SIGABRT; xfail = the equality above) |
| `test_lexlat_k2.py`, shared env `speech_llm` | 6 passed, 22 skipped, rc 0 |
| REAL `build-hlg` on a fixture-shaped npz, escape | no `determinize` stage in `stages`; console prints `determinized: False ...`; `build.json` `{"escape": true, "determinized": false}`; HLG 137 states / 4340 arcs |
| REAL `build-hlg`, `--strict-lexicon` | same: no determinize stage, `{"escape": false, "determinized": false}` |
| graph load of `config/sae_4a_lexlat_k2.py` (sis env) | loads, 10 jobs |
| hashes (UNCHANGED) | `LexlatHLGBuildJob.rtX44PBJFNy1`, `LexlatK2ProbeJob.Tkl94t85j4pV` |
| census (UNCHANGED) | `sae_4a_prepro_pack` 133, `sae_4a_budget_pack` 775, `sae_4a_lexlat_probes` 3, `LexiconTrieBuildJob.rlMsnTBSZXsB` |

`__sis_version__` was NOT bumped (both jobs stay at 2): the configured arm is `escape=True`, whose
graph was already built without determinization, so the artefact under `rtX44PBJFNy1` is unchanged
bit for bit, and no `LexlatHLGBuildJob` directory exists on disk at all. Only a `escape=False`
build would now differ, and none has ever run. Say so if you want the version moved anyway.

Not covered: `LexlatK2ProbeJob.run()` (needs RETURNN + GPU + the arm checkpoint), so the probe's
`summary.txt` render is reviewed, not executed -- as in round 1.

## 4. Files touched (all in `recipe/2025-10-speech-llm`)

* `src/speech_llm/sae/emc/lexlat_k2.py`
* `src/speech_llm/sae/emc/lexlat_k2_jobs.py`
* `src/speech_llm/sae/emc/test_lexlat_k2.py`

Nothing else; no config change was needed (the ruling is a property of the build, not a knob).
