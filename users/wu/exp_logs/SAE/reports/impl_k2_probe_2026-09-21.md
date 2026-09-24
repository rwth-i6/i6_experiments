# Implementer report: the k2 settling probe (round 1, handed back mid-build)

**Status: DONE_WITH_CONCERNS.** Every piece asked for exists, is tested and is committed, and the
config loads with both job hashes resolved. Nothing has been launched. Two items of the dispatch
are NOT as specified and are named below (the G route, and the pruning ladder's rung values).

Commit `ffa5501` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`
(explicit paths, not pushed).

## What exists

| file | what it is |
| --- | --- |
| `src/speech_llm/sae/emc/lexlat_k2.py` | the graph and the probe body; importable by BOTH envs (k2 imports are local to the functions that need them) and runnable as a script: `gpu-check`, `build-hlg`, `probe` |
| `src/speech_llm/sae/emc/lexlat_k2_jobs.py` | `LexlatHLGBuildJob` (CPU 16 / 200 GB / 6 h) and `LexlatK2ProbeJob` (1 GPU, mem 64, gpu_mem 96, 2 h) |
| `.../librispeech/configs/config_sae_4a_lexlat_k2_v1.py` | one entry point `py()`, registering exactly those two jobs |
| `config/sae_4a_lexlat_k2.py` (setup dir, untracked) | the shim defining `py` / `run` |
| `src/speech_llm/sae/emc/test_lexlat_k2.py` | the six checks |

Hashes, from a graph load of the shim's config in the shared env:

* `speech_llm/sae/emc/lexlat_k2_jobs/LexlatHLGBuildJob.BoWs93v20S6x`
* `speech_llm/sae/emc/lexlat_k2_jobs/LexlatK2ProbeJob.7eYgCXeFxMDQ`

Census: the banked `LexiconTrieBuildJob` re-resolves to `rlMsnTBSZXsB`, unmoved. No banked module
was edited (both new modules are new files), so no banked `__sis_version__` can have moved; the
prepro / budget / probes counts were not re-run as a full census and remain to be confirmed by the
coordinator if wanted.

## The two environments

`k2` with CUDA exists only in `/e/scratch/spell/wu24/envs/sae_k2`
(`reports/impl_k2_build_2026-09-21.md`); sisyphus workers run under the shared env. Both jobs do
their own half under the shared env and hand the k2 half to
`/e/scratch/spell/wu24/envs/sae_k2/bin/python .../lexlat_k2.py <sub-command>`, with `PYTHONPATH`
set to the recipe `src` root ALONE. `ENV_PYTHON` is in both job hashes, so a rebuilt environment
at another path is a different measurement. The probe job runs
`/e/scratch/spell/wu24/envs/k2_gpu_check.py` FIRST and dies with that script's own output on a
non-zero exit.

## H: what the reading actually is

`lattice.py`'s three arcs, on the recognizer axis alone: **blank** is dead (`_arc_weights` fills
`w_blank` with `NEG_INF` outside `topology == "ctc"`); **repeat** re-emits `last(h)`, opens no token
and is available from `f = 1`; **emit** of `k == last(h)` needs `f = 0`, and only a blank can reset
`f`, so after the first emission no symbol -- SIL included, the `k != sil_id` exemption in
`_prior_history` being CTC-only -- may open a second token in a row. The frame string's token
string is therefore the adjacent-run collapse, which is exactly `h_topology`.

**The one reading choice**: `d_min = 2` is a duration on the 50 Hz unit clock (on `s`), not on the
recognizer clock; expressed on the recognizer axis it is `ceil(2 / 3) = 1` frame, so at this bed
`h_topology` carries NO minimum-duration chain and is the plain run-collapse machine. The real
`d_min >= 2` constraint lives in the segment leg, which this probe does not price at all.
`emission_min_frames` / `h_topology(min_frames=...)` implement the general case and are tested.

## Tests, and what they establish

Run under the scratch env (`/e/scratch/spell/wu24/envs/sae_k2/bin/python -m pytest
recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_lexlat_k2.py -q`):
**22 passed, 1 skipped.** Under the shared training env (k2 absent): **5 passed, 18 skipped.**

* **H is the bed's machine, not a claim about it.** `H . linear(y)` intersected with random
  emissions, log-semiring, against `blankfree_seed.transcript_logprob` -- the bed's own exact
  "sum the paths whose adjacent-run collapse equals this target, without a blank" -- over 5 token
  strings x 2 frame counts, agreeing to < 1e-4. Plus: every 4-frame string over 3 symbols is
  accepted and its aux-label string is `collapse_adjacent` of it; at `min_frames = 2` a one-frame
  run has no path.
* **L round-trips** every disambiguation-free pronunciation of the tiny fixture (4 of them; the
  dispatch asked for 5 and the fixture offers 4 -- a pronunciation that is a proper prefix of
  another reaches L's final state only after its `#k`, which is not part of a phone string).
* **G matches KenLM** on 3 sentences, best path, to < 1e-4 nats.
* **THE EQUIVALENCE**: the compiled `HLG`'s max-plus total score on two-word utterances equals
  `lexlat.string_best_segmentation(escape=False)` plus `sil_model_log_prob(2)` to < 1e-4 (measured
  5.71e-9 on the earlier hand run), and `log-sum >= max` holds. This is the E-1 machinery and it
  ties H, L, G and `compile_hlg` end to end to the phase's own scorer.
* the trie round-trip, icefall's disambiguation rule, the numpy twin of `lexlat.lm_score`, and the
  pruning ladder (theta = 0 bit-identical; arc count monotone over 0/0.2/0.5/1.0/2.0/5.0; every
  probed state sums to 1 after the back-off weights are re-derived).

## Deviations from the dispatch, to be ruled on

1. **G does not come from an ARPA.** The banked `LexiconTrieBuildJob.rlMsnTBSZXsB` work directory
   has been cleaned: `words_o3.arpa` is gone, and only `word_lm.bin` (KenLM binary) and the parsed
   CSR automaton in `lexlat_resources.npz` survive. The CSR was verified lossless
   (13,964,716 arcs = 151,734 + 3,393,577 + 10,419,405), so `g_fsa` converts THOSE tables -- the
   same object the phase's own scorer reads. `kaldilm` 1.15.4 DID install into the scratch env
   (aarch64 wheel) and was used as an independent cross-check on a tiny ARPA by hand: the
   difference `(kaldilm - mine)` equalled exactly KenLM's `log p(</s> | ctx)` to 5.8e-7 / 4.2e-6 /
   1.3e-6, which is the documented end-of-sentence convention (`kaldilm` folds `</s>` into the
   final weight; the bed pre-registers "no end-of-sequence term", so `g_fsa` makes every state
   final at 0 and drops the `<unk>` / `<s>` / `</s>` arcs). That cross-check is IN the test file
   but `@pytest.mark.skip`ped: `kaldilm.arpa2fst` SIGABRTs the interpreter on this aarch64 build
   and takes the whole pytest process with it.
2. **The pruning ladder's rungs are mine.** The dispatch fixes only the rule ("the smallest
   threshold that fits"), so `PRUNE_LADDER = (0.0, 0.5, 2.0)` nats in the config is an
   implementation choice, disclosed in the job's `summary.txt`. Rung 0 is the full banked trigram
   and the job asserts it is first, so a run that succeeds at once used the LM unchanged; the rung
   used and the resulting n-gram counts go into `build.json`. The criterion is the count-free
   Seymore-Rosenfeld back-off gain with back-off renormalisation.

## Unfinished / where the next implementer picks up

* **Neither job has ever run.** The HLG build has only been exercised at toy scale (the earlier
  hand run compiled H 7/36, L 22/62, G 135/1032, LG after determinize 2739/8468, HLG 2085 states /
  11405 arcs). The full-lexicon compile time and memory are UNKNOWN; whether the 13.96M-arc
  trigram fits 4 h / 200 GB is exactly what rung 0 of the ladder is there to find out.
* **`LexlatK2ProbeJob.run` has never been executed.** Its emission-dump leg mirrors E1
  (`raw_dict_to_extern_data`, `rf.init_train_step_run_ctx(train_flag=True)`, the same
  `merge_random_seeds([epoch, index, random_seed])`, the same `_plan` imported from
  `LexlatEfficiencyProbeJob`) but the two-walk loader determinism assert, the `hlg_stats`
  assertions against the live `model.lattice_cfg`, and the subprocess hand-over have only been
  read, never run. First run should be watched for: the parent's CUDA context still resident when
  the child starts (freed with `del model, engine` + `empty_cache()`, context ~0.5 GiB remains),
  and the child's peak-memory numbers therefore being read on a card that is not completely empty.
* **`gpu-check` has never run on a GPU.** `k2.with_cuda` is True and the version is the report's,
  both asserted by a test, but no sm_90 kernel has executed in this session.
* The setup-dir shim `config/sae_4a_lexlat_k2.py` is untracked (the setup dir is not a git
  repository); recreate it with one import line if the workspace is rebuilt.
* The skipped `kaldilm` cross-check: re-enable only if it is run out of process (it aborts the
  interpreter).

## Reads, not decisions

`summary.txt` prints PASS/FAIL per `max_active` against E1's bar as written (<= 1202 s per
sub-epoch on the sum rule, <= 80 GiB reserved), and says on the same page that what is priced is
the LEXICON LEG alone -- no reverse segment score, no phone trigram, no optimizer step -- so a PASS
licenses the two-graph design amendment and its review, never an arm, and a FAIL licenses "not
funding route A at this cost" and nothing about the lexicon's value.
