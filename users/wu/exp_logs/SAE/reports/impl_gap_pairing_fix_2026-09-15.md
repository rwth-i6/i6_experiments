# S3 gap row pairing fix + BT probe rho  (implementer, 2026-09-15)

Commit `d69f938ea65c9e408bf515d3774ac78822d27866` on `haotian_modality_matching_jupiter`
in `recipe/2025-10-speech-llm` (staged by explicit path, NOT pushed).
Another implementer's `config_sae_1g_v1.py` modification and the untracked
`config_sae_3e1_d6_swap_cont_v1.py` were left alone.

## (1) `src/speech_llm/sae/emc/s3_jobs.py` -- `S3DerangementGapJob.run`, `_rows` (body only)

```python
_, _, rows = evaluate(model, items, per_utterance=True)
assert sorted(r["index"] for r in rows) == list(range(len(items)))
return {tags[r["index"]]: r for r in rows}      # was: {t: r for t, r in zip(tags, rows)}
```

`reverse.evaluate` batches through `_length_buckets` (sort key `(len(z), len(y))`) and stamps each
row with its item index, so the row order is the bucket order, not the `tags` order. Modelled on
`bt_probe.py:1427-1437`. No signature / parameter / output change, so the job hash is unchanged.

Impact of the old code, measured on the new fixture (buggy vs fixed, same fixture):
per-utterance json rows carried another utterance's `own`, `deranged` and `frames` under a tag
(e.g. frames 20 reported for the 44-frame utterance), and the per-utterance deltas were assigned to
the wrong speaker cluster -- macro CI95 `[-0.0185, +0.0938]` buggy vs `[+0.0271, +0.0482]` fixed.
`own/deranged/gap per frame` are sums over the same set and were NOT affected (both `+0.016794`
here). Where two utterances tie on frame count the own/deranged row orders differ from each other
too, so the paired delta itself is wrong; with distinct frame counts the permutation is common to
both calls and only the tag attribution and the clustering move.

## (2) `.../librispeech/configs/config_sae_4a_bt_probe_v1.py:78`

`RHO_HZ = None` -> `RHO_HZ = 9.6619373279`, with the comment tracing it to
`analysis/out/rho.rate_term.txt` (`phones_per_word(T_phi)` 3.5784953066 x `WORDS_PER_SEC` 2.7 =
0.193238746559 tokens per 50 Hz frame), identical to `RATE_RHO_HZ` in
`config_sae_4a_s3b_rate_v1.py:88`. Nothing else in the file touched.

`rho_hz` is hashed: the eight arms are now
`BtProbeJob.{1E6mfTahsKLZ, 3BFG7tATDxqI, 5owmxhkPXkdf, 91VDNBrLup2B, 96K3LFZ7brhJ, ZXlljMDUea5m,
rNJbMa2sG9ia, uXVX60ASzhKI}` (previously `hyZPr4oNVP8D, 2FfxpVDvPK70, UmX3p8W82zz3, OP7BVhVoS26C,
CyQ2eUKpwsGi, qIuqW55Emayf, opqSeQlbHnm3, evGHzEhtYGxH`). `find work -name 'BtProbeJob.*'` returns
nothing, so no arm is orphaned and nothing is re-funded. With rho no longer `None`, `bt_probe.py:1376,1485,1521`
takes the band branch (`[0.6 x rho, 1.5 x rho]` = `[5.797, 14.493]` /s by that line's arithmetic)
and the `takes_off` clause instead of the "undetermined" reason; not exercised at this rho value
here (the dry run in `test_bt_probe` uses `rho_hz = 10.0`).

## (3) `src/speech_llm/sae/emc/test_s3_jobs.py`

New `test_s3_gap_rows_are_keyed_by_evaluate_index`: four utterances with deliberately non-monotone
frame counts (44 / 20 / 60 / 30, bucket order `[1, 3, 0, 2]`). It calls the REAL `reverse.evaluate`
(the shared primitive, per the "test the call into shared primitives" memory), asserts the returned
row order is not the item order, then checks (a) the `tags[r["index"]]` map and (b) the job's own
`per_utterance.json` give each tag its own frame count. `_fixture` gained an optional per-tag
`frames` argument; the default path is byte-identical to before.

## Checks run (conda env `/e/project1/spell/wu24/env/conda/envs/speech_llm`, from `recipe/2025-10-speech-llm/src`)

| module | result |
|---|---|
| `speech_llm.sae.emc.test_s3_jobs` | 5/5 pass ("all s3_jobs tests passed") |
| `speech_llm.sae.emc.test_bt_probe` | pass ("all bt_probe tests passed") |
| `speech_llm.sae.emc.test_reverse` | pass |
| `speech_llm.sae.emc.test_s1a_job` | pass |
| regression: fix reverted to `zip(tags, rows)` | the new test FAILS as intended (frames 20 for the 44-frame tag); fix restored and re-run |
| config import, `sis_env` python | `RHO_HZ = 9.6619373279` |
| `config_sae_4a_bt_probe_v1.build()`, `sis_env` python | 8 distinct `BtProbeJob`s, all with `rho_hz = 9.6619373279` (graph built in memory only, nothing launched) |

Not done (out of scope, held by another implementer): `prefix_lm/model/train_steps/sae_emc.py`,
`emc_train_jobs.py`. This report is not committed (the dispatch named only the speech-llm repo).
Nothing in the dispatch was left undetermined.
