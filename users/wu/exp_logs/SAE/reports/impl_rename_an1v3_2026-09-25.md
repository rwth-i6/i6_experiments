# Implementation report: SAE_4A_rename AN-1 V3, J with a 4-gram / 5-gram LM term (2026-09-25)

Status: DONE. The job is built, tested on the real inputs and committed. It has not been launched.

## What was built
- Job `speech_llm/sae/emc/key_objective_v3_jobs/KeyObjectiveV3Job.N7q9hgRBDrFw`. It is one mini task on the "short" engine (the login node): cpu 1, mem 8 GB, time 1 h, no GPU. It writes `report.txt` and `table.json`.
- Files, all new. No existing file was edited.
  - `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/prior_high_order.py` (183 lines, pure numpy): counts orders 1-5 on the window, fits the tables, scores tokens, and computes held-out perplexity.
  - `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/key_objective_v3_jobs.py` (357 lines): the job. It sets no `__sis_version__`.
  - `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_key_objective_v3.py` (308 lines): the test, with parts unit, real, graph, hash and run.
  - `.../librispeech/configs/config_sae_4a_rename_an1v3_v1.py` (58 lines): the config. It builds only the V3 job. It reuses AN-1's `_pin` keys, adds a pin for the window `SampleLinesJob.orN768ARKwlt/output/text.phn.gz`, and pins AN-1's `table.json` from erCoRAmZID5a.
  - `config/sae_4a_rename_an1v3.py`: a shim in the setup dir. It is not tracked, the same as AN-1's shim.
- Commit `1add18a8` on branch `haotian_modality_matching_jupiter`. The four recipe paths were staged explicitly. Nothing was pushed.

## Estimator
- It is interpolated Witten-Bell, the same estimator the trigram prior uses (`prior.py` docstring), extended to higher orders.
- Formula: p_n(w|h) = (c(h,w) + T(h) p_{n-1}(w|h')) / (N(h) + T(h)).
  - h' is the context h with its oldest symbol dropped.
  - A context that was never seen backs off entirely to p_{n-1}.
  - p_1 backs off to the uniform distribution. It is `from_counts`' formula, copied verbatim.
- Every order is built by the same function, `prior._witten_bell`, applied to blocks of rows. No new constant is introduced.
- Counts use `fit_prior`'s read on the uniform-sample window: the first 1,000,000 + 10,000 lines, with every line whose index is a multiple of 101 held out, tokens mapped by `_to_ids`, and each line padded with 4 BOS symbols.
- V3a = lm_4 + emis + dur and V3b = lm_5 + emis + dur.
  - lm_n is the n-gram log-probability over J's own tokens (`absorb_segments` followed by `split_tokens`), with the history padded by BOS at the start of each utterance and no EOS term.
  - emis and dur are `_j_pair`'s terms, unchanged.
  - The keys, the competitor rule, the derangements and the null corpus are AN-1's (`KeyObjectiveScreenJob.key_set`).

## Checks (all passed)
- **Checks inside the job.** Any failure raises before an output is written.
  - Fit identity: the refit's orders 1-3 equal the banked `prior.npz` of RtzbESkOedsT exactly (max |diff| 0), and its order-3 counts equal the banked `tri_counts`.
  - Order-3 reproduction: J_3 - V0 has max |diff| 8.9e-16 over 652 comparisons (326 rows x 2 sides).
  - V0 equals AN-1's `J_V0`, lm, emis and dur exactly (max |diff| 0).
  - AN-1's same-scale check passes: 628 comparisons with max |diff| 0.
  - The competitor list equals AN-1's 187.
- **unit:** on toy data the counts equal `prior.count_ngrams` and a hand Counter; the order 1-3 tables equal `from_counts` bit for bit; the order-4 tables equal a scalar recursion to 1e-13; the token index equals a hand loop at orders 1-4.
- **real:**
  - The fit takes 17 s with a peak of 3.1 GB.
  - Held-out perplexity by order: 26.73, 14.14, 9.561 (all three match `prior.stats.txt`), then 7.20 at order 4 and 5.98 at order 5.
  - On the first 20k lines of the window, 12 order-5 cells and their order-4 backoffs equal a scalar recursion to 1e-12. One of the 12 has an unseen context.
  - The token path at order 3 equals `PhoneNgramPrior.log_prob(order=3)` on 5 held-out utterances, and orders 4 and 5 equal a hand loop to 1e-12.
  - On the full sides, gold's J_3 minus `_j_pair`'s J is at most 8.9e-16.
- **graph:** the graph has 3 jobs, and only the V3 job is unfinished. All 25 inputs exist, and the task routes to "short" with no GPU.
- **hash:** the AN-1 config still builds `KeyObjectiveScreenJob.erCoRAmZID5a`, and its graph (erCoRAmZID5a, CvDisjointSegmentsJob.PvgJ79Qc1Nro, BlankfreeDurationPriorMeanJob.ReQtJKYpZgsN) is unchanged and finished.
- **run:** the full job ran in-process into a temp dir on the real inputs. It took 408 s with a peak RSS of 3.11 GB.
  - Its output is test output only and is not banked. The log is in the session scratchpad at `an1v3/an1v3_run.log`.
  - After that run I relabelled one render string (the held-out stride label). That edit was checked with py_compile and the graph test (the id did not change), but not with another full run.

## Undetermined or assumed
1. The 4-gram and 5-gram tables are fit inside the job, not in a separate prior job. They are not saved, because the 5-gram table is 0.9 GB. A later TP-C build would refit them; the fit is deterministic and takes 17 s.
2. The rank is printed as 1 + the number of competitors at or above gold, out of 188 (the audit's convention).

## Launch (not done)
From the setup dir, on the login node jpbl-s02-03:
`/e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis manager -r config/sae_4a_rename_an1v3.py`

Expect about 7 minutes and a peak of about 3.1 GB. The ask is 8 GB.
