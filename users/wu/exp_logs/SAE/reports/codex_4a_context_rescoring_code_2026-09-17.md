DONE_WITH_CONCERNS

Read-only feasibility inspection; no implementation, experiment, or active-training change.

Paths below are relative to `recipe/2025-10-speech-llm/src/speech_llm/sae/` unless stated otherwise.

Existing seams:
- `emc/lattice.py:895` `forward_log_z(..., table_out=...)` exports forward masses; `:987` `lattice_forward_backward` returns aggregate frame/segment arc marginals (`LatticeOutput`, :397). There is no complete posterior-path sampler. A new backward sampler could use the exported masses and the existing arc weights, retaining CTC frames, token-emission decisions, and reverse durations. Independently sampling returned frame/segment marginals would not produce a legal joint sample. `reverse.sample` (:368) generates fresh synthetic units given phones; it does not sample this observed-utterance lattice posterior.
- No complete-candidate beam/n-best API was found inside `emc/`. Its decode jobs describe greedy decoding (`emc_train_jobs.py:330,1756`). `channel_h.prefix_beam_decode` (:471) produces n-best with LM callbacks, but operates on the older channel-H HMM/emission topology, not the EMC CTC/reverse lattice. Reusing its output would require an explicit proposal adapter; it is not the current posterior.
- No fixed-phone-sequence, same-temperature, same-band joint marginal API exists. Reusable inputs are recognizer `log_q`, `build_segment_table` (:414), duration legality and lattice arc conventions. A new constrained DP needs phone position in place of unrestricted history, retaining CTC repeat/blank state and reverse offset. It must sum compatible path/segmentation assignments at the specified temperature, not multiply two existing likelihood calls.

Why ordinary CTC(y|x) times reverse.forward_logsum(z,y,eta) is different:
1. The band couples CTC time to reverse consumed frames, so legal assignments are not a Cartesian product of CTC paths and reverse segmentations.
2. Temperature acts on EACH joint assignment before summation; `(sum q * sum p)^(1/tau)` does not equal `sum (q*p)^(1/tau)`. `reverse.forward_logsum` (:299) has neither temperature nor CTC-time/band inputs. Even without coupling, one would need separately tempered path sums, not powered ordinary marginals.
3. `lattice.py:45–74` permits adjacent SIL as either a new token or a repeat. One frame path therefore corresponds to multiple consistent phone strings, unlike strict standard CTC collapse. The exact implemented latent is (frame path, consistent token string, reverse segmentation). Its band is the state invariant `|s-t|<=W` at every state; emitted segments compare with `t+1`. A replacement must preserve this convention.

For fixed y, the prior factors out as `P(y)^(beta/tau)`. Thus an exact constrained joint score would permit arbitrary full-sequence LM scoring over a FINITE candidate set, but the outer candidate sum remains approximate. Alternatively, complete samples from the trigram posterior could receive a sequence-level LM/prior ratio; this still needs a new sampler, scorer and estimator and is not exact neural-LM integration. If replacing the prior, the log ratio is `beta/tau * (log P_new(y)-log P3(y))`; multiplying a new LM without removing P3 is a different objective.

`PriorHistory` (:251–289) is structurally restricted: `next_h[h,k]=succ[last(h)]*n_ctx+k`; registered histories are bigram/class-trigram/trigram. It cannot accept a neural hidden state or arbitrary full prefix as a plug-in table. `reports/estimate_prior_order_2026-09-15.md` discusses FFBS/importance sampling as a proposal with weight-degeneracy estimates, not an implemented API or measured accuracy result. Its old runtime estimates precede later D3/D4 work and are not current benchmarks.

No `firstorder`/`first_order` hook was found in EMC, SAE, the speech_llm tree, or the searched wu recipe Python files. Separate generic prefix-LM rescoring files exist under `speech_llm/prefix_lm/model/rescoring/`; their existence is not evidence of EMC training integration. The adjacent channel-H LM callback is decoding only. `reports/codex_4a_lm_prior_evidence_2026-09-17.md` records no higher-than-trigram/neural/word LM cold-training result.

Minimum concrete integration boundary: the existing `log_q` + reverse segment table + current arc conventions. Missing components are either complete joint posterior sampling, or complete candidate generation plus a fixed-y constrained joint DP, together with sequence LM scoring and a deliberately approximate outer estimator. Current beta=1, tau 8→2, band25 and 41 CTC categories remain unchanged; this report licenses no new result claim.

/e/project1/spell/wu24/2026-07-13_unsupervised/reports/codex_4a_context_rescoring_code_2026-09-17.md
