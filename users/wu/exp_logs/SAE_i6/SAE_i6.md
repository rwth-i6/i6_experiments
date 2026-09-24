# SAE_i6 — unsupervised ASR through an exact-marginal cycle, on the i6 cluster

Continuation of phase 4A of the SAE campaign (JUPITER, 2026-07 to 2026-09-24), moved to i6 for
compute quota. Phase 4A is the whole of SAE_i6; the other SAE phases (speech-LLM GRPO etc.) are out
of scope. Basis: `SAE_i6_ref.md` (setup, protocol, banked numbers, constraints, established results)
and its topic files `SAE_i6_ref_*.md`. JUPITER logs: `exp_logs/SAE/` (frozen provenance).

## Objective

Pure unsupervised phone recognition without GANs: make the cycle (recognizer theta + semi-Markov
reverse model phi + frozen text prior, trained jointly through an exact marginal) take off from a
cold start on LibriSpeech train-clean-100, i.e. leave the content-free band (dev-other PER
0.83-0.91) toward the wav2vec-U 2.0 reference (0.214). Seeded (10 h supervised) refinement is a
disclosed parallel analysis, never cold-start progress.

## Constraints

`SAE_i6_ref.md` section 2 governs: label quarantine, no GAN, final or label-free checkpoints only,
paired speaker-clustered comparisons, one delta per arm with an own `ctrl_20`, uniform-sample
n-grams, N = 20 sub-epochs, float64 DP, pre-registered gates. i6: L40S 46 GB GPUs, 5 per user.
Project rules (documents, commits, env): `CLAUDE.md` in the setup dir.

## Queue

1. **P0 port (ACTIVE)**, `SAE_i6_P0.md`: env, i6 settings, reproduce `ctrl_20`, `k2lat_20_ma3000`
   and the gold-phi init; verify every component and loss term (gates G0.R1-R3, G0.V).
2. After P0: resume the cold line from where JUPITER stopped (`SAE_i6_ref_lexlat_v2.md`, "Open at
   the move": the L2-1 wave with durinit / 12 sub-epochs, A14 (i), A17 (i)-(iii), A16 (b) stage 1-2,
   A18 bridges, A19). Which of these, and in what order, is decided after P0 with the user's
   priorities; most have no entry point in the port yet.

## Phase pointers

- **P0 port and verification** — OPEN, cost screen running (input graph + ctrl_20) — `SAE_i6_P0.md`.
