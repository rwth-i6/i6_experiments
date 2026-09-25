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
n-grams, N = 20 sub-epochs, float64 DP, pre-registered gates. i6: L40S 46 GB GPUs, 5 per user; V100 32 GB,
32 of them with no per-user cap. The V100 fits the reference batch and is 1.32x faster per step (`SAE_i6_P0.md`,
Results, cost screen). Trainings run on V100 from 2026-09-25 (user decision). A pack runs its arm AND its own control on the same GPU type.
Project rules (documents, commits, env): `CLAUDE.md` in the setup dir.

## Queue

1. **P0 port (ACTIVE)**, `SAE_i6_P0.md`: env, i6 settings, reproduce `ctrl_20`, `k2lat_20_ma3000`
   and the gold-phi init; verify every component and loss term (gates G0.R1-R3, G0.V).
2. After P0: resume the cold line from where JUPITER stopped (`SAE_i6_ref_lexlat_v2.md`, "Open at
   the move": the L2-1 wave with durinit / 12 sub-epochs, A14 (i), A17 (i)-(iii), A16 (b) stage 1-2,
   A18 bridges, A19). Which of these, and in what order, is decided after P0 with the user's
   priorities; most have no entry point in the port yet. Ported and matching the banked jobs: A10 12
   sub-epochs, duration prior, 285-set S reader and selection, gold phi, p0, rt ladder. Not ported: A10
   48 sub-epochs and its K* reader; A13 260-set reads; generative PER on dev-other (genmarg refuses it,
   and refuses `sil_run_collapse`); a ladder read outside the 285 set (25 of its utterances are fit
   items); none of the 34 SAE modules added on JUPITER after c49559ce (A11-A20, AN-0..6, TP0), on which
   the rename evidence rests (`exp_logs/SAE/SAE_4A_rename.md`, d9b7e5362). The JUPITER orchestrator
   will put a phi-init plan on the branch (`reports/jupiter_port_review_2026-09-25.md`).

## Phase pointers

- **P0 port and verification** — OPEN, the four trainings running (ctrl_20 on L40S; ctrl_20_s1,
  ctrl_20_rc and k2lat on V100) — `SAE_i6_P0.md`.
