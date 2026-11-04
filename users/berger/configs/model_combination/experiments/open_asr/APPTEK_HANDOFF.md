# AppTek evaluation: company-side handoff

Recorded: 2026-11-04. This is a preparation and decision plan, not a company
deployment. AppTek data and experiments require a separate setup on company
servers. No proprietary recordings, transcripts or credentials belong in this
public-data workspace.

## Start preparation now

Preparation can run alongside public-data Steps 1–4. It does not depend on the
remaining full-suite GPU jobs.

1. Choose a representative, manageable company development pilot, plus an
   untouched evaluation split. Keep recordings, speakers and related material
   together when splitting. Record any prior use of the evaluation data.
2. Establish the permitted research use and internal handling/export rules.
   Keep audio, transcripts, examples, hypotheses, n-best lists and caches on
   company infrastructure; export only approved aggregate results.
3. Inventory recording lengths, domains, channels, sample rates, reference
   conventions and segmentation. Preserve stable recording and segment IDs.
   Fix the evaluation protocol before looking at model-specific errors.
4. Recreate the runtime on company servers: pinned model/tokenizer revisions,
   RASR commit/build options, containers, normalizer and alignment versions.
   Use company-local caches and storage. Public Sisyphus absolute paths and
   job hashes are not portable; reconstruct jobs from explicit configuration.
5. Verify common audio decoding and scoring with a small smoke test. Decide
   how to handle punctuation, numbers, partial outputs and truncations.
   Retain raw hypotheses as well as normalized scoring artifacts.

The existing analysis interface uses stable `id`, `group_id`, `chunk_index`
and `reference` fields, with `audio_seconds` recorded at inference time.
Declare which fields identify input chunks versus official scoring units.
Preserve hypothesis-to-reference coverage and configuration provenance.

## Start native baselines before porting additional search

Once the data/scorer smoke test passes, run both standalone constituents on the
company development pilot. This can start before the public full-suite study
finishes. Inspect WER, I/D/S, runtime, memory and paired error complementarity.
Standalone Speech-Qwen must use an explicit, comparable decoding budget.

First compare on identical fixed segments. Recording-grouped scoring of many
short inputs does not test long-form recognition. For natural long recordings,
separate segmentation and context: common fixed segments versus the long-form
pipeline, and history reset versus carried context where supported. Report
output limits, retained partial outputs, truncations, repetitions and
hallucinations alongside WER. Start small before scaling.

## Decision gates

| Gate | Evidence needed | Next action |
| --- | --- | --- |
| Data/scorer readiness | Stable splits, common preprocessing and verified scoring | Native company-dev baselines and a small pilot |
| Public Step 3 | Better alternatives absent from AM n-best versus present but misranked | Choose search direction/candidate generation versus score calibration before porting expensive search |
| Public Steps 2/4 and company-dev complementarity | Useful headroom and informative source/duration/rarity patterns | Choose the model pair and evaluation conditions worth expanding |
| Company-dev tuning complete | Frozen checkpoints, search parameters, scoring and primary metrics | Run untouched company evaluation once |

If candidate coverage is the main limitation, prioritize a union-pool or
Speech-LLM-led pilot. If ranking is the main limitation, test calibration on
company development data before implementing another search. A union-pool
system needs comparable scores from both models for every candidate; the
reference-dependent augmented oracle is not deployable.

The confirmatory comparison must include each standalone constituent, the
better standalone result, selected fusion and its computational cost. Do not
select configurations on the company test set. Public test results already
used during exploration remain exploratory; simply moving to company data
does not remove the need for a development/evaluation split.
