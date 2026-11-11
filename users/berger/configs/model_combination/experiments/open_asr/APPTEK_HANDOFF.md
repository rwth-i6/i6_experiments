# AppTek model-combination experiments: company-side handoff

Recorded 2026-11-04; updated 2026-11-11 after the public-data review and repository split.
This is a setup brief for a separate company-side checkout. No company jobs have
been created or submitted from the public setup. Preparation can start while the
last public SPGISpeech jobs are running; switching the public live checkout still
waits for its workers to finish.

## Objective and current scope

Test when combining pretrained ASR models improves on the **better standalone
constituent**, and whether the improvement justifies its computational cost.
Start with English data, matching the current public study. Other languages need
a separate choice of ASR models and evaluation protocol.

The public results justify transferring the existing algorithms. New calibration,
union-pool, reversed-search and controlled-noise studies are **optional**, not
prerequisites for AppTek evaluation. Do not port the full historical tuning grid.

Useful observations from the [completed-result review](COMPLETED_REPORT_REVIEW_20261110.md):

- Speech-Qwen 1.7B generally gives stronger fusion than 0.6B or text-Qwen.
- Two-pass rescoring is a competitive, cheaper baseline for one-pass fusion.
- Fusion can beat a stronger standalone Speech-Qwen on Monsoon, but loses to it
  on Earnings22 and GigaSpeech. This is a conditional benefit, not a universal gain.
- AMI does **not** show the previously suspected Parakeet search degradation:
  native Parakeet is 14.34%, AM-only beam search 13.29%. The 9.43% value was
  Speech-Qwen 0.6B, misread from a table of pairwise minimum scores.
- The completed short-segment experiments do not establish long-form behavior.

## 1. Code and runtime to transfer

Use the public `rwth-i6/i6_experiments` repository, branch **`berger_dev`**.
The cleaned `berger_template_setups` branch contains generic examples and does
not contain the research fusion/evaluation pipeline needed here.

| Location | Reusable component |
| --- | --- |
| `users/berger/data/` | Dataset preparation, reference grouping and provenance helpers |
| `users/berger/model_pipelines/common/` | Native adapters, one-pass fusion, n-best rescoring, scoring and analysis |
| `users/berger/configs/model_combination/experiments/open_asr/` | Public recipe/config builders to adapt |
| `users/berger/configs/model_combination/` | Modular launchers, tests and research notes |

Keep the branch-local import compatibility layer enabled. Historical
`example_setups.seq2seq_rasr_2025.*` imports resolve to moved research modules;
canonical `users.berger.*` imports alias the same classes. See the
[layout and migration guide](../../README.md) and
[migration validation](../../MIGRATION_VALIDATION.md).

Create a company-local dataset adapter and AppTek-only entrypoints, for example
`apptek_dev.py` and `apptek_eval.py`. These are **files to create in the company
setup**, not existing public launchers. Reuse explicit job/config builders rather
than invoking `open_asr.run_all()` or replaying the public controlled-evaluation
selection file: those reconstruct public corpora and absolute job paths.

Record all dependency revisions and necessary patches. The public setup's
`i6_core` checkout has local changes, including CMake support; they were not
committed by the `i6_experiments` cleanup. Inspect and transfer the required
patches separately. The public settings currently use a Torch 2.8/ONNX 1.26
Apptainer image; record its actual digest and installed package versions rather
than assuming the filename establishes equivalence. Current pretrained adapters
explicitly require `transformers==5.16.1` and use FP32 with TF32 disabled.

Set company-local model/data caches, work/output/alias directories, scheduler
queues and container paths. Do not reuse `/u/berger` or `/work/asr4/berger` paths.
Public Sisyphus job hashes and compiled `.so` files are not portable artifacts.
Rebuild RASR against the company Python/CUDA/ONNX environment.

### Model pins used in the public comparison

| Model | Hugging Face repository | Recorded revision |
| --- | --- | --- |
| Granite CTC | `ibm-granite/granite-speech-5.0-470m-turboctc-nc` | `1ea89b7` |
| Parakeet RNN-T | `nvidia/parakeet-rnnt-1.1b` | `2acc4c61eede2f52ddefe74935de1930a9064d4a` |
| Speech-Qwen 0.6B | `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` |
| Speech-Qwen 1.7B | `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` |

Granite's original recorded revision is short; resolve and record the full
commit when materializing the company checkpoint. Speech-Qwen here means the
**audio-conditioned ASR checkpoint**, not text-only Qwen3 Base.

### RASR implementations used by the selected public jobs

| Purpose | Pinned commit |
| --- | --- |
| Granite dynamic one-pass | `9804f5aa545d1fcf69df4843382a41242eb9f722` |
| Parakeet dynamic one-pass and its n-best recipe | `d000386c08a9b9213238e3cafdeb4291dc0b1775` |
| Granite tree n-best generation with score-accounting fixes | `ff206f90c64fa5c44fa1acb7059f23722bfefc8c` |

These are distinct historical implementations, not interchangeable builds.
Run the corresponding topology, multi-token/casing and one-best/n-best parity
gates against the **same binary** used for decoding. A newer unified branch can
be used only after these checks and a public-data smoke comparison. The build
options recorded by these jobs include `-DMODULE_TENSORFLOW=Off`,
`-DMODULE_LM_TFRNN=Off`, and `-DMARCH=x86-64-v3`; adapt the architecture flag to
the company nodes and record that change.

The research [test guide](../../TESTING.md) describes CPU checks and explicit
native asset variables. Tiny CPU tests do not validate production GPU memory,
provider availability, full-checkpoint WER or runtime. Run a small real-model
GPU smoke test before scaling.

## 2. Prepare data and scoring

1. Choose a manageable company development pilot and an untouched evaluation
   split. Split related recordings/speakers together and record prior use.
2. Inventory language, domain, channels, sample rates, duration, segmentation and
   reference conventions. Keep company audio, references, hypotheses and caches
   on company infrastructure; share only results permitted for the study.
3. Feed every model the same decoded audio and segment boundaries. Match the
   model adapters' 16 kHz mono input, without lossy re-encoding, preemphasis or
   extra peak normalization. Retain original audio provenance and conversion
   settings; any required channel selection/downmix must be common to all models.
4. Implement the prepared-data interface: `out_dataset` (a saved HF Dataset with
   `id`, `audio`, `text`), `out_references` (JSONL), and `out_manifest` (dataset,
   sampling rate, counts and audio provenance). This is a local storage interface;
   company data need not be published to Hugging Face.
5. Each reference row contains `id`, `group_id`, `chunk_index`, `reference`.
   For independent segments use `group_id=id`, `chunk_index=0`. For recording-level
   scoring, give chunks the same group and full recording reference, with ordered
   unique indices, as required by `benchmark_scoring.py`. Do not repeat a segment
   reference as though it were the full recording reference. Preserve coverage.
6. Retain raw and normalized hypotheses. For English comparability, use
   `ScoreBenchmarkJob` and the updated normalizer revision
   `48219c6028db0517d704600d92f31edfc96e8c23`, including compound-equivalence scoring.
   Document any additional official company scoring separately. Do not silently
   switch to the older normalizer behind some public leaderboard numbers.

Choose scoring units and statistical clusters explicitly from company metadata.
The public `cluster_id()` heuristic recognizes public identifier formats; it must
be adapted to company recording/speaker IDs rather than treating every short
segment as independent. Keep insertions, deletions, substitutions and denominators,
not only rounded WER. Frequency/rare-word bins require an independent company
training-text source; LibriSpeech frequency is not generic rarity. Proper names
need independent labels if they are to be analyzed as a category.

## 3. Initial experiment batch

Start with common fixed segments on development data. Suggested core:

| Condition | Purpose |
| --- | --- |
| Native Granite and native Parakeet | Standalone controls |
| Native Speech-Qwen 1.7B greedy and beam 8, length penalty 0 | Strong standalone auxiliary controls |
| Granite/Parakeet AM-only RASR search | Separate search effects from model combination |
| Each AM + Speech-Qwen 1.7B, one-pass | Transfer existing fusion |
| Each AM + Speech-Qwen 1.7B, two-pass | Accuracy/runtime comparison and cached diagnostics |

Add 0.6B standalone/fusion or text-Qwen only if model size or LM type is a
company-side research question. Do not assume 1.7B is always better on company
data. Compare each fusion against the better of its own standalone constituents,
with both native and search-control scores visible.

Use the frozen public settings as **initial transfer settings**, not claimed
company-domain optima:

| Setting | Granite + Speech-Qwen 1.7B | Parakeet + Speech-Qwen 1.7B |
| --- | --- | --- |
| One-pass auxiliary scale | 0.3 | 0.25 |
| One-pass word penalty | 0 | -0.5 |
| Final / pre-LLM beam | 32 / 128 | 8 / 32 |
| Final / pre-LLM threshold | 6 / 6 | 3 / 3 |
| AM prior correction | 0.1 in selected recipe | None in selected recipe |
| Word casing | Preserve Granite surface | Lowercase and capitalized variants |
| Two-pass candidate cap | 50 | 50 |
| Two-pass auxiliary scale | 0.5 | 0.3 |
| Two-pass word penalty | -1 | -1 |
| Two-pass formatting | Initial capital plus terminal period | Initial capital plus terminal period |

Reconstruct the full method-specific config, including first-pass generation,
blank index, repetition behavior, recombination and length limits; the table is
not a complete decoder configuration. Native RNN-T and CTC do not share topology.
Share first-pass candidate lists across auxiliary checkpoints/settings. Sum
continuation NLLs with EOS and retain the recorded first-pass costs; do not
average by token count or invent missing AM scores. Check scaled n-best/one-best
score parity before rescoring.

For Granite prior correction, explicitly choose either the frozen public prior
as a transfer condition or a new prior estimated from separate company training
(or designated calibration) audio. Do not estimate it from untouched evaluation
data or silently substitute a company prior while calling it the same condition.

Use the current FP32 scoring and batch size 8 policy initially. Preserve the
per-method casing policy in the table; text-only and speech auxiliary comparisons
for a given AM must use the same casing strategy. Initialize librasr logging,
keep stepwise statistics disabled, and retain partial hypotheses on output-limit
termination instead of appending a fabricated EOS. Record truncations and caps.

## 4. Measure, then decide whether to expand

Measure complete end-to-end RTF (both passes for rescoring), WER/I/D/S, host and
GPU memory where actually measured, truncations and paired corrections/regressions.
Use the same GPU/container for speed comparisons; requested GPU RAM is not peak
usage. Start on a small development subset and measure duration/RTF before
requesting a large job. The public target is RTF <1.5, but transfer settings exceed
that on some corpora. Company queue/runtime limits must be configured locally;
no sharding is introduced by this handoff.

The fixed transfer batch can be reported as such. Any adjustment of scales,
pruning, priors or output limits uses company development data only; freeze the
choices before running untouched evaluation. There is no need to wait for a
public-data calibration or reversed-search project.

For later natural long-form work, compare fixed common segments with genuinely
long model inputs, and separate segmentation effects from history carry-over.
Recording-grouped scoring of many short inputs is not long-form recognition.
Begin with a few recordings and inspect deletions, repetitions and output limits.

## 5. Analysis fixes and deliverables

Use `berger_dev` with the multi-fusion attribution fix from `3954b21c6` (and the
AMI interpretation correction from `def2a4033`). Earlier complementarity summaries
assigned the last fusion's counters to every fusion in a panel. WERs and standalone
pair oracles were unaffected; multi-system correction/regression counts must be
regenerated. This is a CPU analysis fix, not a reason to rerun ASR inference.

The first company-side handoff should produce:

- A local runtime manifest: code/dependency/model/RASR revisions, patches,
  container digest, precision, GPU type, data/scoring versions and seeds.
- Data/scorer and real-model smoke checks, plus native/topology/n-best gates.
- A development table with both standalone and fusion controls, I/D/S and cost.
- Paired error/complementarity analysis and a list of missing or failed conditions.
- A frozen evaluation configuration and explicit remaining research questions.

Company dataset preparation, paths, scheduler settings, statistical grouping and
entrypoints still need implementation on company servers. The public handoff is
not a turnkey AppTek config, and no private credentials or dataset knowledge are
assumed here.
