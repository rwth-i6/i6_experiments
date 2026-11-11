# AppTek feedback follow-up (2026-11-11)

This update was prepared in the isolated `berger_dev` worktree. The live public
checkout and its running jobs were not modified.

## API parity is a required measurement, not established yet

`9804f5aa` and `d000386c` have incompatible Python scorer APIs and boundary
configuration keys. The old Granite jobs must not be pointed at the newer binary.
`CtcQwenRecogJob` uses the request-based API and keeps Granite surface casing;
`CtcLlmGateJob` checks CTC repeats, blank-separated repeats, words and EOS costs.
The native gate passes in a fresh subprocess on the pinned d000386c build.

New public-only entrypoint: `config/ctc_api_parity.py`. By default it runs only
Granite + Speech-Qwen 1.7B on full LibriSpeech dev-other, using the exact selected
prior, acoustic model, auxiliary model, dataset, beam/thresholds and FP32 settings.
It reuses the completed old-API run, creates one new GPU recognition, scores it,
and compares ID coverage, token paths, raw/normalized transcripts, cumulative
AM/LM costs on equal paths and exact edit counts. Its report does not claim parity
from equal rounded WER. Report: `reports/asr-lm/ctc-api-parity-librispeech-dev-other.md`.

Use the isolated research checkout with the public setup's dependencies and
settings to load this entrypoint; do not switch the live checkout while workers
are running or start a second manager writing its outputs concurrently. With
`config` pointing to the research launchers:

```bash
./sis m config/ctc_api_parity.py
```

`ASR_CTC_PARITY_DATASETS` optionally adds `loquacious-dev.short` (comma-separated).
`ASR_CTC_PARITY_TRANSFORMERS` defaults to `5.16.1` and is a hashed, enforced input
of the new recognition. A separate 5.19.0 run needs that environment and a separate
job. Accepting both library versions is not proof of numerical parity between
them. This shell has no CUDA or sbatch: the GPU comparison is prepared, **not run**.
Do not pool AppTek/new-API Granite numbers with old-API public results until the
comparison has been reviewed.

## Frozen settings and prior are now in the repository

`frozen_public_conditions.json` contains all 12 fast selected families from the
handoff, with both development corpora's stage constructor parameters, repositories,
full decoder-config text, configuration SHAs, public artifact provenance and paths
requiring relocation. It is an explicit audit manifest, not a company replay
launcher: its public paths must be replaced by company-local artifacts.

The selected Granite AM-only control was `CompactGraniteNbestJob(n=50)`, with the
LibriSpeech **open fallback tree**, LM scale zero, beta 1, alpha 2, beam and word-end
beam 256, thresholds 16, and `sentence-end-fall-back=no`. It was not plain greedy
lexicon-free decoding or the new CTC job's `kind="zero"` condition. Keep native
controls and do not conflate these different AM-only searches.

The exact public prior is committed at
`users/berger/data/assets/granite_ctc/librispeech_train_clean_100_prior.json`, with
metadata/checksum, label inventory and training sample IDs. It has 16,384 values.
Granite's fully resolved revision is
`1ea89b77a944f4a5ea26e5fcf36273e008594436`; historical download-job revision arguments
remain short to preserve existing IDs, while the manifest records the full commit.

Speech-Qwen beam 8/length penalty 0 is an **explicit** `SpeechQwenBeamJob` condition,
not the default native job. Initially it was dev-only. The completed frozen
selected-evaluation replay now contains it on held-out corpora too (e.g. AMI
1.7B is `SpeechQwenBeamJob.gYmZWX2uOj9M`, beam 8, length penalty 0). The company
config must still register that baseline explicitly; ordinary `qwen_speech.run()`
does not imply that beam policy. The frozen manifest includes its stage parameters.

## Imports, environment and job identity

`tools` no longer imports CMakeJob or constructs the default clone/compile jobs
at module import. The default tool bundle is lazy; explicit build factories import
CMakeJob only when called. Config-only import was tested with CMakeJob removed.
This does not add CMakeJob to public i6_core: compilation still needs the local
extension or an externally supplied binary.

The incoming compatibility commits changed 785 of 1,197 checked public IDs because
the builders hash runtime source bytes, despite unchanged constructor defaults.
`bootstrap.source_bytes()` now applies a versioned, SHA-guarded reverse-edit registry
for only the reviewed compatibility/default-preserving changes. It reconstructs
the historical fingerprint bytes, not execution source, and validates their SHA.
Unknown/future edits fingerprint normally; new CTC parity code fingerprints its
actual runtime source. This keeps the checked original public graph reusable.
It is not a blanket hash override or a promise about arbitrary AppTek graphs.
A graph construction check is not observation of the running manager, which has
intentionally not pulled these changes.

The common version checker accepts 5.16.1/5.19.0. `HfBenchmarkRecogJob` still hashes
and enforces its explicit `transformers_version` argument (default 5.16.1); set it
to 5.19.0 on AppTek. Other runtime builders must record the chosen environment in
their provenance/code version; sharing constructor defaults does not make different
library environments experimentally identical.

`GraniteRawPiecesJob`, `HuggingFaceCtcVocabJob`, `ScoreBenchmarkJob` and normalized
hypothesis comparison now request worker tasks instead of mini-tasks. The worker
image must contain their dependencies. Install
`requirements-scoring.txt` (`kaldialign==0.12.0`) into AppTek v19; do not install
packages dynamically inside inference jobs. Native gates/tests need separate
processes: two SearchAlgorithm instances can trip Flf::Lexicon's singleton check.

## Safe resume for the request-based jobs

`RnntQwenRecogJob` and `CtcQwenRecogJob` now accept **opt-in** `resume=True`.
The default remains false to preserve existing constructor behavior and the
checked public IDs. AppTek source-version builders that hash the new CTC module
may still create new identities; check that graph separately. The flag
and resume-helper version affect opt-in job identity. The new parity job opts in.
Completed independent segments are kept in an ordered JSONL prefix with a sidecar
binding the dataset IDs/fingerprint, job identity, decoder config, vocabulary and
prior. Changed metadata, duplicate/out-of-order IDs and unidentified old outputs
are rejected. A torn final line is discarded and that segment is recomputed.
RTF totals carry forward completed segments; failure/reload overhead and discarded
partial-segment work are not counted as completed inference. The Sisyphus task is
marked resumable when opted in.

Tests simulate an interrupted job and verify no reencoding of completed segments,
no duplicate hypotheses, preserved timing totals, identity rejection and torn-write
recovery. This is **per-segment**, not within-recording decoder/KV-state resume.
Native and old-API jobs remain unchanged and restart from scratch. Output cleanup
must preserve the hypothesis file and its `.resume.json` sidecar.

## RASR merge note (reported by AppTek; not compiled here)

AppTek master contains GitHub master through `8fe741b1`. A textual merge of
`5705b9e7` is not a successful build: AppTek's `LanguageModelLookahead` constructor
needs a lexicon argument absent from the call introduced by GitHub `842410a2`.
The AppTek extra-lexicon update also needs to rebuild/invalidate the new lookahead
tables. Resolve both in the company RASR tree and test lexicon updates before
using that merged decoder. This experiment-repository update does not change
RASR or claim that merge compiles. Keep the known pinned binaries for parity.

## Validation on the public side

All 190 tests passed, including the real native CTC gate in its own process,
config import without CMakeJob, source-fingerprint invalidation, prior/config
assets, comparison of token paths and interrupted/resumed job execution. The
final graph probe is byte-identical to the original 1,197-job inventory. The
parity entrypoint constructs seven jobs (one GPU recognition plus its pinned
build/gate/scoring/comparison dependencies) without rebuilding the full suite.
The GPU result and an actual reloaded-manager inspection remain pending.

A standalone batch script is also supplied so the parity check can run now
without switching the live checkout or starting another Sisyphus manager:

```bash
sbatch /work/asr4/berger/refactor/asr-20261104/research/i6_experiments/users/berger/configs/model_combination/scripts/run_ctc_api_parity.sbatch
```

It uses the existing public image/binary/model/dataset paths, runs native stages
in separate processes and writes only under
`/work/asr4/berger/refactor/asr-20261104/ctc-api-parity-dev-other`. It enables
per-segment resume. Keep that isolated checkout fixed while the batch runs.
The standalone CPU gate has been executed successfully; its GPU stage has not.
