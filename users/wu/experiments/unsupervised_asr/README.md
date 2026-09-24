# unsupervised_asr: phase 4a, the exact-marginal cycle (EMC)

A self-contained sisyphus setup for phase 4a of the unsupervised ASR (SAE) campaign. It trains two
models together on unlabelled LibriSpeech train-clean-100:
- a recognizer theta: a blank-free convolutional phone recognizer on frozen wav2vec 2.0 features;
- a segmental reverse model phi, from phones to k-means units.

The text side is a phone trigram prior and, in the k2 arms, a word-level HLG graph.

Every input is built from public raw sources:
- openslr LibriSpeech;
- the LibriSpeech LM corpus and lexicon;
- the Hugging Face wav2vec 2.0 checkpoint;
- MFA alignments, used only for PER references and the analysis-only pieces.

The package imports only the standard library, third-party packages, `i6_core`,
`i6_experiments.common`, `returnn`, `sisyphus` and itself.

## Layout

| path | content |
|---|---|
| `inputs.py` | wires every input once: `get_inputs()`, `get_seed_inputs()`, `get_graph(kind)` |
| `config/` | sisyphus entry points, each with a `py()`: `base`, `k2_word_lm`, `lexlat_v2`, `supervised_init`. The shared read wiring is in `common.py` |
| `default_tools.py` | interpreters, the RETURNN root, KenLM, ffmpeg and the HF cache, and the `settings.py` keys that set them |
| `env/` | the conda env: `environment.yml`, `install_env.sh` (creates the env, with sequitur and k2) and `build_k2.sh` (k2 source build). See "One env" |
| `phones.py` | the phone set: 39 ARPAbet monophones plus `SIL` |
| `data/` | ogg zips (pinned ffmpeg, checked by `ffmpeg_pin.py`), split ids, the shipped train shard and 10 h seed id lists, the CV split, rVAD streams, speaker eta, MFA gold, phone targets |
| `w2v2/` | wav2vec 2.0 large-lv60 `hidden_states[15]` dumps (`ReturnnForwardJobV2`) and the k-means units |
| `lm/` | lexicon and g2p, phone text, the prior's line sample, the phone trigram prior, word LMs, lexicon trie, HLG builds (in-house and official) |
| `model/` | RETURNN-side model code: recognizer, reverse model, lattice, EMC / blank-free model, train step, k2 lexical lattice |
| `training/` | config builder, schedules, arm presets, flat init, checkpoint slicing, the RETURNN patch |
| `reverse_model/` | duration prior, phi-first EM (L2-1), genmarg reads and selection, supervised inits, the L2-0 ladder |
| `analysis/` | posterior dump, greedy PER, derangement / decode gap, paired PER delta, JSD reads |
| `tests/` | CPU unit and graph-build tests (`pytest tests/`) |
| `vad_port.py` | the old top-level rVAD port. It is kept only because `experiments/ssl/analysis/real_repr_probe.py` loads it by file path; the package itself uses `data/vad_port.py` |

## Quick start

1. Create a setup dir with `recipe/` holding `i6_core`, `i6_experiments` and `returnn`, and with
   `tools/sisyphus`, at the commits in "Environment pins" (i6_core at least `25db695`). Run every
   sisyphus command from the setup dir. Create the conda env with `env/install_env.sh` (see "One
   env").
2. Set the server's paths in `settings.py`. The package reads these keys:
   - `FFMPEG_BINARY` (required): the ffmpeg that encodes the LibriSpeech audio. The graph build
     raises without it. The reference is the conda build of ffmpeg 7.1.1. See "The ffmpeg pin check"
     below.
   - `FFMPEG_PIN_ACCEPT` (optional, unset by default): a label that lets an ffmpeg which fails the pin
     check run anyway, as a separate audio generation. See "The ffmpeg pin check" below.
   - `SAE_PYTHON` (optional): the conda env's python, `<env_prefix>/bin/python`.
   - `K2_PYTHON` (optional): the python of a separate k2 env. Without it the k2 jobs use
     `SAE_PYTHON`, which is right for the env `env/install_env.sh` creates. The reference cluster
     must set it, because its main env has no k2 (see "One env").
   - `HF_HOME` (optional): the Hugging Face cache.
   - `KENLM_BINARY_PATH` (optional): a prebuilt KenLM `bin/`, built from `default_tools.KENLM_COMMIT`.
     Set it where the job env cannot compile KenLM. Without it, `CompileKenLMJob` builds KenLM.
   - `G2P_PATH` and `G2P_PYTHON`: sequitur `g2p.py` and its python, for i6_core's g2p jobs.

   Without `SAE_PYTHON` or `HF_HOME`, the reference cluster's path in `default_tools.py` is used;
   without `K2_PYTHON`, `SAE_PYTHON` is used. Every one of these paths carries a fixed
   `hash_overwrite`, so setting them moves no job hash. `FFMPEG_PIN_ACCEPT` is the exception: it moves the hashes on purpose. `RETURNN_PYTHON_EXE`
   and `RETURNN_ROOT` are not read: every RETURNN job states its interpreter and root explicitly.
3. Route the download jobs to a node with internet access. `DownloadHuggingFaceSnapshotJob`
   (`data/hf_hub.py`) fetches the wav2vec 2.0 checkpoint and the MFA alignments from the Hugging Face
   Hub, and i6_core's `DownloadJob` / `DownloadLibriSpeechCorpusJob` fetch the openslr files. On a
   cluster whose compute nodes are offline, these fail unless `settings.py` sends them to an online
   node. The reference settings run every job whose class name starts with `DownloadHuggingFace` on
   the login node, with `HF_HUB_OFFLINE` off.
4. Start a manager on one or more entry points:

   ```
   tools/sisyphus/sis manager recipe/i6_experiments/users/wu/experiments/unsupervised_asr/config/base.py
   tools/sisyphus/sis manager recipe/i6_experiments/users/wu/experiments/unsupervised_asr/config/k2_word_lm.py
   ```

   Sisyphus imports the module and calls its `py()`. To call another function, or `py()` with
   arguments, write a small config in the setup's own `config/`. For example:

   ```python
   def py():
       from i6_experiments.users.wu.experiments.unsupervised_asr.config import lexlat_v2
       lexlat_v2.ladder()
   ```

### Entry points

| module | `py()` builds | other functions |
|---|---|---|
| `config/base.py` | `ctrl_20` and `ctrl_20_x60` (the control, phone trigram only), their reads and the input manifests | `ctrl_arms(inputs)` |
| `config/k2_word_lm.py` | the default arm `k2_word_lm`; the presets `k2lat_20_ma3000`, `k2lat_20_ma3000_x60` and `off4_k2lat_20`; the controls; the paired PER deltas and JS rows against the control of the same length. `py(phone_trigram=...)` sets the default arm's phone-trigram mode | `k2_arms(inputs, phone_trigram=...)` |
| `config/lexlat_v2.py` | the L2-1 stage-1 probe: `uniform` / `durinit` / `durfrz` x seeds 1 and 2, with genmarg reads of every sub-epoch | `wave(phi_c_source=, duration=, num_subepochs=)`: the L2-1 wave and its label-free selection. `ladder()`: L2-0, analysis only |
| `config/supervised_init.py` | ANALYSIS ONLY: the 10 h supervised reverse init (gold phi) and p0 | `gold_phi()`, `p0()` |

All reads are on dev-other:
- greedy PER at every kept checkpoint;
- at the final checkpoint only, the derangement gap, the D4 decode gap and the JS rows.

The kept checkpoints are sub-epochs 1, 4, 10 and 20, plus 30, 40, 50 and 60 in the x60 runs.

The wave's arms are trainings of their own. Every wave config carries the inert key
`phi_first.WAVE_KEY`, so at `num_subepochs=4` no wave arm shares a job with a probe run, as in the
source, where the probe and the wave were separate job classes.

## The ffmpeg pin check

The audio depends on the ffmpeg build, and the ffmpeg path enters the job hash only through a fixed
label. So `data/ffmpeg_pin.py`'s `FfmpegPinCheckJob` runs before any encode:
- it encodes all 2864 dev-other recordings with i6_core's `BlissChangeEncodingJob` command and
  `FFMPEG_BINARY`, on 4 processes (about 1 min on the reference login node; the job asks for 4 CPUs,
  4 GB and 1 h);
- it decodes each result to 16-bit PCM with the same binary;
- it compares each PCM sha256 with the shipped reference list
  `data/ffmpeg_pin_dev_other_pcm_sha256.txt.gz` (one line per recording). The list was computed
  with the reference cluster's conda ffmpeg 7.1.1, and it equals the decode of the banked audio for
  all 2864 recordings.

The check covers the whole of dev-other because a build can change some recordings and leave others
bit-identical: an alternative libvorbis build changed 73% of a dev-other sample, yet left
116-288045-0002 to 0004 bit-identical. train-clean-100 is not checked; dev-other stands for it.

The job writes every recording's PCM sha256 to `pcm_sha256.txt` and the `ffmpeg -version` output to
`ffmpeg_version.txt`. On a mismatch it fails, giving the number of changed recordings, the first few
names and the `ffmpeg -version` line; `pin_check.txt` lists every changed recording. On success it
writes `ffmpeg`, a wrapper that execs the checked binary. The encode jobs take that wrapper as their
ffmpeg, so no encode can run before the check passes. The check's hash holds the pin label and the
reference list's sha256, not the binary's path, so every server whose ffmpeg passes shares the
downstream hashes.

### When the check fails: `FFMPEG_PIN_ACCEPT`

The comparison is bit-exact. The references come from an aarch64 (GH200) build, so on another CPU
architecture, such as x86_64, even ffmpeg 7.1.1 may fail it. When that happens the audio really is
different, so failing is the default. First try an ffmpeg that passes. If none does and you accept
working on different audio, set a label in `settings.py`, for example
`FFMPEG_PIN_ACCEPT = "x86_64-conda-ffmpeg-7.1.1"`. Then:
- the check still encodes and decodes every recording, and writes their PCM sha256 and the
  `ffmpeg -version` output to its outputs;
- a mismatch is recorded in `pin_check.txt` with the same numbers and marked `ACCEPTED`, instead of
  failing the job;
- the label enters the check job's hash, so every downstream ogg, feature, unit and model hash
  changes.

Results under a label are a different audio generation. They are **not** the banked audio, so compare
them with the banked numbers only as a reproduction on other audio. They never share a job with the
banked-audio graph. Leave the key unset for the strict default. An encode that fails outright (for
example, an ffmpeg without libvorbis) still fails under a label.

## The k2 default

The default k2 arm is `k2_word_lm`. It uses:
- the official LibriSpeech 4-gram HLG, pruned at theta 5.0, with escape resources from
  `LexlatOfficialResourcesJob`;
- max_active 1000;
- `phone_trigram="rampout"`: the phone trigram runs at full weight, then ramps to 0 as the word graph
  ramps in from sub-epoch 8. This is D15's schedule.

**This combination was never run as such, and it has no banked number.**

The other modes of the default arm:
- `phone_trigram="full"` gives `off4_k2lat_20`'s treatment;
- `phone_trigram="off"` leaves no LM term at all before the on-set. It is not a word LM replacing
  the phone LM.

The presets reproduce the banked arms:
- `k2lat_20_ma3000`: in-house word trigram graph, max_active 3000, full phone trigram;
- `off4_k2lat_20`: official 4-gram graph, max_active 1000, full phone trigram.

## Banked numbers (reference cluster)

| run | read | value |
|---|---|---|
| `ctrl_20` | dev-other greedy PER, sub-epoch 20 | 0.874568 |
| `k2lat_20_ma3000` | dev-other greedy PER, sub-epoch 20 | 0.818615 |
| `ctrl_20_x60` | dev-other greedy PER, sub-epoch 60 | 0.873490 |
| `k2lat_20_ma3000_x60` | dev-other greedy PER, sub-epoch 60 | 0.823991 |
| supervised init (gold phi) | held-out NLL per frame (`dev_loss_nll_per_frame`), epoch 8 | 3.2888 |

The banked x60 numbers come from a 20-sub-epoch run resumed for 40 more sub-epochs. The port instead
runs one 60-sub-epoch job with the concatenated schedules. The config content is the same, but the
trajectory is not bit-identical.

## Analysis-only pieces (they use transcripts)

These pieces never initialise, select or gate a main-line arm:
- `config/supervised_init.py`: the gold phi and p0;
- `inputs.get_seed_inputs()`: the 10 h seed's MFA gold, its split and its targets;
- `reverse_model/supervised*.py` and `reverse_model/p0*.py`;
- `reverse_model/ladder.py` and `config/lexlat_v2.ladder()`: the L2-0 competence ladder.

The dev MFA gold (`GoldPhonesJob`) is used only as a PER reference.

## Open values (left undecided on purpose)

1. `reverse_model/phi_first.WAVE_DURATION_SETTING`: `durinit` or `durfrz`. It comes from the probe
   read (A9).
2. `reverse_model/phi_first.WAVE_NUM_SUBEPOCHS`: a multiple of 4. It comes from the A10 read.
3. phi_c's source checkpoint for the wave. The banked one is `k2lat_20_x60` at sub-epoch 60, an arm
   that has no preset here.

`build_wave` refuses while value 1 or 2 is None. `config/lexlat_v2.wave()` takes all three as required
arguments. `py()` never builds the wave or the ladder.

One more value is open. The L2-0 ladder's random-init phi rung is left as None by `ladder.ladder_phis`,
because no writer for a standalone random-init phi checkpoint is registered.

## Environment pins

The reference env is the conda env `speech_llm` on the reference cluster, on aarch64 with GH200 GPUs.
`env/environment.yml` pins its versions. It contains:
- python 3.11 (3.11.15), torch 2.7.1 with CUDA 12.6 (the conda-forge build), transformers 5.9.0,
  datasets 4.8.5 and huggingface_hub 1.16.1;
- numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, scikit-learn 1.8.0, pandas 3.0.3, pyarrow 24.0.0,
  soundfile 0.13.1, typeguard 4.5.2 and dm-tree 0.1.9;
- **rVADfast 0.0.5**;
- **ffmpeg 7.1.1** (the conda build). `FFMPEG_BINARY` must point to it, or to another build that
  passes the pin check;
- **`black`** (26.3.1), importable and on the job PATH, because i6_core's `ReturnnConfig.write`
  calls it;
- sequitur-g2p 1.0.1668.30, built against SWIG 3.0.12 with `pip install --no-build-isolation`. SWIG
  4.2 and later produce broken wrappers. The PyPI x86_64 wheel of this version was generated by SWIG
  4.2.1, so it must be built from source (`--no-binary sequitur-g2p`);
- k2 at `ec31d2c96e04665eadb276ebcfd436892f74b0aa` (`1.24.4.dev20260921+cuda12.6.torch2.7.1` on
  the reference cluster), and kaldilm 1.15.4.

**RETURNN** is rwth-i6/returnn at `00171dfe252eaabc71f6c1fa5a5a910c11a14788`, plus
`training/returnn_local_fixes.patch`, applied by `CloneGitRepositoryJob(patches=...)`. The patch
carries three fixes:
- the padding-ratio statistic in `engine.py`;
- `weights_only=False` for the optimizer state in `updater.py`, which every resume needs under
  torch 2.6 or later;
- `numpy.frombuffer` in `task_system.py`.

**KenLM** is kpu/kenlm at `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`.

**Recipe commits.** The tests and the config loads ran against:
- **i6_core** `ca161b7d35b5bcc971409ac3e982cdb6d425e69c`. The minimum is
  `25db6956ba1ab67ca947d84cc96017694c269c65` (#575), which adds `KenLMplzJob(discount_fallback=...)`.
  `lm/word_lm.py` passes that argument, so an older i6_core raises TypeError at graph build;
- **sisyphus** `ddcd0281834841873562fecfc8f60c1828504cf5`;
- **RETURNN** `00171dfe252eaabc71f6c1fa5a5a910c11a14788` with `training/returnn_local_fixes.patch`
  applied, as above.

## One env

The package runs in one conda env. `env/install_env.sh <env_prefix>` creates it:
1. the env from `env/environment.yml`: conda-forge, hand-pinned to the reference env's versions.
   It is not a `conda env export`, which does not solve on x86_64;
2. sequitur-g2p from source against SWIG 3.0.12, as above;
3. k2: a prebuilt CUDA wheel from the official index (<https://k2-fsa.github.io/k2/cuda.html>) that
   matches the env's torch, CUDA and python, if one exists. k2-fsa publishes these for linux x86_64
   only. Otherwise the script builds k2 from source with `env/build_k2.sh` (`K2_INSTALL=source`
   forces this);
4. a self-check: the imports of the package and RETURNN, the sequitur wrapper's SWIG version, a small
   k2 CPU op, `black --version` and `ffmpeg -version`.

The cluster-specific values are environment variables, documented at the top of each script:
`CONDA_BIN`, `MODULES` (on JUPITER `"Stages/2025 GCC/13.3.0 CUDA/12 CMake/3.30.3"`), `CUDA_ARCH`
(default and JUPITER: 90), `K2_INSTALL`, `K2_WHEEL_VERSION`, `K2_SRC_DIR` and `MAKE_JOBS`. The
install needs a C++ compiler (sequitur, the kenlm python module) and, for a k2 source build, nvcc and
CMake.

Then set `SAE_PYTHON = "<env_prefix>/bin/python"` and `FFMPEG_BINARY = "<env_prefix>/bin/ffmpeg"` in
`settings.py`. `K2_PYTHON` is needed only if k2 lives in a separate env.

**The k2 build.** The prebuilt wheels are not built from the pinned commit. On 2026-09-24 the
newest torch 2.7.1 wheel, `1.24.4.dev20260625+cuda12.6.torch2.7.1` (cp311), reported git `4ef277c4`,
from a wheel-build branch that forked before the pinned commit. The script prints a warning when the installed k2 is not from
`ec31d2c96e04665eadb276ebcfd436892f74b0aa`; use `K2_INSTALL=source` to build the pinned commit.
`env/build_k2.sh` is the reference cluster's build: k2 at that commit, for CUDA 12, sm_90 and torch
2.7.1, installed with `python setup.py install`, with the build environment of the reference
`k2_env.sh` (`-UTORCH_LIBRARY`, tests and benchmarks off, `-L<env>/lib` for the linker). The
reference build reached this configuration over several attempts in one build directory; a clean
build with it has not been run.

**Why JUPITER has two envs.** There is no aarch64 CUDA wheel of k2, so k2 was built from source into
a conda clone of the main env (`/e/project1/spell/wu24/envs/sae_k2`), to leave the main env that
live experiments use untouched. The two envs differ only by k2, kaldilm and fsspec (2026.2.0 in the
main env, 2026.4.0 in the k2 env). So on JUPITER `settings.py` must set `K2_PYTHON` to
`/e/project1/spell/wu24/envs/sae_k2/bin/python`.

The k2 training arms run RETURNN under the k2 python, and the HLG builds run their child scripts
under it too. Because of those child scripts, the package `__init__.py` files must stay import-free.

## Reproducibility limits

- **GPU nondeterminism.** Training is not bit-reproducible on GPU. On this bed, two runs of one
  config differ by a measured 0.01 to 0.03 PER by sub-epoch 4. Compare a reproduction against that
  spread, not against equality.
- **The audio depends on the pinned ffmpeg.**
  - With ffmpeg 7.1.1 and i6's encode, the Ogg audio was sample-identical to the banked audio on 16
    checked dev-other utterances.
  - A zip encoded by another ffmpeg differed on most dev-other utterances (median 62 dB SNR). That
    changes the features without any hash change, so `FfmpegPinCheckJob` stops such a build before
    any encode. It checks every dev-other recording, but not train-clean-100.
  - The rVAD job checks the banked totals (`data.vad.BANKED_VAD_COUNTS`) and raises on a mismatch.
    train-clean-100 itself has not been re-encoded and compared.
- **Features.** The port's wav2vec 2.0 forward was checked on CPU against the banked GPU dump. It
  matched to the CPU-vs-GPU floor (relative Frobenius difference around 1e-3), not bit for bit.
- **E60.** The x60 runs are single 60-sub-epoch jobs; the banked E60 runs were resumed from
  sub-epoch 20.
