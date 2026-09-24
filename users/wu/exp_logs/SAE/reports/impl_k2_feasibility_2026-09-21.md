# k2 (k2-fsa) GPU feasibility on JUPITER GH200 / aarch64 — 2026-09-21

Feasibility check only. Nothing installed, nothing built, no Slurm job, no `squeue`.
Only artifact written outside this file: a 2.6 MB wheel downloaded into the session
scratchpad for inspection (not installed).

## 0. Environment facts (measured)

| fact | value | how measured |
|---|---|---|
| platform | `aarch64`, RHEL 9.8, kernel 5.14.0 | `uname -m`, `/etc/os-release` |
| GPU (login) | NVIDIA GH200 480GB, sm_90, driver 595.71.05 (CUDA 13.2 driver API) | `nvidia-smi` |
| training env | `/e/project1/spell/wu24/env/conda/envs/speech_llm` | memory `sae-impl-toolchain` |
| python | 3.11.15 | `python -c` in that env |
| torch | **2.7.1**, `torch.version.cuda` = **12.6**, cudnn 9.10.2 | `python -c "import torch; ..."` |
| torch provenance | conda-forge `pytorch-2.7.1-cuda126_generic_py311`, `libtorch-2.7.1-cuda126_generic` | `conda-meta/` |
| torchaudio | 2.7.1 (conda-forge), **CPU-only build** | see §4 |
| login `nvcc` | none on PATH; env has `cuda-nvcc-tools 12.6.85` (`$ENV/bin/nvcc`, V12.6.85) | `nvcc --version` |
| login `cmake` | **not on PATH**; available as modules (§2) | `cmake --version` |
| login `gcc` | **14.3.0** (GCCcore/14.3.0 in PATH) | `gcc --version` |
| existing k2 | none anywhere (`find / -maxdepth 4 -name 'k2*' -type d` empty; no `k2`/`_k2` in either conda env) | `find` |

## 1. Prebuilt wheels for linux aarch64 — NONE with CUDA

k2's CUDA wheel index (`https://k2-fsa.github.io/k2/cuda.html`, the flat find-links page
behind `.../installation/pre-compiled-cuda-wheels-linux/index.html`) lists **2191 wheel
links; the platform tag is x86_64 in 100% of them**:

```
632  linux_x86_64.whl
2760 manylinux_2_17_x86_64.manylinux2014_x86_64.whl
236  manylinux2014_x86_64.manylinux_2_17_x86_64.whl
754  manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```
(counts are over href occurrences; each wheel appears twice). `grep -i 'aarch64|arm64|sbsa'`
over the whole page returns **0 matches**.

For our torch there are 90 CUDA entries, e.g. the otherwise perfect match
`k2-1.24.4.dev20260625+cuda12.6.torch2.7.1-cp311-cp311-manylinux_2_27_x86_64...whl`
(cuda 11.8 / 12.6 / 12.8 x cp39..cp313) — **x86_64 only**, unusable here.

PyPI `k2`: latest is **1.24.1** (2023-era), files are macOS x86_64 wheels plus
`py3.X-none-any` stubs; no linux wheel, no usable sdist. Dead end.

anaconda.org channel `k2-fsa`: latest `1.24.3.dev20230508`, subdirs `linux-64, osx-64,
win-64` — **no linux-aarch64**, and torch-era 2.0. Dead end. `conda-forge` has no `k2`.

**A CPU aarch64 wheel does exist and matches us exactly** (see §5).

## 2. Source build requirements vs. what this machine has

k2 from-source docs require: C++14+ compiler (GCC >= 7), CMake (>= 3.8 per
`CMakeLists.txt:16`), CUDA toolkit + cuDNN, python3, a torch install. `setup.py` passes
`-DK2_ENABLE_TESTS=OFF -DK2_ENABLE_BENCHMARK=OFF -DCMAKE_BUILD_TYPE=Release` and
`-DCMAKE_CXX_STANDARD=17` for torch 2.1..2.12 (C++20 only from torch 2.13).

| requirement | status on this machine |
|---|---|
| CUDA toolkit matching torch's 12.6 | **YES** — module `CUDA/12` in stage 2025 = `/e/software/default/stages/2025/software/CUDA/12`, `nvcc V12.6.20`, ELF aarch64, `targets/sbsa-linux`, full `include/cuda_runtime.h` + `lib64/libcublas.so`. Same 12.6 minor as `torch.version.cuda`. |
| CMake | **YES** — `CMake/3.29.3-GCCcore-13.3.0`, `3.30.3-GCCcore-13.3.0` (stage 2025), `CMake/3.31.8`, `4.0.3` (stage 2026). Use a 3.x one (k2 declares `cmake_minimum_required(3.8)`, which CMake 4 rejects). |
| host compiler | **CONSTRAINT** — CUDA 12.6 `include/crt/host_config.h:141` = `#if __GNUC__ > 13 ... #error gcc versions later than 13 are not supported`. The default login GCC is **14.3.0**, so a naive build fails. `GCCcore/13.3.0` exists at `/e/software/default/stages/2025/software/GCCcore/13.3.0` -> build with `CC/CXX` and `-DCMAKE_CUDA_HOST_COMPILER` pointed at it. Not a blocker, but must be set explicitly. |
| cuDNN | present in the env (`cudnn 9.10.2`, `libcudnn-dev`) | 
| sm_90 support in k2 | **YES** — `CMakeLists.txt:314-317` appends arch 90 for `CUDA_VERSION >= 11.8`. |
| network for build | outbound https works from the login node (all fetches above succeeded). |

Notes / risks, not blockers:
- The env's own `nvcc` (conda `cuda-nvcc-tools 12.6.85`) is **not sufficient**: the env has
  no `cuda_runtime.h` (neither `include/` nor `targets/sbsa-linux/include/`) and no
  `libcublas.so`/`libcusolver.so` dev links. k2's docs also state outright that "cudatoolkit
  installed by `conda install` cannot be used to compile k2". Use the module toolkit.
- k2's docs recommend a **pip** torch and warn against conda torch; ours is conda-forge
  `libtorch-2.7.1-cuda126_generic`. Linking k2 against conda-forge libtorch on aarch64 is
  unverified — this is the single biggest unknown of the build.
- GPU arch auto-detection on aarch64 has a live bug: k2 issue **#1354** (open) "CUDA
  architecture detection fails for NVIDIA GB10 / sm_121 on ARM aarch64" — `cuda_select_nvcc_arch_flags`
  mis-parses capability `12.1` and emits `arch=compute_20,code=sm_121`. Our sm_90 is in the
  known-arch table, but pass the arch explicitly anyway (`-DK2_BUILD_FOR_ALL_ARCHS=OFF` plus
  an explicit `CUDA_ARCH_LIST`/`TORCH_CUDA_ARCH_LIST=9.0`) rather than relying on auto-detect.
  k2 CI has `ubuntu-arm64-cpu-wheels.yml` (CPU) and `ubuntu-cuda-wheels.yml` (`runs-on: ubuntu-latest`,
  x86) — **no CI covers aarch64+CUDA**, so we would be the first-party tester.

## 3. Is a source build a 1-2 h job in a cloned scratch env? — estimate: YES, ~1 h

Repo census (`git/trees/master?recursive=1`): 122 `.cu` + 43 `.cc`, of which 54 are test
files; **107 non-test non-benchmark translation units** (csrc 54, python 32, torch 21).
Tests and benchmarks are off by default in `setup.py`, so all 107 and only those compile.
Single target arch (sm_90) means one `-gencode` pass per TU. On Grace (72 cores) with
`K2_MAKE_ARGS="-j16"` and heavily templated CUB-based CUDA TUs at ~1-4 min each, the
compile is ~20-45 min, plus cmake configure and link: **~30-60 min wall**, comfortably
inside a 1-2 h budget, with the env clone (`conda create --clone speech_llm -p <scratch>`)
on top. No k2 build logs exist on this system to calibrate against (none found).

Caveats on the budget: it assumes the ABI/link step against conda-forge libtorch works
first try. A mismatch there (or an arch-flag fight) can eat the whole budget; recommend
timeboxing the configure step separately. Also note `$SCRATCH` is **unset** in a
non-login shell here; project scratch lives under `/p/scratch/<project>`, and per the
standing note scratch is a 90-day lease. `conda` and `mamba` are both available at
`/e/project1/spell/wu24/env/conda/bin/`. The env is large (a `du -sh` over it did not
finish in 90 s on GPFS), so budget clone time and inodes; the alternative is a fresh small
env with only `torch==2.7.1` rather than a full clone.

**Do not build on the login node** beyond a configure smoke test; a build with `-j16`
belongs on a compute node (not launched here, per instructions).

## 4. torchaudio fallback — NOT AVAILABLE

`torchaudio 2.7.1` in the env is a **CPU-only** build:
- `torchaudio/lib/` contains only `libtorchaudio.so`, `libtorchaudio_sox.so`,
  `_torchaudio.so`, `_torchaudio_sox.so`.
- `ldd libtorchaudio.so` lists no `libcudart`/`libtorch_cuda`; `nm -D` has **0** undefined
  cuda symbols and **0** `cuda_ctc`/`cuctc` defined symbols.
- Registered op strings: only `torchaudio::forced_align` (CPU kernel). `forced_align` is a
  Viterbi forced alignment, not a pruned lattice and not log-semiring — it cannot stand in
  for `intersect_dense_pruned` + `get_tot_scores(log_semiring=True)`.
- `torchaudio.models.decoder` exposes `ctc_decoder`, `cuda_ctc_decoder`, `CUCTCDecoder` in
  **python only**: `_cuda_ctc_decoder.py:10` does
  `torchaudio._extension._load_lib("libctc_prefix_decoder")` and that shared object is
  absent, and the flashlight lexicon decoder's `_torchaudio_decoder` module is absent too
  (`ModuleNotFoundError`). Both decoders will fail at construction. Neither is a pruned
  lattice/log-semiring tool anyway.

Not present in the env: `flashlight`, `gtn`, `pynini`, `kaldifst`, `kaldilm`, `openfst`, `k2`.

## 5. Fallbacks that DO exist

1. **k2 CPU on aarch64, exact match, install-in-seconds.** k2's CPU index has 972 aarch64
   wheel links, including
   `k2-1.24.4.dev20260625+cpu.torch2.7.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`
   at `https://huggingface.co/csukuangfj2/k2/resolve/main/cpu/1.24.4.dev20260625/linux-arm64/`.
   Verified by download (HTTP 200, 2.62 MB) and inspection, **not installed**:
   `Requires-Dist: torch==2.7.1` (our exact version), `_k2.cpython-311-aarch64-linux-gnu.so`
   with ~0 cuda symbols, `k2/autograd.py` defines `intersect_dense_pruned`, `k2/fsa.py`
   defines `get_tot_scores`. manylinux_2_28 vs RHEL 9.8 glibc 2.34: fine.
   This gives the **API and correctness surface immediately** (in a cloned//scratch env) and a
   CPU timing baseline, but it does **not** answer the GPU timing question.
2. **Offline H.L.G construction, CPU:** conda-forge ships `pynini 2.1.7` and `openfst 1.8.4`
   for **linux-aarch64** — installable into a scratch env. `gtn` is **not** on conda-forge
   (404) and would need a source build (CPU, small).
3. If k2 CPU is enough for the graph side, `kaldifst`/`kaldilm` (icefall's usual HLG tools)
   were not checked for aarch64 wheels; worth a look before hand-rolling with pynini.

## 6. Bottom line

- GPU k2 **cannot** be obtained from any prebuilt artifact on this architecture.
- A CUDA source build is **plausible and un-blocked**: every requirement (CUDA 12.6 aarch64
  toolkit with headers, CMake 3.29/3.30, GCC 13.3, sm_90 in k2's arch table) is present on
  this machine. The one thing that must be set by hand is the host compiler (GCC <= 13,
  default is 14.3.0), and the one real unknown is linking against conda-forge libtorch.
- Estimated ~30-60 min build (107 TUs, single arch, tests off) => a 1-2 h slot including the
  scratch env clone is realistic, with the caveat above.
