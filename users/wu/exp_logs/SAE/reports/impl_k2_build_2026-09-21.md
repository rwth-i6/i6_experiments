# k2 (k2-fsa) CUDA source build on JUPITER GH200 / aarch64 — 2026-09-21

Follow-up to `impl_k2_feasibility_2026-09-21.md`. **Result: SUCCESS.** k2 with CUDA is
built and installed into a scratch clone of the training env. No Slurm job, no `squeue`,
no background process; everything ran in the foreground on the login node.

## 0. Artifacts

| artifact | path |
|---|---|
| cloned conda env | `/e/scratch/spell/wu24/envs/sae_k2` (9.9 GB) |
| build env script | `/e/scratch/spell/wu24/envs/k2_env.sh` |
| k2 source tree | `/e/scratch/spell/wu24/envs/k2_src` (237 MB incl. `build/`) |
| build log | `/e/scratch/spell/wu24/envs/k2_build.log` |
| env clone log | `/e/scratch/spell/wu24/envs/clone.log` |
| CPU check (run, passed) | `/e/scratch/spell/wu24/envs/k2_cpu_check.py` |
| **GPU check (for the executor)** | `/e/scratch/spell/wu24/envs/k2_gpu_check.py` |

`$SCRATCH` is not exported on this host; the per-project variable is
`SCRATCH_spell=/e/scratch/spell`, so "`$SCRATCH/envs`" was resolved to
`/e/scratch/spell/wu24/envs`. **Scratch is a 90-day lease** — and `_k2*.so` carries an
*absolute* RPATH into `/e/scratch/spell/wu24/envs/sae_k2`, so the env cannot simply be
moved later; a relocation means re-running the build (~9 min) at the new prefix.

## 1. Environment clone (conda, not venv)

The training env `/e/project1/spell/wu24/env/conda/envs/speech_llm` is a **conda** env
(356 `conda-meta/*.json`, no `pyvenv.cfg`).

```
conda create --yes --clone /e/project1/spell/wu24/env/conda/envs/speech_llm \
             -p /e/scratch/spell/wu24/envs/sae_k2
```
took **106 s** (09:51:28 → 09:53:14). Verified faithful: `torch 2.7.1 / cuda 12.6 /
cudnn 9.10.2`, python 3.11.15, and `numpy, transformers, sentencepiece, sklearn,
datasets, peft` all import in both the source env and the clone (`returnn` is absent
in **both** — it comes from `PYTHONPATH` at job time, not from the env).

### Blocker hit: conda 26.3.2 cannot clone an env containing an *epoch* package

The first two attempts died with
```
SpecNotFoundInPackageCache: Missing package cache records for:
  conda-forge/linux-aarch64::x264==1!164.3095=h4e544f5_2
```
Root cause (conda 26.3.2 bug, `conda/misc.py::_get_package_record_from_specs`): the
MatchSpec built from the package URL carries the **un-quoted** filename
`x264-1!164.3095-h4e544f5_2.tar.bz2`, while the package-cache record carries the
**URL-quoted** name `x264-1%21164.3095-h4e544f5_2.tar.bz2`. Every other field matches;
only `fn` fails, so no package with an epoch (`1!`) in its version can ever be resolved.

Work-around (touches **only** scratch, never the shared env or its cache): a private
package-cache dir `/e/scratch/spell/wu24/envs/pkgs` holding a copy of the x264 package
under the un-quoted name, with `fn` in `info/repodata_record.json` rewritten to match,
plus an empty `urls.txt` (conda's cache "magic file" — without it the dir is treated as
non-existent and is skipped by `PackageCacheData.query_all`). Then
`CONDA_PKGS_DIRS='/e/scratch/spell/wu24/envs/pkgs,/e/project1/spell/wu24/env/conda/pkgs'`
(**comma**-separated; a colon-separated value is silently taken as one path name).

Shared env prefix verified untouched afterwards (`find … -newermt "2026-09-21 09:00"`
returns nothing). The shared *package cache* was read during the clone, which refreshed
some directory mtimes there; no package was added or removed.

## 2. Toolchain

`module` is an exported bash function and works in non-interactive shells, **but a piped
`module load` runs in a subshell and silently has no effect**, and `Stages/2025` must be
loaded in its own `module load` before the modules it exposes are visible.

```
module --force purge
module load Stages/2025          # deprecated-stage warning is expected
module load GCC/13.3.0           # -> GCCcore/.13.3.0 (hidden module), gcc/g++ 13.3.0
module load CUDA/12              # nvcc V12.6.20, CUDA_HOME=/e/software/default/stages/2025/software/CUDA/12
module load CMake/3.30.3         # CMake 3.30.3 (CMake 4 would reject k2's cmake_minimum_required(3.8))
```
`make` resolves to `/usr/bin/make` (GNU Make 4.3) after the purge — fine.

`libstdc++` ABI census (decisive later): GCCcore/13.3.0 tops out at `GLIBCXX_3.4.32` /
`CXXABI_1.3.14`; the conda env ships `libstdc++.so.6.0.34` with `GLIBCXX_3.4.34` /
`CXXABI_1.3.15`; `libtorch_cpu.so` needs at most `GLIBCXX_3.4.32`.

Full, reproducible environment: `/e/scratch/spell/wu24/envs/k2_env.sh`. Its
`K2_CMAKE_ARGS` is
```
-UTORCH_LIBRARY -DCMAKE_BUILD_TYPE=Release
-DK2_ENABLE_TESTS=OFF -DK2_ENABLE_BENCHMARK=OFF
-DK2_BUILD_FOR_ALL_ARCHS=OFF -DCUDA_GPU_DETECT_OUTPUT=9.0 -DCMAKE_CUDA_ARCHITECTURES=90
-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=<GCCcore 13.3.0 g++>
-DCMAKE_C_COMPILER=<GCCcore 13.3.0 gcc> -DCMAKE_CXX_COMPILER=<GCCcore 13.3.0 g++>
-DPYTHON_EXECUTABLE=/e/scratch/spell/wu24/envs/sae_k2/bin/python
-DCMAKE_EXE_LINKER_FLAGS=-L/e/scratch/spell/wu24/envs/sae_k2/lib
-DCMAKE_SHARED_LINKER_FLAGS=-L/e/scratch/spell/wu24/envs/sae_k2/lib
```
with `CUDACXX`/`CMAKE_CUDA_COMPILER`, `CMAKE_CUDA_ARCHITECTURES=90`,
`TORCH_CUDA_ARCH_LIST=9.0`, `MAKEFLAGS=-j8`, `K2_MAKE_ARGS=-j8` also exported.

**Arch pinned explicitly** (feasibility note on k2 issue #1354): k2 calls
`cuda_select_nvcc_arch_flags()` with no argument, which falls through to `Auto` and
probes the local GPU. `CUDA_DETECT_INSTALLED_GPUS` honours a pre-set
`CUDA_GPU_DETECT_OUTPUT` cache entry, so `-DCUDA_GPU_DETECT_OUTPUT=9.0` short-circuits
the probe. (`-DCMAKE_CUDA_ARCHITECTURES=90` alone is *not* enough — `CMakeLists.txt:368`
overwrites `CMAKE_CUDA_ARCHITECTURES` from its own arch list.) The value 9.0/90 is the
sm_90 from the dispatch/feasibility report. Confirmed in the log — the only gencode
emitted is
`-gencode arch=compute_90,code=sm_90`, and `K2_COMPUTE_ARCHS: 90`.

## 3. k2 version selected

**No k2 *tag* supports torch 2.7.** The newest tag, `v1.24.4`
(`9bad91bb5522471adf23ce025807cdb00a3008f3`), is from **2023-09-26**, i.e. the torch-2.1
era. Support for torch 2.7 exists only on `master`, which is also where the
`+cpu.torch2.7.1` aarch64 wheels inspected in the feasibility report come from.

Built at **master HEAD `ec31d2c96e04665eadb276ebcfd436892f74b0aa`** (2026-07-10,
"Provide pre-built wheels for PyTorch 2.13.0"). Justification from the source's own
version tables: `scripts/github_actions/generate_build_matrix.py:325` lists torch
`2.7.1` with `cuda ["11.8", "12.6", "12.8"]` (default 12.6) and python 3.9–3.13 — our
exact combination — and `setup.py:153-158` selects `-DCMAKE_CXX_STANDARD=17` for
`2.1 <= torch < 2.13` (C++20 only from 2.13).

Resulting package version: **`1.24.4.dev20260921+cuda12.6.torch2.7.1`**
(`k2.version.__version__ == "1.24.4"`, the dev suffix is generated from the build date).

## 4. Build

`python setup.py install` (it `cd`s into `build/temp.linux-aarch64-cpython-311`, runs
`cmake` then `make -j8 install`, so it is **incremental and resumable** — a `timeout`-killed
run simply continues). Configure downloads `moderngpu`/`cub`/`pybind11`; the login node
has outbound https.

Wall time, in `timeout 560` foreground chunks:

| step | wall | outcome |
|---|---|---|
| attempt 1 | 441 s (09:57:15 → 10:04:36) | configure + compile to 100 % of `_k2`; **failed** linking `k2/torch/bin/*` |
| attempt 2 | 47 s | reconfigure failure (see fix 2) |
| attempt 3 | 35 s | relink + install, **exit 0** |
| **total build** | **≈ 523 s (8.7 min)** | |

Env clone (106 s) + `git clone` of k2 on top; the whole task fitted well inside the 2 h
budget, and inside the feasibility estimate of 30–60 min.

### Fix 1 (the anticipated libtorch-linkage failure — it is really a libstdc++ failure)

```
ld: /e/scratch/spell/wu24/envs/sae_k2/lib/libnccl.so.2:
    undefined reference to `__cxa_call_terminate@CXXABI_1.3.15'
```
on the seven C++ demo executables (`ctc_decode`, `hlg_decode`, `ngram_lm_rescore`,
`attention_rescore`, `online_decode`, `rnnt_demo`, `pruned_stateless_transducer`).
`libk2_torch.so`, `libk2_torch_api.so` and `_k2` itself linked cleanly — shared-library
links do not have to resolve a dependency's own undefined symbols, executables do. The
conda-forge `libnccl 2.30.4` was built with GCC 14 and needs `CXXABI_1.3.15`, which the
GCCcore/13.3.0 `libstdc++` does not provide.

Fix: put the conda env's own (newer) `libstdc++.so.6.0.34` on the link search path —
`-DCMAKE_EXE_LINKER_FLAGS=-L<env>/lib` (and the same for `CMAKE_SHARED_LINKER_FLAGS`, so
every link resolves libstdc++ from the same env that hosts k2 at run time). Verified in
isolation by re-running the failing `ctc_decode` link command with the single extra `-L`
before changing the build. This is safe in the other direction: objects compiled by GCC
13.3 require at most `GLIBCXX_3.4.32`, which the env's 6.0.34 provides.

**So the feasibility report's "single biggest unknown", linking k2 against conda-forge
libtorch on aarch64, is resolved: it works.** The only ABI friction came from NCCL, and
one `-L` settles it.

### Fix 2 (k2 cannot be re-configured in place)

The second `cmake` run in an existing build dir fails with
```
CMake Error at cmake/torch.cmake:119 (set_property):
  set_property could not find TARGET torch_cuda.
```
`cmake/torch.cmake:15` guards `find_package(Torch REQUIRED)` with
`if(NOT DEFINED TORCH_LIBRARY)`, but `TORCH_LIBRARY` is a **cache** variable written by
the first `find_package`, so on every later configure `find_package` is skipped and the
`torch_cuda`/`torch_cpu` imported targets do not exist. This is upstream behaviour,
independent of our environment, and it breaks *any* rebuild/resume.

Fix: prepend **`-UTORCH_LIBRARY`** to `K2_CMAKE_ARGS`, which drops that cache entry
before each configure and lets `find_package(Torch)` re-create the targets. With this in
place `python setup.py install` can be re-run any number of times, which is what makes
the timeboxed-chunk strategy work.

No k2 source file was patched. (`setup.py` does append a `__dev_version__` line to
`k2/python/k2/__init__.py` on every invocation; a pristine copy is kept at
`/e/scratch/spell/wu24/envs/k2_init_pristine.py` and was restored before each run.)

## 5. Verification

### Import + CUDA flag (login node, no modules loaded)
```
$ /e/scratch/spell/wu24/envs/sae_k2/bin/python -c "import k2, torch; print(k2.__dev_version__, k2.with_cuda)"
1.24.4.dev20260921+cuda12.6.torch2.7.1 True
```
Note `k2.__version__` does **not** exist in this version (it is `k2.__dev_version__`, and
`k2.version.__version__ == "1.24.4"`).

### CPU functional check — PASSED
`/e/scratch/spell/wu24/envs/k2_cpu_check.py`: 3 real states (0,1,2) + final state 3, blank
self-loops and labels 1,2; `k2.create_fsa_vec` → `k2.arc_sort` → `k2.DenseFsaVec` over a
`(1, 6, 3)` log-softmax tensor with `requires_grad=True` → `k2.intersect_dense_pruned`
(`search_beam=20, output_beam=8, min_active_states=30, max_active_states=10000`) →
`get_tot_scores(log_semiring=True)` → `backward()`:
```
tot_scores: [-5.2523875]
grad norm: 2.0919947624206543
CPU CHECK OK  k2 1.24.4.dev20260921+cuda12.6.torch2.7.1 with_cuda True
```
Both the score and the gradient are finite and the gradient is non-zero (asserted).
A first draft without the blank self-loops returned `-inf`/zero gradient, because the
graph then admits no 6-frame path — a reminder that an empty intersection is silent.

### Runtime dependency resolution
`readelf -d _k2.cpython-311-aarch64-linux-gnu.so` →
`RPATH=[$ORIGIN:<env>/site-packages/torch/lib:<env>/lib:$ORIGIN/k2/lib64]`, and
`env -i PATH=/usr/bin ldd …` shows **no** unresolved library: `libcudart.so.12`,
`libnvrtc.so.12`, `libcublas(Lt).so.12`, `libcusparse.so.12`, `libcurand.so.10`,
`libnccl.so.2` all resolve from the cloned env. **No `module load` and no
`LD_LIBRARY_PATH` are needed at run time** — which matters because the sisyphus worker
env only has `PATH=conda:/usr/bin`.

### GPU check — NOT run here (belongs in a Slurm job)
`/e/scratch/spell/wu24/envs/k2_gpu_check.py` is the same graph on `cuda:0`; it prints the
device name, `tot_scores`, the gradient norm, and asserts finiteness/non-zero gradient.
Run it as
```
/e/scratch/spell/wu24/envs/sae_k2/bin/python /e/scratch/spell/wu24/envs/k2_gpu_check.py
```
It only `py_compile`s here. **A clean CPU import and a clean build prove nothing about
sm_90 kernels actually executing**; the login-node result above is a CPU result.

## 6. Open items for the caller

- The GPU check is unrun. Until an executor runs it inside a GPU job, "k2 works on
  GH200" is unverified.
- k2 has **no CI covering aarch64 + CUDA** (feasibility report §2), so this build is
  self-tested only; consider running k2's own python test-suite on GPU before funding
  anything on top of it (it was built with `K2_ENABLE_TESTS=OFF`, so that would be a
  separate build).
- Nothing was committed: this round changed no file under `recipe/`, only scratch
  artifacts and this report, and no repository/branch was named in the dispatch.
