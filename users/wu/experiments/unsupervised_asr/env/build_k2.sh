#!/usr/bin/env bash
# Build k2 from source at the reference commit and install it into an existing conda env.
#
# Usage: build_k2.sh <env_prefix>
#
# This is the reference cluster's (JUPITER) build of k2 1.24.4.dev20260921+cuda12.6.torch2.7.1,
# from the package README: k2 ec31d2c96e04665eadb276ebcfd436892f74b0aa for CUDA 12, sm_90 and torch
# 2.7.1, installed with `python setup.py install`.  install_env.sh calls it when no prebuilt CUDA
# wheel fits (always on aarch64: k2-fsa publishes linux x86_64 CUDA wheels only).  The env must
# already hold torch.
#
# Settings (environment variables; the JUPITER values are the examples):
#   CUDA_ARCH   GPU compute capability without the dot. Default: 90. JUPITER (GH200): 90
#   MODULES     modules to load first (`module --force purge; module load $MODULES`); they must
#               provide gcc/g++, CUDA (nvcc and CUDA_HOME) and CMake. Default: empty (the toolchain
#               is taken from PATH; CUDA_HOME defaults to the parent of `nvcc`'s bin/).
#               JUPITER: "Stages/2025 GCC/13.3.0 CUDA/12 CMake/3.30.3"
#   K2_SRC_DIR  where k2 is cloned and built. Default: ./k2_src
#   MAKE_JOBS   parallel make jobs. Default: 8 (the reference build)
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <env_prefix>" >&2
    exit 2
fi
K2_ENV="$1"
CUDA_ARCH="${CUDA_ARCH:-90}"
MODULES="${MODULES:-}"
K2_SRC_DIR="${K2_SRC_DIR:-$PWD/k2_src}"
MAKE_JOBS="${MAKE_JOBS:-8}"
K2_COMMIT="ec31d2c96e04665eadb276ebcfd436892f74b0aa"

die() { echo "build_k2.sh: $*" >&2; exit 1; }

[[ -x "$K2_ENV/bin/python" ]] || die "$K2_ENV/bin/python not found"
[[ "$CUDA_ARCH" =~ ^[0-9]{2,3}$ ]] || die "CUDA_ARCH must be like 90 or 120, got '$CUDA_ARCH'"
# 90 -> 9.0, 120 -> 12.0
CUDA_ARCH_DOT="${CUDA_ARCH%?}.${CUDA_ARCH: -1}"

if [[ -n "$MODULES" ]]; then
    # Lmod's shell function is not written for `set -u`
    set +u
    module --force purge
    # shellcheck disable=SC2086  # MODULES is a word list
    module load $MODULES
    set -u
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    nvcc_path="$(command -v nvcc)" || die "no nvcc on PATH and CUDA_HOME unset"
    CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
fi
export CUDA_HOME

# ---- build environment (the reference k2_env.sh) ----------------------------------------------
CC="$(command -v gcc)"
CXX="$(command -v g++)"
export CC CXX
export CUDACXX="$CUDA_HOME/bin/nvcc" CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_DOT"
export MAKEFLAGS="-j$MAKE_JOBS" K2_MAKE_ARGS="-j$MAKE_JOBS"
export PATH="$K2_ENV/bin:$PATH"
# -L$K2_ENV/lib: link against the env's libstdc++ (the env's libnccl needs CXXABI_1.3.15, newer than
# the reference GCC 13.3 toolchain)
export K2_CMAKE_ARGS="-UTORCH_LIBRARY -DCMAKE_BUILD_TYPE=Release \
 -DK2_ENABLE_TESTS=OFF -DK2_ENABLE_BENCHMARK=OFF -DK2_BUILD_FOR_ALL_ARCHS=OFF \
 -DCUDA_GPU_DETECT_OUTPUT=$CUDA_ARCH_DOT -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
 -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=$CXX \
 -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
 -DPYTHON_EXECUTABLE=$K2_ENV/bin/python \
 -DCMAKE_EXE_LINKER_FLAGS=-L$K2_ENV/lib -DCMAKE_SHARED_LINKER_FLAGS=-L$K2_ENV/lib"

# ---- source at the pinned commit ---------------------------------------------------------------
if [[ ! -d "$K2_SRC_DIR/.git" ]]; then
    git clone https://github.com/k2-fsa/k2 "$K2_SRC_DIR"
fi
git -C "$K2_SRC_DIR" checkout "$K2_COMMIT"
[[ "$(git -C "$K2_SRC_DIR" rev-parse HEAD)" == "$K2_COMMIT" ]] || die "k2 checkout is not $K2_COMMIT"

# ---- build and install into the env ------------------------------------------------------------
echo "== building k2 $K2_COMMIT for sm_$CUDA_ARCH into $K2_ENV (CUDA_HOME=$CUDA_HOME)"
(cd "$K2_SRC_DIR" && "$K2_ENV/bin/python" setup.py install)

(cd "$(mktemp -d)" && "$K2_ENV/bin/python" -c '
import k2
from k2.version import __git_sha1__
print("k2", k2.__dev_version__, "| git", __git_sha1__, "| with_cuda", k2.with_cuda)')
