#!/usr/bin/env bash
# Run from training/qat on a GPU node. Optional argument: w4, w8, or both.
set -euo pipefail
PYTHON=/work/smt4/azim.javed/envs/base_env/bin/python
ROOT=/work/smt4/azim.javed/sisyphus_work/i6_core/tools/git/CloneGitRepositoryJob.pxBD5qpvkBp6/output/synaptogen_ml
CASE=${1:-both}
if [[ "$CASE" != w4 && "$CASE" != w8 && "$CASE" != both ]]; then
    printf 'Usage: bash diagnostics/run_joiner_conversion_gpu.sh [w4|w8|both]\n' >&2
    exit 2
fi
if [[ "$CASE" == w4 || "$CASE" == both ]]; then
    "$PYTHON" -u diagnostics/probe_joiner_conversion.py \
        --synaptogen-root "$ROOT" \
        --digital /work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.rvKceYv76piK/output/models/epoch.2000.pt \
        --converted alias/memristor_conversion/qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_memristor_num_cycles_1/output/memristor_converted_model.pt \
        --bits 4 --device cuda | tee diagnostics/joiner_conversion_w4_gpu.log
fi
if [[ "$CASE" == w8 || "$CASE" == both ]]; then
    "$PYTHON" -u diagnostics/probe_joiner_conversion.py \
        --synaptogen-root "$ROOT" \
        --digital /work/smt4/azim.javed/sisyphus_work/i6_core/returnn/training/ReturnnTrainingJob.s9sXAl53j9fG/output/models/epoch.2000.pt \
        --converted alias/memristor_conversion/qat_ffnn_transducer_full_quant_v2_lowbs_memristor_num_cycles_1/output/memristor_converted_model.pt \
        --bits 8 --device cuda | tee diagnostics/joiner_conversion_w8_gpu.log
fi
