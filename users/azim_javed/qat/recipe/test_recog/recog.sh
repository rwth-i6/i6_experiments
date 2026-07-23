#!/usr/bin/env bash
set -ueo pipefail

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

apptainer exec -B /work/asr4/berger -B /u/berger --env PYTHONNOUSERSITE=1 /work/asr4/berger/apptainer/images/torch-2.8_onnx-1.22.sif python3 recog.py
