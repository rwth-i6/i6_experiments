#!/usr/bin/env bash
set -euo pipefail

IMAGE="/work/asr4/hwu/apptainer/images/u22_pytorch2.2_onnx_flashlight_0402v1_librasr.sif"
LM_UTIL="/u/hwu/repositories/rasr/src/Tools/Lm/lm-util.linux-x86_64-standard"

if [[ -x /bin/apptainer ]]; then
  APPTAINER_BIN="/bin/apptainer"
elif [[ -x /usr/bin/apptainer ]]; then
  APPTAINER_BIN="/usr/bin/apptainer"
else
  echo "Could not find apptainer binary" >&2
  exit 127
fi

exec "$APPTAINER_BIN" exec \
  -B /u/hwu \
  -B /work/asr4 \
  -B /tmp \
  -B /var/tmp \
  "$IMAGE" \
  "$LM_UTIL" \
  "$@"
