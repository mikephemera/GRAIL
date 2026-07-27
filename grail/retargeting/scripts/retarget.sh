#!/bin/bash
# Retarget GRAIL HOI motions to the G1 robot using GMR.
#
# Usage:  bash grail/retargeting/scripts/retarget.sh <data_dir> <output_folder> [--zero_out_wrist]
# Example: bash grail/retargeting/scripts/retarget.sh data/genhoi/benchmark_v3 benchmark_v3_0203
#
# Extra args after <output_folder> are forwarded to grail.retargeting.retarget (e.g., --zero_out_wrist
# for terrain/sitting data, --no_g1_proportions for non-G1-proportioned SMPLX).

set -euo pipefail

DATA_DIR="${1:?data directory (e.g. data/genhoi/<dataset>/generation/4dhoi_recon_valid/Hunyuan)}"
OUTPUT_FOLDER="${2:?output folder name under data/motion_lib/}"
shift 2

OUTPUT_BASE="data/motion_lib/${OUTPUT_FOLDER}"

# Activate the retargeting env when conda is available. Retarget-only Docker
# images may install into the active Python instead; in that case keep going.
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/root/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
fi

if command -v conda &>/dev/null; then
    conda activate "${GRAIL_SONIC_ENV:-sonic}"
else
    echo ">>> [skip conda] using active Python: $(python -c 'import sys; print(sys.executable)')"
fi

# GMR opens a mujoco viewer — ensure DISPLAY is set for headless runs.
export DISPLAY="${DISPLAY:-:1}"

python -m grail.retargeting.retarget \
    --data_dir "${DATA_DIR}" \
    --all \
    --robot unitree_g1 \
    --output_dir "${OUTPUT_BASE}" \
    --no_viewer \
    "$@"

echo "Retarget output: ${OUTPUT_BASE}/{robot,objects,object_usd,meta}/"
