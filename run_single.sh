#!/bin/bash
set -euo pipefail

DEFAULT_SCENARIO=/home/yyin5/mack/datasets_mack_raw/HUGSIM-public/ss/scenarios/kitti360/scene-10320_10520-easy-00.yaml

if [ -z "${1-}" ]; then
    echo "Usage: $0 <agent_name> [scenario_yaml] [sim_cuda] [ad_cuda]"
    echo "  scenario_yaml defaults to ${DEFAULT_SCENARIO}"
    exit 1
fi
AGENT=${1}
SCENARIO=${2:-${DEFAULT_SCENARIO}}
sim_cuda=${3:-0}
ad_cuda=${4:-0}

# .../ss/scenarios/<dataset>/<scene>.yaml -> <dataset>, which selects the base/camera configs
DATASET=$(basename "$(dirname "${SCENARIO}")")

echo "→ ${AGENT} on ${DATASET}: $(basename "${SCENARIO}")"

CUDA_VISIBLE_DEVICES=${sim_cuda} \
python closed_loop.py --scenario_path "${SCENARIO}" \
            --base_path "./configs/sim/${DATASET}_base.yaml" \
            --camera_path "./configs/sim/${DATASET}_camera.yaml" \
            --kinematic_path "./configs/sim/kinematic.yaml" \
            --ad "${AGENT}" \
            --ad_cuda ${ad_cuda}
