#!/bin/bash
set -euo pipefail

if [ -z "${1-}" ] || [ -z "${2-}" ] || [ -z "${3-}" ]; then
    echo "Usage: $0 <agent_name> <sim_cuda> <ad_cuda> [scenario_order] [split]"
    echo "  scenario_order: forward (default) or reverse"
    echo "  split:          public (default) or private"
    exit 1
fi

AGENT=${1}
sim_cuda=${2}
ad_cuda=${3}
scenario_order=${4:-forward}
split=${5:-public}

for DATASET in nuscenes kitti360 pandaset waymo; do
    ./run_dataset.sh "${AGENT}" "${DATASET}" "${sim_cuda}" "${ad_cuda}" "${scenario_order}" "${split}"
done
